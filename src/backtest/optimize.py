"""
Optuna Hyperparameter-Optimierung — Bayessche Suche mit Walk-Forward-CV.

Jede Objective-Funktion:
1. Sampelt Hyperparameter via trial.suggest_*()
2. Iteriert über (subsampled) Walk-Forward-Folds
3. Trainiert das Modell pro Fold auf dem Train-Window
4. Berechnet annualisierten OOS-Sharpe aus den Portfolio-Returns
5. Gibt den medianen OOS-Sharpe über alle Folds zurück (Optuna maximiert)

Kein Look-Ahead-Bias: Optuna sieht ausschließlich OOS-Metriken.
Jeder Fold trainiert unabhängig. Die Fold-Splits sind identisch zu
denen im finalen Walk-Forward-Lauf (walk_forward.py).
"""

import warnings
import numpy as np
import pandas as pd
import optuna
import os

from src.data.labels.resolver import compute_supervised_labels, resolve_label_col
from src.backtest.walk_forward import walk_forward_splits

# ============================================================================
# Helper-Funktionen
# ============================================================================

def _compute_oos_sharpe(
    daily_returns: np.ndarray,
    trading_days_per_year: int = 252,
) -> float:
    """Annualisierter Sharpe Ratio aus täglichen Netto-Returns."""
    if len(daily_returns) < 20:
        return -999.0
    std = np.std(daily_returns)
    if std == 0:
        return 0.0
    return float((np.mean(daily_returns) / std) * np.sqrt(trading_days_per_year))


def _fold_portfolio_returns(
    df_test: pd.DataFrame,
    signal: pd.Series,
    signal_shift: int = 1,
    fee: float = 0.001,
) -> np.ndarray:
    """
    Berechnet tägliche Netto-Portfolio-Returns für einen OOS-Fold.

    Repliziert die Logik aus engine.backtest():
    - Signal um signal_shift Tage verschieben (Look-Ahead-Vermeidung)
    - Signal=0 → Portfolio-Return, Signal=1 → Cash-Return
    - Transaktionskosten bei Signalwechseln abziehen
    """
    trading_signal = signal.shift(signal_shift).fillna(0)
    trades = trading_signal.diff().fillna(0).abs()

    strategy_returns = np.where(
        trading_signal == 0,
        df_test["Returns"].values,
        df_test["Cash_Returns"].values,
    )
    return strategy_returns - (trades.values * fee)


def _subsample_splits(
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    every_nth: int | None,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Jede n-te Fold auswählen für schnellere Optimierung."""
    if every_nth is None or every_nth <= 1:
        return splits
    return splits[::every_nth]


def _generate_hmm_labels(df_train, df_test, cfg):
    from src.models.hmm import train_hmm_fold

    hmm_cfg = cfg.models.hmm
    probs, signal_test, signal_train = train_hmm_fold(
        features_df_train=df_train[hmm_cfg.features],
        features_df_test=df_test[hmm_cfg.features],
        returns_train=df_train["Returns"],
        n_components=hmm_cfg.n_components,
        covariance_type=hmm_cfg.covariance_type,
        n_iter=hmm_cfg.n_iter,
        random_state=hmm_cfg.random_state,
        threshold=hmm_cfg.threshold,
    )

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["HMM_Signal"] = signal_train.values
    df_test["HMM_Signal"] = signal_test.values
    return df_train, df_test


# ============================================================================
# Objective-Funktionen pro Modell
# ============================================================================

def objective_msm(
    trial: optuna.Trial,
    df: pd.DataFrame,
    splits: list,
    fee: float,
    signal_shift: int,
) -> float:
    """MSM: k_regimes und Threshold optimieren."""
    from src.models.msm import train_msm_fold

    k_regimes = 2
    threshold = trial.suggest_float("threshold", 0.3, 0.7, step=0.05)

    fold_sharpes = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        try:
            df_train = df.loc[train_idx]
            df_test = df.loc[test_idx]

            probs, signal, _ = train_msm_fold(
                returns_train=df_train["Returns"],
                returns_test=df_test["Returns"],
                k_regimes=k_regimes,
                switching_variance=True,
                threshold=threshold,
            )

            oos_rets = _fold_portfolio_returns(df_test, signal, signal_shift, fee)
            fold_sharpes.append(_compute_oos_sharpe(oos_rets))
        except Exception as e:
            warnings.warn(f"MSM Trial {trial.number}, Fold {fold_id}: {e}")
            fold_sharpes.append(-999.0)

        # Pruning: Zwischenergebnis melden
        trial.report(np.median(fold_sharpes), fold_id)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.median(fold_sharpes))


def objective_hmm(
    trial: optuna.Trial,
    df: pd.DataFrame,
    splits: list,
    cfg,
    fee: float,
    signal_shift: int,
) -> float:
    """HMM: n_components, covariance_type und Threshold optimieren."""
    from src.models.hmm import train_hmm_fold

    n_components = 2
    covariance_type = trial.suggest_categorical(
        "covariance_type", ["full", "diag", "tied"],
    )
    threshold = trial.suggest_float("threshold", 0.3, 0.7, step=0.05)

    hmm_features = cfg.models.hmm.features

    fold_sharpes = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        try:
            df_train = df.loc[train_idx]
            df_test = df.loc[test_idx]

            probs, signal, _ = train_hmm_fold(
                features_df_train=df_train[hmm_features],
                features_df_test=df_test[hmm_features],
                returns_train=df_train["Returns"],
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=cfg.models.hmm.n_iter,
                random_state=cfg.models.hmm.random_state,
                threshold=threshold,
            )

            oos_rets = _fold_portfolio_returns(df_test, signal, signal_shift, fee)
            fold_sharpes.append(_compute_oos_sharpe(oos_rets))
        except Exception as e:
            warnings.warn(f"HMM Trial {trial.number}, Fold {fold_id}: {e}")
            fold_sharpes.append(-999.0)

        trial.report(np.median(fold_sharpes), fold_id)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.median(fold_sharpes))


def objective_lstm(
    trial: optuna.Trial,
    df: pd.DataFrame,
    splits: list,
    cfg,
    fee: float,
    signal_shift: int,
) -> float:
    """
    LSTM: window_size, units, learning_rate, dropout, epochs optimieren.

    Hinweis: learning_rate wird über ein Keras-Optimizer-Objekt an
    train_lstm_fold übergeben. Keras model.compile() akzeptiert sowohl
    Strings ("adam") als auch Optimizer-Instanzen.
    """
    from src.models.lstm import train_lstm_fold
    from tensorflow.keras.optimizers import Adam

    window_size = trial.suggest_int("window_size", 20, 120, step=10)
    units_l1 = trial.suggest_categorical("units_l1", [16, 32, 64])
    units_l2 = trial.suggest_categorical("units_l2", [32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
    epochs = trial.suggest_int("epochs", 10, 50, step=5)
    threshold = trial.suggest_float("threshold", 0.3, 0.7, step=0.05)

    lstm_cfg = cfg.models.lstm
    features = cfg.features.model_features
    labels_col = resolve_label_col(cfg)

    fold_sharpes = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        try:
            df_train = df.loc[train_idx]
            df_test = df.loc[test_idx]

            # HMM-Labels für diesen Fold generieren
            df_train, df_test = _generate_hmm_labels(df_train, df_test, cfg)

            probs_raw, pred_idx = train_lstm_fold(
                df_train=df_train,
                df_test=df_test,
                features=features,
                labels_col=labels_col,
                window_size=window_size,
                units_l1=units_l1,
                units_l2=units_l2,
                return_sequences=lstm_cfg.return_sequences,
                dropout=dropout,
                dense=lstm_cfg.dense,
                activation=lstm_cfg.activation,
                optimizer=Adam(learning_rate=learning_rate),
                metrics=lstm_cfg.metrics,
                epochs=epochs,
                batch_size=lstm_cfg.batch_size,
                validation_split=lstm_cfg.validation_split,
                verbose=0,
            )

            signal = (probs_raw >= threshold).astype(int)
            signal_series = pd.Series(signal, index=pred_idx)
            df_test_aligned = df_test.loc[pred_idx]
            oos_rets = _fold_portfolio_returns(
                df_test_aligned, signal_series, signal_shift, fee,
            )
            fold_sharpes.append(_compute_oos_sharpe(oos_rets))
        except Exception as e:
            warnings.warn(f"LSTM Trial {trial.number}, Fold {fold_id}: {e}")
            fold_sharpes.append(-999.0)

        trial.report(np.median(fold_sharpes), fold_id)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.median(fold_sharpes))


def objective_transformer(
    trial: optuna.Trial,
    df: pd.DataFrame,
    splits: list,
    cfg,
    fee: float,
    signal_shift: int,
) -> float:
    """
    Transformer: d_model, n_heads, n_layers, learning_rate, dropout, epochs.

    Constraint: d_model muss durch n_heads teilbar sein.
    """
    from src.models.transformer import train_transformer_fold

    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    n_layers = trial.suggest_int("n_layers", 1, 4)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.3, step=0.05)
    epochs = trial.suggest_int("epochs", 20, 80, step=10)
    window_size = trial.suggest_int("window_size", 20, 120, step=10)
    threshold = trial.suggest_float("threshold", 0.3, 0.7, step=0.05)

    # Constraint: d_model % n_heads == 0
    if d_model % n_heads != 0:
        raise optuna.TrialPruned()

    t_cfg = cfg.models.transformer
    features = cfg.features.model_features
    labels_col = resolve_label_col(cfg)
    
    if cfg.labels.supervised_label_source != "hmm":
        df = df.copy()
        df["Supervised_Label"] = compute_supervised_labels(df, cfg)

    fold_sharpes = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        try:
            df_train = df.loc[train_idx]
            df_test = df.loc[test_idx]

            # HMM-Labels für diesen Fold generieren
            df_train, df_test = _generate_hmm_labels(df_train, df_test, cfg)

            probs_raw, pred_idx = train_transformer_fold(
                df_train=df_train,
                df_test=df_test,
                features=features,
                labels_col=labels_col,
                window_size=window_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=t_cfg.batch_size,
                validation_split=t_cfg.validation_split,
                verbose=0,
            )

            signal = (probs_raw >= threshold).astype(int)
            signal_series = pd.Series(signal, index=pred_idx)
            df_test_aligned = df_test.loc[pred_idx]
            oos_rets = _fold_portfolio_returns(
                df_test_aligned, signal_series, signal_shift, fee,
            )
            fold_sharpes.append(_compute_oos_sharpe(oos_rets))
        except Exception as e:
            warnings.warn(f"Transformer Trial {trial.number}, Fold {fold_id}: {e}")
            fold_sharpes.append(-999.0)

        trial.report(np.median(fold_sharpes), fold_id)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # GPU-Speicher freigeben nach jedem Fold
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return float(np.median(fold_sharpes))


# ============================================================================
# Orchestrierung
# ============================================================================

_OBJECTIVE_MAP = {
    "MSM": objective_msm,
    "HMM": objective_hmm,
    "LSTM": objective_lstm,
    "Transformer": objective_transformer,
}


def run_optimization(
    model_name: str,
    df: pd.DataFrame,
    cfg,
    n_trials: int = 50,
    every_nth_fold: int | None = None,
    storage: str | None = None,
) -> optuna.Study:
    """
    Optuna-Study für ein einzelnes Modell ausführen.

    Parameter
    ---------
    model_name : str
        "MSM", "HMM", "LSTM" oder "Transformer".
    df : pd.DataFrame
        Feature-engineerter DataFrame (Silver-Schicht) mit DatetimeIndex.
    cfg : PipelineConfig
        Zentrale Konfiguration.
    n_trials : int
        Anzahl Optuna-Trials.
    every_nth_fold : int | None
        Nur jeden n-ten Fold verwenden (Speed). None = alle Folds.
    storage : str | None
        Optuna Storage-URL (z.B. "sqlite:///optuna.db").
        None = In-Memory.

    Rückgabe
    --------
    optuna.Study mit .best_params und .best_value.
    """
    if model_name not in _OBJECTIVE_MAP:
        raise ValueError(
            f"Unbekanntes Modell '{model_name}'. "
            f"Verfügbar: {list(_OBJECTIVE_MAP.keys())}"
        )

    # Optuna-Logging auf Warnungen reduzieren
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Walk-Forward-Splits generieren (identisch zum finalen Lauf)
    wf = cfg.walk_forward
    splits = walk_forward_splits(
        index=df.index,
        mode=wf.mode,
        train_window_years=wf.train_window_years,
        test_window_months=wf.test_window_months,
        step_months=wf.step_months,
        min_train_years=wf.min_train_years,
    )
    splits = _subsample_splits(splits, every_nth_fold)
    print(f"\n{'='*60}")
    print(f"Optimierung: {model_name} | {n_trials} Trials | {len(splits)} Folds")
    print(f"{'='*60}")

    # Storage-Pfad relativ zum Projekt-Root auflösen
    if storage and storage.startswith("sqlite:///") and not os.path.isabs(storage[10:]):
        from pathlib import Path
        project_root = Path(cfg.paths.data_dir).resolve().parent
        db_path = project_root / storage[10:]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path}"

    # Study erstellen
    study = optuna.create_study(
        direction="maximize",
        study_name=f"opt_{model_name}",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        ),
    )

    # Transaktionskosten aus Config
    fee = cfg.backtesting.transaction_cost_bps / 10_000
    signal_shift = cfg.backtesting.signal_shift

    # Objective-Funktion mit gebundenen Parametern
    if model_name in ("MSM",):
        objective = lambda trial: _OBJECTIVE_MAP[model_name](
            trial, df, splits, fee, signal_shift,
        )
    else:
        objective = lambda trial: _OBJECTIVE_MAP[model_name](
            trial, df, splits, cfg, fee, signal_shift,
        )

    # Default-Parameter als ersten Trial einspeisen (Baseline)
    default_params = _get_default_params(model_name, cfg)
    if default_params:
        study.enqueue_trial(default_params)

    # Bereits abgeschlossene + geprunte Trials zählen
    done = len([t for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.PRUNED)])
    remaining = max(0, n_trials - done)

    if remaining == 0:
        print(f"  ➜ {model_name}: Bereits {done}/{n_trials} Trials vorhanden — überspringe.")
    else:
        print(f"  ➜ {done} Trials vorhanden, starte {remaining} weitere.")
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    # Ergebnis ausgeben
    print(f"\n--- {model_name}: Beste Parameter ---")
    print(f"  Sharpe (Median OOS): {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  Trials: {len(study.trials)} "
          f"(davon {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))} gepruned)")

    # Visualisierungen speichern (immer — auch bei Skip)
    try:
        from src.backtest.plots import save_optuna_plots
        save_optuna_plots(study, model_name, cfg)
    except ImportError:
        warnings.warn("Plotly/Kaleido nicht installiert — Optuna-Plots übersprungen.")

    return study

def optimize_all(
    df: pd.DataFrame,
    cfg,
    n_trials: int = 50,
    every_nth_fold: int | None = None,
    models: list[str] | None = None,
    storage: str | None = None,
) -> dict[str, optuna.Study]:
    """
    Alle (oder ausgewählte) Modelle sequenziell optimieren.

    Reihenfolge: MSM → HMM → LSTM → Transformer
    (HMM vor DL, da DL-Labels auf HMM basieren.)

    Parameter
    ---------
    models : list[str] | None
        Zu optimierende Modelle. None = alle vier.
    Übrige Parameter : siehe run_optimization.

    Rückgabe
    --------
    Dict[model_name → optuna.Study].
    """
    if models is None:
        models = ["MSM", "HMM", "LSTM", "Transformer"]

    studies = {}
    for model_name in models:
        studies[model_name] = run_optimization(
            model_name=model_name,
            df=df,
            cfg=cfg,
            n_trials=n_trials,
            every_nth_fold=every_nth_fold,
            storage=storage,
        )

    # Zusammenfassung
    print(f"\n{'='*60}")
    print("Optimierung abgeschlossen — Zusammenfassung")
    print(f"{'='*60}")
    for name, study in studies.items():
        print(f"\n{name}:")
        print(f"  Best Sharpe: {study.best_value:.4f}")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

    return studies

def _get_default_params(model_name: str, cfg) -> dict | None:
    """Aktuelle Config-Werte als Optuna-Trial-Dict."""
    if model_name == "MSM":
        return {
            "threshold": cfg.models.msm.threshold,          # 0.5
        }
    if model_name == "HMM":
        return {
            "covariance_type": cfg.models.hmm.covariance_type,  # "full"
            "threshold": cfg.models.hmm.threshold,          # 0.5
        }
    if model_name == "LSTM":
        c = cfg.models.lstm
        return {
            "window_size": c.window_size,       # 60
            "units_l1": c.units_l1,             # 32
            "units_l2": c.units_l2,             # 64
            "learning_rate": c.learning_rate,   # 0.001
            "dropout": c.dropout,               # 0.2
            "epochs": c.epochs,                 # 30
            "threshold": cfg.models.msm.threshold,  # 0.5
        }
    if model_name == "Transformer":
        c = cfg.models.transformer
        return {
            "d_model": c.d_model,               # 64
            "n_heads": c.n_heads,               # 4
            "n_layers": c.n_layers,             # 2
            "dim_feedforward": c.dim_feedforward, # 128
            "learning_rate": c.learning_rate,   # 0.0001
            "dropout": c.dropout,               # 0.1
            "epochs": c.epochs,                 # 50
            "window_size": c.window_size,       # 60
            "threshold": cfg.models.msm.threshold,  # 0.5
        }
    return None