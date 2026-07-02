"""
Optuna hyperparameter optimization: Bayesian search with walk-forward CV.

Each objective function:
1. Samples hyperparameters via trial.suggest_*()
2. Iterates over the (subsampled) walk-forward folds
3. Trains the model per fold on the train window
4. Computes the annualized OOS Sharpe from the portfolio returns
5. Returns the median OOS Sharpe over all folds (Optuna maximizes)

No look-ahead bias: Optuna only observes OOS metrics.
Each fold trains independently. The fold splits are identical to
those in the final walk-forward run (walk_forward.py).
"""

import warnings
import numpy as np
import pandas as pd
import optuna
import os

from src.data.labels.resolver import compute_supervised_labels, resolve_label_col
from src.backtest.walk_forward import walk_forward_splits

# ============================================================================
# Helper functions
# ============================================================================

def _compute_oos_sharpe(
    daily_returns: np.ndarray,
    trading_days_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio from daily net returns."""
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
    Computes daily net portfolio returns for an OOS fold.

    Replicates the logic from engine.backtest():
    - Shift the signal by signal_shift days (look-ahead prevention)
    - Signal=0 → portfolio return, signal=1 → cash return
    - Subtract transaction costs on signal switches
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
    """Select every n-th fold for faster optimization."""
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
# Objective functions per model
# ============================================================================

def objective_msm(
    trial: optuna.Trial,
    df: pd.DataFrame,
    splits: list,
    fee: float,
    signal_shift: int,
) -> float:
    """MSM: optimize k_regimes and threshold."""
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
            warnings.warn(f"MSM trial {trial.number}, fold {fold_id}: {e}")
            fold_sharpes.append(-999.0)

        # Pruning: report the intermediate result
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
    """HMM: optimize n_components, covariance_type, and threshold."""
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
            warnings.warn(f"HMM trial {trial.number}, fold {fold_id}: {e}")
            fold_sharpes.append(-999.0)

        trial.report(np.median(fold_sharpes), fold_id)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.median(fold_sharpes))

def objective_hmm_uni(
    trial: optuna.Trial,
    df: pd.DataFrame,
    splits: list,
    cfg,
    fee: float,
    signal_shift: int,
) -> float:
    """
    HMM_Uni: optimize the threshold only.

    covariance_type is NOT tuned: with univariate input (returns only),
    full/diag/tied are identical (1x1 covariance). The search space is
    therefore congruent with the MSM objective; a fair basis for the
    architecture comparison (Issue #3).
    """
    from src.models.hmm import train_hmm_fold

    threshold = trial.suggest_float("threshold", 0.3, 0.7, step=0.05)

    uni_cfg = cfg.models.hmm_uni

    fold_sharpes = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        try:
            df_train = df.loc[train_idx]
            df_test = df.loc[test_idx]

            probs, signal, _ = train_hmm_fold(
                features_df_train=df_train[uni_cfg.features],
                features_df_test=df_test[uni_cfg.features],
                returns_train=df_train["Returns"],
                n_components=uni_cfg.n_components,
                covariance_type=uni_cfg.covariance_type,
                n_iter=uni_cfg.n_iter,
                random_state=uni_cfg.random_state,
                threshold=threshold,
            )

            oos_rets = _fold_portfolio_returns(df_test, signal, signal_shift, fee)
            fold_sharpes.append(_compute_oos_sharpe(oos_rets))
        except Exception as e:
            warnings.warn(f"HMM_Uni trial {trial.number}, fold {fold_id}: {e}")
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
    LSTM: optimize window_size, units, learning_rate, dropout, epochs.

    Note: learning_rate is passed to train_lstm_fold via a Keras optimizer
    object. Keras model.compile() accepts both strings ("adam") and
    optimizer instances.
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

    # Precompute the supervised labels once globally (analogous to the
    # Transformer objective). Without this block, the column is missing in
    # the fold and every trial raises a KeyError.
    if cfg.labels.supervised_label_source != "hmm":
        df = df.copy()
        if "Supervised_Label" not in df.columns:
            df["Supervised_Label"] = compute_supervised_labels(df, cfg)

    fold_sharpes = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        try:
            df_train = df.loc[train_idx]
            df_test = df.loc[test_idx]

            # Generate HMM labels for this fold
            if cfg.labels.supervised_label_source == "hmm":
                df_train, df_test = _generate_hmm_labels(df_train, df_test, cfg)

            # train_lstm_fold returns (probs, pred_idx, weights); weights is
            # needed for warm starts, irrelevant in the HPO context.
            probs_raw, pred_idx, _ = train_lstm_fold(
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
            warnings.warn(f"LSTM trial {trial.number}, fold {fold_id}: {e}")
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

    Constraint: d_model must be divisible by n_heads.
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
        if "Supervised_Label" not in df.columns:
            df["Supervised_Label"] = compute_supervised_labels(df, cfg)

    fold_sharpes = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        try:
            df_train = df.loc[train_idx]
            df_test = df.loc[test_idx]

            # Generate HMM labels for this fold
            df_train, df_test = _generate_hmm_labels(df_train, df_test, cfg)

            # train_transformer_fold returns (probs, pred_idx, state_dict);
            # state_dict is for warm starts, irrelevant in the HPO context.
            probs_raw, pred_idx, _ = train_transformer_fold(
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
            warnings.warn(f"Transformer trial {trial.number}, fold {fold_id}: {e}")
            fold_sharpes.append(-999.0)

        trial.report(np.median(fold_sharpes), fold_id)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Free GPU memory after each fold
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return float(np.median(fold_sharpes))


# ============================================================================
# Orchestration
# ============================================================================

_OBJECTIVE_MAP = {
    "MSM": objective_msm,
    "HMM": objective_hmm,
    "HMM_Uni": objective_hmm_uni,
    "LSTM": objective_lstm,
    "Transformer": objective_transformer,
}


def _resolve_from_cfg(cfg, attr: str, model_name: str) -> int | None:
    """
    Reads a per-model value from cfg.optimization.<attr>.

    Expects the new structure with cfg.optimization.n_trials_per_model and
    cfg.optimization.every_nth_fold_per_model, which are loaded as
    SimpleNamespace in the config loader. Returns None if the entry is missing.
    """
    container = getattr(cfg.optimization, attr, None)
    if container is None:
        return None
    return getattr(container, model_name, None)


def run_optimization(
    model_name: str,
    df: pd.DataFrame,
    cfg,
    n_trials: int | None = None,
    every_nth_fold: int | None = None,
    storage: str | None = None,
) -> optuna.Study:
    """
    Run an Optuna study for a single model.

    Parameters
    ----------
    model_name : str
        "MSM", "HMM", "LSTM", or "Transformer".
    df : pd.DataFrame
        Feature-engineered DataFrame (Silver layer) with a DatetimeIndex.
    cfg : PipelineConfig
        Central configuration.
    n_trials : int | None
        Number of Optuna trials. None = read from
        cfg.optimization.n_trials_per_model[model_name].
    every_nth_fold : int | None
        Use only every n-th fold (speed). None = read from
        cfg.optimization.every_nth_fold_per_model[model_name]
        (fallback: all folds if not configured).
    storage : str | None
        Optuna storage URL (e.g. "sqlite:///optuna.db").
        None = in-memory.

    Returns
    -------
    optuna.Study with .best_params and .best_value.
    """
    if model_name not in _OBJECTIVE_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_OBJECTIVE_MAP.keys())}"
        )

    # Resolve defaults from the config if not passed explicitly
    if n_trials is None:
        n_trials = _resolve_from_cfg(cfg, "n_trials_per_model", model_name)
        if n_trials is None:
            raise ValueError(
                f"n_trials not passed and cfg.optimization."
                f"n_trials_per_model.{model_name} is missing."
            )
    if every_nth_fold is None:
        every_nth_fold = _resolve_from_cfg(
            cfg, "every_nth_fold_per_model", model_name,
        )

    # Reduce Optuna logging to warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Generate the walk-forward splits (identical to the final run)
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
    print(f"Optimization: {model_name} | {n_trials} trials | {len(splits)} folds")
    print(f"{'='*60}")

    # Resolve the storage path relative to the project root
    if storage and storage.startswith("sqlite:///") and not os.path.isabs(storage[10:]):
        from pathlib import Path
        project_root = Path(cfg.paths.data_dir).resolve().parent
        db_path = project_root / storage[10:]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path}"

    # SQLite on Docker bind mounts is fragile under write pressure
    # (intermittent "disk I/O error"). WAL + busy_timeout + NORMAL synchronous
    # mitigates this; pre-ping against stale-pool issues.
    storage_arg = storage
    if isinstance(storage, str) and storage.startswith("sqlite:///"):
        from sqlalchemy import event

        rdb_storage = optuna.storages.RDBStorage(
            url=storage,
            heartbeat_interval=60,
            grace_period=120,
            engine_kwargs={
                "connect_args": {"timeout": 30, "check_same_thread": False},
                "pool_pre_ping": True,
            },
        )

        # PRAGMAs only on the Optuna engine, not globally (avoids
        # multiple registration with optimize_all over 4 models).
        def _set_sqlite_pragma(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            try:
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.execute("PRAGMA temp_store=MEMORY")
            finally:
                cur.close()

        event.listen(rdb_storage.engine, "connect", _set_sqlite_pragma)
        # Also apply to the already open connection from the pool
        with rdb_storage.engine.connect() as _conn:
            _conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            _conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
            _conn.exec_driver_sql("PRAGMA busy_timeout=30000")

        storage_arg = rdb_storage

    # Create the study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"opt_{model_name}",
        storage=storage_arg,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        ),
    )

    # Transaction costs from the config
    fee = cfg.backtesting.transaction_cost_bps / 10_000
    signal_shift = cfg.backtesting.signal_shift

    # Objective function with bound parameters
    if model_name in ("MSM",):
        objective = lambda trial: _OBJECTIVE_MAP[model_name](
            trial, df, splits, fee, signal_shift,
        )
    else:
        objective = lambda trial: _OBJECTIVE_MAP[model_name](
            trial, df, splits, cfg, fee, signal_shift,
        )

    # Enqueue the default parameters as the first trial (baseline)
    default_params = _get_default_params(model_name, cfg)
    if default_params:
        study.enqueue_trial(default_params)

    # Count already completed + pruned trials
    done = len([t for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.PRUNED)])
    remaining = max(0, n_trials - done)

    if remaining == 0:
        print(f"  ➜ {model_name}: {done}/{n_trials} trials already present, skipping.")
    else:
        print(f"  ➜ {done} trials present, starting {remaining} more.")
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    # Print the result
    print(f"\n--- {model_name}: best parameters ---")
    print(f"  Sharpe (median OOS): {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  Trials: {len(study.trials)} "
          f"(of which {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))} pruned)")

    # Save the visualizations (always, even on skip)
    try:
        from src.backtest.plots import save_optuna_plots
        save_optuna_plots(study, model_name, cfg)
    except ImportError:
        warnings.warn("Plotly/Kaleido not installed, skipping Optuna plots.")

    return study

def optimize_all(
    df: pd.DataFrame,
    cfg,
    n_trials: int | None = None,
    every_nth_fold: int | None = None,
    models: list[str] | None = None,
    storage: str | None = None,
) -> dict[str, optuna.Study]:
    """
    Optimize all (or selected) models sequentially.

    Order: MSM → HMM → LSTM → Transformer

    Parameters
    ----------
    models : list[str] | None
        Models to optimize. None = all four.
    n_trials : int | None
        Explicit override for ALL models. None = read per model from
        cfg.optimization.n_trials_per_model (thesis default:
        50 for MSM/HMM, 30 for LSTM/Transformer).
    every_nth_fold : int | None
        Explicit override for ALL models. None = read per model from
        cfg.optimization.every_nth_fold_per_model.
    Remaining parameters : see run_optimization.

    Returns
    -------
    Dict[model_name → optuna.Study].
    """
    if models is None:
        models = ["MSM", "HMM", "HMM_Uni", "LSTM", "Transformer"]

    studies = {}
    for model_name in models:
        studies[model_name] = run_optimization(
            model_name=model_name,
            df=df,
            cfg=cfg,
            n_trials=n_trials,            # None → run_optimization reads from the config
            every_nth_fold=every_nth_fold,  # None → run_optimization reads from the config
            storage=storage,
        )

    # Summary
    print(f"\n{'='*60}")
    print("Optimization complete: summary")
    print(f"{'='*60}")
    for name, study in studies.items():
        print(f"\n{name}:")
        print(f"  Best Sharpe: {study.best_value:.4f}")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

    return studies

def _format_param_value(v) -> str:
    """Format a hyperparameter value readably for Markdown tables."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")  # narrow no-break space
    if isinstance(v, float):
        a = abs(v)
        if a > 0 and (a < 1e-3 or a >= 1e6):
            return f"{v:.3e}"
        # 4 significant digits, without superfluous zeros
        return f"{float(f'{v:.4g}')}"
    return str(v)


def save_optuna_best_params(
    studies: dict[str, "optuna.Study"],
    cfg,
    metric_label: str = "Sharpe (Median OOS)",
) -> str:
    """
    Persists the best hyperparameters of all Optuna studies as Markdown.

    Format: overview table (model · best score · ✓/✂/total)
    plus one parameter table per model with readably formatted values.
    Target path: cfg.asset_path("optuna_best_params").
    """
    from pathlib import Path
    import datetime
    import optuna as _optuna

    lines: list[str] = [
        "# Optuna: Best Hyperparameters",
        "",
        f"_Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_  ",
        f"Optimization metric: **{metric_label}**",
        "",
        "## Overview",
        "",
        "| Model | Best Score | ✓ Complete | ✗ Pruned | Total |",
        "|:---|---:|---:|---:|---:|",
    ]
    for name, study in studies.items():
        n_complete = sum(
            1 for t in study.trials
            if t.state == _optuna.trial.TrialState.COMPLETE
        )
        n_pruned = sum(
            1 for t in study.trials
            if t.state == _optuna.trial.TrialState.PRUNED
        )
        lines.append(
            f"| **{name}** | {study.best_value:.4f} | "
            f"{n_complete} | {n_pruned} | {len(study.trials)} |"
        )
    lines.append("")

    for name, study in studies.items():
        lines.append(f"### {name}: Best Score `{study.best_value:.4f}`")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|:---|---:|")
        for k, v in study.best_params.items():
            lines.append(f"| `{k}` | `{_format_param_value(v)}` |")
        lines.append("")

    path = Path(cfg.asset_path("optuna_best_params"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ {path}")
    return str(path)


def _get_default_params(model_name: str, cfg) -> dict | None:
    """Current config values as an Optuna trial dict."""
    if model_name == "MSM":
        return {
            "threshold": cfg.models.msm.threshold,          # 0.5
        }
    if model_name == "HMM":
        return {
            "covariance_type": cfg.models.hmm.covariance_type,  # "full"
            "threshold": cfg.models.hmm.threshold,          # 0.5
        }
    if model_name == "HMM_Uni":
        return {
            "threshold": cfg.models.hmm_uni.threshold,
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
