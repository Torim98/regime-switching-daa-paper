"""Walk-Forward-Validierung — Splitter und Helper für rollierende OOS-Evaluation."""

import warnings
import pandas as pd
from pandas.tseries.offsets import DateOffset
import hashlib
import json
from src.data.labels.resolver import compute_supervised_labels, resolve_label_col


def walk_forward_splits(
    index: pd.DatetimeIndex,
    mode: str,
    train_window_years: int,
    test_window_months: int,
    step_months: int,
    min_train_years: int,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generiert Walk-Forward-Splits über einen DatetimeIndex.

    Parameter
    ---------
    index : pd.DatetimeIndex
        Vollständiger Zeitindex der Datenreihe (z.B. df.index).
    mode : str
        "rolling"   = Train-Fenster konstanter Länge, wandert mit.
        "expanding" = Train-Fenster wächst monoton ab Start.
    train_window_years : int
        Länge des Train-Fensters in Jahren (nur bei mode="rolling" relevant).
    test_window_months : int
        Länge eines OOS-Test-Folds in Monaten.
    step_months : int
        Schrittweite zwischen aufeinanderfolgenden Test-Fenster-Starts.
        step_months == test_window_months → disjunkte (nicht-überlappende) Folds.
    min_train_years : int
        Mindest-Trainingsdatenmenge (in Jahren) für den ersten Fold (mode="expanding").

    Rückgabe
    --------
    list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
        Liste von (train_idx, test_idx)-Paaren. Indizes sind echte
        DatetimeIndex-Slices aus dem übergebenen index; robust gegenüber
        Reindexing und Period/Datetime-Konvertierungen.

    Garantien
    ---------
    - train_idx und test_idx überlappen NICHT (train endet strikt vor test).
    - Bei step_months == test_window_months sind die Test-Bereiche aller
      Folds disjunkt (kein Doppel-Sampling).
    - Folds mit leerem Train- oder Test-Bereich werden übersprungen.

    Voraussetzungen
    --------
    - index ist monoton steigend und enthält Handelstage (Wochenenden / Feiertage
      lückenhaft, das ist ok; die Selektion erfolgt per Datums-Maske).
    """
    if mode not in ("rolling", "expanding"):
        raise ValueError(f"mode muss 'rolling' oder 'expanding' sein, war: {mode}")
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if not index.is_monotonic_increasing:
        raise ValueError("index muss monoton steigend sein.")

    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []

    start = index.min()
    end = index.max()

    # Erster Test-Beginn: nach Ablauf des initialen Trainingsfensters
    if mode == "rolling":
        first_test_start = start + DateOffset(years=train_window_years)
    else:  # expanding
        first_test_start = start + DateOffset(years=min_train_years)

    current_test_start = first_test_start

    while current_test_start + DateOffset(months=test_window_months) <= end + DateOffset(days=1):
        current_test_end = current_test_start + DateOffset(months=test_window_months)

        # Train-Fenster bestimmen
        if mode == "rolling":
            train_start = current_test_start - DateOffset(years=train_window_years)
        else:  # expanding
            train_start = start

        # Indizes per Datums-Maske selektieren
        # Train: [train_start, current_test_start) — strikt VOR Test
        # Test:  [current_test_start, current_test_end)
        train_mask = (index >= train_start) & (index < current_test_start)
        test_mask = (index >= current_test_start) & (index < current_test_end)

        train_idx = index[train_mask]
        test_idx = index[test_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        current_test_start = current_test_start + DateOffset(months=step_months)

    # --- Partialer letzter Fold: restliche Daten nutzen ---
    if current_test_start < end:
        if mode == "rolling":
            train_start = current_test_start - DateOffset(years=train_window_years)
        else:
            train_start = start
        
        train_mask = (index >= train_start) & (index < current_test_start)
        test_mask = (index >= current_test_start)  # bis zum Ende der Daten
        
        train_idx = index[train_mask]
        test_idx = index[test_mask]
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def summarize_splits(
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
) -> pd.DataFrame:
    """
    Erstellt eine Übersichtstabelle der Walk-Forward-Splits.

    Pro Fold: train_start, train_end, test_start, test_end, n_train, n_test.
    Nützlich für Sanity-Checks (Überlappungs-Prüfung, Fold-Anzahl, Fenster-Größen)
    und als Datenquelle für die Walk-Forward-Schema-Visualisierung in Schritt 2.
    """
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        rows.append({
            "fold": fold_id,
            "train_start": train_idx.min(),
            "train_end": train_idx.max(),
            "test_start": test_idx.min(),
            "test_end": test_idx.max(),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })
    return pd.DataFrame(rows).set_index("fold")


def assert_no_leakage(
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
) -> None:
    """
    Sanity-Check: verifiziert, dass kein Train-Fenster in den zugehörigen
    Test-Bereich hineinreicht. Wirft AssertionError bei Verletzung.

    Aufrufen direkt nach walk_forward_splits() im Notebook, bevor irgendein
    Training startet — schützt vor subtilen Off-by-One-Bugs in der
    Datums-Logik.
    """
    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        train_max = train_idx.max()
        test_min = test_idx.min()
        assert train_max < test_min, (
            f"Fold {fold_id}: Train-Ende ({train_max}) liegt nicht strikt vor "
            f"Test-Beginn ({test_min}) — Look-Ahead-Verdacht!"
        )

def run_walk_forward(
    df: pd.DataFrame,
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    cfg,
    models_to_run: list[str],
) -> pd.DataFrame:
    """
    Walk-Forward mit parallelisierten CPU-Modellen (MSM, HMM) und
    sequentiellem DL-Training (LSTM, Transformer) auf der GPU.

    Parallelisierung gilt ausschliesslich innerhalb der CPU-Fold-Schleife;
    Ergebnisse sind bit-identisch zur sequentiellen Variante, da jeder Fold
    einen eigenen RandomState und keinen Shared State hat.
    """
    import warnings
    import logging
    from src.backtest.parallel import run_folds_parallel
    from src.models.lstm import train_lstm_fold
    from src.models.transformer import train_transformer_fold

    logger = logging.getLogger("model_service")
    n_jobs = getattr(cfg.walk_forward, "n_jobs", -1)

    # 1. Supervised-Labels einmalig fuer den gesamten DF erzeugen
    supervised_label_source = cfg.labels.supervised_label_source
    result_df = df.copy()
    if supervised_label_source != "hmm":
        df = df.copy()
        df["Supervised_Label"] = compute_supervised_labels(df, cfg)
        result_df["Supervised_Label"] = df["Supervised_Label"]

    for m in models_to_run:
        result_df[f"{m}_Prob"]   = pd.Series(dtype=float, index=df.index)
        result_df[f"{m}_Signal"] = pd.Series(dtype=float, index=df.index)

    failed_folds = {m: 0 for m in models_to_run}

    # 2. CPU-Modelle parallel ueber alle Folds
    logger.info(
        f"Walk-Forward CPU-Phase start: n_jobs={n_jobs}, folds={len(splits)}, "
        f"models={[m for m in models_to_run if m in ('MSM', 'HMM')]}"
    )
    parallel_results = run_folds_parallel(
        df, splits,
        msm_cfg=cfg.models.msm if "MSM" in models_to_run else None,
        hmm_cfg=cfg.models.hmm if "HMM" in models_to_run else None,
        n_jobs=n_jobs,
    )
    for model_name, fold_results in parallel_results.items():
        for r in fold_results:
            if not r["ok"]:
                warnings.warn(f"[{model_name}] Fold failed: {r['error']}")
                failed_folds[model_name] += 1
                continue
            result_df.loc[r["test_idx"], f"{model_name}_Prob"]   = r["probs"].values
            result_df.loc[r["test_idx"], f"{model_name}_Signal"] = r["signal"].values
    logger.info("Walk-Forward CPU-Phase done")

    # 3. DL-Modelle sequentiell (GPU-gebunden)
    features = cfg.features.model_features
    label_col = resolve_label_col(cfg)

    logger.info(
        f"Walk-Forward DL-Phase start: folds={len(splits)}, "
        f"models={[m for m in models_to_run if m in ('LSTM', 'Transformer')]}"
    )
    # Warm-Start zwischen Folds: Gewichte aus Fold N-1 als Initialisierung fuer
    # Fold N verwenden (Rolling-Window -> ~90% Train-Overlap -> legitim, weil
    # Fold N-1 die Fold-N-Testdaten nie gesehen hat).
    # Beim ersten Fold bzw. nach Fehlschlaegen: Kaltstart (state = None).
    lstm_state = None
    transformer_state = None
    dl_warm_start = getattr(cfg.walk_forward, "dl_warm_start", False)
    epochs_warm = getattr(cfg.walk_forward, "dl_warm_start_epochs", None)

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        df_train = df.loc[train_idx]
        df_test  = df.loc[test_idx]

        if "LSTM" in models_to_run:
            try:
                c = cfg.models.lstm
                probs_raw, pred_idx, lstm_state = train_lstm_fold(
                    df_train=df_train, df_test=df_test,
                    features=features, labels_col=label_col,
                    window_size=c.window_size, units_l1=c.units_l1, units_l2=c.units_l2,
                    return_sequences=c.return_sequences, dropout=c.dropout,
                    dense=c.dense, activation=c.activation, optimizer=c.optimizer,
                    metrics=c.metrics, epochs=c.epochs, batch_size=c.batch_size,
                    validation_split=c.validation_split, verbose=0,
                    init_weights=lstm_state if dl_warm_start else None,
                    epochs_warm=epochs_warm if (dl_warm_start and lstm_state is not None) else None,
                )
                signal = (probs_raw >= c.threshold).astype(int)
                result_df.loc[pred_idx, "LSTM_Prob"]   = probs_raw
                result_df.loc[pred_idx, "LSTM_Signal"] = signal
            except Exception as e:
                import traceback
                warnings.warn(f"[LSTM] Fold {fold_id} failed: {type(e).__name__}: {e}")
                if failed_folds["LSTM"] < 2:
                    traceback.print_exc()
                failed_folds["LSTM"] += 1
                # Warm-Start verwerfen, damit der naechste Fold wieder kalt startet.
                lstm_state = None

        if "Transformer" in models_to_run:
            try:
                c = cfg.models.transformer
                probs_raw, pred_idx, transformer_state = train_transformer_fold(
                    df_train=df_train, df_test=df_test,
                    features=features, labels_col=label_col,
                    window_size=c.window_size, d_model=c.d_model, n_heads=c.n_heads,
                    n_layers=c.n_layers, dim_feedforward=c.dim_feedforward,
                    dropout=c.dropout, learning_rate=c.learning_rate,
                    epochs=c.epochs, batch_size=c.batch_size,
                    validation_split=c.validation_split, verbose=0,
                    init_state_dict=transformer_state if dl_warm_start else None,
                    epochs_warm=epochs_warm if (dl_warm_start and transformer_state is not None) else None,
                )
                signal = (probs_raw >= c.threshold).astype(int)
                result_df.loc[pred_idx, "Transformer_Prob"]   = probs_raw
                result_df.loc[pred_idx, "Transformer_Signal"] = signal
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            except Exception as e:
                import traceback
                warnings.warn(f"[Transformer] Fold {fold_id} failed: {type(e).__name__}: {e}")
                if failed_folds["Transformer"] < 2:
                    traceback.print_exc()
                failed_folds["Transformer"] += 1
                # Warm-Start verwerfen, damit der naechste Fold wieder kalt startet.
                transformer_state = None

    logger.info("Walk-Forward DL-Phase done")

    # 4. Abschluss-Report
    print(f"\n=== Walk-Forward abgeschlossen ===")
    for model_name, n_failed in failed_folds.items():
        n_oos = result_df[f"{model_name}_Signal"].notna().sum()
        print(f"  {model_name}: {n_oos} OOS-Tage, {n_failed} Folds fehlgeschlagen")

    return result_df
    
def _walk_forward_fingerprint(cfg, df_shape: tuple, df_index_hash: str) -> str:
    """
    Erzeugt einen deterministischen Hash über alle Parameter, die das
    Walk-Forward-Ergebnis beeinflussen. Ändert sich ein Parameter,
    wird der Cache invalidiert.
    """
    params = {
        "mode": cfg.walk_forward.mode,
        "train_window_years": cfg.walk_forward.train_window_years,
        "test_window_months": cfg.walk_forward.test_window_months,
        "step_months": cfg.walk_forward.step_months,
        "min_train_years": cfg.walk_forward.min_train_years,
        "df_shape": list(df_shape),
        "df_index_hash": df_index_hash,
        # Modell-Hyperparameter, die das Ergebnis ändern
        "msm_k": cfg.models.msm.k_regimes,
        "msm_threshold": cfg.models.msm.threshold,
        "hmm_n_components": cfg.models.hmm.n_components,
        "hmm_n_iter": cfg.models.hmm.n_iter,
        "lstm_window": cfg.models.lstm.window_size,
        "lstm_epochs": cfg.models.lstm.epochs,
        "lstm_units_l1": cfg.models.lstm.units_l1,
        "lstm_batch_size": cfg.models.lstm.batch_size,
        "transformer_window": cfg.models.transformer.window_size,
        "transformer_epochs": cfg.models.transformer.epochs,
        "transformer_d_model": cfg.models.transformer.d_model,
        "transformer_batch_size": cfg.models.transformer.batch_size,
        "dl_warm_start": getattr(cfg.walk_forward, "dl_warm_start", False),
        "supervised_label_source": cfg.labels.supervised_label_source,
        "pag_soss_params": vars(cfg.labels.pagan_sossounov),
        "p2t_params": vars(cfg.labels.peak_to_trough),
    }
    raw = json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_walk_forward_cache(
    test_df: pd.DataFrame,
    fingerprint: str,
    cache_path: str,
) -> None:
    """Speichert OOS-Ergebnisse + Fingerprint als Parquet mit Metadaten."""
    test_df.attrs["wf_fingerprint"] = fingerprint
    test_df.to_parquet(cache_path)
    # Fingerprint separat als .txt speichern (Parquet attrs gehen bei manchen
    # Engines verloren)
    with open(cache_path + ".fingerprint", "w") as f:
        f.write(fingerprint)
    print(f"  Walk-Forward-Cache gespeichert: {cache_path}")


def load_walk_forward_cache(
    cache_path: str,
    expected_fingerprint: str,
) -> pd.DataFrame | None:
    """
    Lädt Cache wenn vorhanden UND Fingerprint übereinstimmt.
    Gibt None zurück wenn Cache ungültig/nicht vorhanden.
    """
    import os
    fp_path = cache_path + ".fingerprint"

    if not os.path.exists(cache_path) or not os.path.exists(fp_path):
        return None

    with open(fp_path, "r") as f:
        stored_fp = f.read().strip()

    if stored_fp != expected_fingerprint:
        print(f"  Walk-Forward-Cache ungültig (Fingerprint mismatch). Re-Training nötig.")
        return None

    print(f"  Walk-Forward-Cache geladen: {cache_path}")
    return pd.read_parquet(cache_path)