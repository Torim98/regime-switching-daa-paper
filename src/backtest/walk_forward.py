"""Walk-Forward-Validierung — Splitter und Helper für rollierende OOS-Evaluation."""

import warnings
import pandas as pd
from pandas.tseries.offsets import DateOffset
import hashlib
import json


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
    Iteriert über alle Walk-Forward-Folds und aggregiert OOS-Vorhersagen
    der angegebenen Modelle zu durchgehenden Serien.

    Parameter
    ---------
    df : pd.DataFrame
        Vollständiger Feature-DataFrame (DatetimeIndex), enthält alle Spalten
        die irgendein Modell braucht (Features, Returns, Labels-Spalte etc.).
    splits : Liste von (train_idx, test_idx)-Paaren aus walk_forward_splits().
    cfg : zentrale PipelineConfig (für Modell-Hyperparameter).
    models_to_run : Liste der Modellnamen, z.B. ["MSM", "HMM", "LSTM", "Transformer"].

    Rückgabe
    --------
    pd.DataFrame
        Kopie von df, ergänzt um {Model}_Prob und {Model}_Signal Spalten
        über den vollständigen OOS-Bereich. Train-Only-Bereiche bleiben NaN.

    Verhalten bei Fold-Fehlern
    --------------------------
    - Fängt Konvergenz-/Numerik-Fehler einzelner Modelle pro Fold ab.
    - Schreibt NaN für den fehlgeschlagenen Bereich, gibt eine Warnung aus.
    - Andere Modelle und andere Folds laufen weiter.
    """
    # Lokale Imports (zirkuläre Imports vermeiden)
    from src.models.msm import train_msm_fold
    from src.models.hmm import train_hmm_fold
    from src.models.lstm import train_lstm_fold
    from src.models.transformer import train_transformer_fold

    # Ergebnis-Container
    result_df = df.copy()
    for model_name in models_to_run:
        result_df[f"{model_name}_Prob"] = pd.Series(dtype=float, index=df.index)
        result_df[f"{model_name}_Signal"] = pd.Series(dtype=float, index=df.index)

    # Hyperparameter-Shortcuts
    features = cfg.features.model_features
    failed_folds = {m: 0 for m in models_to_run}

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n=== Fold {fold_id}/{len(splits)} | "
              f"Train {train_idx.min().date()}–{train_idx.max().date()} "
              f"({len(train_idx)} d) | "
              f"Test {test_idx.min().date()}–{test_idx.max().date()} "
              f"({len(test_idx)} d) ===")

        df_train = df.loc[train_idx].copy()
        df_test = df.loc[test_idx].copy()

        # ---------- MSM ----------
        if "MSM" in models_to_run:
            try:
                msm_cfg = cfg.models.msm
                probs, signal, signal_train = train_msm_fold(
                    returns_train=df_train["Returns"],
                    returns_test=df_test["Returns"],
                    k_regimes=msm_cfg.k_regimes,
                    switching_variance=msm_cfg.switching_variance,
                    threshold=msm_cfg.threshold,
                )
                result_df.loc[test_idx, "MSM_Prob"] = probs
                result_df.loc[test_idx, "MSM_Signal"] = signal

                # MSM-Train-Signal als Label-Spalte in df_train injizieren
                # (LSTM/Transformer lesen diese Spalte als labels_col)
                df_train["MSM_Signal"] = signal_train
                df_test["MSM_Signal"] = signal
            except Exception as e:
                warnings.warn(f"  [MSM] Fold {fold_id} failed: {e}")
                failed_folds["MSM"] += 1

        # ---------- HMM ----------
        if "HMM" in models_to_run:
            try:
                hmm_cfg = cfg.models.hmm
                probs, signal, signal_train = train_hmm_fold(
                    features_df_train=df_train[hmm_cfg.features],
                    features_df_test=df_test[hmm_cfg.features],
                    returns_train=df_train["Returns"],
                    n_components=hmm_cfg.n_components,
                    covariance_type=hmm_cfg.covariance_type,
                    n_iter=hmm_cfg.n_iter,
                    random_state=hmm_cfg.random_state,
                    threshold=hmm_cfg.threshold,
                )
                result_df.loc[test_idx, "HMM_Prob"] = probs
                result_df.loc[test_idx, "HMM_Signal"] = signal

                # HMM-Labels als Spalte in df_train/df_test injizieren
                # (LSTM/Transformer lesen "HMM_Signal" als labels_col)
                df_train["HMM_Signal"] = signal_train.values
                df_test["HMM_Signal"] = signal.values
            except Exception as e:
                warnings.warn(f"  [HMM] Fold {fold_id} failed: {e}")
                failed_folds["HMM"] += 1

        # ---------- LSTM ----------
        if "LSTM" in models_to_run:
            if "HMM_Signal" not in df_train.columns:
                warnings.warn(
                    f"  [LSTM] Fold {fold_id} skipped: HMM-Labels nicht verfügbar."
                )
                failed_folds["LSTM"] += 1
            else:
                try:
                    lstm_cfg = cfg.models.lstm
                    probs_raw, pred_idx = train_lstm_fold(
                    df_train=df_train,
                    df_test=df_test,
                    features=features,
                    labels_col=lstm_cfg.labels,
                    window_size=lstm_cfg.window_size,
                    units_l1=lstm_cfg.units_l1,
                    units_l2=lstm_cfg.units_l2,
                    return_sequences=lstm_cfg.return_sequences,
                    dropout=lstm_cfg.dropout,
                    dense=lstm_cfg.dense,
                    activation=lstm_cfg.activation,
                    optimizer=lstm_cfg.optimizer,
                    metrics=lstm_cfg.metrics,
                    epochs=lstm_cfg.epochs,
                    batch_size=lstm_cfg.batch_size,
                    validation_split=lstm_cfg.validation_split,
                    verbose=0,
                    )
                    signal = (probs_raw >= lstm_cfg.threshold).astype(int)
                    result_df.loc[pred_idx, "LSTM_Prob"] = probs_raw
                    result_df.loc[pred_idx, "LSTM_Signal"] = signal
                except Exception as e:
                    warnings.warn(f"  [LSTM] Fold {fold_id} failed: {e}")
                    failed_folds["LSTM"] += 1

        # ---------- Transformer ----------
        if "Transformer" in models_to_run:
            if "HMM_Signal" not in df_train.columns:
                warnings.warn(
                    f"  [Transformer] Fold {fold_id} skipped: HMM-Labels nicht verfügbar."
                )
                failed_folds["Transformer"] += 1
            else:
                try:
                    t_cfg = cfg.models.transformer
                    probs_raw, pred_idx = train_transformer_fold(
                        df_train=df_train,
                        df_test=df_test,
                        features=features,
                        labels_col=t_cfg.labels,
                        window_size=t_cfg.window_size,
                        d_model=t_cfg.d_model,
                        n_heads=t_cfg.n_heads,
                        n_layers=t_cfg.n_layers,
                        dim_feedforward=t_cfg.dim_feedforward,
                        dropout=t_cfg.dropout,
                        learning_rate=t_cfg.learning_rate,
                        epochs=t_cfg.epochs,
                        batch_size=t_cfg.batch_size,
                        validation_split=t_cfg.validation_split,
                        verbose=0,
                    )
                    signal = (probs_raw >= t_cfg.threshold).astype(int)
                    result_df.loc[pred_idx, "Transformer_Prob"] = probs_raw
                    result_df.loc[pred_idx, "Transformer_Signal"] = signal

                    # GPU-Speicher freigeben
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                except Exception as e:
                    warnings.warn(f"  [Transformer] Fold {fold_id} failed: {e}")
                    failed_folds["Transformer"] += 1

    # --- Abschluss-Report ---
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
        "transformer_window": cfg.models.transformer.window_size,
        "transformer_epochs": cfg.models.transformer.epochs,
        "transformer_d_model": cfg.models.transformer.d_model,
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