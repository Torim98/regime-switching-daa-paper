"""Walk-forward validation: splitter and helpers for rolling OOS evaluation."""

import warnings
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from types import SimpleNamespace
from src.data.labels.resolver import (
    compute_supervised_labels,
    compute_supervised_labels_asof,
    resolve_label_col,
)


def walk_forward_splits(
    index: pd.DatetimeIndex,
    mode: str,
    train_window_years: int,
    test_window_months: int,
    step_months: int,
    min_train_years: int,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generates walk-forward splits over a DatetimeIndex.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Full time index of the data series (e.g. df.index).
    mode : str
        "rolling"   = training window of constant length, moves along.
        "expanding" = training window grows monotonically from the start.
    train_window_years : int
        Length of the training window in years (only relevant for mode="rolling").
    test_window_months : int
        Length of one OOS test fold in months.
    step_months : int
        Step size between the starts of consecutive test windows.
        step_months == test_window_months → disjoint (non-overlapping) folds.
    min_train_years : int
        Minimum amount of training data (in years) for the first fold (mode="expanding").

    Returns
    -------
    list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
        List of (train_idx, test_idx) pairs. Indices are true
        DatetimeIndex slices from the provided index; robust against
        reindexing and period/datetime conversions.

    Guarantees
    ----------
    - train_idx and test_idx do NOT overlap (train ends strictly before test).
    - With step_months == test_window_months, the test ranges of all
      folds are disjoint (no double sampling).
    - Folds with an empty train or test range are skipped.

    Requirements
    ------------
    - index is monotonically increasing and contains trading days (gaps for
      weekends/holidays are fine; selection is done via date masks).
    """
    if mode not in ("rolling", "expanding"):
        raise ValueError(f"mode must be 'rolling' or 'expanding', was: {mode}")
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if not index.is_monotonic_increasing:
        raise ValueError("index must be monotonically increasing.")

    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []

    start = index.min()
    end = index.max()

    # First test start: after the initial training window has elapsed
    if mode == "rolling":
        first_test_start = start + DateOffset(years=train_window_years)
    else:  # expanding
        first_test_start = start + DateOffset(years=min_train_years)

    current_test_start = first_test_start

    while current_test_start + DateOffset(months=test_window_months) <= end + DateOffset(days=1):
        current_test_end = current_test_start + DateOffset(months=test_window_months)

        # Determine the training window
        if mode == "rolling":
            train_start = current_test_start - DateOffset(years=train_window_years)
        else:  # expanding
            train_start = start

        # Select indices via date masks
        # Train: [train_start, current_test_start), strictly BEFORE test
        # Test:  [current_test_start, current_test_end)
        train_mask = (index >= train_start) & (index < current_test_start)
        test_mask = (index >= current_test_start) & (index < current_test_end)

        train_idx = index[train_mask]
        test_idx = index[test_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        current_test_start = current_test_start + DateOffset(months=step_months)

    # --- Partial last fold: use the remaining data ---
    if current_test_start < end:
        if mode == "rolling":
            train_start = current_test_start - DateOffset(years=train_window_years)
        else:
            train_start = start

        train_mask = (index >= train_start) & (index < current_test_start)
        test_mask = (index >= current_test_start)  # up to the end of the data

        train_idx = index[train_mask]
        test_idx = index[test_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def summarize_splits(
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
) -> pd.DataFrame:
    """
    Creates an overview table of the walk-forward splits.

    Per fold: train_start, train_end, test_start, test_end, n_train, n_test.
    Useful for sanity checks (overlap check, fold count, window sizes)
    and as the data source for the walk-forward schema visualization in step 2.
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
    Sanity check: verifies that no training window extends into the
    corresponding test range. Raises AssertionError on violation.

    Call directly after walk_forward_splits() in the code, before any
    training starts; protects against subtle off-by-one bugs in the
    date logic.
    """
    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        train_max = train_idx.max()
        test_min = test_idx.min()
        assert train_max < test_min, (
            f"Fold {fold_id}: train end ({train_max}) does not lie strictly before "
            f"test start ({test_min}): possible look-ahead!"
        )

def run_walk_forward(
    df: pd.DataFrame,
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    cfg,
    models_to_run: list[str],
) -> pd.DataFrame:
    """
    Walk-forward with parallelized CPU models (MSM, HMM) and
    sequential DL training (LSTM, Transformer) on the GPU.

    Parallelization applies exclusively within the CPU fold loop;
    results are bit-identical to the sequential variant, since every fold
    has its own RandomState and no shared state.
    """
    import warnings
    import logging
    from src.backtest.parallel import run_folds_parallel
    from src.models.lstm import train_lstm_fold
    from src.models.transformer import train_transformer_fold

    logger = logging.getLogger("model_service")
    n_jobs = getattr(cfg.walk_forward, "n_jobs", -1)

    # 1. Create the supervised labels once for the entire DF
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

    # 2. CPU models in parallel over all folds
    logger.info(
        f"Walk-forward CPU phase start: n_jobs={n_jobs}, folds={len(splits)}, "
        f"models={[m for m in models_to_run if m in ('MSM', 'HMM', 'HMM_Uni')]}"
    )
    parallel_results = run_folds_parallel(
        df, splits,
        msm_cfg=cfg.models.msm if "MSM" in models_to_run else None,
        hmm_cfg=cfg.models.hmm if "HMM" in models_to_run else None,
        hmm_uni_cfg=cfg.models.hmm_uni if "HMM_Uni" in models_to_run else None,
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
    logger.info("Walk-forward CPU phase done")

    # 3. DL models sequentially (GPU-bound)
    features = cfg.features.model_features
    label_col = resolve_label_col(cfg)

    logger.info(
        f"Walk-forward DL phase start: folds={len(splits)}, "
        f"models={[m for m in models_to_run if m in ('LSTM', 'Transformer')]}"
    )
    # Warm start between folds: use the weights from fold N-1 as initialization
    # for fold N (rolling window -> ~90% train overlap -> legitimate, since
    # fold N-1 has never seen the fold-N test data).
    # For the first fold or after failures: cold start (state = None).
    #
    # Seed-averaged ensemble: the seed-sensitivity analysis showed LSTM and
    # Transformer swing between retrainings (headline CV ~0.12). We therefore
    # train dl_ensemble_size members per fold with distinct global seeds and
    # average their OOS probabilities before thresholding, which shrinks the
    # prediction variance ~1/sqrt(N). Each member keeps its OWN warm-start state
    # across folds (states are lists of length N). dl_ensemble_size=1 reduces to
    # a single, now seed-pinned (hence reproducible) model.
    from src.backtest.hpo_analysis import _set_global_seeds

    dl_warm_start = getattr(cfg.walk_forward, "dl_warm_start", False)
    epochs_warm = getattr(cfg.walk_forward, "dl_warm_start_epochs", None)
    n_ens = max(1, int(getattr(cfg.walk_forward, "dl_ensemble_size", 1)))
    seed_base = int(getattr(cfg.walk_forward, "dl_ensemble_seed_base", 0))
    logger.info(f"Walk-forward DL ensemble size: {n_ens} (seed base {seed_base})")

    lstm_states = [None] * n_ens
    transformer_states = [None] * n_ens

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        df_train = df.loc[train_idx].copy()
        df_test  = df.loc[test_idx]

        # External turning-point labels must be recomputed as of each training
        # cutoff. The global label stored in result_df is descriptive only and
        # may use later observations; it must never enter fold training.
        if supervised_label_source != "hmm":
            df_train["Supervised_Label"] = compute_supervised_labels_asof(
                df, train_idx, cfg,
            )

        if "LSTM" in models_to_run:
            try:
                c = cfg.models.lstm
                member_probs, new_states, pred_idx = [], [], None
                for m in range(n_ens):
                    _set_global_seeds(seed_base + m)
                    st = lstm_states[m]
                    probs_raw, pred_idx, st_new = train_lstm_fold(
                        df_train=df_train, df_test=df_test,
                        features=features, labels_col=label_col,
                        window_size=c.window_size, units_l1=c.units_l1, units_l2=c.units_l2,
                        return_sequences=c.return_sequences, dropout=c.dropout,
                        dense=c.dense, activation=c.activation, optimizer=c.optimizer,
                        learning_rate=c.learning_rate,
                        metrics=c.metrics, epochs=c.epochs, batch_size=c.batch_size,
                        validation_split=c.validation_split, verbose=0,
                        init_weights=st if dl_warm_start else None,
                        epochs_warm=epochs_warm if (dl_warm_start and st is not None) else None,
                    )
                    member_probs.append(np.asarray(probs_raw, dtype=float))
                    new_states.append(st_new)
                lstm_states = new_states
                probs_mean = np.mean(member_probs, axis=0)
                signal = (probs_mean >= c.threshold).astype(int)
                result_df.loc[pred_idx, "LSTM_Prob"]   = probs_mean
                result_df.loc[pred_idx, "LSTM_Signal"] = signal
            except Exception as e:
                import traceback
                warnings.warn(f"[LSTM] Fold {fold_id} failed: {type(e).__name__}: {e}")
                if failed_folds["LSTM"] < 2:
                    traceback.print_exc()
                failed_folds["LSTM"] += 1
                # Discard the warm starts so that the next fold starts cold again.
                lstm_states = [None] * n_ens

        if "Transformer" in models_to_run:
            try:
                c = cfg.models.transformer
                member_probs, new_states, pred_idx = [], [], None
                for m in range(n_ens):
                    _set_global_seeds(seed_base + m)
                    st = transformer_states[m]
                    probs_raw, pred_idx, st_new = train_transformer_fold(
                        df_train=df_train, df_test=df_test,
                        features=features, labels_col=label_col,
                        window_size=c.window_size, d_model=c.d_model, n_heads=c.n_heads,
                        n_layers=c.n_layers, dim_feedforward=c.dim_feedforward,
                        dropout=c.dropout, learning_rate=c.learning_rate,
                        epochs=c.epochs, batch_size=c.batch_size,
                        validation_split=c.validation_split, verbose=0,
                        init_state_dict=st if dl_warm_start else None,
                        epochs_warm=epochs_warm if (dl_warm_start and st is not None) else None,
                    )
                    member_probs.append(np.asarray(probs_raw, dtype=float))
                    new_states.append(st_new)
                transformer_states = new_states
                probs_mean = np.mean(member_probs, axis=0)
                signal = (probs_mean >= c.threshold).astype(int)
                result_df.loc[pred_idx, "Transformer_Prob"]   = probs_mean
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
                # Discard the warm starts so that the next fold starts cold again.
                transformer_states = [None] * n_ens

    logger.info("Walk-forward DL phase done")

    # 4. Final report
    print(f"\n=== Walk-forward complete ===")
    for model_name, n_failed in failed_folds.items():
        n_oos = result_df[f"{model_name}_Signal"].notna().sum()
        print(f"  {model_name}: {n_oos} OOS days, {n_failed} folds failed")

    return result_df

_WF_CACHE_SCHEMA_VERSION = 2
_WF_IMPLEMENTATION_FILES = (
    "src/backtest/walk_forward.py",
    "src/backtest/parallel.py",
    "src/backtest/hpo_analysis.py",  # global seed setup used by production DL
    "src/models/common.py",
    "src/models/msm.py",
    "src/models/hmm.py",
    "src/models/lstm.py",
    "src/models/transformer.py",
    "src/data/labels/resolver.py",
    "src/data/labels/pagan_sossounov.py",
    "src/data/labels/peak_to_trough.py",
)
_WF_RUNTIME_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "statsmodels",
    "hmmlearn",
    "tensorflow",
    "keras",
    "torch",
)


def _canonicalize_fingerprint_value(value):
    """Convert config namespaces and scalar types to stable JSON values."""
    if isinstance(value, SimpleNamespace):
        return {
            key: _canonicalize_fingerprint_value(val)
            for key, val in sorted(vars(value).items())
        }
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_fingerprint_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_fingerprint_value(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _dataframe_fingerprint(df: pd.DataFrame) -> dict:
    """Hash index, schema, and every input value used by walk-forward."""
    row_hashes = pd.util.hash_pandas_object(df, index=True, categorize=True)
    values_hash = hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()
    schema = {
        "columns": [str(col) for col in df.columns],
        "dtypes": [str(dtype) for dtype in df.dtypes],
        "index_name": str(df.index.name),
        "index_dtype": str(df.index.dtype),
        "shape": list(df.shape),
    }
    return {"schema": schema, "values_sha256": values_hash}


def _implementation_fingerprint() -> dict[str, str]:
    """Hash the source files that can alter cached model probabilities."""
    project_root = Path(__file__).resolve().parents[2]
    hashes = {}
    for relative in _WF_IMPLEMENTATION_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Walk-forward fingerprint source file missing: {path}"
            )
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _runtime_fingerprint() -> dict:
    """Capture numerical-library versions that can change fitted outputs."""
    packages = {}
    for package in _WF_RUNTIME_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "missing"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": packages,
    }


def _walk_forward_fingerprint(cfg, df: pd.DataFrame) -> str:
    """
    Deterministic provenance hash over effective configuration, complete input
    data (schema, index and values), and the implementation source files that
    can influence probabilities. Any such change invalidates the cache.
    """
    provenance = {
        "cache_schema_version": _WF_CACHE_SCHEMA_VERSION,
        "effective_config": {
            "data": _canonicalize_fingerprint_value(cfg.data),
            "features": _canonicalize_fingerprint_value(cfg.features),
            "portfolio": _canonicalize_fingerprint_value(cfg.portfolio),
            "models": _canonicalize_fingerprint_value(cfg.models),
            "walk_forward": _canonicalize_fingerprint_value(cfg.walk_forward),
            "labels": _canonicalize_fingerprint_value(cfg.labels),
        },
        "input_dataframe": _dataframe_fingerprint(df),
        "implementation": _implementation_fingerprint(),
        "runtime": _runtime_fingerprint(),
    }
    raw = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_walk_forward_cache(
    test_df: pd.DataFrame,
    fingerprint: str,
    cache_path: str,
) -> None:
    """Saves the OOS results + fingerprint as Parquet with metadata."""
    test_df.attrs["wf_fingerprint"] = fingerprint
    test_df.to_parquet(cache_path)
    # Store the fingerprint separately as .txt (Parquet attrs are lost with
    # some engines)
    with open(cache_path + ".fingerprint", "w") as f:
        f.write(fingerprint)
    print(f"  Walk-forward cache saved: {cache_path}")


def load_walk_forward_cache(
    cache_path: str,
    expected_fingerprint: str,
) -> pd.DataFrame | None:
    """
    Loads the cache if it exists AND the fingerprint matches.
    Returns None if the cache is invalid/missing.
    """
    import os
    fp_path = cache_path + ".fingerprint"

    if not os.path.exists(cache_path) or not os.path.exists(fp_path):
        return None

    with open(fp_path, "r") as f:
        stored_fp = f.read().strip()

    if stored_fp != expected_fingerprint:
        print(f"  Walk-forward cache invalid (fingerprint mismatch). Retraining required.")
        return None

    print(f"  Walk-forward cache loaded: {cache_path}")
    return pd.read_parquet(cache_path)
