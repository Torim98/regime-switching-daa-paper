"""
Optuna hyperparameter optimization: Bayesian / grid search with walk-forward CV.

Design (revised for the paper, Issue #5)
----------------------------------------
Each objective function:
1. Samples hyperparameters from cfg.optimization.search_spaces via trial.suggest_*()
2. Iterates over the (optionally date-restricted, subsampled) walk-forward folds
3. Trains the model per fold on the train window (DL folds warm-start from the
   previous fold exactly as in the final walk-forward run)
4. Concatenates the OOS net returns of all folds into ONE pooled OOS series
5. Computes the configured risk metric on that pooled series (Optuna maximizes)

Why a pooled OOS series instead of the fold-wise median Sharpe:
The fold-median aggregates over a sequence of mostly bullish 12-month test
windows, so rare crisis folds only weakly shape the objective (thesis ch. 4.4).
Path-dependent tail metrics (Ulcer, Martin, Calmar, MaxDD) computed on the
pooled series are instead driven by exactly those crisis segments, which aligns
the objective with the paper's SORR / tail-risk goal.

Objective metric (cfg.optimization.metric):
    "sharpe" | "sortino" | "calmar" | "martin" | "ulcer" | "max_drawdown"
All metrics are normalized to "higher = better" internally, so the study
direction is always "maximize". The full metric vector of every trial is stored
as user_attrs, which lets the paper build an objective-sensitivity table
(best config under Martin vs. Sharpe vs. Sortino) without re-running the search.

Samplers:
- Econometric models (cfg.optimization.grid_models) use GridSampler, i.e. the
  low-dimensional space is evaluated exhaustively -> "global optimum within the
  grid" rather than "a sample of the space".
- DL models use a multivariate TPESampler.

Selection vs. evaluation (Issue #5):
cfg.optimization.tune_until restricts the HPO to folds whose test window ends on
or before that date (development period; e.g. Dotcom + GFC). The final
walk-forward run (walk_forward.py) is untouched and still uses all folds, so the
holdout folds (e.g. COVID, 2022) remain selection-free.

No look-ahead bias: Optuna only observes OOS metrics. Each fold's labels and
scaler are fit on the train window only. The fold splits are identical to those
in the final walk-forward run (walk_forward.py).
"""

import warnings
import math
from typing import Callable
import numpy as np
import pandas as pd
import optuna
import os

from src.data.labels.resolver import compute_supervised_labels_asof, resolve_label_col
from src.backtest.walk_forward import walk_forward_splits
from src.backtest.engine import BacktestState, compute_strategy_log_returns

# ============================================================================
# Objective metrics (all normalized to "higher = better")
# ============================================================================

# Sign that maps each raw metric onto a maximization score.
# ulcer is a loss (lower = better) -> maximize its negative.
# max_drawdown is already <= 0 (less negative = better) -> maximize as-is.
_MAXIMIZE_SIGN = {
    "sharpe": 1.0,
    "sortino": 1.0,
    "calmar": 1.0,
    "martin": 1.0,
    "max_drawdown": 1.0,
    "ulcer": -1.0,
}
ALLOWED_METRICS = tuple(_MAXIMIZE_SIGN.keys())

# Worst possible maximization score for failed / degenerate trials.
_SENTINEL = -1.0e9


def _ulcer_from_equity(equity: np.ndarray) -> float:
    """Ulcer index (RMS drawdown, in %) via the canonical evaluation helper."""
    # Reuse the single source of truth so the HPO objective and the reported
    # Ulcer index in the results are computed identically.
    from src.backtest.evaluation import ulcer_index
    return float(ulcer_index(pd.Series(equity)))


def compute_oos_metrics(
    daily_log_returns: np.ndarray,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    """
    Full risk-metric vector for a pooled OOS net-return series.

    Input convention: `daily_log_returns` are LOG returns, i.e. exactly what
    engine.compute_strategy_log_returns produces (and what every caller in this
    package passes). They are converted to simple returns here, so the whole
    metric vector matches engine.calculate_annualized_metrics, which works on
    pct_change of the capital curve (canonical Sharpe (1966) convention).

    Compounding log returns as if they were simple ones understates growth by
    the volatility drag (~sigma^2/2) and deepens every drawdown-based metric,
    with a penalty that grows with volatility -- for the Martin objective that
    is a second, unintended volatility penalty on top of the Ulcer denominator.

    Returns sharpe, sortino, calmar, martin, ulcer, max_drawdown and cagr.
    Metrics are defined so that (except ulcer) higher = better; ulcer and
    max_drawdown are reported as their natural raw values.
    """
    rets = np.asarray(daily_log_returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    # Log -> simple; equity below is then the exact capital curve.
    rets = np.expm1(rets)
    n = len(rets)
    empty = {k: 0.0 for k in ("sharpe", "sortino", "calmar",
                              "martin", "ulcer", "max_drawdown", "cagr")}
    if n < 20:
        return empty

    tdpy = trading_days_per_year
    mean, sd = rets.mean(), rets.std()
    sharpe = float(mean / sd * np.sqrt(tdpy)) if sd > 0 else 0.0

    downside = rets[rets < 0]
    dsd = downside.std() * np.sqrt(tdpy) if downside.size > 0 else 0.0
    sortino = float(mean * tdpy / dsd) if dsd > 0 else 0.0

    equity = np.cumprod(1.0 + rets)     # rets are simple here (see docstring)
    total = float(equity[-1])
    cagr = float(total ** (tdpy / n) - 1.0) if total > 0 else -1.0

    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    mdd = float(dd.min())
    calmar = float(cagr / max(abs(mdd), 1e-6))

    ulcer = _ulcer_from_equity(equity)          # in percent
    # Floor the Ulcer index so a (near) drawdown-free path does not blow the
    # Martin ratio up to +inf and dominate the search numerically.
    martin = float(cagr * 100.0 / max(ulcer, 0.1))

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "martin": martin,
        "ulcer": ulcer,
        "max_drawdown": mdd,
        "cagr": cagr,
    }


def _objective_score(metrics: dict[str, float], metric: str) -> float:
    """Map the raw metric vector onto a single maximization score."""
    return _MAXIMIZE_SIGN[metric] * metrics[metric]


# ============================================================================
# Search-space parsing (cfg.optimization.search_spaces -> trial.suggest_*)
# ============================================================================

def _suggest_from_spec(trial: optuna.Trial, name: str, spec) -> object:
    """Sample one hyperparameter from its YAML spec (SimpleNamespace)."""
    kind = spec.type
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec.choices))
    if kind == "int":
        step = int(getattr(spec, "step", 1))
        return trial.suggest_int(name, int(spec.low), int(spec.high), step=step)
    if kind == "float":
        if getattr(spec, "log", False):
            return trial.suggest_float(name, float(spec.low), float(spec.high), log=True)
        step = getattr(spec, "step", None)
        if step is not None:
            return trial.suggest_float(name, float(spec.low), float(spec.high), step=float(step))
        return trial.suggest_float(name, float(spec.low), float(spec.high))
    raise ValueError(f"Unknown search-space type '{kind}' for parameter '{name}'.")


def _suggest_space(trial: optuna.Trial, space_ns) -> dict:
    """Sample every parameter of a model's search space, preserving YAML order."""
    return {name: _suggest_from_spec(trial, name, spec)
            for name, spec in vars(space_ns).items()}


def _build_grid(space_ns) -> dict[str, list]:
    """
    Expand a search-space spec into an explicit grid for optuna.GridSampler.

    Only categorical and stepped-float / stepped-int parameters are supported
    here (which is all the econometric models use). A continuous parameter in a
    grid model would raise, since it cannot be enumerated.
    """
    grid: dict[str, list] = {}
    for name, spec in vars(space_ns).items():
        kind = spec.type
        if kind == "categorical":
            grid[name] = list(spec.choices)
        elif kind == "int":
            step = int(getattr(spec, "step", 1))
            grid[name] = list(range(int(spec.low), int(spec.high) + 1, step))
        elif kind == "float":
            step = getattr(spec, "step", None)
            if step is None:
                raise ValueError(
                    f"Grid model parameter '{name}' is a continuous float without "
                    f"'step'; cannot enumerate. Add a step or use a TPE model."
                )
            n = int(round((float(spec.high) - float(spec.low)) / float(step)))
            grid[name] = [round(float(spec.low) + i * float(step), 10)
                          for i in range(n + 1)]
        else:
            raise ValueError(f"Unsupported grid type '{kind}' for '{name}'.")
    return grid


# ============================================================================
# Fold helpers
# ============================================================================

def _fold_portfolio_returns(
    df_test: pd.DataFrame,
    signal: pd.Series,
    signal_shift: int = 1,
    fee: float = 0.001,
    state: BacktestState | None = None,
) -> tuple[np.ndarray, BacktestState]:
    """
    Computes daily net portfolio returns for an OOS fold and returns the
    execution state needed by the chronologically following fold.

    Replicates the logic from engine.backtest():
    - Shift the signal by signal_shift days (look-ahead prevention)
    - Signal=0 -> portfolio return, signal=1 -> cash return
    - Subtract transaction costs on signal switches
    """
    return compute_strategy_log_returns(
        df=df_test,
        signal=signal,
        signal_shift=signal_shift,
        fee=fee,
        state=state,
    )


def _subsample_splits(
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    every_nth: int | None,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Select every n-th fold for faster optimization."""
    if every_nth is None or every_nth <= 1:
        return splits
    return splits[::every_nth]


def _filter_splits_until(
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    tune_until,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Keep only folds whose OOS test window ends on or before `tune_until`
    (development period for HPO, Issue #5). None / "" -> no restriction.
    """
    if tune_until is None or (isinstance(tune_until, str) and not tune_until.strip()):
        return splits
    cutoff = pd.Timestamp(tune_until)
    return [(tr, te) for (tr, te) in splits if te.max() <= cutoff]


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
        n_init=getattr(hmm_cfg, "n_init", 1),
    )

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["HMM_Signal"] = signal_train.values
    df_test["HMM_Signal"] = signal_test.values
    return df_train, df_test


# ---- Pooled-OOS accumulation, pruning and finalization ---------------------

def _maybe_prune(trial, pooled, metric, fold_id, prune_enabled):
    """Report the running pooled score and prune if the pruner requests it."""
    if not prune_enabled:
        return
    running = (_objective_score(compute_oos_metrics(np.concatenate(pooled)), metric)
               if pooled else _SENTINEL)
    trial.report(running, fold_id)
    if trial.should_prune():
        raise optuna.TrialPruned()


def _finalize(trial, pooled, metric, n_failed, n_folds):
    """Compute the pooled metric vector, log all metrics, return the score."""
    if not pooled or n_failed >= math.ceil(0.5 * max(n_folds, 1)):
        return _SENTINEL
    pooled_rets = np.concatenate(pooled)
    metrics = compute_oos_metrics(pooled_rets)
    for k, v in metrics.items():
        trial.set_user_attr(k, float(v))
    trial.set_user_attr("n_oos_days", int(len(pooled_rets)))
    trial.set_user_attr("n_failed_folds", int(n_failed))
    return _objective_score(metrics, metric)


def _dl_warm_cfg(cfg):
    """(dl_warm_start, epochs_warm, max_epochs) shared by the DL objectives."""
    dl_warm = getattr(cfg.walk_forward, "dl_warm_start", False)
    epochs_warm = getattr(cfg.walk_forward, "dl_warm_start_epochs", None)
    max_epochs = int(getattr(cfg.optimization, "dl_max_epochs", 150))
    return dl_warm, epochs_warm, max_epochs


# ============================================================================
# Objective functions per model
# ============================================================================

def objective_msm(
    trial: optuna.Trial,
    df: pd.DataFrame,
    splits: list,
    cfg,
    fee: float,
    signal_shift: int,
    metric: str,
    prune_enabled: bool,
) -> float:
    """MSM: optimize threshold (k_regimes fixed at 2)."""
    from src.models.msm import train_msm_fold

    params = _suggest_space(trial, cfg.optimization.search_spaces.MSM)
    threshold = params["threshold"]
    k_regimes = 2

    pooled, n_failed, portfolio_state = [], 0, None
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
            fold_rets, portfolio_state = _fold_portfolio_returns(
                df_test, signal, signal_shift, fee, portfolio_state,
            )
            pooled.append(fold_rets)
        except Exception as e:
            warnings.warn(f"MSM trial {trial.number}, fold {fold_id}: {e}")
            n_failed += 1
            portfolio_state = None

        _maybe_prune(trial, pooled, metric, fold_id, prune_enabled)

    return _finalize(trial, pooled, metric, n_failed, len(splits))


def objective_hmm(
    trial: optuna.Trial,
    df: pd.DataFrame,
    splits: list,
    cfg,
    fee: float,
    signal_shift: int,
    metric: str,
    prune_enabled: bool,
) -> float:
    """HMM: optimize covariance_type and threshold (n_components fixed at 2)."""
    from src.models.hmm import train_hmm_fold

    params = _suggest_space(trial, cfg.optimization.search_spaces.HMM)
    covariance_type = params["covariance_type"]
    threshold = params["threshold"]
    n_components = 2
    hmm_features = cfg.models.hmm.features

    pooled, n_failed, portfolio_state = [], 0, None
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
                n_init=getattr(cfg.models.hmm, "n_init", 1),
            )
            fold_rets, portfolio_state = _fold_portfolio_returns(
                df_test, signal, signal_shift, fee, portfolio_state,
            )
            pooled.append(fold_rets)
        except Exception as e:
            warnings.warn(f"HMM trial {trial.number}, fold {fold_id}: {e}")
            n_failed += 1
            portfolio_state = None

        _maybe_prune(trial, pooled, metric, fold_id, prune_enabled)

    return _finalize(trial, pooled, metric, n_failed, len(splits))


def objective_hmm_uni(
    trial: optuna.Trial,
    df: pd.DataFrame,
    splits: list,
    cfg,
    fee: float,
    signal_shift: int,
    metric: str,
    prune_enabled: bool,
) -> float:
    """
    HMM_Uni: optimize the threshold only.

    covariance_type is NOT tuned: with univariate input (returns only),
    full/diag/spherical are identical state-specific 1x1 covariances. ``tied``
    is deliberately excluded because it forces one shared variance. The space is
    therefore congruent with the MSM objective; a fair basis for the
    architecture comparison (Issue #3).
    """
    from src.models.hmm import train_hmm_fold

    params = _suggest_space(trial, cfg.optimization.search_spaces.HMM_Uni)
    threshold = params["threshold"]
    uni_cfg = cfg.models.hmm_uni

    pooled, n_failed, portfolio_state = [], 0, None
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
                n_init=getattr(uni_cfg, "n_init", 1),
            )
            fold_rets, portfolio_state = _fold_portfolio_returns(
                df_test, signal, signal_shift, fee, portfolio_state,
            )
            pooled.append(fold_rets)
        except Exception as e:
            warnings.warn(f"HMM_Uni trial {trial.number}, fold {fold_id}: {e}")
            n_failed += 1
            portfolio_state = None

        _maybe_prune(trial, pooled, metric, fold_id, prune_enabled)

    return _finalize(trial, pooled, metric, n_failed, len(splits))


def objective_lstm(
    trial: optuna.Trial,
    df: pd.DataFrame,
    splits: list,
    cfg,
    fee: float,
    signal_shift: int,
    metric: str,
    prune_enabled: bool,
) -> float:
    """
    LSTM: optimize window_size, units_l1/l2, batch_size, learning_rate,
    dropout and threshold. Epochs are NOT tuned; a large dl_max_epochs is
    combined with the trainer's early stopping. DL folds warm-start from the
    previous fold, mirroring the final walk-forward run exactly.
    """
    from src.models.lstm import train_lstm_fold

    p = _suggest_space(trial, cfg.optimization.search_spaces.LSTM)
    window_size = p["window_size"]
    units_l1 = p["units_l1"]
    units_l2 = p["units_l2"]
    batch_size = p["batch_size"]
    learning_rate = p["learning_rate"]
    dropout = p["dropout"]
    threshold = p["threshold"]

    lstm_cfg = cfg.models.lstm
    features = cfg.features.model_features
    labels_col = resolve_label_col(cfg)
    dl_warm, epochs_warm, max_epochs = _dl_warm_cfg(cfg)

    pooled, n_failed, portfolio_state = [], 0, None
    lstm_state = None
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        try:
            df_train = df.loc[train_idx].copy()
            df_test = df.loc[test_idx]

            if cfg.labels.supervised_label_source == "hmm":
                df_train, df_test = _generate_hmm_labels(df_train, df_test, cfg)
            else:
                df_train["Supervised_Label"] = compute_supervised_labels_asof(
                    df, train_idx, cfg,
                )

            probs_raw, pred_idx, lstm_state = train_lstm_fold(
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
                optimizer=lstm_cfg.optimizer,
                learning_rate=learning_rate,
                metrics=lstm_cfg.metrics,
                epochs=max_epochs,
                batch_size=batch_size,
                validation_split=lstm_cfg.validation_split,
                verbose=0,
                init_weights=lstm_state if dl_warm else None,
                epochs_warm=epochs_warm if (dl_warm and lstm_state is not None) else None,
            )

            signal = (probs_raw >= threshold).astype(int)
            signal_series = pd.Series(signal, index=pred_idx)
            df_test_aligned = df_test.loc[pred_idx]
            fold_rets, portfolio_state = _fold_portfolio_returns(
                df_test_aligned, signal_series, signal_shift, fee, portfolio_state,
            )
            pooled.append(fold_rets)
        except Exception as e:
            warnings.warn(f"LSTM trial {trial.number}, fold {fold_id}: {e}")
            n_failed += 1
            lstm_state = None  # cold restart for the next fold
            portfolio_state = None

        _maybe_prune(trial, pooled, metric, fold_id, prune_enabled)

    return _finalize(trial, pooled, metric, n_failed, len(splits))


def objective_transformer(
    trial: optuna.Trial,
    df: pd.DataFrame,
    splits: list,
    cfg,
    fee: float,
    signal_shift: int,
    metric: str,
    prune_enabled: bool,
) -> float:
    """
    Transformer: optimize window_size, the (d_model, n_heads) pair, n_layers,
    dim_feedforward, batch_size, learning_rate, dropout and threshold.

    The divisibility constraint d_model % n_heads == 0 is guaranteed by
    construction: only valid pairs are offered as a single categorical
    'dmodel_nheads' (e.g. "64-8"), so no trial is wasted on a pruned constraint
    violation and the pruning statistics stay interpretable. Epochs are not
    tuned (early stopping + dl_max_epochs). DL folds warm-start.
    """
    from src.models.transformer import train_transformer_fold

    p = _suggest_space(trial, cfg.optimization.search_spaces.Transformer)
    d_model, n_heads = (int(x) for x in p["dmodel_nheads"].split("-"))
    n_layers = p["n_layers"]
    dim_feedforward = p["dim_feedforward"]
    batch_size = p["batch_size"]
    learning_rate = p["learning_rate"]
    dropout = p["dropout"]
    window_size = p["window_size"]
    threshold = p["threshold"]

    t_cfg = cfg.models.transformer
    features = cfg.features.model_features
    labels_col = resolve_label_col(cfg)
    dl_warm, epochs_warm, max_epochs = _dl_warm_cfg(cfg)

    pooled, n_failed, portfolio_state = [], 0, None
    transformer_state = None
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        try:
            df_train = df.loc[train_idx].copy()
            df_test = df.loc[test_idx]

            if cfg.labels.supervised_label_source == "hmm":
                df_train, df_test = _generate_hmm_labels(df_train, df_test, cfg)
            else:
                df_train["Supervised_Label"] = compute_supervised_labels_asof(
                    df, train_idx, cfg,
                )

            probs_raw, pred_idx, transformer_state = train_transformer_fold(
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
                epochs=max_epochs,
                batch_size=batch_size,
                validation_split=t_cfg.validation_split,
                verbose=0,
                init_state_dict=transformer_state if dl_warm else None,
                epochs_warm=epochs_warm if (dl_warm and transformer_state is not None) else None,
            )

            signal = (probs_raw >= threshold).astype(int)
            signal_series = pd.Series(signal, index=pred_idx)
            df_test_aligned = df_test.loc[pred_idx]
            fold_rets, portfolio_state = _fold_portfolio_returns(
                df_test_aligned, signal_series, signal_shift, fee, portfolio_state,
            )
            pooled.append(fold_rets)
        except Exception as e:
            warnings.warn(f"Transformer trial {trial.number}, fold {fold_id}: {e}")
            n_failed += 1
            transformer_state = None  # cold restart for the next fold
            portfolio_state = None

        _maybe_prune(trial, pooled, metric, fold_id, prune_enabled)

        # Free GPU memory after each fold
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return _finalize(trial, pooled, metric, n_failed, len(splits))


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


def _resolve_from_cfg(cfg, attr: str, model_name: str):
    """Read a per-model value from cfg.optimization.<attr>.<model_name>."""
    container = getattr(cfg.optimization, attr, None)
    if container is None:
        return None
    return getattr(container, model_name, None)


def _resolve_metric(cfg, metric: str | None) -> str:
    """Resolve and validate the objective metric."""
    if metric is None:
        metric = getattr(cfg.optimization, "metric", "sharpe")
    metric = str(metric).lower()
    if metric not in ALLOWED_METRICS:
        raise ValueError(
            f"Unknown optimization metric '{metric}'. Allowed: {list(ALLOWED_METRICS)}"
        )
    return metric


def _make_sampler(cfg, model_name: str, use_grid: bool, space_ns):
    """Build the GridSampler (econometric) or TPESampler (DL) for a model."""
    sampler_cfg = getattr(cfg.optimization, "sampler", None)
    seed = int(getattr(sampler_cfg, "seed", 42)) if sampler_cfg else 42

    if use_grid:
        grid = _build_grid(space_ns)
        grid_size = int(np.prod([len(v) for v in grid.values()]))
        return optuna.samplers.GridSampler(grid, seed=seed), grid_size

    multivariate = bool(getattr(sampler_cfg, "multivariate", True)) if sampler_cfg else True
    group = bool(getattr(sampler_cfg, "group", True)) if sampler_cfg else True
    startup_ns = getattr(sampler_cfg, "n_startup_trials", None) if sampler_cfg else None
    n_startup = int(getattr(startup_ns, model_name, 20)) if startup_ns else 20
    sampler = optuna.samplers.TPESampler(
        multivariate=multivariate,
        group=group,
        seed=seed,
        n_startup_trials=n_startup,
    )
    return sampler, None


def _make_pruner(cfg, prune_enabled: bool):
    """MedianPruner if pruning is enabled, otherwise NopPruner."""
    if not prune_enabled:
        return optuna.pruners.NopPruner()
    pcfg = getattr(cfg.optimization, "pruning", None)
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(getattr(pcfg, "n_startup_trials", 5)) if pcfg else 5,
        n_warmup_steps=int(getattr(pcfg, "n_warmup_steps", 10)) if pcfg else 10,
    )


def run_optimization(
    model_name: str,
    df: pd.DataFrame,
    cfg,
    n_trials: int | None = None,
    every_nth_fold: int | None = None,
    storage: str | None = None,
    metric: str | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> optuna.Study:
    """
    Run an Optuna study for a single model.

    Parameters
    ----------
    model_name : str
        "MSM", "HMM", "HMM_Uni", "LSTM", or "Transformer".
    df : pd.DataFrame
        Feature-engineered DataFrame (Silver layer) with a DatetimeIndex.
    cfg : PipelineConfig
        Central configuration.
    n_trials : int | None
        Number of trials for TPE models. None = read from
        cfg.optimization.n_trials_per_model[model_name]. Ignored for grid
        models (the grid size is used instead).
    every_nth_fold : int | None
        Use only every n-th fold. None = read from
        cfg.optimization.every_nth_fold_per_model[model_name].
    storage : str | None
        Optuna storage URL (e.g. "sqlite:///optuna.db"). None = in-memory.
    metric : str | None
        Objective metric. None = cfg.optimization.metric.
    should_stop : Callable[[], bool] | None
        Cooperative cancellation hook. Checked once after every completed trial;
        when it returns True the study stops gracefully. All finished trials stay
        persisted in the Optuna storage, so a re-run resumes from exactly this
        point (the completed trials are skipped, only the remaining ones run).

    Returns
    -------
    optuna.Study with .best_params and .best_value (the maximization score of
    the configured metric). Per-trial user_attrs hold the full metric vector.
    """
    if model_name not in _OBJECTIVE_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(_OBJECTIVE_MAP.keys())}"
        )

    metric = _resolve_metric(cfg, metric)

    # --- Grid vs. TPE, sampler, trial budget, pruning ---
    grid_models = set(getattr(cfg.optimization, "grid_models", []) or [])
    use_grid = model_name in grid_models
    space_ns = getattr(cfg.optimization.search_spaces, model_name)
    sampler, grid_size = _make_sampler(cfg, model_name, use_grid, space_ns)

    if use_grid:
        n_trials = grid_size          # exhaustive
        prune_enabled = False         # never prune an exhaustive grid
    else:
        if n_trials is None:
            n_trials = _resolve_from_cfg(cfg, "n_trials_per_model", model_name)
            if n_trials is None:
                raise ValueError(
                    f"n_trials not passed and cfg.optimization."
                    f"n_trials_per_model.{model_name} is missing."
                )
        pcfg = getattr(cfg.optimization, "pruning", None)
        prune_enabled = bool(getattr(pcfg, "enabled", False)) if pcfg else False

    pruner = _make_pruner(cfg, prune_enabled)

    if every_nth_fold is None:
        every_nth_fold = _resolve_from_cfg(cfg, "every_nth_fold_per_model", model_name)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # --- Walk-forward splits: identical to the final run, then restricted ---
    wf = cfg.walk_forward
    splits = walk_forward_splits(
        index=df.index,
        mode=wf.mode,
        train_window_years=wf.train_window_years,
        test_window_months=wf.test_window_months,
        step_months=wf.step_months,
        min_train_years=wf.min_train_years,
    )
    n_total = len(splits)
    tune_until = getattr(cfg.optimization, "tune_until", None)
    splits = _filter_splits_until(splits, tune_until)
    n_dev = len(splits)
    splits = _subsample_splits(splits, every_nth_fold)

    print(f"\n{'='*60}")
    print(f"Optimization: {model_name} | metric={metric} | "
          f"{'grid' if use_grid else 'TPE'} | {n_trials} trials | {len(splits)} folds")
    if tune_until:
        print(f"  Selection window: folds ending <= {tune_until} "
              f"({n_dev}/{n_total} folds; holdout kept selection-free)")
    print(f"{'='*60}")

    # --- Resolve the storage path relative to the project root ---
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
        with rdb_storage.engine.connect() as _conn:
            _conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            _conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
            _conn.exec_driver_sql("PRAGMA busy_timeout=30000")

        storage_arg = rdb_storage

    # --- Study (versioned name so a changed search space / metric never ---
    # --- resumes into a study built under different distributions) ---
    suffix = getattr(cfg.optimization, "study_suffix", None)
    study_name = f"opt_{model_name}" + (f"_{suffix}" if suffix else "")
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_arg,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    study.set_user_attr("metric", metric)
    study.set_user_attr("tune_until", str(tune_until) if tune_until else "")

    fee = cfg.backtesting.transaction_cost_bps / 10_000
    signal_shift = cfg.backtesting.signal_shift

    objective = lambda trial: _OBJECTIVE_MAP[model_name](
        trial, df, splits, cfg, fee, signal_shift, metric, prune_enabled,
    )

    # Enqueue the current config as a baseline trial (TPE only). For grid
    # models the baseline is already a grid point, and enqueuing off-grid
    # points would confuse the GridSampler's coverage bookkeeping.
    if not use_grid:
        default_params = _get_default_params(model_name, cfg)
        if default_params:
            study.enqueue_trial(default_params, skip_if_exists=True)

    done = len([t for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.PRUNED)])
    remaining = max(0, n_trials - done)

    # Cooperative stop: a callback runs after every completed trial and asks
    # study.stop() to end the loop gracefully. The current trial always finishes
    # first, so its result is persisted too and a resume never re-runs it.
    callbacks = None
    if should_stop is not None:
        def _stop_callback(study_, trial_):
            if should_stop():
                study_.stop()
        callbacks = [_stop_callback]

    if remaining == 0:
        print(f"  -> {model_name}: {done}/{n_trials} trials already present, skipping.")
    elif should_stop is not None and should_stop():
        print(f"  -> {model_name}: stop already requested, not starting new trials.")
    else:
        print(f"  -> {done} trials present, starting {remaining} more.")
        study.optimize(objective, n_trials=remaining, show_progress_bar=True,
                       callbacks=callbacks)
        if should_stop is not None and should_stop():
            n_done = len([t for t in study.trials
                          if t.state in (optuna.trial.TrialState.COMPLETE,
                                         optuna.trial.TrialState.PRUNED)])
            print(f"  -> {model_name}: stopped on request at {n_done}/{n_trials} "
                  f"trials; re-run to resume from here.")

    print(f"\n--- {model_name}: best parameters ({metric}) ---")
    print(f"  {metric} score: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  Trials: {len(study.trials)} "
          f"(of which {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))} pruned)")

    try:
        from src.backtest.plots import save_optuna_plots
        save_optuna_plots(study, model_name, cfg)
    except ImportError:
        warnings.warn("Plotly/Kaleido not installed, skipping Optuna plots.")

    _maybe_apply_best_params(cfg, model_name, storage_arg)

    return study


def _maybe_apply_best_params(cfg, model_name: str, storage) -> None:
    """Write this model's best params into config.yaml, if enabled.

    Runs after the model's last trial so the subsequent pipeline steps (train,
    backtest, evaluation) pick the tuned values up without a manual copy. Kept
    best-effort on purpose: HPO is by far the most expensive step in the
    pipeline, and losing a finished sweep to a config-writing problem would be
    the worst possible trade. A failure is reported and the study is returned
    regardless -- the values remain recoverable from the Optuna DB and from
    assets/optuna_best_params.md.
    """
    if not getattr(cfg.optimization, "apply_best_params", False):
        return
    try:
        # Imported lazily: hpo_analysis imports this module at import time.
        from src.backtest.hpo_analysis import apply_best_params
        changes = apply_best_params(
            cfg, models=[model_name], storage=storage,
        )
        print(f"  -> {model_name}: applied {len(changes)} params to config.yaml.")
    except Exception as e:
        warnings.warn(
            f"[{model_name}] could not write best params to config.yaml: "
            f"{type(e).__name__}: {e}. The study is intact; apply manually via "
            f"`python -m src.backtest.hpo_analysis apply`."
        )


def optimize_all(
    df: pd.DataFrame,
    cfg,
    n_trials: int | None = None,
    every_nth_fold: int | None = None,
    models: list[str] | None = None,
    storage: str | None = None,
    metric: str | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, optuna.Study]:
    """
    Optimize all (or selected) models sequentially.

    Order: MSM -> HMM -> HMM_Uni -> LSTM -> Transformer

    Parameters
    ----------
    models : list[str] | None
        Models to optimize. None = all five.
    n_trials, every_nth_fold, metric : optional overrides for ALL models.
        None = read per model from the config.
    should_stop : Callable[[], bool] | None
        Cooperative cancellation hook (see run_optimization). Besides ending the
        current model's study early, a set flag also skips the remaining models
        so the whole optimize-all run returns promptly. Every completed study
        stays persisted, so a re-run resumes model by model.
    Remaining parameters : see run_optimization.

    Returns
    -------
    Dict[model_name -> optuna.Study].
    """
    if models is None:
        models = ["MSM", "HMM", "HMM_Uni", "LSTM", "Transformer"]

    studies = {}
    for model_name in models:
        studies[model_name] = run_optimization(
            model_name=model_name,
            df=df,
            cfg=cfg,
            n_trials=n_trials,
            every_nth_fold=every_nth_fold,
            storage=storage,
            metric=metric,
            should_stop=should_stop,
        )
        if should_stop is not None and should_stop():
            print(f"\nOptimization stopped on request after {model_name}; "
                  f"{len(studies)}/{len(models)} models done, the rest is skipped.")
            break

    print(f"\n{'='*60}")
    print("Optimization complete: summary")
    print(f"{'='*60}")
    for name, study in studies.items():
        m = study.user_attrs.get("metric", "score")
        print(f"\n{name} (best {m}: {study.best_value:.4f}):")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

    return studies


def _format_param_value(v) -> str:
    """Format a hyperparameter value readably for Markdown tables."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")  # narrow no-break space
    if isinstance(v, float):
        a = abs(v)
        if a > 0 and (a < 1e-3 or a >= 1e6):
            return f"{v:.3e}"
        return f"{float(f'{v:.4g}')}"
    return str(v)


def save_optuna_best_params(
    studies: dict[str, "optuna.Study"],
    cfg,
    metric_label: str | None = None,
) -> str:
    """
    Persists the best hyperparameters of all Optuna studies as Markdown.

    Overview table (model, best score, complete/pruned/total) plus, per model,
    a parameter table and the secondary risk metrics of the best trial (Sharpe,
    Sortino, Calmar, Martin, Ulcer, MaxDD) taken from its user_attrs. The
    secondary metrics support the objective-sensitivity discussion (Issue #5).
    Target path: cfg.asset_path("optuna_best_params").
    """
    from pathlib import Path
    import datetime
    import optuna as _optuna

    metric = getattr(cfg.optimization, "metric", "sharpe")
    if metric_label is None:
        metric_label = f"{metric} (pooled OOS)"

    _sec_keys = ["sharpe", "sortino", "calmar", "martin", "ulcer", "max_drawdown", "cagr"]

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

        attrs = study.best_trial.user_attrs
        if any(k in attrs for k in _sec_keys):
            lines.append("Secondary metrics of the best trial (pooled OOS):")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|:---|---:|")
            for k in _sec_keys:
                if k in attrs:
                    lines.append(f"| {k} | {attrs[k]:.4f} |")
            lines.append("")

    path = Path(cfg.asset_path("optuna_best_params"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ {path}")
    return str(path)


def _get_default_params(model_name: str, cfg) -> dict | None:
    """
    Current config values as an Optuna trial dict (TPE baseline). Keys must
    match the new search-space parameter names (cfg.optimization.search_spaces).
    """
    if model_name == "LSTM":
        c = cfg.models.lstm
        return {
            "window_size": c.window_size,
            "units_l1": c.units_l1,
            "units_l2": c.units_l2,
            "batch_size": c.batch_size,
            "learning_rate": c.learning_rate,
            "dropout": c.dropout,
            "threshold": c.threshold,
        }
    if model_name == "Transformer":
        c = cfg.models.transformer
        return {
            "window_size": c.window_size,
            "dmodel_nheads": f"{c.d_model}-{c.n_heads}",
            "n_layers": c.n_layers,
            "dim_feedforward": c.dim_feedforward,
            "batch_size": c.batch_size,
            "learning_rate": c.learning_rate,
            "dropout": c.dropout,
            "threshold": c.threshold,
        }
    # Grid models (MSM/HMM/HMM_Uni) do not enqueue a baseline.
    return None
