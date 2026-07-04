"""
Post-HPO analysis and finalization utilities (Issue #5).

Four building blocks, all operating on the persisted Optuna studies
(models/optuna_studies.db) and reusing the fold machinery from optimize.py so
that the numbers are identical to the search itself:

1. convergence_review    - best-so-far curve, fANOVA importance and an
                           edge-of-range check per model (no retraining).
2. apply_best_params     - write the best params of every study into
                           config.yaml via targeted, comment-preserving edits.
3. evaluate_params /
   multiseed_reeval       - re-run a FIXED hyperparameter set (optionally with
                           several global seeds) to quantify DL seed sensitivity.
4. deflated_sharpe_ratio /
   pbo_cscv               - price in the number of tested configs (DSR) and the
                           probability of backtest overfitting (CSCV/PBO).
5. objective_sensitivity  - best config under each candidate metric (Martin vs
                           Sharpe vs Sortino ...), valued across all metrics, to
                           show the ranking is not objective-cherry-picked.

CLI:
    python -m src.backtest.hpo_analysis review
    python -m src.backtest.hpo_analysis apply [--dry-run]
    python -m src.backtest.hpo_analysis multiseed --model LSTM --top 5 --seeds 5
    python -m src.backtest.hpo_analysis dsr --model LSTM
    python -m src.backtest.hpo_analysis pbo --model LSTM --top 20
    python -m src.backtest.hpo_analysis sensitivity [--save]
"""

import math
import warnings
import numpy as np
import pandas as pd
import optuna

from src.backtest import optimize as O
from src.backtest.walk_forward import walk_forward_splits


# ============================================================================
# Study access
# ============================================================================

def study_name_for(cfg, model_name: str) -> str:
    suffix = getattr(cfg.optimization, "study_suffix", None)
    return f"opt_{model_name}" + (f"_{suffix}" if suffix else "")


def load_study(cfg, model_name: str, storage: str | None = None) -> optuna.Study:
    if storage is None:
        storage = f"sqlite:///{cfg.model_path('optuna_db')}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.load_study(study_name=study_name_for(cfg, model_name), storage=storage)


def _complete_trials(study) -> list:
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]


# ============================================================================
# 1. Convergence + edge-of-range review (cheap, no retraining)
# ============================================================================

def _search_space(cfg, model_name):
    return getattr(cfg.optimization.search_spaces, model_name)


def _edge_flags(cfg, model_name: str, best_params: dict) -> list[str]:
    """
    Flag numeric best params sitting AT or within one grid step of a search
    bound (suggests the range may be too narrow and should be extended).
    Categorical parameters are not flagged: their 'ends' are unordered and
    cannot be extended, so an edge there carries no range signal.
    """
    space = _search_space(cfg, model_name)
    flags = []
    for name, spec in vars(space).items():
        if name not in best_params:
            # Transformer stores d_model/n_heads jointly as dmodel_nheads.
            continue
        val = best_params[name]
        kind = spec.type
        if kind == "float":
            low, high = float(spec.low), float(spec.high)
            step = float(getattr(spec, "step", (high - low) / 20.0))
            eps = step * 1e-6
            if val <= low + step + eps:
                flags.append(f"{name}={val} near LOWER bound {low}")
            elif val >= high - step - eps:
                flags.append(f"{name}={val} near UPPER bound {high}")
        elif kind == "int":
            low, high = int(spec.low), int(spec.high)
            step = int(getattr(spec, "step", 1))
            if val <= low + step:
                flags.append(f"{name}={val} near LOWER bound {low}")
            elif val >= high - step:
                flags.append(f"{name}={val} near UPPER bound {high}")
    return flags


def convergence_review(cfg, models: list[str] | None = None,
                       storage: str | None = None) -> pd.DataFrame:
    """
    Per model: best value, #complete/#pruned, whether the best value was found
    late (convergence proxy), top fANOVA importances and edge-of-range flags.
    """
    if models is None:
        models = list(cfg.optimization.models)

    rows = []
    for m in models:
        try:
            study = load_study(cfg, m, storage)
        except KeyError:
            rows.append({"model": m, "status": "no study"})
            continue
        complete = _complete_trials(study)
        if not complete:
            rows.append({"model": m, "status": "no complete trials"})
            continue

        best = study.best_trial
        values = [t.value for t in complete]
        best_at = best.number
        n_total = len(study.trials)
        # Fraction of the search elapsed before the best value appeared: a high
        # value means the search was still improving late -> budget may be tight.
        conv_frac = best_at / max(n_total - 1, 1)

        try:
            imp = optuna.importance.get_param_importances(
                study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42),
            )
            top_imp = ", ".join(f"{k}={v:.2f}" for k, v in list(imp.items())[:3])
        except Exception as e:
            top_imp = f"n/a ({type(e).__name__})"

        flags = _edge_flags(cfg, m, best.params)
        rows.append({
            "model": m,
            "metric": study.user_attrs.get("metric", ""),
            "best_value": round(best.value, 4),
            "best_trial": best_at,
            "conv_frac": round(conv_frac, 2),
            "n_complete": len(complete),
            "n_pruned": sum(1 for t in study.trials
                            if t.state == optuna.trial.TrialState.PRUNED),
            "top_importance": top_imp,
            "edge_flags": "; ".join(flags) if flags else "-",
        })
    return pd.DataFrame(rows).set_index("model")


# ============================================================================
# 2. Best-param transfer: studies -> config.yaml (comment-preserving)
# ============================================================================

def _fmt_yaml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return f'"{v}"'


def _study_updates(model_name: str, best_params: dict) -> list[tuple[str, str, object]]:
    """(model_block, key, value) edits for one study's best params."""
    p = best_params
    if model_name == "MSM":
        return [("msm", "threshold", p["threshold"])]
    if model_name == "HMM":
        return [("hmm", "covariance_type", p["covariance_type"]),
                ("hmm", "threshold", p["threshold"])]
    if model_name == "HMM_Uni":
        return [("hmm_uni", "threshold", p["threshold"])]
    if model_name == "LSTM":
        keys = ["window_size", "units_l1", "units_l2", "batch_size",
                "learning_rate", "dropout", "threshold"]
        return [("lstm", k, p[k]) for k in keys]
    if model_name == "Transformer":
        d_model, n_heads = (int(x) for x in p["dmodel_nheads"].split("-"))
        return [
            ("transformer", "window_size", p["window_size"]),
            ("transformer", "d_model", d_model),
            ("transformer", "n_heads", n_heads),
            ("transformer", "n_layers", p["n_layers"]),
            ("transformer", "dim_feedforward", p["dim_feedforward"]),
            ("transformer", "batch_size", p["batch_size"]),
            ("transformer", "dropout", p["dropout"]),
            ("transformer", "learning_rate", p["learning_rate"]),
            ("transformer", "threshold", p["threshold"]),
        ]
    return []


def _block_bounds(lines: list[str], model_block: str) -> tuple[int, int]:
    """
    Index range [start+1, end) of the keys inside `  <model_block>:` under the
    top-level `models:` mapping (2-space model indent, 4-space keys).
    """
    start = None
    for i, ln in enumerate(lines):
        if ln.rstrip() == f"  {model_block}:":
            start = i
            break
    if start is None:
        raise KeyError(f"model block '  {model_block}:' not found in config.yaml")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[j]) - len(lines[j].lstrip())
        if indent <= 2:  # next model block or dedent out of models:
            end = j
            break
    return start + 1, end


def _set_scalar(lines: list[str], model_block: str, key: str, value) -> bool:
    """In-place edit of `    <key>: <value>` inside a model block, keeping the
    inline comment. Returns True on change."""
    lo, hi = _block_bounds(lines, model_block)
    target = f"{key}:"
    for i in range(lo, hi):
        stripped = lines[i].lstrip()
        if not stripped.startswith(target):
            continue
        indent = lines[i][:len(lines[i]) - len(stripped)]
        # Preserve any inline comment (do not treat '#' inside quotes; values
        # here are numbers or simple quoted strings, so first '#' is the comment).
        comment = ""
        hash_pos = lines[i].find("#")
        if hash_pos != -1:
            comment = "  " + lines[i][hash_pos:].rstrip("\n")
        lines[i] = f"{indent}{key}: {_fmt_yaml_scalar(value)}{comment}\n"
        return True
    return False


def apply_best_params(cfg, config_path: str | None = None,
                      models: list[str] | None = None,
                      storage: str | None = None,
                      dry_run: bool = False) -> list[str]:
    """
    Write the best params of every study into config.yaml (comment- and
    anchor-preserving). Returns the list of applied 'model.key: value' changes.
    Validates the result via yaml.safe_load before writing.
    """
    import yaml
    from pathlib import Path

    if config_path is None:
        config_path = str(cfg._path)
    if models is None:
        models = list(cfg.optimization.models)

    path = Path(config_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    changes = []
    for m in models:
        try:
            study = load_study(cfg, m, storage)
        except KeyError:
            warnings.warn(f"[apply_best_params] no study for {m}, skipped.")
            continue
        try:
            best_params = study.best_params
        except ValueError:
            warnings.warn(f"[apply_best_params] {m} has no completed trials, skipped.")
            continue
        for block, key, value in _study_updates(m, best_params):
            if _set_scalar(lines, block, key, value):
                changes.append(f"{block}.{key}: {_fmt_yaml_scalar(value)}")
            else:
                warnings.warn(f"[apply_best_params] {block}.{key} not found; skipped.")

    new_text = "".join(lines)
    # Fail loudly rather than write a broken config.
    yaml.safe_load(new_text)

    if dry_run:
        print("[apply_best_params] DRY RUN, no file written. Changes:")
    else:
        path.write_text(new_text, encoding="utf-8")
        print(f"[apply_best_params] wrote {len(changes)} changes to {path}")
    for c in changes:
        print(f"    {c}")
    return changes


# ============================================================================
# 3. Fixed-param (multi-seed) re-evaluation
# ============================================================================

def _set_global_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _hpo_splits(cfg, df) -> list:
    """Reproduce the exact HPO fold set: all folds, restricted by tune_until."""
    wf = cfg.walk_forward
    splits = walk_forward_splits(
        index=df.index, mode=wf.mode,
        train_window_years=wf.train_window_years,
        test_window_months=wf.test_window_months,
        step_months=wf.step_months, min_train_years=wf.min_train_years,
    )
    splits = O._filter_splits_until(splits, getattr(cfg.optimization, "tune_until", None))
    return splits


def evaluate_params(model_name: str, df: pd.DataFrame, cfg, params: dict,
                    splits: list | None = None, seed: int | None = None
                    ) -> tuple[dict, list[np.ndarray]]:
    """
    Run the pooled-OOS fold loop for a FIXED hyperparameter set (no Optuna
    trial). Mirrors the objective in optimize.py but with fixed params and an
    optional global seed. Returns (metric_vector, per_fold_return_arrays).
    """
    if seed is not None:
        _set_global_seeds(seed)
    if splits is None:
        splits = _hpo_splits(cfg, df)

    fee = cfg.backtesting.transaction_cost_bps / 10_000
    signal_shift = cfg.backtesting.signal_shift
    pooled: list[np.ndarray] = []

    if model_name in ("MSM", "HMM", "HMM_Uni"):
        pooled = _eval_econometric(model_name, df, cfg, params, splits, fee, signal_shift)
    elif model_name == "LSTM":
        pooled = _eval_lstm(df, cfg, params, splits, fee, signal_shift)
    elif model_name == "Transformer":
        pooled = _eval_transformer(df, cfg, params, splits, fee, signal_shift)
    else:
        raise ValueError(f"unknown model {model_name}")

    metrics = O.compute_oos_metrics(np.concatenate(pooled)) if pooled else \
        {k: 0.0 for k in ("sharpe", "sortino", "calmar", "martin",
                          "ulcer", "max_drawdown", "cagr")}
    return metrics, pooled


def _eval_econometric(model_name, df, cfg, params, splits, fee, signal_shift):
    pooled = []
    for train_idx, test_idx in splits:
        df_train, df_test = df.loc[train_idx], df.loc[test_idx]
        try:
            if model_name == "MSM":
                from src.models.msm import train_msm_fold
                _, signal, _ = train_msm_fold(
                    returns_train=df_train["Returns"], returns_test=df_test["Returns"],
                    k_regimes=2, switching_variance=True, threshold=params["threshold"],
                )
            else:
                from src.models.hmm import train_hmm_fold
                mc = cfg.models.hmm if model_name == "HMM" else cfg.models.hmm_uni
                cov = params.get("covariance_type", mc.covariance_type)
                _, signal, _ = train_hmm_fold(
                    features_df_train=df_train[mc.features],
                    features_df_test=df_test[mc.features],
                    returns_train=df_train["Returns"],
                    n_components=mc.n_components, covariance_type=cov,
                    n_iter=mc.n_iter, random_state=mc.random_state,
                    threshold=params["threshold"],
                )
            pooled.append(O._fold_portfolio_returns(df_test, signal, signal_shift, fee))
        except Exception as e:
            warnings.warn(f"{model_name} eval fold failed: {e}")
    return pooled


def _eval_lstm(df, cfg, params, splits, fee, signal_shift):
    from src.models.lstm import train_lstm_fold
    from tensorflow.keras.optimizers import Adam
    from src.data.labels.resolver import compute_supervised_labels, resolve_label_col

    features = cfg.features.model_features
    labels_col = resolve_label_col(cfg)
    lc = cfg.models.lstm
    dl_warm, epochs_warm, max_epochs = O._dl_warm_cfg(cfg)

    if cfg.labels.supervised_label_source != "hmm":
        df = df.copy()
        if "Supervised_Label" not in df.columns:
            df["Supervised_Label"] = compute_supervised_labels(df, cfg)

    pooled, state = [], None
    for train_idx, test_idx in splits:
        df_train, df_test = df.loc[train_idx], df.loc[test_idx]
        try:
            if cfg.labels.supervised_label_source == "hmm":
                df_train, df_test = O._generate_hmm_labels(df_train, df_test, cfg)
            probs, pred_idx, state = train_lstm_fold(
                df_train=df_train, df_test=df_test, features=features, labels_col=labels_col,
                window_size=params["window_size"], units_l1=params["units_l1"],
                units_l2=params["units_l2"], return_sequences=lc.return_sequences,
                dropout=params["dropout"], dense=lc.dense, activation=lc.activation,
                optimizer=Adam(learning_rate=params["learning_rate"]), metrics=lc.metrics,
                epochs=max_epochs, batch_size=params["batch_size"],
                validation_split=lc.validation_split, verbose=0,
                init_weights=state if dl_warm else None,
                epochs_warm=epochs_warm if (dl_warm and state is not None) else None,
            )
            sig = pd.Series((probs >= params["threshold"]).astype(int), index=pred_idx)
            pooled.append(O._fold_portfolio_returns(df_test.loc[pred_idx], sig, signal_shift, fee))
        except Exception as e:
            warnings.warn(f"LSTM eval fold failed: {e}")
            state = None
    return pooled


def _eval_transformer(df, cfg, params, splits, fee, signal_shift):
    from src.models.transformer import train_transformer_fold
    from src.data.labels.resolver import compute_supervised_labels, resolve_label_col

    features = cfg.features.model_features
    labels_col = resolve_label_col(cfg)
    tc = cfg.models.transformer
    dl_warm, epochs_warm, max_epochs = O._dl_warm_cfg(cfg)
    d_model, n_heads = (int(x) for x in params["dmodel_nheads"].split("-"))

    if cfg.labels.supervised_label_source != "hmm":
        df = df.copy()
        if "Supervised_Label" not in df.columns:
            df["Supervised_Label"] = compute_supervised_labels(df, cfg)

    pooled, state = [], None
    for train_idx, test_idx in splits:
        df_train, df_test = df.loc[train_idx], df.loc[test_idx]
        try:
            if cfg.labels.supervised_label_source == "hmm":
                df_train, df_test = O._generate_hmm_labels(df_train, df_test, cfg)
            probs, pred_idx, state = train_transformer_fold(
                df_train=df_train, df_test=df_test, features=features, labels_col=labels_col,
                window_size=params["window_size"], d_model=d_model, n_heads=n_heads,
                n_layers=params["n_layers"], dim_feedforward=params["dim_feedforward"],
                dropout=params["dropout"], learning_rate=params["learning_rate"],
                epochs=max_epochs, batch_size=params["batch_size"],
                validation_split=tc.validation_split, verbose=0,
                init_state_dict=state if dl_warm else None,
                epochs_warm=epochs_warm if (dl_warm and state is not None) else None,
            )
            sig = pd.Series((probs >= params["threshold"]).astype(int), index=pred_idx)
            pooled.append(O._fold_portfolio_returns(df_test.loc[pred_idx], sig, signal_shift, fee))
        except Exception as e:
            warnings.warn(f"Transformer eval fold failed: {e}")
            state = None
    return pooled


def multiseed_reeval(model_name: str, df: pd.DataFrame, cfg, top: int = 5,
                     seeds: int | list[int] = 5, metric: str | None = None,
                     storage: str | None = None) -> pd.DataFrame:
    """
    Re-run the top-N configs of a study with several global seeds and report
    mean/std of every metric. Guards DL rankings against seed sensitivity.
    """
    metric = O._resolve_metric(cfg, metric)
    seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    study = load_study(cfg, model_name, storage)
    complete = sorted(_complete_trials(study), key=lambda t: t.value, reverse=True)
    top_trials = complete[:top]
    splits = _hpo_splits(cfg, df)

    rows = []
    for rank, t in enumerate(top_trials, start=1):
        scores = {k: [] for k in ("martin", "sharpe", "sortino", "calmar",
                                   "ulcer", "max_drawdown", "cagr")}
        for s in seed_list:
            metrics, _ = evaluate_params(model_name, df, cfg, t.params, splits, seed=s)
            for k in scores:
                scores[k].append(metrics[k])
        row = {"rank": rank, "trial": t.number,
               f"{metric}_hpo": round(t.value, 4)}
        for k, vals in scores.items():
            row[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{k}_std"] = round(float(np.std(vals)), 4)
        rows.append(row)
        print(f"  {model_name} rank {rank} (trial {t.number}): "
              f"{metric} {row[f'{metric}_mean']}±{row[f'{metric}_std']} over {len(seed_list)} seeds")
    return pd.DataFrame(rows).set_index("rank")


# ============================================================================
# 4. Deflated Sharpe Ratio + PBO (CSCV)
# ============================================================================

_EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(sr_std: float, n_trials: int) -> float:
    """E[max Sharpe] over n_trials iid trials (Bailey & Lopez de Prado 2014)."""
    from scipy.stats import norm
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(sr_hat_pp: float, sr_star_pp: float, n_obs: int,
                          skew: float, kurt: float) -> float:
    """
    DSR = P(true SR > 0) after deflating for multiple testing.
    sr_hat_pp / sr_star_pp are PER-PERIOD Sharpe ratios (not annualized),
    kurt is the non-excess (Pearson) kurtosis.
    """
    from scipy.stats import norm
    if n_obs < 3:
        return float("nan")
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr_hat_pp + (kurt - 1.0) / 4.0 * sr_hat_pp ** 2))
    num = (sr_hat_pp - sr_star_pp) * math.sqrt(n_obs - 1)
    return float(norm.cdf(num / denom))


def dsr_for_study(model_name: str, df: pd.DataFrame, cfg,
                  storage: str | None = None) -> dict:
    """
    DSR for the selected (best) config of a study. The trial Sharpe user_attrs
    (annualized, pooled OOS) define the multiple-testing distribution; the best
    config is re-evaluated once to get its per-period Sharpe, skew and kurtosis.
    """
    from scipy.stats import skew as _skew, kurtosis as _kurt

    study = load_study(cfg, model_name, storage)
    complete = _complete_trials(study)
    sr_ann = [t.user_attrs.get("sharpe") for t in complete
              if t.user_attrs.get("sharpe") is not None]
    if len(sr_ann) < 2:
        return {"model": model_name, "dsr": float("nan"), "note": "too few trials"}

    tdpy = cfg.evaluation.mcs.trading_days_per_year
    sr_pp = np.array(sr_ann) / math.sqrt(tdpy)          # per-period
    sr_std = float(np.std(sr_pp, ddof=1))
    n_trials = len(sr_pp)

    metrics, pooled = evaluate_params(model_name, df, cfg, study.best_params)
    rets = np.concatenate(pooled) if pooled else np.array([])
    if len(rets) < 3:
        return {"model": model_name, "dsr": float("nan"), "note": "no returns"}
    sr_hat_pp = float(np.mean(rets) / np.std(rets)) if np.std(rets) > 0 else 0.0
    sr_star_pp = expected_max_sharpe(sr_std, n_trials)
    dsr = deflated_sharpe_ratio(
        sr_hat_pp, sr_star_pp, len(rets),
        float(_skew(rets)), float(_kurt(rets, fisher=False)),
    )
    return {
        "model": model_name,
        "n_trials": n_trials,
        "sr_ann_best": round(sr_hat_pp * math.sqrt(tdpy), 4),
        "sr_star_ann": round(sr_star_pp * math.sqrt(tdpy), 4),
        "dsr": round(dsr, 4),
        "significant_5pct": bool(dsr > 0.95),
    }


def pbo_cscv(perf_matrix: np.ndarray, n_splits: int = 10) -> float:
    """
    Probability of Backtest Overfitting via Combinatorially Symmetric CV
    (Bailey et al. 2017). perf_matrix: (T_slices, N_configs) performance (higher
    better). Returns PBO = P(best-IS config underperforms OS median).
    """
    from itertools import combinations
    from scipy.stats import rankdata

    M = np.asarray(perf_matrix, dtype=float)
    T, N = M.shape
    if T < n_splits or n_splits % 2 != 0:
        n_splits = max(2, (T // 2) * 2 if T >= 4 else 2)
    # Partition the T rows into S disjoint contiguous blocks.
    blocks = np.array_split(np.arange(T), n_splits)
    logits = []
    half = n_splits // 2
    for is_idx in combinations(range(n_splits), half):
        is_rows = np.concatenate([blocks[b] for b in is_idx])
        os_rows = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_idx])
        is_perf = M[is_rows].mean(axis=0)
        os_perf = M[os_rows].mean(axis=0)
        best_is = int(np.argmax(is_perf))
        # Rank of the IS-best config among OS performance (1 = worst).
        os_rank = rankdata(os_perf)[best_is]
        w = os_rank / (N + 1)                     # relative rank in (0,1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1.0 - w)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0.0))          # fraction with OS rank in lower half


def build_perf_matrix(model_name: str, df: pd.DataFrame, cfg, top: int = 20,
                      storage: str | None = None) -> tuple[np.ndarray, list]:
    """
    (T_folds, N_configs) matrix of per-fold Sharpe for the top-N configs of a
    study, for PBO. Re-evaluates each config once over the HPO folds.
    """
    study = load_study(cfg, model_name, storage)
    complete = sorted(_complete_trials(study), key=lambda t: t.value, reverse=True)
    top_trials = complete[:top]
    splits = _hpo_splits(cfg, df)

    cols = []
    for t in top_trials:
        _, pooled = evaluate_params(model_name, df, cfg, t.params, splits)
        per_fold = [O.compute_oos_metrics(r)["sharpe"] if len(r) >= 20 else 0.0
                    for r in pooled]
        cols.append(per_fold)
    # Align on the shortest fold count (DL folds may drop a degenerate fold).
    min_folds = min(len(c) for c in cols) if cols else 0
    mat = np.array([c[:min_folds] for c in cols]).T   # (folds, configs)
    return mat, [t.number for t in top_trials]


# ============================================================================
# 5. Objective-sensitivity table (which config wins under which metric)
# ============================================================================

# Columns reported per config (cagr is display-only, not an optimization target).
_SENSITIVITY_COLS = ["martin", "sharpe", "sortino", "calmar",
                     "ulcer", "max_drawdown", "cagr"]
# Metrics one could optimize for (all normalized to maximize by _objective_score).
_CANDIDATE_METRICS = ["martin", "sharpe", "sortino", "calmar", "ulcer", "max_drawdown"]


def _metrics_of(trial) -> dict:
    """Metric vector logged as user_attrs during the search."""
    return {k: trial.user_attrs[k] for k in _SENSITIVITY_COLS}


def objective_sensitivity(cfg, models: list[str] | None = None,
                          storage: str | None = None,
                          candidate_metrics: list[str] | None = None
                          ) -> dict[str, pd.DataFrame]:
    """
    For each model and each candidate objective metric, find the trial that
    would have been selected if the search had optimized THAT metric, and report
    the selected config's value under every metric.

    This quantifies how metric-dependent the model's chosen config is: if the
    Sharpe-optimal and Martin-optimal configs coincide (same trial), the ranking
    is robust to the objective choice; if they diverge, the table shows exactly
    how much each metric gains/loses under an alternative objective. Uses only
    the per-trial user_attrs logged during the search (no retraining).

    Returns {model -> DataFrame indexed by 'optimized_for'}.
    """
    if models is None:
        models = list(cfg.optimization.models)
    if candidate_metrics is None:
        candidate_metrics = _CANDIDATE_METRICS

    out: dict[str, pd.DataFrame] = {}
    for m in models:
        try:
            study = load_study(cfg, m, storage)
        except KeyError:
            continue
        # Only trials that carry the full metric vector (failed/sentinel trials
        # never set the user_attrs and are excluded).
        trials = [t for t in _complete_trials(study)
                  if all(k in t.user_attrs for k in _SENSITIVITY_COLS)]
        if not trials:
            out[m] = pd.DataFrame()
            continue

        objective = study.user_attrs.get("metric",
                                         getattr(cfg.optimization, "metric", "martin"))
        best_under = {mk: max(trials, key=lambda t: O._objective_score(_metrics_of(t), mk))
                      for mk in candidate_metrics}
        obj_best = best_under.get(objective) or \
            max(trials, key=lambda t: O._objective_score(_metrics_of(t), objective))

        rows = []
        for mk in candidate_metrics:
            t = best_under[mk]
            row = {"optimized_for": mk, "trial": t.number,
                   "same_as_objective": t.number == obj_best.number}
            row.update({k: round(t.user_attrs[k], 4) for k in _SENSITIVITY_COLS})
            rows.append(row)
        df_out = pd.DataFrame(rows).set_index("optimized_for")
        df_out.attrs["objective"] = objective
        df_out.attrs["params"] = {mk: best_under[mk].params for mk in candidate_metrics}
        df_out.attrs["n_trials"] = len(trials)
        out[m] = df_out
    return out


def save_objective_sensitivity(cfg, tables: dict[str, pd.DataFrame],
                               path: str | None = None) -> str:
    """Write the objective-sensitivity tables as Markdown (asset for the paper)."""
    from pathlib import Path
    import datetime

    if path is None:
        path = cfg.asset_path("objective_sensitivity")

    lines = [
        "# Objective Sensitivity of the Selected Hyperparameters",
        "",
        f"_Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_  ",
        "Best config under each candidate metric, valued across all metrics "
        "(from the search trials' logged OOS metrics; no retraining). "
        "`same_as_objective` marks configs identical to the actual objective's pick.",
        "",
    ]
    for model, tbl in tables.items():
        if tbl.empty:
            lines.append(f"## {model}\n\n_No completed trials with metrics._\n")
            continue
        obj = tbl.attrs.get("objective", "")
        lines.append(f"## {model} (objective: {obj}, {tbl.attrs.get('n_trials', '?')} trials)")
        lines.append("")
        lines.append(tbl.to_markdown())
        lines.append("")
        params = tbl.attrs.get("params", {})
        distinct = {}
        for mk, p in params.items():
            distinct.setdefault(tuple(sorted(p.items())), (mk, p))
        lines.append("Selected configs:")
        for _, (mk, p) in distinct.items():
            pretty = ", ".join(f"{k}={v}" for k, v in p.items())
            lines.append(f"- best under **{mk}**: {pretty}")
        lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ {path}")
    return str(path)


# ============================================================================
# 6. Orchestrator: generate all HPO report assets (Markdown)
# ============================================================================

def _write_md(cfg, asset_key: str, title: str, body: str) -> str:
    from pathlib import Path
    import datetime
    path = Path(cfg.asset_path(asset_key))
    text = (f"# {title}\n\n"
            f"_Generated at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}_\n\n"
            f"{body}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  ✓ {path}")
    return str(path)


def generate_hpo_reports(cfg, df=None, scope: str = "cheap",
                         models: list[str] | None = None, storage: str | None = None,
                         pbo_top: int = 15, ms_top: int = 5, ms_seeds: int = 5) -> dict:
    """
    Generate the post-HPO report assets as Markdown (embedded in statistics.md
    and shown in the dashboard).

    scope="cheap": convergence review + objective sensitivity. Reads the logged
        trial user_attrs only, no retraining (seconds).
    scope="full":  additionally DSR, PBO and multi-seed reeval, which RE-TRAIN
        models via evaluate_params (minutes-hours for the DL models on GPU;
        requires df).

    Returns {report -> asset path}.
    """
    if models is None:
        models = list(cfg.optimization.models)
    summary: dict = {}

    conv = convergence_review(cfg, models=models, storage=storage)
    summary["convergence"] = _write_md(
        cfg, "hpo_convergence", "HPO Convergence & Edge-of-Range Review",
        conv.to_markdown() if not conv.empty else "_No studies with completed trials._",
    )

    tables = objective_sensitivity(cfg, models=models, storage=storage)
    summary["sensitivity"] = save_objective_sensitivity(cfg, tables)

    if scope != "full":
        return summary
    if df is None:
        raise ValueError("scope='full' requires df (models are re-trained).")

    dsr_rows = []
    for m in models:
        try:
            dsr_rows.append(dsr_for_study(m, df, cfg, storage=storage))
        except Exception as e:
            warnings.warn(f"DSR {m} failed: {e}")
    dsr_df = pd.DataFrame(dsr_rows).set_index("model") if dsr_rows else pd.DataFrame()
    summary["dsr"] = _write_md(
        cfg, "hpo_dsr", "Deflated Sharpe Ratio (multiple-testing adjusted)",
        (dsr_df.to_markdown() if not dsr_df.empty else "_n/a_")
        + "\n\nDSR = P(true Sharpe > 0) after deflating the best trial's Sharpe for the "
          "number of tested configs. `significant_5pct` = DSR > 0.95.",
    )

    pbo_rows = []
    for m in models:
        try:
            mat, _ = build_perf_matrix(m, df, cfg, top=pbo_top, storage=storage)
            if mat.size == 0 or mat.shape[1] < 2:
                continue
            pbo_rows.append({"model": m, "folds": mat.shape[0],
                             "configs": mat.shape[1], "pbo": round(pbo_cscv(mat), 3)})
        except Exception as e:
            warnings.warn(f"PBO {m} failed: {e}")
    pbo_df = pd.DataFrame(pbo_rows).set_index("model") if pbo_rows else pd.DataFrame()
    summary["pbo"] = _write_md(
        cfg, "hpo_pbo", "Probability of Backtest Overfitting (CSCV)",
        (pbo_df.to_markdown() if not pbo_df.empty else "_n/a_")
        + f"\n\nPBO over the top-{pbo_top} configs per model (per-fold Sharpe matrix, "
          "CSCV). Lower is better; PBO > 0.5 flags overfitting.",
    )

    dl_models = [m for m in models if m in ("LSTM", "Transformer")]
    parts = []
    for m in dl_models:
        try:
            msr = multiseed_reeval(m, df, cfg, top=ms_top, seeds=ms_seeds, storage=storage)
            parts.append(f"### {m}\n\n{msr.to_markdown()}")
        except Exception as e:
            warnings.warn(f"multiseed {m} failed: {e}")
    summary["multiseed"] = _write_md(
        cfg, "hpo_multiseed", f"Multi-Seed Re-Evaluation (top-{ms_top}, {ms_seeds} seeds)",
        "\n\n".join(parts) if parts else "_No DL studies available._",
    )
    return summary


# ============================================================================
# CLI
# ============================================================================

def _load_cfg_and_df():
    from config.config_loader import PipelineConfig
    cfg = PipelineConfig()
    df = pd.read_parquet(cfg.data_path("feature_engineered"))
    return cfg, df


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Post-HPO analysis (Issue #5).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("review")
    ap = sub.add_parser("apply"); ap.add_argument("--dry-run", action="store_true")
    ms = sub.add_parser("multiseed")
    ms.add_argument("--model", required=True); ms.add_argument("--top", type=int, default=5)
    ms.add_argument("--seeds", type=int, default=5)
    ds = sub.add_parser("dsr"); ds.add_argument("--model", required=True)
    pb = sub.add_parser("pbo")
    pb.add_argument("--model", required=True); pb.add_argument("--top", type=int, default=20)
    se = sub.add_parser("sensitivity"); se.add_argument("--save", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("--scope", choices=["cheap", "full"], default="cheap")
    args = parser.parse_args(argv)

    cfg, df = _load_cfg_and_df()

    if args.cmd == "review":
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(convergence_review(cfg).to_string())
    elif args.cmd == "apply":
        apply_best_params(cfg, dry_run=args.dry_run)
    elif args.cmd == "multiseed":
        print(multiseed_reeval(args.model, df, cfg, top=args.top, seeds=args.seeds).to_string())
    elif args.cmd == "dsr":
        print(dsr_for_study(args.model, df, cfg))
    elif args.cmd == "pbo":
        mat, trials = build_perf_matrix(args.model, df, cfg, top=args.top)
        print(f"{args.model}: PBO={pbo_cscv(mat):.3f} over {mat.shape[0]} folds x {mat.shape[1]} configs")
    elif args.cmd == "sensitivity":
        tables = objective_sensitivity(cfg)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            for model, tbl in tables.items():
                obj = tbl.attrs.get("objective", "") if not tbl.empty else ""
                print(f"\n=== {model} (objective: {obj}) ===")
                print(tbl.to_string() if not tbl.empty else "  no completed trials")
        if args.save:
            save_objective_sensitivity(cfg, tables)
    elif args.cmd == "report":
        summary = generate_hpo_reports(cfg, df=df, scope=args.scope)
        print("\nHPO reports written:")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
