"""
Seed-sensitivity quantification of the PRODUCTION config (retraining stability).

Motivation
----------
Re-running the walk-forward pipeline changes the equity curves of the two deep
learning models (LSTM, Transformer) noticeably, while the econometric models
stay put. This module measures HOW MUCH each model moves when only the random
source of its estimator changes, on the exact production hyperparameters from
config.yaml and the full walk-forward fold set (no tune_until restriction).

What is varied per model family (this is the crux):

* MSM        : nothing. MarkovRegression.fit() is deterministic MLE without any
               random_state, so a single evaluation is reported as the (trivial)
               zero-variance control. Confirms the reproducibility claim.
* HMM/HMM_Uni: the GaussianHMM EM initialization (random_state). hmmlearn takes
               an EXPLICIT random_state argument and ignores the global numpy
               seed, so global reseeding alone would falsely report zero
               variance. We therefore override cfg.models.<hmm>.random_state per
               run to probe the local-optimum sensitivity of the EM fit.
* LSTM/Transf: the seed base of the production ensemble. The pipeline does not
               run a single network; walk_forward trains dl_ensemble_size
               members per fold and averages their probabilities. Each run here
               therefore trains a COMPLETE ensemble from a disjoint seed block
               (run k -> seeds k*N .. k*N+N-1), so the reported spread is the
               spread of the estimator that produces the headline results. With
               dl_ensemble_size=1 this degenerates to varying a single global
               seed, which is the pre-ensemble behaviour.
               Cost note: N times the DL training per run.

The metric vector (sharpe, sortino, calmar, martin, ulcer, max_drawdown, cagr)
is computed on the pooled OOS net-return series, identical in definition to the
HPO objective and to engine.calculate_annualized_metrics, so the numbers line
up with the rest of the paper.

CLI
---
    python -m src.backtest.seed_sensitivity run --seeds 5
    python -m src.backtest.seed_sensitivity run --models LSTM Transformer --seeds 10 --save
"""

import logging
import warnings
import numpy as np
import pandas as pd

from src.backtest.walk_forward import walk_forward_splits
from src.backtest.hpo_analysis import evaluate_params

# Reuse the model-service logger (same name as walk_forward): its StreamHandler
# flushes on every emit, so progress shows live in the Docker logs, unlike a
# block-buffered bare print() in a non-TTY container.
logger = logging.getLogger("model_service")

# Metrics reported per model, in display order. Higher is better except ulcer
# and max_drawdown, which are natural raw values.
_METRICS = ("cagr", "martin", "sharpe", "sortino", "calmar", "ulcer", "max_drawdown")

# Headline metrics whose coefficient of variation drives the sensitivity flag.
_HEADLINE = ("cagr", "martin")

_ALL_MODELS = ("MSM", "HMM", "HMM_Uni", "LSTM", "Transformer")
_DETERMINISTIC = ("MSM",)              # no random source at all
_INIT_SENSITIVE = ("HMM", "HMM_Uni")   # EM init via random_state
_GLOBAL_SEED = ("LSTM", "Transformer")  # global RNG


# ============================================================================
# Production hyperparameters -> the params dict evaluate_params expects
# ============================================================================

def _production_params(cfg, model_name: str) -> dict:
    """
    Translate the production config of one model into the flat params dict that
    evaluate_params / the _eval_* helpers consume (same keys as an Optuna trial).
    """
    if model_name == "MSM":
        # _eval_econometric hardcodes k_regimes=2, switching_variance=True,
        # matching the config; only the threshold is read from params.
        return {"threshold": cfg.models.msm.threshold}

    if model_name in ("HMM", "HMM_Uni"):
        mc = cfg.models.hmm if model_name == "HMM" else cfg.models.hmm_uni
        return {"threshold": mc.threshold, "covariance_type": mc.covariance_type}

    if model_name == "LSTM":
        lc = cfg.models.lstm
        return {
            "window_size": lc.window_size,
            "units_l1": lc.units_l1,
            "units_l2": lc.units_l2,
            "dropout": lc.dropout,
            "learning_rate": lc.learning_rate,
            "batch_size": lc.batch_size,
            "threshold": lc.threshold,
        }

    if model_name == "Transformer":
        tc = cfg.models.transformer
        return {
            "window_size": tc.window_size,
            # _eval_transformer parses "d_model-n_heads"
            "dmodel_nheads": f"{tc.d_model}-{tc.n_heads}",
            "n_layers": tc.n_layers,
            "dim_feedforward": tc.dim_feedforward,
            "dropout": tc.dropout,
            "learning_rate": tc.learning_rate,
            "batch_size": tc.batch_size,
            "threshold": tc.threshold,
        }

    raise ValueError(f"unknown model {model_name}")


def _full_splits(cfg, df) -> list:
    """Full walk-forward fold set (headline configuration, no tune_until cut)."""
    wf = cfg.walk_forward
    return walk_forward_splits(
        index=df.index, mode=wf.mode,
        train_window_years=wf.train_window_years,
        test_window_months=wf.test_window_months,
        step_months=wf.step_months, min_train_years=wf.min_train_years,
    )


# ============================================================================
# One evaluation, with the correct random source varied per model family
# ============================================================================

def _dl_ensemble_size(cfg) -> int:
    """Production ensemble size for LSTM/Transformer (walk_forward config)."""
    return max(1, int(getattr(cfg.walk_forward, "dl_ensemble_size", 1)))


def _eval_one(model_name: str, df, cfg, params: dict, splits: list, seed: int) -> dict:
    """
    Evaluate one model once, varying the random source that actually matters:

    * HMM/HMM_Uni: override cfg.models.<hmm>.random_state = seed (the EM init).
      The global seed is set too, but the EM init is what moves the fit.
    * LSTM/Transformer: vary the seed base of the PRODUCTION ensemble. Run `k`
      uses the disjoint seed block `k*N .. k*N+N-1`, so no two runs share a
      member. This is what makes the reported spread the spread of the
      estimator that actually produces the headline results; varying a single
      model's global seed would measure a configuration the pipeline never uses
      (and overstate the variance by roughly sqrt(N)).
    * MSM: deterministic; seed ignored.
    """
    if model_name in _INIT_SENSITIVE:
        mc = cfg.models.hmm if model_name == "HMM" else cfg.models.hmm_uni
        old_rs = mc.random_state
        mc.random_state = int(seed)
        try:
            metrics, _ = evaluate_params(model_name, df, cfg, params, splits, seed=int(seed))
        finally:
            mc.random_state = old_rs
        return metrics

    if model_name in _DETERMINISTIC:
        metrics, _ = evaluate_params(model_name, df, cfg, params, splits, seed=None)
        return metrics

    # LSTM / Transformer: seed-averaged production ensemble.
    n_ens = _dl_ensemble_size(cfg)
    metrics, _ = evaluate_params(
        model_name, df, cfg, params, splits, seed=int(seed),
        ensemble_size=n_ens, ensemble_seed_base=int(seed) * n_ens,
    )
    return metrics


# ============================================================================
# Public API
# ============================================================================

def seed_sensitivity(
    df: pd.DataFrame,
    cfg,
    models: list[str] | None = None,
    seeds: int | list[int] = 5,
) -> pd.DataFrame:
    """
    Quantify how much each model's OOS metrics move when only its random source
    changes, on the production config and the full fold set.

    Deterministic models (MSM) are evaluated once and reported with zero spread.
    Stochastic models are evaluated over every seed.

    Returns a tidy DataFrame indexed by (model, metric) with columns
    mean, std, min, max, cv (coefficient of variation = std / |mean|), n_seeds.
    """
    models = list(models) if models else list(_ALL_MODELS)
    seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    splits = _full_splits(cfg, df)

    n_ens = _dl_ensemble_size(cfg)
    logger.info(
        f"Seed-sensitivity start: models={models}, seeds={seed_list}, "
        f"folds={len(splits)}, dl_ensemble_size={n_ens}. Each DL run retrains "
        f"the full {n_ens}-member ensemble (GPU-bound: "
        f"{len(seed_list) * n_ens * len(splits)} trainings per DL model)."
    )

    rows = []
    for model_name in models:
        params = _production_params(cfg, model_name)

        if model_name in _DETERMINISTIC:
            # A single run fully characterizes a deterministic model.
            logger.info(f"[{model_name}] deterministic, single evaluation...")
            m = _eval_one(model_name, df, cfg, params, splits, seed=seed_list[0])
            logger.info(
                f"[{model_name}] done: cagr={m['cagr']:.4f}, martin={m['martin']:.4f}"
            )
            samples = {k: [m[k]] for k in _METRICS}
            used_seeds = 1
        else:
            src = "EM init (random_state)" if model_name in _INIT_SENSITIVE else "global RNG"
            logger.info(
                f"[{model_name}] evaluating {len(seed_list)} seeds, varying {src}..."
            )
            samples = {k: [] for k in _METRICS}
            for i, s in enumerate(seed_list, start=1):
                m = _eval_one(model_name, df, cfg, params, splits, seed=s)
                for k in _METRICS:
                    samples[k].append(m[k])
                logger.info(
                    f"[{model_name}] seed {s} ({i}/{len(seed_list)}): "
                    f"cagr={m['cagr']:.4f}, martin={m['martin']:.4f}, "
                    f"maxDD={m['max_drawdown']:.4f}"
                )
            used_seeds = len(seed_list)

        for k in _METRICS:
            vals = np.asarray(samples[k], dtype=float)
            mean = float(vals.mean())
            std = float(vals.std())
            cv = float(std / abs(mean)) if abs(mean) > 1e-12 else 0.0
            rows.append({
                "model": model_name, "metric": k,
                "mean": round(mean, 4), "std": round(std, 4),
                "min": round(float(vals.min()), 4), "max": round(float(vals.max()), 4),
                "cv": round(cv, 4), "n_seeds": used_seeds,
            })
        hi = np.mean([
            (np.std(samples[k]) / abs(np.mean(samples[k]))) if abs(np.mean(samples[k])) > 1e-12 else 0.0
            for k in _HEADLINE
        ])
        logger.info(
            f"[{model_name}] headline CV (mean over {list(_HEADLINE)}) = {hi:.3f} "
            f"over {used_seeds} seed(s) -> {_sensitivity_verdict(hi)}"
        )

    logger.info("Seed-sensitivity done.")
    return pd.DataFrame(rows).set_index(["model", "metric"])


# ============================================================================
# Markdown report
# ============================================================================

def _sensitivity_verdict(cv: float) -> str:
    """Coarse label for the headline coefficient of variation."""
    if cv < 0.02:
        return "stable"
    if cv < 0.10:
        return "moderate"
    return "unstable"


def to_markdown(result: pd.DataFrame, seed_list: list[int],
                ensemble_size: int = 1) -> str:
    """Render the seed-sensitivity result as a Markdown report."""
    lines = []
    lines.append(f"# Seed Sensitivity of the Production Config ({len(seed_list)} runs)\n")
    if ensemble_size > 1:
        dl_src = (
            f"the seed base of the {ensemble_size}-member production ensemble "
            f"for LSTM and Transformer (run k trains a complete ensemble from "
            f"the disjoint seed block k*{ensemble_size}..k*{ensemble_size}+"
            f"{ensemble_size - 1}, so the spread is that of the estimator "
            f"behind the headline results, not of a single network the "
            f"pipeline never uses)"
        )
    else:
        dl_src = (
            "the global RNG (weight init, batch shuffle, dropout) for LSTM and "
            "Transformer -- note dl_ensemble_size is 1, so this IS the "
            "production estimator"
        )
    lines.append(
        "Each model is re-run on the production hyperparameters and the full "
        "walk-forward fold set, varying only its random source: the EM "
        f"initialization (random_state) for HMM and HMM_Uni, {dl_src}. MSM is "
        "deterministic and shown as the zero-variance control. Metrics are the "
        "pooled OOS values (CV = std / |mean|).\n"
    )

    # Headline summary table: one row per model.
    lines.append("## Summary (headline metrics)\n")
    lines.append("| Model | CAGR mean | CAGR std | CAGR CV | Martin mean | Martin std | Martin CV | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for model in result.index.get_level_values("model").unique():
        cagr = result.loc[(model, "cagr")]
        martin = result.loc[(model, "martin")]
        head_cv = float(np.mean([cagr["cv"], martin["cv"]]))
        lines.append(
            f"| {model} | {cagr['mean']:.4f} | {cagr['std']:.4f} | {cagr['cv']:.3f} "
            f"| {martin['mean']:.4f} | {martin['std']:.4f} | {martin['cv']:.3f} "
            f"| {_sensitivity_verdict(head_cv)} |"
        )
    lines.append("")

    # Per-model detail tables.
    lines.append("## Per-model detail\n")
    for model in result.index.get_level_values("model").unique():
        sub = result.loc[model]
        n_seeds = int(sub["n_seeds"].iloc[0])
        lines.append(f"### {model} ({n_seeds} run(s))\n")
        lines.append("| Metric | mean | std | min | max | CV |")
        lines.append("|---|---|---|---|---|---|")
        for metric in _METRICS:
            r = sub.loc[metric]
            lines.append(
                f"| {metric} | {r['mean']:.4f} | {r['std']:.4f} | "
                f"{r['min']:.4f} | {r['max']:.4f} | {r['cv']:.3f} |"
            )
        lines.append("")

    lines.append(
        "Reading the verdict: 'stable' (headline CV < 0.02) means a single seed "
        "is representative; 'moderate' (< 0.10) warrants reporting a seed band; "
        "'unstable' (>= 0.10) means a single run must NOT be reported as a "
        "point estimate. Because the DL rows already measure the seed-averaged "
        "production ensemble, an 'unstable' verdict there cannot be fixed by "
        "averaging more of the same: report the band, raise "
        "walk_forward.dl_ensemble_size (variance shrinks ~1/sqrt(N)), or treat "
        "the model as not reliably estimable on this sample. For HMM, "
        "best-of-k by train log-likelihood (n_init) is the corresponding "
        "remedy.\n"
    )
    return "\n".join(lines)


def run_and_write(
    df: pd.DataFrame,
    cfg,
    models: list[str] | None = None,
    seeds: int | list[int] = 5,
    save: bool = False,
) -> pd.DataFrame:
    """Run the quantification, print the summary, optionally write the asset."""
    seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    result = seed_sensitivity(df, cfg, models=models, seeds=seed_list)
    if save:
        md = to_markdown(result, seed_list, _dl_ensemble_size(cfg))
        out = cfg.asset_path("seed_sensitivity")
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info(f"Seed-sensitivity report written: {out}")
    return result


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
    # Standalone CLI: the model_service logger has no handler outside the service,
    # so wire a console handler here to surface the progress logs.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(
        description="Seed-sensitivity quantification of the production config."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", default=None,
                   help="subset of MSM HMM HMM_Uni LSTM Transformer (default: all)")
    r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--save", action="store_true", help="write assets/seed_sensitivity.md")
    args = parser.parse_args(argv)

    cfg, df = _load_cfg_and_df()

    if args.cmd == "run":
        result = run_and_write(df, cfg, models=args.models, seeds=args.seeds, save=args.save)
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(result.to_string())


if __name__ == "__main__":
    main()
