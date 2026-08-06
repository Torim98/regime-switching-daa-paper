from fastapi import APIRouter, HTTPException
from config.config_loader import PipelineConfig
from src.backtest.sorr import run_sorr_simulation, build_sorr_scenarios, build_sorr_summary
from src.backtest.evaluation import (
    evaluate_strategies, run_monte_carlo_simulation,
    # Issue #13: extended evaluation
    add_ulcer_to_table,
    compute_classification_metrics, plot_confusion_matrices, plot_roc_pr_curves,
    churning_stats, threshold_sensitivity,
    time_to_recovery, switch_timing_vs_peak,
    depletion_rate_with_ci,
    compare_bootstrap_methods, bootstrap_robustness_summary,
    test_h1_drawdown, test_h2_transformer, plot_mcs_violins,
    plot_depletion_forest, plot_h1_forest, plot_h2_forest,
    plot_risk_return_positioning,
    break_even_transaction_cost, plot_break_even, withdrawal_sensitivity,
    plot_regime_probability_heatmap,
)
from src.backtest.reporting import generate_statistics_report
from src.backtest.bear_coverage import generate_bear_coverage_report
from src.backtest.engine import (
    run_all_backtests, backtest,
    calculate_performance_summary,
    calculate_annualized_metrics,
    calculate_crisis_performance,
    calculate_rolling_sharpe,
)
from src.backtest.plots import (
    plot_equity_curves, plot_transaction_costs, plot_sorr_scenario,
    plot_mcs_boxplots, plot_mcs_paths, plot_mcs_quantiles,
    plot_rolling_sharpe, plot_drawdown,
)
from src.data.labels import load_nber_recession
from pathlib import Path
import pandas as pd
import numpy as np
import os
import time
import logging

logger = logging.getLogger("backtest_service")

router = APIRouter(prefix="/backtest", tags=["backtest"])

def get_cfg():
    return PipelineConfig()

@router.post("/run")
def run_backtest():
    """Run backtesting + SORR for all models."""
    start = time.time()
    cfg = get_cfg()
    logger.info("Starting backtesting...")

    test_df = pd.read_parquet(cfg.data_path("test_data"))

    # Walk-forward: common OOS window (all models must have a signal)
    if cfg.walk_forward.enabled:
        signal_cols = [c for c in test_df.columns if c.endswith("_Signal")]
        n_before = len(test_df)
        test_df = test_df.dropna(subset=signal_cols, how="any").copy()
        logger.info(
            f"Walk-forward OOS window: {n_before} → {len(test_df)} rows "
            f"({test_df.index.min().date()} → {test_df.index.max().date()})"
        )

    # Backtesting
    backtesting_results, backtesting_costs = run_all_backtests(
        test_df=test_df,
        fee_rate=cfg.transaction_cost_rate,
        signal_shift=cfg.backtesting.signal_shift,
    )

    # Performance Summary
    performance_summary = calculate_performance_summary(
        backtesting_results,
        initial_capital=float(cfg.backtesting.sorr.scenarios.Standard.initial_capital),
    )
    performance_summary.to_markdown(cfg.asset_path("performance_summary"))

    # Annualized metrics
    annualized = calculate_annualized_metrics(backtesting_results)
    annualized.to_markdown(cfg.asset_path("annualized_metrics"))

    crisis_windows = {
        name: tuple(w)
        for name, w in vars(cfg.evaluation.extended.crisis_windows).items()
    }
    crisis = calculate_crisis_performance(backtesting_results, crisis_windows)
    if not crisis.empty:
        crisis.to_markdown(cfg.asset_path("crisis_performance"))

    rolling_sharpe = calculate_rolling_sharpe(backtesting_results)
    plot_rolling_sharpe(rolling_sharpe, cfg.color_map,
                        cfg.asset_path("rolling_sharpe"))
    plot_drawdown(backtesting_results, cfg.color_map,
                  cfg.asset_path("drawdown"))

    # Persist
    Path(cfg.data_path("backtesting_results")).parent.mkdir(parents=True, exist_ok=True)
    backtesting_results.to_parquet(cfg.data_path("backtesting_results"))
    backtesting_costs.to_parquet(cfg.data_path("backtesting_costs"))

    # Plots (existing)
    plot_equity_curves(backtesting_results, cfg.color_map,
                       cfg.asset_path("equity_curves"),
                       initial_capital=float(cfg.backtesting.sorr.scenarios.Standard.initial_capital))
    plot_transaction_costs(backtesting_costs, cfg.transaction_cost_rate,
                           cfg.color_map,
                           cfg.asset_path("transaction_costs"))

    # SORR for all scenarios
    scenarios = build_sorr_scenarios(cfg.backtesting.sorr.scenarios)
    sorr_summaries = []
    backtesting_sorr = pd.DataFrame(index=backtesting_results.index)

    for name, params in scenarios.items():
        logger.info(f"Running SORR scenario: {name}")
        sim_results = run_sorr_simulation(
            backtesting_results, test_df,
            params["start"], params["withdrawal"], params["fee"],
        )
        for col in sim_results.columns:
            backtesting_sorr[f"{name}_{col}"] = sim_results[col]

        plot_sorr_scenario(sim_results, name, params, cfg.color_map,
                           cfg.asset_path(f"sorr_sim_{name.lower()}"))
        sorr_summaries.extend(build_sorr_summary(sim_results, name))

    backtesting_sorr.to_parquet(cfg.data_path("backtesting_sorr"))
    sorr_df = pd.DataFrame(sorr_summaries).set_index(["Scenario", "Strategy"])
    sorr_df.to_markdown(cfg.asset_path("sorr_summary"), index=True)

    elapsed = time.time() - start
    logger.info(f"Backtesting complete in {elapsed:.1f}s")

    return {
        "status": "ok",
        "strategies": list(backtesting_results.columns),
        "rows": len(backtesting_results),
    }

@router.post("/evaluate")
def evaluate():
    """Performance metrics + Monte Carlo simulation + statistics report."""
    start = time.time()
    cfg = get_cfg()
    logger.info("Starting evaluation + MCS...")

    backtesting_results = pd.read_parquet(cfg.data_path("backtesting_results"))
    backtesting_costs = pd.read_parquet(cfg.data_path("backtesting_costs"))
    test_df = pd.read_parquet(cfg.data_path("test_data"))

    # Signal mapping (dynamic)
    signals_to_count = pd.DataFrame(index=test_df.index)
    for sig_col in [c for c in test_df.columns if c.endswith("_Signal")]:
        model_name = sig_col.rsplit("_", 1)[0]
        signals_to_count[model_name] = test_df[sig_col]

    evaluation_table = evaluate_strategies(backtesting_results, signals_to_count, backtesting_costs)
    evaluation_table.to_markdown(cfg.asset_path("evaluation_table"), index=True)

    # MCS
    scenarios = build_sorr_scenarios(cfg.backtesting.sorr.scenarios)
    daily_rets = backtesting_results.pct_change().dropna()

    mcs_cfg = cfg.evaluation.mcs
    all_mc_summaries, mcs = run_monte_carlo_simulation(
        daily_rets=daily_rets,
        test_df=test_df,
        scenarios=scenarios,
        n_simulations=mcs_cfg.n_paths,
        block_size=mcs_cfg.block_length,
        random_seed=mcs_cfg.random_seed,
        sim_years=mcs_cfg.sim_years,
        trading_days_per_year=mcs_cfg.trading_days_per_year,
        bootstrap_method=getattr(mcs_cfg, "bootstrap_method", "block"),
        n_plot_paths=getattr(mcs_cfg, "n_plot_paths", 1000),
    )

    # Persist the MCS plot subsample (legacy parquet schema; inference below
    # uses mcs.finals / mcs.max_drawdowns, which cover ALL paths)
    mcs_results = mcs.sample_paths_frame()
    mcs_results.to_parquet(cfg.data_path("mcs_data"))

    if all_mc_summaries:
        mc_summary_df = pd.DataFrame(all_mc_summaries).set_index(["Scenario", "Strategy"])
        mc_summary_df.to_markdown(cfg.asset_path("mcs_summary"))

    # MCS Plots
    scenarios_list = list(vars(cfg.backtesting.sorr.scenarios).keys())
    strategies = list(backtesting_results.columns)
    total_days = mcs_cfg.sim_years * mcs_cfg.trading_days_per_year

    boxplot_template = os.path.join(str(cfg._base_dir / "assets"), "mcs_boxplot_{}.png")
    plot_mcs_boxplots(mcs.finals, daily_rets.columns, scenarios, mcs_cfg.sim_years,
                      boxplot_template)

    # Simulations start on the day after the data cutoff
    from datetime import datetime
    sim_start_year = datetime.strptime(cfg.data.end_date, "%Y-%m-%d").year + 1

    plot_mcs_paths(
        mcs_results, scenarios_list, strategies, cfg.color_map,
        cfg.asset_path("mcs_paths"),
        trading_days_per_year=mcs_cfg.trading_days_per_year,
        start_year=sim_start_year,
    )
    plot_mcs_quantiles(
        mcs_results, scenarios_list, strategies, total_days, cfg.color_map,
        cfg.asset_path("mcs_quantiles"),
        trading_days_per_year=mcs_cfg.trading_days_per_year,
        start_year=sim_start_year,
    )

    # --- Issue #13: Extended Evaluation ---
    ext = cfg.evaluation.extended
    models = list(ext.f1_models)
    logger.info("Running Issue #13 extended evaluation...")

    # 1) Ulcer index → extend the evaluation table
    evaluation_table = add_ulcer_to_table(backtesting_results, evaluation_table)
    evaluation_table.to_markdown(cfg.asset_path("evaluation_table"), index=True)

    # 2) Classification vs. NBER
    nber = load_nber_recession(test_df.index, source=ext.nber_source)
    class_tbl, cms = compute_classification_metrics(test_df, nber, models)
    class_tbl.to_markdown(cfg.asset_path("classification_metrics"))
    plot_confusion_matrices(cms, cfg.asset_path("confusion_matrices"))
    plot_roc_pr_curves(
        test_df, nber, models, cfg.color_map,
        cfg.asset_path("roc_curves"), cfg.asset_path("pr_curves"),
    )

    # 3) Churning + threshold sensitivity
    churn = churning_stats(
        test_df, models, cfg.transaction_cost_rate,
        min_phase_days=ext.whipsaw_min_phase_days,
    )
    churn.to_markdown(cfg.asset_path("churning_stats"))

    # 3b) Bear-market coverage of the walk-forward folds (Issue #8)
    bear_cov_path = generate_bear_coverage_report(cfg)
    logger.info(f"Bear-coverage diagnostic written: {bear_cov_path}")

    for m, grid in vars(ext.threshold_grid).items():
        ts = threshold_sensitivity(
            test_df, backtest, m, list(grid),
            cfg.transaction_cost_rate, cfg.backtesting.signal_shift,
            initial_capital=float(cfg.backtesting.sorr.scenarios.Standard.initial_capital),
        )
        ts.to_markdown(cfg.asset_path("threshold_sensitivity").replace("{model}", m))

    # 4) Regime probability heatmap
    plot_regime_probability_heatmap(
        test_df, models, cfg.asset_path("regime_probability_heatmap"),
    )

    # 5) Time-to-recovery + switch timing
    for m in ["Buy_Hold"] + models:
        if m not in backtesting_results.columns:
            continue
        ttr = time_to_recovery(backtesting_results[m], min_dd=ext.ttr_min_dd)
        ttr.to_markdown(
            cfg.asset_path("ttr_table").replace("{model}", m), index=False,
        )

    crisis_windows = {name: tuple(w) for name, w in vars(ext.crisis_windows).items()}
    switch_rows = []
    for m in models:
        t = switch_timing_vs_peak(test_df, backtesting_results, m, crisis_windows)
        if not t.empty:
            t.insert(0, "Model", m)
            switch_rows.append(t.reset_index())
    if switch_rows:
        pd.concat(switch_rows, ignore_index=True).to_markdown(
            cfg.asset_path("switch_timing"), index=False,
        )

    # 6) MCS: depletion CIs + H1/H2 + violin plots
    # (all on the full path set via MCSResult.finals / .max_drawdowns)
    finals = mcs.finals

    dep = depletion_rate_with_ci(finals, alpha=ext.alpha)
    dep.to_markdown(cfg.asset_path("depletion_ci"))

    regime_models = [m for m in strategies if m != "Buy_Hold"]

    # H1/H2 are reported for EVERY scenario, not just the headline one. The
    # MCS MaxDD is measured on the capital path after withdrawals, so at an
    # overstretched rate a merely underperforming strategy is driven to -100%
    # mechanically and H1 degenerates into a survival test. Which models pass
    # therefore depends strongly on the withdrawal rate, and reporting a single
    # scenario would present that dependence as a clean rejection.
    # `ext.hypothesis_scenario` still designates the headline scenario; the
    # extra rows are what makes its result interpretable.
    def _by_scenario(fn, index_col: str) -> pd.DataFrame:
        frames = []
        for sc in scenarios_list:
            tbl = fn(sc)
            if tbl is None or tbl.empty:
                continue
            tbl = tbl.reset_index()
            tbl.insert(0, "Scenario", sc)
            tbl.insert(1, "Headline", "yes" if sc == ext.hypothesis_scenario else "")
            frames.append(tbl)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).set_index(["Scenario", index_col])

    h1 = _by_scenario(
        lambda sc: test_h1_drawdown(
            mcs.max_drawdowns, scenario=sc,
            regime_models=regime_models, alpha=ext.alpha,
        ),
        "Model",
    )
    if not h1.empty:
        h1.to_markdown(cfg.asset_path("h1_drawdown"))

    h2_challenger = "Transformer"
    h2_competitors = tuple(
        m for m in regime_models if m != h2_challenger
    )
    h2 = _by_scenario(
        lambda sc: test_h2_transformer(
            finals, scenario=sc,
            challenger=h2_challenger, competitors=h2_competitors,
            alpha=ext.alpha,
        ),
        "Comparison",
    )
    if not h2.empty:
        h2.to_markdown(cfg.asset_path("h2_transformer"))

    violin_template = os.path.join(str(cfg._base_dir / "assets"), "mcs_violin_{}.png")
    plot_mcs_violins(
        finals, scenarios_list, strategies, cfg.color_map, violin_template,
    )

    # Hypothesis figures, built from the same arrays as the tables above.
    plot_depletion_forest(
        finals, scenarios_list, strategies,
        cfg.asset_path("mcs_depletion_forest"), alpha=ext.alpha,
    )
    plot_h1_forest(
        mcs.max_drawdowns, scenarios_list, regime_models,
        cfg.asset_path("mcs_h1_forest"), alpha=ext.alpha,
    )
    plot_h2_forest(
        finals, scenarios_list, h2_challenger, h2_competitors,
        cfg.asset_path("mcs_h2_forest"), alpha=ext.alpha,
    )
    plot_risk_return_positioning(
        finals, [sc for sc in scenarios_list if sc != "Standard"], strategies,
        cfg.asset_path("risk_return_positioning"),
        sim_years=mcs_cfg.sim_years, alpha=ext.alpha,
    )

    # 7) Break-even transaction costs
    be_tbl, be_curves = break_even_transaction_cost(
        test_df, backtest, backtesting_results["Buy_Hold"],
        [m for m in models if f"{m}_Signal" in test_df.columns],
        list(ext.fee_grid_bps),
        cfg.backtesting.signal_shift,
    )
    be_tbl.to_markdown(cfg.asset_path("break_even_table"))
    plot_break_even(
        be_curves, float(backtesting_results["Buy_Hold"].iloc[-1]),
        cfg.color_map, cfg.asset_path("break_even_plot"),
    )

    # 8) Withdrawal rate sensitivity
    wdraw = withdrawal_sensitivity(
        backtesting_results, test_df, run_sorr_simulation,
        base_scenario={
            "start": cfg.backtesting.sorr.scenarios.Standard.initial_capital,
            "fee":   cfg.backtesting.sorr.scenarios.Standard.liquidity_fee,
        },
        rates=tuple(ext.withdrawal_rates),
    )
    wdraw.to_markdown(cfg.asset_path("withdrawal_sensitivity"))

    logger.info("Issue #13 extended evaluation done")

    # Generate the statistics report
    generate_report()

    elapsed = time.time() - start
    logger.info(f"Evaluation complete in {elapsed:.1f}s")

    return {
        "status": "ok",
        "evaluation": evaluation_table.to_dict(),
        "mcs_scenarios": len(all_mc_summaries),
    }

@router.post("/report")
def generate_report():
    """Generate docs/statistics.md."""
    cfg = get_cfg()
    logger.info("Generating statistics report...")

    stats_md = generate_statistics_report(cfg)

    output_path = cfg.asset_path("statistics_output")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(stats_md)

    logger.info(f"Statistics report saved to {output_path}")
    return {"status": "ok", "path": output_path}

@router.post("/bootstrap-robustness")
def bootstrap_robustness(n_paths: int | None = None):
    """Bootstrap robustness comparison (Issue #7, Arbeitspaket 4 Teil A).

    Runs the MCS twice on the existing return/signal paths, once with the
    fixed-length block bootstrap and once with the stationary bootstrap
    (Politis & Romano 1994), using the same seed and n_paths. Writes a
    side-by-side comparison of depletion rate (Wilson CI) and median terminal
    capital per scenario and strategy to assets/bootstrap_robustness.md.

    No model re-training is involved; only the resampling scheme changes.
    Optional `n_paths` overrides the config value (useful for a quick check).
    """
    start = time.time()
    cfg = get_cfg()
    logger.info("Starting bootstrap robustness comparison (block vs. stationary)...")

    backtesting_results = pd.read_parquet(cfg.data_path("backtesting_results"))
    test_df = pd.read_parquet(cfg.data_path("test_data"))

    scenarios = build_sorr_scenarios(cfg.backtesting.sorr.scenarios)
    daily_rets = backtesting_results.pct_change().dropna()

    mcs_cfg = cfg.evaluation.mcs
    paths = int(n_paths) if n_paths else mcs_cfg.n_paths

    def _run(method: str) -> dict:
        # Same seed for both methods -> paired, reproducible comparison.
        # n_plot_paths=0: the comparison only needs terminal capitals,
        # so no full path histories are kept at all.
        _, mcs = run_monte_carlo_simulation(
            daily_rets=daily_rets,
            test_df=test_df,
            scenarios=scenarios,
            n_simulations=paths,
            block_size=mcs_cfg.block_length,
            random_seed=mcs_cfg.random_seed,
            sim_years=mcs_cfg.sim_years,
            trading_days_per_year=mcs_cfg.trading_days_per_year,
            bootstrap_method=method,
            n_plot_paths=0,
        )
        return mcs.finals

    finals_block = _run("block")
    finals_stationary = _run("stationary")

    table = compare_bootstrap_methods(
        finals_block, finals_stationary, alpha=cfg.evaluation.extended.alpha,
    )
    summary = bootstrap_robustness_summary(finals_block, finals_stationary)

    out_path = cfg.asset_path("bootstrap_robustness")
    header = (
        "# Bootstrap Robustness: Block vs. Stationary\n\n"
        f"Monte-Carlo depletion analysis on {paths:,} paths per cell, "
        f"seed {mcs_cfg.random_seed}, mean block length {mcs_cfg.block_length} "
        "trading days. Both runs share the same seed and paths; only the "
        "resampling scheme differs (fixed-length block bootstrap vs. the "
        "stationary bootstrap of Politis & Romano 1994).\n\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        # disable_numparse: keep the signed "+x.xx" Delta strings verbatim
        # (tabulate would otherwise reparse them and drop the sign/decimals).
        f.write(table.to_markdown(disable_numparse=True))
        f.write("\n\n" + summary + "\n")

    elapsed = time.time() - start
    logger.info(f"Bootstrap robustness comparison written to {out_path} in {elapsed:.1f}s")
    return {"status": "ok", "path": out_path, "n_paths": paths, "cells": len(table)}

@router.get("/results")
def get_results():
    """Evaluation table as JSON."""
    cfg = get_cfg()
    try:
        evaluation_md = Path(cfg.asset_path("evaluation_table")).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(404, "No evaluation results. Run /backtest/evaluate first.")
    return {"evaluation_table_md": evaluation_md}