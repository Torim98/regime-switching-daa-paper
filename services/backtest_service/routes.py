from fastapi import APIRouter, HTTPException
from config.config_loader import PipelineConfig
from src.backtest.sorr import run_sorr_simulation, build_sorr_scenarios, build_sorr_summary
from src.backtest.evaluation import (
    evaluate_strategies, run_monte_carlo_simulation,
    # Issue #13 — Extended Evaluation
    add_ulcer_to_table,
    compute_classification_metrics, plot_confusion_matrices, plot_roc_pr_curves,
    churning_stats, threshold_sensitivity,
    time_to_recovery, switch_timing_vs_peak,
    mcs_final_capitals, depletion_rate_with_ci,
    test_h1_drawdown, test_h2_transformer, plot_mcs_violins,
    break_even_transaction_cost, plot_break_even, withdrawal_sensitivity,
    plot_regime_probability_heatmap,
)
from src.backtest.reporting import generate_statistics_report
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
    """Backtesting + SORR aller Modelle durchführen."""
    start = time.time()
    cfg = get_cfg()
    logger.info("Starting backtesting...")

    test_df = pd.read_parquet(cfg.data_path("test_data"))

    # Walk-Forward: gemeinsames OOS-Fenster (alle Modelle müssen Signal haben)
    if cfg.walk_forward.enabled:
        signal_cols = [c for c in test_df.columns if c.endswith("_Signal")]
        n_before = len(test_df)
        test_df = test_df.dropna(subset=signal_cols, how="any").copy()
        logger.info(
            f"Walk-Forward OOS-Fenster: {n_before} → {len(test_df)} Zeilen "
            f"({test_df.index.min().date()} → {test_df.index.max().date()})"
        )

    # Backtesting
    backtesting_results, backtesting_costs = run_all_backtests(
        test_df=test_df,
        fee_rate=cfg.transaction_cost_rate,
        signal_shift=cfg.backtesting.signal_shift,
    )

    # Performance Summary
    performance_summary = calculate_performance_summary(backtesting_results)
    performance_summary.to_markdown(cfg.asset_path("performance_summary"))

    # Annualisierte Metriken
    annualized = calculate_annualized_metrics(backtesting_results)
    annualized.to_markdown(cfg.asset_path("annualized_metrics"))

    crisis = calculate_crisis_performance(backtesting_results)
    if not crisis.empty:
        crisis.to_markdown(cfg.asset_path("crisis_performance"))

    rolling_sharpe = calculate_rolling_sharpe(backtesting_results)
    plot_rolling_sharpe(rolling_sharpe, cfg.color_map,
                        cfg.asset_path("rolling_sharpe"))
    plot_drawdown(backtesting_results, cfg.color_map,
                  cfg.asset_path("drawdown"))

    # Persistieren
    Path(cfg.data_path("backtesting_results")).parent.mkdir(parents=True, exist_ok=True)
    backtesting_results.to_parquet(cfg.data_path("backtesting_results"))
    backtesting_costs.to_parquet(cfg.data_path("backtesting_costs"))

    # Plots (bisherig)
    plot_equity_curves(backtesting_results, cfg.color_map,
                       cfg.asset_path("equity_curves"))
    plot_transaction_costs(backtesting_costs, cfg.transaction_cost_rate,
                           cfg.color_map,
                           cfg.asset_path("transaction_costs"))

    # SORR für alle Szenarien
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
    sorr_df = pd.DataFrame(sorr_summaries).set_index(["Szenario", "Strategie"])
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
    """Performance-Metriken + Monte Carlo Simulation + Statistics Report."""
    start = time.time()
    cfg = get_cfg()
    logger.info("Starting evaluation + MCS...")

    backtesting_results = pd.read_parquet(cfg.data_path("backtesting_results"))
    backtesting_costs = pd.read_parquet(cfg.data_path("backtesting_costs"))
    test_df = pd.read_parquet(cfg.data_path("test_data"))

    # Signal-Zuordnung (dynamisch wie im Notebook)
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
    all_mc_summaries, mcs_paths = run_monte_carlo_simulation(
        daily_rets=daily_rets,
        test_df=test_df,
        scenarios=scenarios,
        n_simulations=mcs_cfg.n_paths,
        block_size=mcs_cfg.block_length,
        random_seed=mcs_cfg.random_seed,
        sim_years=mcs_cfg.sim_years,
        trading_days_per_year=mcs_cfg.trading_days_per_year,
    )

    # MCS Daten persistieren
    mcs_results = pd.DataFrame(mcs_paths)
    mcs_results.to_parquet(cfg.data_path("mcs_data"))

    if all_mc_summaries:
        mc_summary_df = pd.DataFrame(all_mc_summaries).set_index(["Szenario", "Strategie"])
        mc_summary_df.to_markdown(cfg.asset_path("mcs_summary"))

    # MCS Plots
    scenarios_list = list(vars(cfg.backtesting.sorr.scenarios).keys())
    strategies = list(backtesting_results.columns)
    total_days = mcs_cfg.sim_years * mcs_cfg.trading_days_per_year

    boxplot_template = os.path.join(str(cfg._base_dir / "assets"), "mcs_boxplot_{}.png")
    plot_mcs_boxplots(mcs_paths, daily_rets.columns, scenarios, mcs_cfg.sim_years,
                      boxplot_template)

    # Simulationen starten am Tag nach dem Daten-Cutoff
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

    # 1) Ulcer Index → Evaluation-Tabelle erweitern
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

    # 3) Churning + Threshold-Sensitivitaet
    churn = churning_stats(
        test_df, models, cfg.transaction_cost_rate,
        min_phase_days=ext.whipsaw_min_phase_days,
    )
    churn.to_markdown(cfg.asset_path("churning_stats"))
    for m, grid in vars(ext.threshold_grid).items():
        ts = threshold_sensitivity(
            test_df, backtest, m, list(grid),
            cfg.transaction_cost_rate, cfg.backtesting.signal_shift,
        )
        ts.to_markdown(cfg.asset_path("threshold_sensitivity").replace("{model}", m))

    # 4) Regime-Wahrscheinlichkeits-Heatmap
    plot_regime_probability_heatmap(
        test_df, models, cfg.asset_path("regime_probability_heatmap"),
    )

    # 5) Time-to-Recovery + Switch-Timing
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
            t.insert(0, "Modell", m)
            switch_rows.append(t.reset_index())
    if switch_rows:
        pd.concat(switch_rows, ignore_index=True).to_markdown(
            cfg.asset_path("switch_timing"), index=False,
        )

    # 6) MCS: Depletion-CIs + H1/H2 + Violin-Plots
    finals = mcs_final_capitals(mcs_paths, scenarios_list, strategies)

    dep = depletion_rate_with_ci(finals, alpha=ext.alpha)
    dep.to_markdown(cfg.asset_path("depletion_ci"))

    regime_models = [m for m in strategies if m != "Buy_Hold"]
    h1 = test_h1_drawdown(
        mcs_paths, scenario=ext.hypothesis_scenario,
        regime_models=regime_models, alpha=ext.alpha,
    )
    h1.to_markdown(cfg.asset_path("h1_drawdown"))

    h2 = test_h2_transformer(
        finals, scenario=ext.hypothesis_scenario, alpha=ext.alpha,
    )
    h2.to_markdown(cfg.asset_path("h2_transformer"))

    violin_template = os.path.join(str(cfg._base_dir / "assets"), "mcs_violin_{}.png")
    plot_mcs_violins(
        finals, scenarios_list, strategies, cfg.color_map, violin_template,
    )

    # 7) Break-Even-Transaktionskosten
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

    # 8) Entnahmeraten-Sensitivitaet
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

    # Statistics Report generieren (wie Notebook 99)
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
    """docs/statistics.md generieren (wie Notebook 99)."""
    cfg = get_cfg()
    logger.info("Generating statistics report...")

    stats_md = generate_statistics_report(cfg)

    output_path = cfg.asset_path("statistics_output")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(stats_md)

    logger.info(f"Statistics report saved to {output_path}")
    return {"status": "ok", "path": output_path}

@router.get("/results")
def get_results():
    """Evaluation-Tabelle als JSON."""
    cfg = get_cfg()
    try:
        evaluation_md = Path(cfg.asset_path("evaluation_table")).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(404, "No evaluation results. Run /backtest/evaluate first.")
    return {"evaluation_table_md": evaluation_md}