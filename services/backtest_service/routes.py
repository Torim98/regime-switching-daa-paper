from fastapi import APIRouter, HTTPException
from config.config_loader import PipelineConfig
from src.backtest.engine import run_all_backtests, calculate_performance_summary
from src.backtest.sorr import run_sorr_simulation, build_sorr_scenarios, build_sorr_summary
from src.backtest.evaluation import evaluate_strategies, run_monte_carlo_simulation
from src.backtest.reporting import generate_statistics_report
from src.backtest.plots import (
    plot_equity_curves, plot_transaction_costs, plot_sorr_scenario,
    plot_mcs_boxplots, plot_mcs_paths, plot_mcs_quantiles,
)
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

    # Backtesting
    backtesting_results, backtesting_costs = run_all_backtests(
        test_df=test_df,
        fee_rate=cfg.transaction_cost_rate,
        signal_shift=cfg.backtesting.signal_shift,
    )

    # Performance Summary
    performance_summary = calculate_performance_summary(backtesting_results)
    performance_summary.to_markdown(cfg.asset_path("performance_summary"))

    # Persistieren
    Path(cfg.data_path("backtesting_results")).parent.mkdir(parents=True, exist_ok=True)
    backtesting_results.to_parquet(cfg.data_path("backtesting_results"))
    backtesting_costs.to_parquet(cfg.data_path("backtesting_costs"))

    # Plots
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
    plot_mcs_paths(mcs_results, scenarios_list, strategies,
                   cfg.color_map, cfg.asset_path("mcs_paths"))
    plot_mcs_quantiles(mcs_results, scenarios_list, strategies, total_days,
                       cfg.color_map, cfg.asset_path("mcs_quantiles"))

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