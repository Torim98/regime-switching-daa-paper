from fastapi import APIRouter, HTTPException
from config.config_loader import PipelineConfig
from src.backtest.engine import run_all_backtests, calculate_performance_summary
from src.backtest.sorr import run_sorr_simulation, build_sorr_scenarios, build_sorr_summary
from src.backtest.evaluation import evaluate_strategies, run_monte_carlo_simulation
from pathlib import Path
import pandas as pd
import numpy as np

router = APIRouter(prefix="/backtest", tags=["backtest"])

def get_cfg():
    return PipelineConfig()

@router.post("/run")
def run_backtest():
    """Backtesting aller Modelle durchführen."""
    cfg = get_cfg()
    
    test_df = pd.read_parquet(cfg.data_path("test_data"))
    
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
    
    return {
        "status": "ok",
        "strategies": list(backtesting_results.columns),
        "rows": len(backtesting_results),
    }

@router.post("/sorr/{scenario}")
def run_sorr(scenario: str):
    """SORR-Simulation für ein Szenario. scenario: Standard|Aggressive|Low_Capital"""
    cfg = get_cfg()
    
    backtesting_results = pd.read_parquet(cfg.data_path("backtesting_results"))
    test_df = pd.read_parquet(cfg.data_path("test_data"))
    
    scenarios = build_sorr_scenarios(cfg.backtesting.sorr.scenarios)
    if scenario not in scenarios:
        raise HTTPException(400, f"Unknown scenario: {scenario}. Available: {list(scenarios.keys())}")
    
    params = scenarios[scenario]
    sim_results = run_sorr_simulation(
        backtesting_results, test_df,
        params["start"], params["withdrawal"], params["fee"],
    )
    
    # Persistieren
    sim_results.to_parquet(cfg.data_path("backtesting_sorr"))
    
    summary = build_sorr_summary(sim_results, scenario)
    return {"status": "ok", "scenario": scenario, "summary": summary}

@router.post("/evaluate")
def evaluate():
    """Performance-Metriken + Monte Carlo Simulation."""
    cfg = get_cfg()
    
    backtesting_results = pd.read_parquet(cfg.data_path("backtesting_results"))
    backtesting_costs = pd.read_parquet(cfg.data_path("backtesting_costs"))
    test_df = pd.read_parquet(cfg.data_path("test_data"))
    
    # Signal-Zuordnung (dynamisch wie im Notebook)
    signals_to_count = pd.DataFrame(index=test_df.index)
    for sig_col in [c for c in test_df.columns if c.endswith('_Signal')]:
        model_name = sig_col.rsplit('_', 1)[0]
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
        mc_summary_df = pd.DataFrame(all_mc_summaries).set_index(['Szenario', 'Strategie'])
        mc_summary_df.to_markdown(cfg.asset_path("mcs_summary"))
    
    return {
        "status": "ok",
        "evaluation": evaluation_table.to_dict(),
        "mcs_scenarios": len(all_mc_summaries),
    }

@router.get("/results")
def get_results():
    """Evaluation-Tabelle als JSON."""
    cfg = get_cfg()
    try:
        evaluation_md = Path(cfg.asset_path("evaluation_table")).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(404, "No evaluation results. Run /backtest/evaluate first.")
    return {"evaluation_table_md": evaluation_md}