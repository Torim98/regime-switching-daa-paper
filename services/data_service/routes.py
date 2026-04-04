from fastapi import APIRouter, HTTPException
from config.config_loader import PipelineConfig
from src.data.ingestion import download_market_data, save_raw_data
from src.data.preprocessing import preprocess_pipeline
from src.data.feature_engineering import engineer_features
from src.data.eda import calculate_descriptive_stats, run_adf_test
from src.data.plots import (
    plot_volatility_clusters, plot_historical_drawdowns,
    plot_capital_curve, plot_feature_correlation,
)
import pandas as pd
import time
import logging

logger = logging.getLogger("data_service")

router = APIRouter(prefix="/data", tags=["data"])

def get_cfg():
    return PipelineConfig()

@router.post("/ingest")
def ingest():
    """Download → Preprocess → Feature Engineering → EDA → Plots → Persist."""
    start = time.time()
    cfg = get_cfg()
    logger.info("Starting data ingestion...")

    # 1. Download
    raw_data = download_market_data(
        tickers=cfg.data.tickers,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    save_raw_data(raw_data, cfg.data_path("raw"))

    # 2. Preprocess
    df = preprocess_pipeline(
        raw_data,
        weight_equity=cfg.portfolio.weight_equity,
        weight_bonds=cfg.portfolio.weight_bonds,
    )
    df.to_parquet(cfg.data_path("preprocessed"))

    # 3. EDA — Deskriptive Statistik + ADF-Tests
    cols_to_analyze = ["Returns_GSPC", "Returns_VUSTX", "Returns", "VIX", "TNX_10Y", "IRX_3M"]
    available_cols = [c for c in cols_to_analyze if c in df.columns]

    desc_stats = calculate_descriptive_stats(df, available_cols)
    desc_stats.to_markdown(cfg.asset_path("eda_descriptive_stats"))

    adf_table = run_adf_test(df, available_cols)
    adf_table.to_markdown(cfg.asset_path("eda_adf_tests"))

    # 4. EDA-Plots (brauchen Returns_GSPC und Returns)
    plot_volatility_clusters(df, cfg.asset_path("eda_volatility_clusters"))
    plot_historical_drawdowns(df, cfg.asset_path("eda_historical_drawdowns"))

    # 5. Feature Engineering
    df = engineer_features(
        df,
        volatility_window=cfg.features.volatility_window,
        sma_window=cfg.features.sma_window,
        momentum_window=cfg.features.momentum_window,
    )
    df.to_parquet(cfg.data_path("feature_engineered"))

    # 6. Feature-Plots
    plot_capital_curve(df, cfg.asset_path("capital_curve"))
    plot_feature_correlation(
        df, cfg.features.model_features,
        cfg.asset_path("feature_correlation_matrix"),
        cfg.asset_path("feature_correlation_table"),
    )

    elapsed = time.time() - start
    logger.info(f"Data ingestion complete: {len(df)} rows, {elapsed:.1f}s")

    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}

@router.get("/features")
def get_features():
    """Feature-DataFrame als JSON."""
    cfg = get_cfg()
    path = cfg.data_path("feature_engineered")
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        raise HTTPException(404, "Feature data not found. Run /data/ingest first.")
    return df.to_dict(orient="split")