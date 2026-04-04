from fastapi import APIRouter, HTTPException
from config.config_loader import PipelineConfig
from src.data.ingestion import download_market_data, save_raw_data
from src.data.preprocessing import preprocess_pipeline
from src.data.feature_engineering import engineer_features
import pandas as pd

router = APIRouter(prefix="/data", tags=["data"])

def get_cfg():
    return PipelineConfig()

@router.post("/ingest")
def ingest():
    """Download → Preprocess → Feature Engineering → Persist."""
    cfg = get_cfg()
    
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
    
    # 3. Feature Engineering
    df = engineer_features(
        df,
        volatility_window=cfg.features.volatility_window,
        sma_window=cfg.features.sma_window,
        momentum_window=cfg.features.momentum_window,
    )
    df.to_parquet(cfg.data_path("feature_engineered"))
    
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