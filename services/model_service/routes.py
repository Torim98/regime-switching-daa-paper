from fastapi import APIRouter, HTTPException
from config.config_loader import PipelineConfig
from src.models.msm import train_msm, load_msm, predict_msm
from src.models.hmm import train_hmm, load_hmm, predict_hmm
from src.models.lstm import train_lstm, load_lstm_model, predict_lstm
from src.models.transformer import train_transformer, load_transformer_model, predict_transformer
from src.models.common import validate_regime_signal
from src.models.plots import (
    plot_msm_regimes, plot_hmm_regimes, plot_dl_model, plot_regime_comparison,
)
from pathlib import Path
import pandas as pd
import time
import logging

router = APIRouter(prefix="/models", tags=["models"])

logger = logging.getLogger("model_service")

def get_cfg():
    return PipelineConfig()

@router.post("/train/{model_name}")

def train_model(model_name: str):
    """Einzelnes Modell trainieren. model_name: msm|hmm|lstm|transformer"""
    start = time.time()
    logger.info(f"Training model: {model_name}")
    
    cfg = get_cfg()
    df = pd.read_parquet(cfg.data_path("feature_engineered"))
    
    # Versuche existierenden test_df zu laden (enthält bereits Signale vorheriger Modelle)
    test_df_path = cfg.data_path("test_data")
    try:
        test_df = pd.read_parquet(test_df_path)
    except FileNotFoundError:
        test_df = None
    
    if model_name == "msm":
        returns = df["Returns"].copy()
        returns.index = pd.DatetimeIndex(returns.index).to_period('B')
        ms_results = train_msm(
            returns=returns,
            k_regimes=cfg.models.msm.k_regimes,
            switching_variance=cfg.models.msm.switching_variance,
            model_file=cfg.model_path("msm"),
        )
        probs, signal = predict_msm(ms_results, cfg.models.msm.threshold)
        probs.index = probs.index.to_timestamp()
        signal.index = signal.index.to_timestamp()
        df["MSM_Prob"] = probs
        df["MSM_Signal"] = signal
        validate_regime_signal(df, "MSM")
        # MSM und HMM operieren auf dem vollen df, test_df wird erst bei LSTM erstellt
        # Aber wir persistieren MSM-Signale im feature_engineered df
        df.to_parquet(cfg.data_path("feature_engineered"))
        
        plot_msm_regimes(df, "MSM", cfg.color_map.get("MSM", "tab:blue"),
                         cfg.asset_path("markov_model"))
        
    elif model_name == "hmm":
        hmm_features = df[cfg.models.hmm.features]
        model, scaler, X_scaled = train_hmm(
            features_df=hmm_features,
            n_components=cfg.models.hmm.n_components,
            covariance_type=cfg.models.hmm.covariance_type,
            n_iter=cfg.models.hmm.n_iter,
            random_state=cfg.models.hmm.random_state,
            model_file=cfg.model_path("hmm"),
            scaler_file=cfg.model_path("scaler_hmm"),
        )
        probs, signal = predict_hmm(model, X_scaled, df["Returns"], cfg.models.hmm.threshold)
        df["HMM_Prob"] = probs
        df["HMM_Signal"] = signal
        validate_regime_signal(df, "HMM")
        df.to_parquet(cfg.data_path("feature_engineered"))
        
        plot_hmm_regimes(df, "HMM", cfg.color_map.get("HMM", "tab:purple"),
                         cfg.asset_path("hmm_regimes"))
        
    elif model_name == "lstm":
        if "MSM_Signal" not in df.columns:
            raise HTTPException(400, "MSM must be trained first (provides labels for LSTM).")
        
        lstm_cfg = cfg.models.lstm
        features = cfg.features.model_features
        window_size = lstm_cfg.window_size
        
        model_file = cfg.model_path("lstm")
        scaler_file = cfg.model_path("scaler_lstm")
        
        _, _, lstm_probs_raw, split = train_lstm(
            df=df, features=features,
            labels_col=lstm_cfg.labels,
            window_size=window_size,
            train_test_split=lstm_cfg.train_test_split,
            units=lstm_cfg.units,
            return_sequences=lstm_cfg.return_sequences,
            dropout=lstm_cfg.dropout,
            dense=lstm_cfg.dense,
            activation=lstm_cfg.activation,
            optimizer=lstm_cfg.optimizer,
            loss=lstm_cfg.loss,
            metrics=lstm_cfg.metrics,
            epochs=lstm_cfg.epochs,
            batch_size=lstm_cfg.batch_size,
            validation_split=lstm_cfg.validation_split,
            verbose=lstm_cfg.verbose,
            model_file=model_file,
            scaler_file=scaler_file,
        )
        
        probs, signal = predict_lstm(lstm_probs_raw, lstm_cfg.threshold)
        
        # test_df erstellen: df zugeschnitten auf X_test Bereich
        test_df = df.iloc[split + window_size:].copy()
        test_df["LSTM_Prob"] = probs
        test_df["LSTM_Signal"] = signal
        validate_regime_signal(test_df, "LSTM")
        
        # test_df persistieren
        from pathlib import Path as P
        P(test_df_path).parent.mkdir(parents=True, exist_ok=True)
        test_df.to_parquet(test_df_path)
        
        plot_dl_model(test_df, "LSTM", cfg.color_map.get("LSTM", "tab:green"),
                      cfg.asset_path("lstm_model"))
        
    elif model_name == "transformer":
        if test_df is None:
            raise HTTPException(400, "LSTM must be trained first (creates test_df).")
        
        t_cfg = cfg.models.transformer
        features = cfg.features.model_features
        window_size = t_cfg.window_size
        
        model_file = cfg.model_path("transformer")
        scaler_file = cfg.model_path("scaler_transformer")
        
        _, _, transformer_probs_raw, split_t = train_transformer(
            df=df, features=features,
            labels_col=t_cfg.labels,
            window_size=window_size,
            train_test_split=t_cfg.train_test_split,
            d_model=t_cfg.d_model,
            n_heads=t_cfg.n_heads,
            n_layers=t_cfg.n_layers,
            dim_feedforward=t_cfg.dim_feedforward,
            dropout=t_cfg.dropout,
            learning_rate=t_cfg.learning_rate,
            epochs=t_cfg.epochs,
            batch_size=t_cfg.batch_size,
            validation_split=t_cfg.validation_split,
            verbose=t_cfg.verbose,
            model_file=model_file,
            scaler_file=scaler_file,
        )
        
        probs, signal = predict_transformer(transformer_probs_raw, t_cfg.threshold)
        
        # In bestehenden test_df schreiben
        test_df["Transformer_Prob"] = probs
        test_df["Transformer_Signal"] = signal
        validate_regime_signal(test_df, "Transformer")
        
        test_df.to_parquet(test_df_path)
        
        plot_dl_model(test_df, "Transformer",
                      cfg.color_map.get("Transformer", "darkorange"),
                      cfg.asset_path("transformer_model"))

        # Regime Comparison (alle 4 Modelle fertig)
        plot_regime_comparison(test_df, cfg.color_map,
                               cfg.asset_path("regime_comparison"))
        
    else:
        raise HTTPException(400, f"Unknown model: {model_name}")

    elapsed = time.time() - start
    logger.info(f"Model {model_name} trained in {elapsed:.1f}s")    
    return {"status": "ok", "model": model_name}

@router.post("/train-all")
def train_all():
    """Alle 4 Modelle sequentiell trainieren (MSM zuerst!)."""
    start = time.time()
    logger.info(f"Training model: {model_name}")
    
    results = []
    for name in ["msm", "hmm", "lstm", "transformer"]:
        result = train_model(name)
        results.append(result)
    
    elapsed = time.time() - start
    logger.info(f"Model {model_name} trained in {elapsed:.1f}s")
    return {"status": "ok", "results": results}

@router.get("/status")
def model_status():
    """Welche Modelle sind persistiert?"""
    cfg = get_cfg()
    status = {}
    for key in ["msm", "hmm", "lstm", "transformer"]:
        path = cfg.model_path(key)
        status[key] = Path(path).exists()
    return status