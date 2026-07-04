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
from src.backtest.plots import plot_walk_forward_schema
from src.data.labels.resolver import compute_supervised_labels, resolve_label_col
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
    """Train a single model. model_name: msm|hmm|hmm_uni|lstm|transformer"""
    start = time.time()
    logger.info(f"Training model: {model_name}")
    
    cfg = get_cfg()
    
    if cfg.walk_forward.enabled:
        raise HTTPException(
            400,
            f"Single-model training is not available in walk-forward mode. "
            f"Use /models/train-all instead."
        )

    df = pd.read_parquet(cfg.data_path("feature_engineered"))
    
    # Try to load an existing test_df (already contains signals of previous models)
    test_df_path = cfg.data_path("test_data")
    try:
        test_df = pd.read_parquet(test_df_path)
    except FileNotFoundError:
        test_df = None
    
    if model_name == "msm":
        msm_cfg = cfg.models.msm
        split_point = int(len(df) * msm_cfg.train_test_split)

        returns = df["Returns"].copy()
        returns.index = pd.DatetimeIndex(returns.index).to_period('B')
        returns_train = returns.iloc[:split_point]
        returns_test = returns.iloc[split_point:]

        ms_results = train_msm(
            returns_train=returns_train,
            k_regimes=msm_cfg.k_regimes,
            switching_variance=msm_cfg.switching_variance,
            model_file=cfg.model_path("msm"),
        )

        probs_train, signal_train, probs_test, signal_test = predict_msm(
            ms_results=ms_results,
            returns_train=returns_train,
            returns_test=returns_test,
            k_regimes=msm_cfg.k_regimes,
            switching_variance=msm_cfg.switching_variance,
            threshold=msm_cfg.threshold,
        )

        # Back to a timestamp index
        probs_train.index = probs_train.index.to_timestamp()
        signal_train.index = signal_train.index.to_timestamp()
        probs_test.index = probs_test.index.to_timestamp()
        signal_test.index = signal_test.index.to_timestamp()

        df.loc[probs_train.index, "MSM_Prob"] = probs_train.values
        df.loc[signal_train.index, "MSM_Signal"] = signal_train.values
        df.loc[probs_test.index, "MSM_Prob"] = probs_test.values
        df.loc[signal_test.index, "MSM_Signal"] = signal_test.values

        validate_regime_signal(df, "MSM")
        df.to_parquet(cfg.data_path("feature_engineered"))

        df_oos = df.iloc[split_point:].copy()
        plot_msm_regimes(df_oos, "MSM", cfg.color_map.get("MSM", "tab:blue"),
                         cfg.asset_path("markov_model"))
        
    elif model_name == "hmm":
        hmm_cfg = cfg.models.hmm
        split_point = int(len(df) * hmm_cfg.train_test_split)

        features_train = df[hmm_cfg.features].iloc[:split_point]
        features_test = df[hmm_cfg.features].iloc[split_point:]
        returns_train = df['Returns'].iloc[:split_point]

        model, scaler = train_hmm(
            features_df_train=features_train,
            n_components=hmm_cfg.n_components,
            covariance_type=hmm_cfg.covariance_type,
            n_iter=hmm_cfg.n_iter,
            random_state=hmm_cfg.random_state,
            model_file=cfg.model_path("hmm"),
            scaler_file=cfg.model_path("scaler_hmm"),
        )

        probs_train, signal_train, probs_test, signal_test = predict_hmm(
            model=model,
            scaler=scaler,
            features_df_train=features_train,
            features_df_test=features_test,
            returns_train=returns_train,
            threshold=hmm_cfg.threshold,
        )

        df.loc[features_train.index, "HMM_Prob"] = probs_train.values
        df.loc[features_train.index, "HMM_Signal"] = signal_train.values
        df.loc[features_test.index, "HMM_Prob"] = probs_test.values
        df.loc[features_test.index, "HMM_Signal"] = signal_test.values

        validate_regime_signal(df, "HMM")
        df.to_parquet(cfg.data_path("feature_engineered"))

        plot_hmm_regimes(df, "HMM", cfg.color_map.get("HMM", "tab:purple"),
                         cfg.asset_path("hmm_regimes"))
        
    elif model_name == "hmm_uni":
        uni_cfg = cfg.models.hmm_uni
        split_point = int(len(df) * uni_cfg.train_test_split)

        features_train = df[uni_cfg.features].iloc[:split_point]
        features_test = df[uni_cfg.features].iloc[split_point:]
        returns_train = df['Returns'].iloc[:split_point]

        model, scaler = train_hmm(
            features_df_train=features_train,
            n_components=uni_cfg.n_components,
            covariance_type=uni_cfg.covariance_type,
            n_iter=uni_cfg.n_iter,
            random_state=uni_cfg.random_state,
            model_file=cfg.model_path("hmm_uni"),
            scaler_file=cfg.model_path("scaler_hmm_uni"),
        )

        probs_train, signal_train, probs_test, signal_test = predict_hmm(
            model=model,
            scaler=scaler,
            features_df_train=features_train,
            features_df_test=features_test,
            returns_train=returns_train,
            threshold=uni_cfg.threshold,
        )

        df.loc[features_train.index, "HMM_Uni_Prob"] = probs_train.values
        df.loc[features_train.index, "HMM_Uni_Signal"] = signal_train.values
        df.loc[features_test.index, "HMM_Uni_Prob"] = probs_test.values
        df.loc[features_test.index, "HMM_Uni_Signal"] = signal_test.values

        validate_regime_signal(df, "HMM_Uni")
        df.to_parquet(cfg.data_path("feature_engineered"))

        plot_hmm_regimes(df, "HMM_Uni", cfg.color_map.get("HMM_Uni", "tab:pink"),
                         cfg.asset_path("hmm_uni_regimes"))

    elif model_name == "lstm":
        if cfg.labels.supervised_label_source == "hmm" and "HMM_Signal" not in df.columns:
            raise HTTPException(400, "HMM must be trained first (provides labels for LSTM).")
        
        lstm_cfg = cfg.models.lstm
        features = cfg.features.model_features
        window_size = lstm_cfg.window_size
        
        model_file = cfg.model_path("lstm")
        scaler_file = cfg.model_path("scaler_lstm")
        
        # Create the supervised labels (replaces the old HMM chain)
        if cfg.labels.supervised_label_source != "hmm":
            df["Supervised_Label"] = compute_supervised_labels(df, cfg)
        label_col = resolve_label_col(cfg)
        
        _, _, lstm_probs_raw, split = train_lstm(
            df=df, features=features,
            labels_col=label_col,
            window_size=window_size,
            train_test_split=lstm_cfg.train_test_split,
            units_l1=lstm_cfg.units_l1,
            units_l2=lstm_cfg.units_l2,
            return_sequences=lstm_cfg.return_sequences,
            dropout=lstm_cfg.dropout,
            dense=lstm_cfg.dense,
            activation=lstm_cfg.activation,
            optimizer=lstm_cfg.optimizer,
            metrics=lstm_cfg.metrics,
            epochs=lstm_cfg.epochs,
            batch_size=lstm_cfg.batch_size,
            validation_split=lstm_cfg.validation_split,
            verbose=lstm_cfg.verbose,
            model_file=model_file,
            scaler_file=scaler_file,
        )
        
        probs, signal = predict_lstm(lstm_probs_raw, lstm_cfg.threshold)
        
        # Create test_df: df trimmed to the X_test range
        test_df = df.iloc[split + window_size:].copy()
        test_df["LSTM_Prob"] = probs
        test_df["LSTM_Signal"] = signal
        validate_regime_signal(test_df, "LSTM")
        
        # Persist test_df
        from pathlib import Path as P
        P(test_df_path).parent.mkdir(parents=True, exist_ok=True)
        test_df.to_parquet(test_df_path)
        
        df_oos = df.iloc[split_point:].copy()
        plot_hmm_regimes(df_oos, "HMM", cfg.color_map.get("HMM", "tab:purple"),
                         cfg.asset_path("hmm_regimes"))
        
    elif model_name == "transformer":
        if cfg.labels.supervised_label_source == "hmm" and "HMM_Signal" not in df.columns:
            raise HTTPException(400, "HMM must be trained first (provides labels for LSTM).")
        
        t_cfg = cfg.models.transformer
        features = cfg.features.model_features
        window_size = t_cfg.window_size
        
        model_file = cfg.model_path("transformer")
        scaler_file = cfg.model_path("scaler_transformer")
        
        if cfg.labels.supervised_label_source != "hmm":
            if "Supervised_Label" not in df.columns:
                df["Supervised_Label"] = compute_supervised_labels(df, cfg)
        label_col = resolve_label_col(cfg)
        
        _, _, transformer_probs_raw, split_t = train_transformer(
            df=df, features=features,
            labels_col=label_col,
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
        
        # Write into the existing test_df
        test_df["Transformer_Prob"] = probs
        test_df["Transformer_Signal"] = signal
        validate_regime_signal(test_df, "Transformer")
        
        test_df.to_parquet(test_df_path)
        
        plot_dl_model(test_df, "Transformer", cfg.plotting.colors.transformer,
                      cfg.asset_path("transformer_model"), cfg=cfg)

        # Regime comparison (all 5 models done)
        plot_regime_comparison(test_df, cfg.color_map,
                               cfg.asset_path("regime_comparison"))
        
    else:
        raise HTTPException(400, f"Unknown model: {model_name}")

    elapsed = time.time() - start
    logger.info(f"Model {model_name} trained in {elapsed:.1f}s")    
    return {"status": "ok", "model": model_name}

@router.post("/train-all")
def train_all():
    """Train all 5 models: single split or walk-forward."""
    start = time.time()
    cfg = get_cfg()

    if cfg.walk_forward.enabled:
        # ============================================================
        # Walk-forward mode: run_walk_forward controls everything
        # ============================================================
        import hashlib
        from src.backtest.walk_forward import (
            walk_forward_splits,
            summarize_splits,
            assert_no_leakage,
            run_walk_forward,
            _walk_forward_fingerprint,
            load_walk_forward_cache,
            save_walk_forward_cache,
        )

        df = pd.read_parquet(cfg.data_path("feature_engineered"))

        logger.info(
            f"Walk-Forward: mode={cfg.walk_forward.mode}, "
            f"train={cfg.walk_forward.train_window_years}y, "
            f"test={cfg.walk_forward.test_window_months}m, "
            f"step={cfg.walk_forward.step_months}m"
        )

        # 1. Generate the splits
        splits = walk_forward_splits(
            index=df.index,
            mode=cfg.walk_forward.mode,
            train_window_years=cfg.walk_forward.train_window_years,
            test_window_months=cfg.walk_forward.test_window_months,
            step_months=cfg.walk_forward.step_months,
            min_train_years=cfg.walk_forward.min_train_years,
        )
        assert_no_leakage(splits)
        logger.info(f"Walk-forward: {len(splits)} folds generated.")

        # 1b. Persist the walk-forward schema as PNG (for statistics.md / dashboard)
        splits_summary = summarize_splits(splits)
        wf_schema_path = cfg.asset_path("walk_forward_schema")
        plot_walk_forward_schema(
            splits_summary=splits_summary,
            save_path=wf_schema_path,
            mode=cfg.walk_forward.mode,
            train_window_years=cfg.walk_forward.train_window_years,
            test_window_months=cfg.walk_forward.test_window_months,
        )
        logger.info(f"Walk-forward schema saved: {wf_schema_path}")

        # 2. Check the cache
        cache_path = cfg.data_path("walk_forward_cache")
        idx_hash = hashlib.sha256(
            df.index.astype(str).str.cat().encode()
        ).hexdigest()[:16]
        fingerprint = _walk_forward_fingerprint(cfg, df.shape, idx_hash)
        logger.info(f"Walk-forward fingerprint: {fingerprint}")

        use_cache = getattr(cfg.walk_forward, "cache_enabled", False)
        cached_df = None
        if use_cache:
            cached_df = load_walk_forward_cache(cache_path, fingerprint)

        if cached_df is not None:
            test_df = cached_df
            logger.info(f"Cache hit! {len(test_df)} OOS rows loaded.")
        else:
            # 3. Run the walk-forward
            test_df = run_walk_forward(
                df=df,
                splits=splits,
                cfg=cfg,
                models_to_run=["MSM", "HMM", "HMM_Uni", "LSTM", "Transformer"],
            )

            # Discard train-only rows
            signal_cols = [c for c in test_df.columns if c.endswith("_Signal")]
            test_df = test_df.dropna(subset=signal_cols, how="all").copy()

            if use_cache:
                save_walk_forward_cache(test_df, fingerprint, cache_path)

        # 4. Validation + individual plots per model
        asset_key_map = {
            "MSM": "markov_model",
            "HMM": "hmm_regimes",
            "HMM_Uni": "hmm_uni_regimes",
            "LSTM": "lstm_model",
            "Transformer": "transformer_model",
        }
        color_defaults = {
            "MSM": "tab:blue",
            "HMM": "tab:purple",
            "HMM_Uni": "tab:pink",
            "LSTM": "tab:green",
            "Transformer": "darkorange",
        }

        for model_name in ["MSM", "HMM", "HMM_Uni", "LSTM", "Transformer"]:
            sig_col = f"{model_name}_Signal"
            if sig_col not in test_df.columns or not test_df[sig_col].notna().any():
                logger.warning(f"{model_name}: no OOS predictions!")
                continue

            sub = test_df.dropna(subset=[sig_col]).copy()
            validate_regime_signal(sub, model_name)
            logger.info(f"{model_name}: {len(sub)} OOS days validated.")

            # Individual plot on the OOS range
            color = cfg.color_map.get(model_name, color_defaults[model_name])
            plot_path = cfg.asset_path(asset_key_map[model_name])

            if model_name == "MSM":
                plot_msm_regimes(sub, model_name, color, plot_path)
            elif model_name == "HMM":
                plot_hmm_regimes(sub, model_name, color, plot_path)
            elif model_name == "HMM_Uni":
                plot_hmm_regimes(sub, model_name, color, plot_path)
            else:  # LSTM / Transformer
                plot_dl_model(sub, model_name, color, plot_path, cfg=cfg)

            logger.info(f"{model_name}: plot saved → {plot_path}")

        # 5. Regime comparison plot
        plot_regime_comparison(test_df, cfg.color_map,
                               cfg.asset_path("regime_comparison"))

        # 6. Persist test_df (for the Backtest Service)
        test_df_path = cfg.data_path("test_data")
        Path(test_df_path).parent.mkdir(parents=True, exist_ok=True)
        test_df.to_parquet(test_df_path)

        elapsed = time.time() - start
        logger.info(f"Walk-Forward complete in {elapsed:.1f}s")
        return {
            "status": "ok",
            "mode": "walk_forward",
            "folds": len(splits),
            "oos_days": len(test_df),
        }

    else:
        # ============================================================
        # Single-split mode: existing logic (sequential)
        # ============================================================
        logger.info("Single split: training MSM → HMM → LSTM → Transformer")
        results = []
        for name in ["msm", "hmm", "hmm_uni", "lstm", "transformer"]:
            result = train_model(name)
            results.append(result)

        elapsed = time.time() - start
        logger.info(f"All models trained in {elapsed:.1f}s")
        return {"status": "ok", "mode": "single_split", "results": results}

@router.get("/status")
def model_status():
    """Which models are persisted?"""
    cfg = get_cfg()
    status = {}
    for key in ["msm", "hmm", "hmm_uni","lstm", "transformer"]:
        path = cfg.model_path(key)
        status[key] = Path(path).exists()
    return status
    
@router.post("/optimize/{model_name}")
async def optimize_model(model_name: str):
    """
    Optuna hyperparameter optimization for a single model.
    Uses walk-forward splits as inner CV.

    Sampler, trial budget, objective metric and the tune_until fold restriction
    are read from the central config (cfg.optimization). No API overrides, so
    the run stays reproducible from config.yaml alone.
    """
    cfg = get_cfg()

    if not cfg.walk_forward.enabled:
        raise HTTPException(400, "Optimization requires walk_forward.enabled = true")

    valid_models = ["MSM", "HMM", "HMM_Uni", "LSTM", "Transformer"]
    if model_name not in valid_models:
        raise HTTPException(400, f"Unknown model. Available: {valid_models}")

    df = pd.read_parquet(cfg.data_path("feature_engineered"))

    from src.backtest.optimize import run_optimization
    study = run_optimization(
        model_name=model_name,
        df=df,
        cfg=cfg,
        # n_trials / every_nth_fold → from the config (per model)
        storage=f"sqlite:///{cfg.model_path('optuna_db')}",
    )

    return {
        "status": "ok",
        "model": model_name,
        "metric": study.user_attrs.get("metric", cfg.optimization.metric),
        "best_score": round(study.best_value, 4),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }

@router.post("/optimize-all")
async def optimize_all_models():
    """
    Optimize all 5 models sequentially.

    Sampler, trial budget, metric and fold restriction are read from the central
    config (cfg.optimization): econometric models are searched exhaustively via
    GridSampler, DL models via a multivariate TPESampler on the pooled-OOS
    objective. See config.yaml (Issue #5) for the full setup.
    """
    cfg = get_cfg()

    if not cfg.walk_forward.enabled:
        raise HTTPException(400, "Optimization requires walk_forward.enabled = true")

    df = pd.read_parquet(cfg.data_path("feature_engineered"))

    from src.backtest.optimize import optimize_all, save_optuna_best_params
    studies = optimize_all(
        df=df,
        cfg=cfg,
        # n_trials / every_nth_fold → per model from the config
        storage=f"sqlite:///{cfg.model_path('optuna_db')}",
    )

    # Persist the best params under assets/
    save_optuna_best_params(studies, cfg)

    return {
        "status": "ok",
        "metric": cfg.optimization.metric,
        "results": {
            name: {
                "best_score": round(s.best_value, 4),
                "best_params": s.best_params,
            }
            for name, s in studies.items()
        },
    }


@router.post("/hpo-analysis")
async def hpo_analysis(scope: str = "cheap"):
    """
    Post-HPO analysis reports (Issue #5). Runs against the persisted Optuna
    studies and writes Markdown assets that statistics.md embeds and the
    dashboard renders.

    scope='cheap' (default): convergence + edge-of-range review and objective
        sensitivity. Reads the logged trial metrics only, no retraining (seconds).
    scope='full': additionally Deflated Sharpe Ratio, PBO/CSCV and multi-seed
        reeval, which re-train the models on the GPU (minutes-hours).
    """
    if scope not in ("cheap", "full"):
        raise HTTPException(400, "scope must be 'cheap' or 'full'")

    cfg = get_cfg()
    from src.backtest.hpo_analysis import generate_hpo_reports

    df = pd.read_parquet(cfg.data_path("feature_engineered")) if scope == "full" else None
    summary = generate_hpo_reports(cfg, df=df, scope=scope)
    return {"status": "ok", "scope": scope, "assets": summary}