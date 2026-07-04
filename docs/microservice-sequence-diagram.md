# Sequence Diagram: Microservice Pipeline

## Inference Run (End-to-End)

```mermaid
sequenceDiagram
    actor Client
    participant DS as Data Service<br/>:8001
    participant MS as Model Service<br/>:8002
    participant BS as Backtest Service<br/>:8003
    participant FS as Filesystem<br/>(Shared Volumes)

    Note over Client,FS: Phase 1: Data Ingestion
    Client->>DS: POST /data/ingest
    DS->>DS: download_market_data (yfinance)
    DS->>FS: raw_data.parquet (Bronze)
    DS->>DS: preprocess_pipeline
    DS->>FS: preprocessed_data.parquet (Silver)
    DS->>DS: engineer_features
    DS->>FS: feature_engineered_data.parquet (Silver)
    DS->>DS: EDA (descriptive stats, ADF tests)
    DS->>FS: EDA plots + markdown (Assets)
    DS-->>Client: 200 OK {rows, columns}
	
    opt Label analysis (optional, one-time)
        Client->>DS: POST /data/label-analysis
        DS->>FS: read test_df_data + raw_data
        DS->>DS: compute_concordance_matrix (MSM, HMM, PagSoss, P2T, LundeT, NBER)
        DS->>FS: label_concordance_matrix.png + label_timeline_comparison.png
        DS-->>Client: 200 OK {concordance, switch_stats}
    end

    Note over Client,FS: Phase 2: Model Training (sequential)
    Client->>MS: POST /models/train/msm
    MS->>FS: read feature_engineered_data
    MS->>MS: train_msm → predict_msm
    MS->>FS: msm_regime_model.pkl + MSM plot
    MS->>FS: feature_engineered_data + MSM_Signal
    MS-->>Client: 200 OK

    Client->>MS: POST /models/train/hmm
    MS->>FS: read feature_engineered_data
    MS->>MS: train_hmm → predict_hmm
    MS->>FS: hmm_regime_model.pkl + HMM plot
    MS->>FS: feature_engineered_data + HMM_Signal
    MS-->>Client: 200 OK

    Client->>MS: POST /models/train/lstm
    MS->>FS: read feature_engineered_data
    MS->>MS: train_lstm → predict_lstm
    MS->>FS: lstm_regime_model.keras + LSTM plot
    MS->>FS: test_df_data.parquet (Silver) ← created here
    MS-->>Client: 200 OK

    Client->>MS: POST /models/train/transformer
    MS->>FS: read feature_engineered + test_df
    MS->>MS: train_transformer → predict_transformer
    MS->>FS: transformer_regime_model.pt + Transformer plot
    MS->>FS: test_df_data + Transformer_Signal
    MS->>FS: regime_comparison plot
    MS-->>Client: 200 OK

    Note over Client,FS: Phase 3: Backtesting & Evaluation
    Client->>BS: POST /backtest/run
    BS->>FS: read test_df_data
    BS->>BS: run_all_backtests
    BS->>FS: backtesting_results + costs (Gold)
    BS->>FS: Equity curves, transaction costs, SORR plots
    BS-->>Client: 200 OK {strategies, rows}

    Client->>BS: POST /backtest/evaluate
    BS->>FS: read backtesting_results + costs + test_df
    BS->>BS: evaluate_strategies
    BS->>BS: run_monte_carlo_simulation
    BS->>FS: MCS data (Gold) + MCS plots
    BS->>BS: generate_statistics_report
    BS->>FS: docs/statistics.md
    BS-->>Client: 200 OK {evaluation, mcs_scenarios}
```

## Walk-Forward Mode (`walk_forward.enabled: true`)

```mermaid
sequenceDiagram
    actor Client
    participant MS as Model Service<br/>:8002
    participant WF as run_walk_forward<br/>(src/backtest)
    participant FS as Filesystem<br/>(Shared Volumes)
    participant BS as Backtest Service<br/>:8003

    Note over Client,FS: Phase 2: Walk-Forward Training
	opt Hyperparameter optimization (optional, one-time)
        Client->>MS: POST /models/optimize-all
        MS->>FS: read feature_engineered_data
        MS->>MS: Optuna grid/TPE on pooled OOS metric (walk-forward CV as inner validation)
        MS->>FS: save optuna_studies.db
        MS-->>Client: 200 OK (best_params per model)
    end
    Client->>MS: POST /models/train-all
    MS->>FS: read feature_engineered_data
    MS->>MS: walk_forward_splits() → 26 folds
    MS->>MS: check cache (fingerprint)
    alt Cache hit
        MS->>FS: load wf_cache.parquet
    else Cache miss
        loop Fold 1..26
            MS->>WF: train_msm_fold(train, test)
            MS->>WF: train_hmm_fold(train, test)
            MS->>WF: train_lstm_fold(train, test)
            MS->>WF: train_transformer_fold(train, test)
            WF->>WF: write OOS predictions to result_df
        end
        MS->>FS: save wf_cache.parquet + .fingerprint
    end
    MS->>FS: test_df_data.parquet (OOS only)
    MS->>FS: regime_comparison plot
    MS-->>Client: 200 OK {mode: walk_forward, folds: 26}

    Note over Client,FS: Phase 3: Backtesting (identical)
    Client->>BS: POST /backtest/run
    BS->>FS: read test_df_data
    BS->>BS: dropna(how="any") → common OOS window
    BS->>BS: run_all_backtests + extended metrics
    BS->>FS: backtesting_results + annualized_metrics + crisis_performance
    BS->>FS: equity_curves + drawdown + rolling_sharpe
    BS-->>Client: 200 OK
```

## Dashboard Service: Control Hub & Visualization

The Dashboard Service (`:8004`) is not a pipeline step but a UI layer on top. It reads the artifacts read-only from the filesystem and proxies pipeline calls to the three services. The following diagram shows the typical interaction paths.

```mermaid
sequenceDiagram
    actor User as Browser<br/>localhost:8004
    participant DB as Dashboard Service<br/>:8004
    participant DS as Data Service<br/>:8001
    participant MS as Model Service<br/>:8002
    participant BS as Backtest Service<br/>:8003
    participant FS as Filesystem<br/>(Shared Volumes)

    Note over User,FS: Visualization (read-only)
    User->>DB: GET /eda
    DB->>DB: Jinja2 render eda.html
    DB-->>User: HTML + Tailwind + Plotly.js
    User->>DB: GET /api/chart/returns?col=Returns&smoothing=20
    DB->>FS: read feature_engineered_data.parquet
    DB->>DB: Plotly Figure → JSON
    DB-->>User: Plotly JSON (rendered client-side)

    Note over User,FS: Control Hub (proxy)
    User->>DB: GET /api/hub/health
    DB->>DS: GET /openapi.json
    DS-->>DB: 200 OK
    DB->>MS: GET /openapi.json
    MS-->>DB: 200 OK
    DB->>BS: GET /openapi.json
    BS-->>DB: 200 OK
    DB-->>User: {data: up, model: up, backtest: up}

    User->>DB: POST /api/hub/call?service=data&path=/data/ingest&method=POST
    DB->>DS: POST /data/ingest (httpx proxy, 8h timeout)
    DS->>FS: write pipeline artifacts
    DS-->>DB: 200 OK {rows, columns}
    DB-->>User: {status_code: 200, ok: true, body: {...}}

    Note over User,FS: Live log streaming (WebSocket)
    User->>DB: WS /ws/logs/data_service.log?tail=500
    DB->>FS: read last 500 lines
    DB-->>User: 500x text frame (initial tail)
    loop Polling (~300 ms)
        DB->>FS: stat + read new bytes
        DB-->>User: text frame (new lines)
    end

    Note over User,FS: Config editor (backup + rollback)
    User->>DB: GET /api/config
    DB->>FS: read config/config.yaml
    DB-->>User: {content: "yaml...", mtime, size}
    User->>DB: POST /api/config {content: "modified YAML"}
    DB->>DB: YAML parse + required-sections check
    DB->>FS: config.yaml → config.YYYYMMDD-HHMMSS.bak
    DB->>FS: atomic write config.yaml
    DB->>DB: PipelineConfig() reload test
    alt Reload OK
        DB-->>User: {status: ok, backup: "...bak"}
    else Reload failed
        DB->>FS: rollback from .bak
        DB-->>User: 422 {detail: "Rollback performed"}
    end
```
