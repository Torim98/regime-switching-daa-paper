# Sequenzdiagramm: Jupyter Notebook Pipeline

## Inferenz-Durchlauf (End-to-End)

```mermaid
sequenceDiagram
    actor User
    participant Master as regime-switching-daa.ipynb<br/>(Papermill Orchestrator)
    participant NB01 as 01_data_preprocessing
    participant NB02 as 02_feature_engineering
    participant NB03 as 03_regime_switching_models
    participant NB04 as 04_backtesting
    participant NB05 as 05_evaluation
    participant NB99 as 99_statistics_md
    participant SRC as src/*<br/>(Shared Library)
    participant FS as Filesystem<br/>(data/ + assets/)

    Note over User,FS: Phase 1: Data Ingestion & Feature Engineering
    User->>Master: Run All (Jupyter / CLI)
    Master->>NB01: papermill execute
    NB01->>SRC: download_market_data (yfinance)
    SRC->>FS: raw_data.parquet (Bronze)
    NB01->>SRC: preprocess_pipeline
    SRC->>FS: preprocessed_data.parquet (Silver)
    NB01->>SRC: EDA (descriptive stats, ADF tests)
    SRC->>FS: EDA plots + markdown (Assets)
    NB01-->>Master: exit 0

    Master->>NB02: papermill execute
    NB02->>FS: read preprocessed_data
    NB02->>SRC: engineer_features
    SRC->>FS: feature_engineered_data.parquet (Silver)
    NB02-->>Master: exit 0

    opt Label-Analyse (manuell, optional — NB01a)
        User->>SRC: run_label_analysis (concordance.py)
        SRC->>FS: read test_df + raw_data
        SRC->>FS: label_concordance_matrix + label_timeline_comparison (Assets)
    end

    Note over User,FS: Phase 2: Model Training (sequentiell)
    Master->>NB03: papermill execute
    NB03->>FS: read feature_engineered_data

    alt walk_forward.enabled = false (Single-Split)
        NB03->>SRC: train_msm -> predict_msm
        SRC->>FS: msm_regime_model.pkl + MSM plot
        NB03->>SRC: train_hmm -> predict_hmm
        SRC->>FS: hmm_regime_model.pkl + HMM plot
        NB03->>SRC: train_lstm -> predict_lstm
        SRC->>FS: lstm_regime_model.keras + LSTM plot
        NB03->>SRC: train_transformer -> predict_transformer
        SRC->>FS: transformer_regime_model.pt + Transformer plot
    else walk_forward.enabled = true
        NB03->>SRC: walk_forward_splits() -> N Folds
        NB03->>SRC: check cache (fingerprint)
        alt Cache Hit
            SRC->>FS: load wf_cache.parquet
        else Cache Miss
            loop Fold 1..N
                SRC->>SRC: train + predict MSM, HMM, LSTM, Transformer
            end
            SRC->>FS: save wf_cache.parquet + .fingerprint
        end
    end
    NB03->>SRC: plot_regime_comparison
    SRC->>FS: regime_comparison plot
    NB03->>FS: test_df_data.parquet (Silver)
    NB03-->>Master: exit 0

    Note over User,FS: Phase 3: Backtesting & Evaluation
    Master->>NB04: papermill execute
    NB04->>FS: read test_df_data
    NB04->>SRC: run_all_backtests
    SRC->>FS: backtesting_results.parquet (Gold)
    SRC->>FS: Equity curves, drawdown, rolling Sharpe plots
    NB04->>SRC: run_sorr_simulation (alle Szenarien)
    SRC->>FS: backtesting_costs + backtesting_sorr (Gold)
    SRC->>FS: SORR plots
    NB04-->>Master: exit 0

    Master->>NB05: papermill execute
    NB05->>FS: read backtesting_results + costs + test_df
    NB05->>SRC: evaluate_strategies
    SRC->>FS: evaluation_table (Assets)
    NB05->>SRC: run_monte_carlo_simulation
    SRC->>FS: mcs_data.parquet (Gold) + MCS plots
    NB05-->>Master: exit 0

    Note over User,FS: Phase 4: Report
    Master->>Master: persist pipeline_timing.md
    Master->>NB99: papermill execute
    NB99->>FS: read all assets + evaluation data
    NB99->>FS: docs/statistics.md
    NB99-->>Master: exit 0
    Master-->>User: Pipeline erfolgreich durchlaufen
```

## Walk-Forward-Modus (`walk_forward.enabled: true`)

```mermaid
sequenceDiagram
    participant NB03a as 03a_hyperparameter_optimization
    participant NB03 as 03_regime_switching_models
    participant SRC as src/*<br/>(Shared Library)
    participant FS as Filesystem<br/>(data/ + models/)

    Note over NB03a,FS: Optional: Hyperparameter-Optimierung (separates Notebook)
    opt Optuna-Optimierung (manuell gestartet)
        NB03a->>FS: read feature_engineered_data
        NB03a->>SRC: optimize_all (alle 4 Modelle)
        Note over SRC: n_trials & every_nth_fold pro Modell<br/>aus config.yaml (50/2 für MSM+HMM,<br/>30/5 für LSTM+Transformer)
        SRC->>SRC: Optuna TPE (Walk-Forward CV als innere Validierung)
        SRC->>FS: optuna_studies.db
        NB03a->>SRC: save_optuna_plots
        SRC->>FS: Optuna-Visualisierungen (Assets)
        NB03a->>SRC: save_optuna_best_params
        SRC->>FS: assets/optuna_best_params.md
    end

    Note over NB03,FS: Walk-Forward-Validierung in NB03
    NB03->>FS: read feature_engineered_data
    NB03->>SRC: walk_forward_splits()
    SRC-->>NB03: splits (N Folds)
    NB03->>SRC: assert_no_leakage(splits)
    NB03->>SRC: _walk_forward_fingerprint(cfg, shape, idx_hash)
    alt Cache Hit
        NB03->>SRC: load_walk_forward_cache
        SRC->>FS: load wf_cache.parquet
    else Cache Miss
        NB03->>SRC: run_walk_forward(df, splits, cfg, models)
        loop Fold 1..N
            SRC->>SRC: train_msm_fold(train, test)
            SRC->>SRC: train_hmm_fold(train, test)
            SRC->>SRC: train_lstm_fold(train, test)
            SRC->>SRC: train_transformer_fold(train, test)
            SRC->>SRC: append OOS predictions
        end
        NB03->>SRC: save_walk_forward_cache
        SRC->>FS: wf_cache.parquet + .fingerprint
    end
    NB03->>SRC: validate_regime_signal (pro Modell)
    NB03->>FS: test_df_data.parquet (OOS only)
```
