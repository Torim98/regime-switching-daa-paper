# Data Architecture: Medallion Model (Bronze / Silver / Gold)

## Overview

Data storage follows a three-tier medallion model.
Each tier represents a defined degree of data processing.

```
Yahoo Finance API
       │
       ▼
  ┌──────────┐
  │  Bronze  │  Raw data: uncleaned, directly from the source
  └────┬─────┘
       │  POST /data/ingest
       ▼
  ┌──────────┐
  │  Silver  │  Cleaned, transformed, and feature-engineered data
  └────┬─────┘
       │  POST /backtest/run + /backtest/evaluate
       ▼
  ┌──────────┐
  │   Gold   │  Results: backtesting, Monte Carlo simulation
  └──────────┘
```

## Directory Structure

```
data/
├── bronze/
│   └── 01_raw_data.parquet
├── silver/
│   ├── 02_preprocessed_data.parquet
│   ├── 03_feature_engineered_data.parquet
│   └── 04_test_df_data.parquet
│   └── wf_cache.parquet
│   └── wf_cache.parquet.fingerprint
└── gold/
    ├── 05_backtesting_results_data.parquet
    ├── 05_backtesting_transaction_costs_data.parquet
    ├── 05_backtesting_sorr_simulation.parquet
    └── 06_mcs_data.parquet
```

## Tier Description

### Bronze: Raw Data

Unmodified market data directly from the Yahoo Finance API.
Contains all original NaNs and gaps.

| File | Created by | Content |
|---|---|---|
| `01_raw_data.parquet` | Data Service | Daily closing prices of all tickers (^GSPC, VUSTX, ^VIX, ^IRX, ^TNX) |

> On every `POST /data/ingest`, a data quality report
> (`assets/data_quality_report.md`) is generated from the Bronze layer. It documents
> coverage, missing values (before cleaning), adjustment plausibility, and the row loss from Bronze to Silver.

### Silver: Cleaned and Transformed Data

Forward fill, dropna, log returns, feature engineering, and
model predictions. Data is analysis-ready but not yet a final result.

| File | Created by | Content |
|---|---|---|
| `02_preprocessed_data.parquet` | Data Service | Portfolio returns, cash returns, VIX, interest rates |
| `03_feature_engineered_data.parquet` | Data Service | Additional features: SMA, volatility, momentum, yield spread |
| `04_test_df_data.parquet` | Model Service | Test dataset with regime predictions of all models |
| `wf_cache.parquet` | Model Service | Walk-forward OOS cache with fingerprint validation (only with `walk_forward.cache_enabled: true`) |

### Gold: Final Results

Aggregated backtesting results and simulations that feed directly
into the evaluation and the thesis.

| File | Created by | Content |
|---|---|---|
| `05_backtesting_results_data.parquet` | Backtest Service | Equity curves and returns of all strategies |
| `05_backtesting_transaction_costs_data.parquet` | Backtest Service | Transaction cost analysis |
| `05_backtesting_sorr_simulation.parquet` | Backtest Service | Sequence-of-returns-risk simulation |
| `06_mcs_data.parquet` | Backtest Service | Monte Carlo simulation paths (plot subsample of `n_plot_paths` capital histories per cell; the full-path inference statistics are computed in-stream and not persisted) |

## Path Management

All file paths are defined centrally in `config/config.yaml` under `paths.files`
and resolved via `cfg.data_path("<key>")`.
Services do not reference hard-coded paths.

## Microservice Access

The Docker-based microservice architecture (see [microservice-architecture.md](microservice-architecture.md)) creates the same files at the same paths. The `data/` directories are shared between host and containers via Docker volume mounts (`./data:/app/data`). The Parquet files are therefore accessible under identical relative paths both locally and inside the containers.

| Service | Tier created | Endpoint |
|---|---|---|
| Data Service (`:8001`) | Bronze + Silver | `POST /data/ingest` |
| Model Service (`:8002`) | Silver | `POST /models/train/{model_name}` |
| Backtest Service (`:8003`) | Gold | `POST /backtest/run` |
