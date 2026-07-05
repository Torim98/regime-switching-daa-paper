# Microservice Architecture

Three services cover the pipeline; a fourth provides the interactive dashboard. All services use the business logic under `src/` and the configuration under `config/config.yaml`.

## Services

| Service | Port | Description |
|---------|------|-------------|
| **Data Service** | 8001 | Data acquisition (yfinance), preprocessing, feature engineering, EDA |
| **Model Service** | 8002 | Training & prediction of all 4 regime-switching models |
| **Backtest Service** | 8003 | Backtesting, SORR simulation, Monte Carlo, reporting |
| **Dashboard Service** | 8004 | Interactive frontend: visualization of all artifacts, control hub for pipeline endpoints (httpx proxy), YAML config editor, WebSocket log streaming. Bound locally only (`127.0.0.1:8004`). Details: [Dashboard Service](dashboard-service.md). |

## Endpoints

### Data Service (`:8001`)

| Method | Path | Description |
|---------|------|-------------|
| POST | `/data/ingest` | Download, preprocessing, feature engineering, EDA plots + stats |
| POST | `/data/label-analysis` | Label analysis of alternative label schemes (for supervised models) |
| GET | `/data/features` | Feature DataFrame as JSON |

### Model Service (`:8002`)

| Method | Path | Description |
|---------|------|-------------|
| POST | `/models/train/{model_name}` | Train a single model (only with `walk_forward.enabled: false`) |
| POST | `/models/train-all` | Train all 4 models; single split or walk-forward depending on config |
| POST | `/models/optimize/{model_name}` | Optuna optimization for one model (grid/TPE, walk-forward as inner CV) |
| POST | `/models/optimize-all` | Optimize all 5 models sequentially |
| POST | `/models/hpo-analysis` | Post-HPO analysis reports (convergence, sensitivity, DSR, PBO, multi-seed) |
| POST | `/models/seed-sensitivity` | Retraining-stability report: per-model seed CV of the OOS metrics (motivates the DL ensemble and HMM multi-start) |
| GET | `/models/status` | Persistence status of all models |

### Backtest Service (`:8003`)

| Method | Path | Description |
|---------|------|-------------|
| POST | `/backtest/run` | Backtesting + equity curves + drawdown + rolling Sharpe + SORR + crisis performance |
| POST | `/backtest/evaluate` | Evaluation table + Monte Carlo simulation + `statistics.md` |
| POST | `/backtest/bootstrap-robustness` | Block vs. stationary bootstrap comparison -> `bootstrap_robustness.md` (Issue #7) |
| POST | `/backtest/report` | Regenerate `statistics.md` |
| GET | `/backtest/results` | Evaluation table as Markdown |

### Dashboard Service (`:8004`)

| Method | Path | Description |
|---------|------|-------------|
| GET | `/`, `/eda`, `/models`, `/backtest`, `/evaluation`, `/hub`, `/config`, `/logs` | HTML pages (Jinja2) |
| GET | `/api/status` | Pipeline artifact overview |
| GET | `/api/chart/{...}` | Plotly JSON for EDA, backtest, regime overlay, MCS quantiles |
| GET/POST | `/api/hub/{catalog,health,call}` | Control hub proxy to `:8001/:8002/:8003` |
| GET/POST | `/api/config`, `/api/config/backups`, `/api/config/restore` | YAML editor with backup + rollback |
| GET | `/api/logs/files`, `/api/logs/snapshot/{file}` | Log listing + initial tail |
| WS | `/ws/logs/{file}?tail=` | Live log streaming (file tail) |

Full list in [dashboard-service.md](dashboard-service.md) and [fastapi-endpoints.md](fastapi-endpoints.md).

## Dependencies

The pipeline services must be called in the following order, since they communicate via the shared filesystem:

```
Data Service → Model Service → Backtest Service
```

The dashboard consumes the artifacts read-only and proxies writing calls (training/backtest) to the pipeline services. It is not a mandatory pipeline step but a control and visualization layer on top.

`docker-compose.yml` maps the dependencies via `depends_on`:
- `model-service` depends_on `data-service`
- `backtest-service` depends_on `model-service`
- `dashboard-service` depends_on `data-service`, `model-service`, `backtest-service`

Within the Model Service, a fixed training order applies:
1. **MSM** (Markov switching): independent
2. **HMM** (hidden Markov): independent, unsupervised baseline
2b. HMM_Uni: univariate ablation variant
3. **LSTM**: supervised on Pagan-Sossounov labels (from `feature_engineered_data`), creates `test_df`
4. **Transformer**: supervised on Pagan-Sossounov labels

With `walk_forward.enabled: true`, `/models/train-all` is called directly and controls all models via `run_walk_forward()`. The individual routes `/models/train/{model_name}` are blocked in this mode (HTTP 400). A fingerprint-based Parquet cache speeds up repeated runs with unchanged configuration.

## Shared Volumes

All services communicate via mounted host directories:

| Volume | Data | Model | Backtest | Dashboard | Content |
|--------|:---:|:---:|:---:|:---:|--------|
| `./data` | R/W | R/W | R | R | Parquet files (medallion: Bronze/Silver/Gold) |
| `./models` | - | R/W | - | R | Persisted model files (.pkl, .keras, .pt) + Optuna SQLite DB |
| `./assets` | R/W | R/W | R/W | R | Plots (PNG) and tables (Markdown) |
| `./config` | R | R | R | **R/W** | `config.yaml` (dashboard writes with `.bak` backup + rollback) |
| `./logs` | R/W | R/W | R/W | R | Service log files (dashboard tails via file tail) |
| `./docs` | - | - | R/W | R | `statistics.md` |

## Logging

Each service writes to its own log file under `logs/`:

- `logs/data_service.log`
- `logs/model_service.log`
- `logs/backtest_service.log`
- `logs/dashboard_service.log`

Format: `YYYY-MM-DD HH:MM:SS,ms | service_name | LEVEL | message`

Time zone: `Europe/Berlin` (configured via the `TZ` environment variable in `docker-compose.yml`)

The Dashboard Service additionally provides a WebSocket at `/ws/logs/{filename}` that streams each of these log files live (file tail), equivalent to `docker compose logs -f`.

## Project Structure

```
regime-switching-daa/
├── src/                              # Shared business logic
│   ├── data/
│   │   ├── ingestion.py              # yfinance download
│   │   ├── preprocessing.py          # Portfolio construction, returns
│   │   ├── feature_engineering.py    # Rolling features (vol, SMA, momentum)
│   │   ├── eda.py                    # Descriptive statistics, ADF tests
│   │   └── plots.py                  # EDA and feature plots
│   ├── models/
│   │   ├── common.py                 # Constants, validate_regime_signal(), create_sequences()
│   │   ├── msm.py                    # Markov-switching model
│   │   ├── hmm.py                    # Hidden Markov model
│   │   ├── lstm.py                   # LSTM network
│   │   ├── transformer.py            # Transformer (PositionalEncoding + classifier)
│   │   └── plots.py                  # Regime plots (MSM, HMM, DL, comparison)
│   └── backtest/
│       ├── engine.py                 # Backtesting logic
│       ├── optimize.py               # Optuna hyperparameter optimization (grid/TPE)
│       ├── hpo_analysis.py           # Post-HPO analysis (convergence, DSR, PBO, multi-seed)
│       ├── seed_sensitivity.py       # Retraining-stability quantification (per-model seed CV)
│       ├── sorr.py                   # SORR simulation
│       ├── evaluation.py             # Strategy evaluation, Monte Carlo
│       ├── reporting.py              # statistics.md generation
│       └── plots.py                  # Equity curves, SORR, MCS plots
│
├── services/                         # FastAPI services
│   ├── __init__.py
│   ├── logging_config.py             # Central logging (file + console)
│   ├── data_service/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── routes.py
│   ├── model_service/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── routes.py
│   ├── backtest_service/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── routes.py
│   └── dashboard_service/            # Interactive frontend (port 8004, dev-only)
│       ├── Dockerfile
│       ├── main.py
│       ├── routes.py                 # HTML pages (Jinja2)
│       ├── data_adapters.py          # Parquet → Plotly JSON (/api/chart/*)
│       ├── hub_api.py                # httpx proxy to data/model/backtest (/api/hub/*)
│       ├── config_api.py             # YAML editor + backup/restore (/api/config/*)
│       ├── websockets.py             # File-tail log streaming (/ws/logs/*)
│       ├── templates/                # 9 Jinja templates (base + 8 pages)
│       └── static/                   # dashboard.css + common.js
│
├── docker-compose.yml
├── pyproject.toml
├── config/                           # config.yaml + config_loader.py
├── jupyter/                          # Exploratory notebooks
├── data/                             # Medallion architecture (bronze/silver/gold)
├── models/                           # Persisted model files
├── assets/                           # Generated plots and tables
├── docs/                             # Project documentation
└── logs/                             # Service log files
```

## Comparison: Microservices vs. Dashboard

| Aspect | Microservice pipeline | Dashboard UI |
|--------|----------------------|--------------|
| Execution | Docker + curl/HTTP | Browser click (control hub) |
| Interactivity | Swagger UI, JSON responses | Plotly charts, forms, live logs |
| Reproducibility | `docker-compose up --build` | `http://localhost:8004/` |
| Business logic | `src/` (identical) | consumes artifacts, proxies services |
| Configuration | `config/config.yaml` (identical) | in-UI editor with validation + rollback |
| Plot generation | `src/*/plots.py` + `matplotlib.use("Agg")` | Plotly.js (client-side, interactive) |
| Data persistence | Parquet (medallion) | read-only consumption |
| Timing report | Logs per service | Live log stream via WebSocket |

See also: [Sequence Diagram: Microservice Pipeline](microservice-sequence-diagram.md) for the detailed flow of a pipeline run and [Dashboard Service](dashboard-service.md) for the architecture and page structure of the frontend.
