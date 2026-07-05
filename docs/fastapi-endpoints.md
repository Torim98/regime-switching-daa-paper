# FastAPI Endpoints Documentation

The `regime-switching-daa` project uses a microservice architecture: three containerized FastAPI services cover the quantitative pipeline; a fourth (dashboard) provides the interactive frontend with control hub, config editor, and live logs.

---

## Data Service (Port: 8001)
*Responsible for data acquisition, preparation, feature engineering, and exploratory data analysis (EDA).*

### `POST /data/ingest`
- **Description**: Starts the entire data pipeline. Downloads historical market data via `yfinance`, performs portfolio construction (preprocessing), and generates indicators/features (volatility, SMA, momentum). Also runs an EDA (descriptive statistics, ADF tests), creates various plots, and stores the intermediate results (Parquet format) according to the medallion architecture.
- Additionally generates a **data quality report** (`assets/data_quality_report.md`): coverage against expected trading days, missing-value counts on the raw data (before ffill/dropna), adjustment plausibility (daily jumps), and the effect of cleaning (Bronze → Silver).

### `POST /data/label-analysis`
- **Description**: Computes the concordance matrix and switch statistics for all regime labelers (MSM, HMM, Pagan-Sossounov, Peak-to-Trough, Lunde-Timmermann, NBER) on the `test_df`. Writes the plots `label_concordance_matrix.png` and `label_timeline_comparison.png` to `assets/` and returns the numerical results as JSON (`{status, elapsed_s, concordance, switch_stats}`). Serves to justify the label choice (Pagan-Sossounov) for LSTM/Transformer. Requires that `/data/ingest` and a subsequent model training run have already been executed (needs `test_df`).

### `GET /data/features`
- **Description**: Returns the fully prepared dataset including all computed features (the "feature-engineered" DataFrame) as a JSON structure (`orient="split"`). Requires that `/data/ingest` has been executed beforehand.

---

## Model Service (Port: 8002)
*Responsible for training, prediction, and persistence of the four machine learning and econometric models.*

### `POST /models/train/{model_name}`
- **Parameters**: `model_name` (string: `msm` | `hmm` | `lstm` | `transformer`)
- **Description**: Trains a single, specified model. **Only available with `walk_forward.enabled: false`.** In walk-forward mode, this endpoint returns HTTP 400 with a hint to use `/models/train-all`.

### `POST /models/train-all`
- **Description**: Trains all four models. With `walk_forward.enabled: false`, sequentially in single-split mode. With `walk_forward.enabled: true`, via the walk-forward engine (`run_walk_forward`) with rolling folds, fingerprint-based cache, and OOS aggregation. In walk-forward mode, additionally returns `folds` and `oos_days`.

### `POST /models/optimize/{model_name}`
- **Parameters**: `model_name` (string: `MSM` | `HMM` | `HMM_Uni` | `LSTM` | `Transformer`)
- **Description**: Runs an Optuna hyperparameter optimization for the specified model. Uses walk-forward splits as inner cross-validation. Sampler (GridSampler for the econometric models, multivariate TPESampler for the DL models), trial budget, objective metric and the `tune_until` fold restriction are read from `config.yaml` (`optimization.*`). The objective is the configured risk metric on the pooled OOS return series. No API overrides, so the run stays reproducible from config alone. Requires `walk_forward.enabled: true`. Results are persisted in `models/optuna_studies.db`. Returns `metric`, `best_score`, `best_params`, and `n_trials`.

### `POST /models/optimize-all`
- **Parameters**: none
- **Description**: Optimizes all five models sequentially (MSM → HMM → HMM_Uni → LSTM → Transformer). Sampler, trial budget, objective metric and the `tune_until` fold restriction are read from `config.yaml` (`optimization.*`): the econometric models are searched exhaustively via GridSampler, the DL models via a multivariate TPESampler on the pooled-OOS objective. Returns `metric` and a dict with `best_score` and `best_params` per model.

### `POST /models/hpo-analysis`
- **Parameters**: `scope` (query, string: `cheap` | `full`, default `cheap`)
- **Description**: Post-HPO analysis over the persisted Optuna studies (Issue #5). `scope=cheap` writes the convergence + edge-of-range review and the objective-sensitivity table (reads the logged trial metrics only, seconds). `scope=full` additionally computes the Deflated Sharpe Ratio, PBO/CSCV and the multi-seed re-evaluation, which re-train the DL models on the GPU (minutes to hours). Writes the Markdown assets (`hpo_convergence.md`, `objective_sensitivity.md`, `hpo_dsr.md`, `hpo_pbo.md`, `hpo_multiseed.md`) embedded in `statistics.md` (section G) and rendered on the Models dashboard page. See [hyperparameter-optimization.md](./hyperparameter-optimization.md). Returns `scope` and the map of written asset paths.

### `GET /models/status`
- **Description**: Returns a boolean (`true`/`false`) for each model indicating whether it has been trained. In single-split mode (`walk_forward.enabled=false`) the filesystem under `models/` is probed for the persisted model file. In walk-forward mode no per-model files are written, so the walk-forward cache (`data/silver/wf_cache.parquet`) is checked instead: a model is `true` when its `{Model}_Signal` column exists and holds at least one non-null value (an all-NaN column, i.e. every fold failed, stays `false`). The response is a flat `{model_key: bool}` map in both modes.

---

## Backtest Service (Port: 8003)
*Responsible for strategy evaluation, Monte Carlo simulations, and the final reporting.*

### `POST /backtest/run`
- **Description**: Runs the historical backtesting. In walk-forward mode, `test_df` is trimmed to the common OOS window (`dropna(how="any")`). In addition to equity curves and transaction costs, produces annualized metrics, a crisis performance table, a rolling Sharpe plot, and a drawdown plot. Runs SORR simulations for all configured scenarios.

### `POST /backtest/evaluate`
- **Description**: Evaluates all strategies in depth. Runs a bootstrap Monte Carlo simulation (block or stationary, selectable via `evaluation.mcs.bootstrap_method`) to test the robustness of the strategies. Produces detailed performance metrics, boxplots, quantile visualizations, and MCS paths, plus the extended-evaluation diagnostics (churning, threshold sensitivity, time-to-recovery, and the walk-forward bear-market coverage table `bear_coverage.md`, Issue #8). Automatically triggers the report generation at the end.

### `POST /backtest/bootstrap-robustness`
- **Description**: Robustness comparison of the two MCS resampling schemes (Issue #7). Re-runs the Monte Carlo simulation twice on the existing return/signal paths, once with the fixed-length block bootstrap and once with the stationary bootstrap (Politis & Romano 1994), using the same seed and `n_paths` (no model re-training). Writes `assets/bootstrap_robustness.md` with depletion rate (Wilson CI) and median terminal capital per scenario and strategy, plus signed difference columns and a data-driven robustness summary. Requires a prior `/backtest/run`.
- **Parameters**: `n_paths` (int, optional) overrides the configured path count for a quick check.

### `POST /backtest/report`
- **Description**: Collects all generated tables, metrics, and Markdown snippets and combines them into a final statistics report (usually `statistics.md`), made available in the `assets/` or `docs/` directory.

### `GET /backtest/results`
- **Description**: Returns the results of the strategy evaluation (the final performance table) as plain text/Markdown in JSON format, so that the results can be queried via the API. Requires a successful run of `/backtest/evaluate`.

---

## Dashboard Service (Port: 8004)
*Interactive frontend with visualization, control hub, config editor, and live log streaming. Dev-only, bound to `127.0.0.1`. Detailed description: [Dashboard Service](dashboard-service.md).*

### HTML Pages (Jinja2)

| Path | Page |
|------|-------|
| `GET /` | Overview (status tiles, artifact grid, coverage map) |
| `GET /hub` | Control hub (all pipeline endpoints per click) |
| `GET /eda` | EDA charts + PNG gallery |
| `GET /models` | Regime detection, label concordance, Optuna heatmaps |
| `GET /backtest` | Equity curves, drawdown, rolling Sharpe, SORR, crisis performance |
| `GET /evaluation` | Full `statistics.md` coverage (MCS, H1/H2, break-even, ...) |
| `GET /config` | Monaco YAML editor for `config.yaml` |
| `GET /logs` | Live log stream (WebSocket) |

### Data Adapters: Parquet to Plotly JSON

All chart endpoints deliver Plotly-compatible JSON that is rendered client-side with Plotly.js. No recomputation; all values come from the Parquet artifacts of the pipeline.

#### `GET /api/status`
- **Description**: Overview of all pipeline artifacts (existence, size in MB, mtime) plus meta info (end date, walk-forward flag, fast-mode flag). Basis for the status tiles and the artifact grid on the overview page.

#### `GET /api/asset/{name}` / `GET /api/markdown/{name}`
- **Description**: Read-only delivery of a PNG or MD asset from `assets/`. Path traversal protection included. `GET /api/markdown/{name}` returns MD content as a JSON payload for client-side rendering with marked.js.

#### `GET /api/chart/returns`
- **Parameters**: `col` (str, default: `Returns`), `smoothing` (int 0–252, default: 0)
- **Description**: Time series of any column from `feature_engineered_data.parquet` with optional moving-average smoothing.

#### `GET /api/chart/feature-correlation`
- **Description**: Correlation matrix (Pearson) over the columns configured in `features.model_features`.

#### `GET /api/chart/capital-curve`
- **Description**: 60/40 benchmark capital curve (cumulative returns).

#### `GET /api/chart/equity-curves`
- **Description**: Equity curves of all strategies from `backtesting_results.parquet`. Requires `/backtest/run`.

#### `GET /api/chart/drawdown`
- **Description**: Drawdown paths of all strategies.

#### `GET /api/chart/rolling-sharpe`
- **Parameters**: `window` (int 21–1260, default: 252)
- **Description**: Rolling Sharpe ratio with configurable window.

#### `GET /api/chart/regime-overlay`
- **Parameters**: `model` (str: `MSM` | `HMM` | `HMM_Uni` | `LSTM` | `Transformer`, default: `MSM`)
- **Description**: 60/40 price overlaid with bear probability and bear-signal bands (red shapes). Shows when and where the selected model detected bear market phases.

#### `GET /api/chart/mcs-quantiles`
- **Parameters**: `scenario` (str, default: `Standard`), `strategy` (str, default: `Transformer`)
- **Description**: Quantile fan (5 / 25 / 50 / 75 / 95%) of the Monte Carlo simulation paths. Requires `/backtest/evaluate`.

### Control Hub Proxy

Calls the pipeline services via `httpx`. Read timeout: 8 h (for walk-forward train-all). Service URLs configurable via environment variables (`DATA_SERVICE_URL`, `MODEL_SERVICE_URL`, `BACKTEST_SERVICE_URL`).

#### `GET /api/hub/catalog`
- **Description**: Returns the complete endpoint catalog (service, path, method, label, description, parameter schema, danger flag) for the dynamic UI rendering of the control hub page.

#### `GET /api/hub/health`
- **Description**: Ping check on all three pipeline services (`/openapi.json` as marker). Returns `{up, status, url}` per service for the health tiles.

#### `POST /api/hub/call`
- **Parameters**: `service` (`data` | `model` | `backtest`), `path` (e.g. `/data/ingest`), `method` (`GET` | `POST`), `query` (optional: JSON string with query params)
- **Description**: Generic proxy call. Used by the UI to trigger any endpoint of the pipeline services. Response: `{status_code, ok, body}`; for non-JSON responses, the body is wrapped as `{"text": ...}`.

### Config Editor

Safety net when writing: (1) YAML parse → (2) required-sections check → (3) `.bak` backup → (4) atomic swap via tempfile → (5) `PipelineConfig()` reload verification → (6) rollback from backup on reload error.

#### `GET /api/config`
- **Description**: Returns the current `config.yaml` as plain text plus meta (path, mtime, size).

#### `POST /api/config`
- **Body**: `{"content": "<entire YAML text>"}`
- **Description**: Saves the submitted YAML text after passing validation. Required sections: `data`, `features`, `portfolio`, `models`, `backtesting`, `walk_forward`, `evaluation`, `paths`, `plotting`. Response: `{status, backup, bytes_written, reloaded}`.

#### `GET /api/config/backups`
- **Description**: List of all `.bak` files in the `config/` directory, sorted by mtime (newest first).

#### `POST /api/config/restore`
- **Body**: `{"name": "config.YYYYMMDD-HHMMSS.bak"}`
- **Description**: Restores a specific backup file as the active `config.yaml`. The previous state is additionally saved as `*.pre-restore.bak`.

### Live Log Streaming

WebSocket-based file tail on `logs/*.log`. Rotation and truncation are detected and signaled with a system line (`[dashboard] file truncated, resume from 0`).

#### `GET /api/logs/files`
- **Description**: List of all available `logs/*.log` files including size (KB) and mtime.

#### `GET /api/logs/snapshot/{filename}`
- **Parameters**: `lines` (int 1–10000, default: 500)
- **Description**: Returns the last N lines of the specified log file without WebSocket (for initial load or snapshots). Path traversal protection active.

#### `WS /ws/logs/{filename}`
- **Parameters**: `tail` (int, default: 200)
- **Description**: First sends the last `tail` lines, then all new lines as text frames via ~300 ms polling. On file truncation, reading restarts from position 0. On `WebSocketDisconnect`, the connection is closed cleanly.
