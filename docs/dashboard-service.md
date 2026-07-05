# Dashboard Service

The **Dashboard Service** (port 8004) is the interactive control and visualization frontend of the project. It covers three tasks at once:

1. **Visualization** of all pipeline artifacts (EDA, regime detection, backtest, evaluation, MCS): fully interactive with Plotly.js.
2. **Control hub**: all FastAPI endpoints of the three pipeline services (`data`, `model`, `backtest`) can be called directly from the UI.
3. **Operational self-service**: edit the configuration (`config.yaml`) and stream live container logs without leaving the browser.

The service is designed **dev-only** and bound exclusively to `127.0.0.1`. Beyond editing `config/config.yaml` (with backup/rollback), the **Full Pipeline Run** orchestrator (see below) can optionally delete selected pipeline artifacts before a run and execute the `jupyter/` figure notebooks in place, so the dashboard also has write access to `data/`, `assets/`, `docs/`, `models/` and `jupyter/`.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  Browser  ──►  Dashboard Service (:8004)                       │
│                                                                │
│                ├─ HTML / Jinja2  (8 pages)                     │
│                ├─ /api/*         (Parquet → Plotly JSON)       │
│                ├─ /api/hub/*     (httpx proxy)  ──► :8001/2/3  │
│                ├─ /api/config/*  (YAML + backup + rollback)    │
│                └─ /ws/logs/*     (WebSocket file tail)         │
└───────────────────────────────────────────────────────────────┘
```

Zero-build frontend stack (everything via CDN, no npm toolchain required):

- **Tailwind CSS** (Play CDN): styling
- **Plotly.js 2.35**: interactive charts
- **Alpine.js 3.x**: UI state
- **HTMX 1.9**: HTML partials
- **Monaco Editor**: YAML editor with syntax highlighting
- **marked.js**: Markdown rendering for the asset gallery

---

## Page Structure

| Route | Page | Content |
|-------|-------|--------|
| `/` | **Overview** | Status tiles (end date, WF, fast mode), pipeline artifact grid, coverage map for `statistics.md` |
| `/hub` | **Control Hub** | **Full Pipeline Run** panel (sequential one-time run of the whole pipeline with per-step progress, optional steps and pre-run cleanup), plus dynamically rendered service cards from `/api/hub/catalog`, health tiles, execute forms with spinner, JSON response viewer |
| `/eda` | **EDA** | Returns chart (column & smoothing picker), feature correlation matrix, 60/40 capital curve, PNG gallery from `assets/` |
| `/models` | **Models** | Regime overlay chart (MSM/HMM/HMM_Uni/LSTM/Transformer), label concordance, model plots, walk-forward schema, OOS bear-market coverage (Issue #8), Optuna heatmaps, HPO analysis reports (convergence, objective sensitivity, DSR/PBO, multi-seed), retraining-stability (seed sensitivity) report |
| `/backtest` | **Backtest** | Equity curves, drawdown, rolling Sharpe (window slider), annualized metrics, crisis performance, SORR scenarios, pipeline timing |
| `/evaluation` | **Evaluation** | Full `statistics.md` coverage: evaluation table, confusion/ROC/PR, churning, switch timing, regime heatmap, threshold sensitivity, TTR, MCS, bootstrap robustness (block vs. stationary, Issue #7), depletion CI, H1/H2 tests, break-even, withdrawal sensitivity |
| `/config` | **Config editor** | Monaco-based YAML editor (Ctrl+S, dirty state, backup drawer, restore functionality) |
| `/logs` | **Live logs** | File tail via WebSocket, file dropdown, regex filter, autoscroll, level coloring (ERROR/WARN/INFO/DEBUG) |

All pages share `base.html` (sidebar navigation, dark mode toggle, build info).

---

## API Endpoints

### Data Adapters (`/api/*`)

Deliver pipeline artifacts as Plotly-compatible JSON. No recomputation. Everything is based on the Parquet files written by the pipeline.

| Method | Path | Description |
|---------|------|-------------|
| GET | `/api/status` | Overview of all pipeline artifacts (existence, size, mtime) |
| GET | `/api/asset/{name}` | Serves PNG/MD from `assets/` (read-only) |
| GET | `/api/markdown/{name}` | MD file as JSON (for client-side rendering with marked.js) |
| GET | `/api/chart/returns?col=&smoothing=` | Return time series (any column, optional MA smoothing) |
| GET | `/api/chart/feature-correlation` | Correlation matrix of the model features |
| GET | `/api/chart/capital-curve` | 60/40 benchmark capital curve |
| GET | `/api/chart/equity-curves` | Equity curves of all strategies (OOS) |
| GET | `/api/chart/drawdown` | Drawdown paths of all strategies |
| GET | `/api/chart/rolling-sharpe?window=` | Rolling Sharpe with configurable window |
| GET | `/api/chart/regime-overlay?model=` | Price + bear probability + signal overlay per model |
| GET | `/api/chart/mcs-quantiles?scenario=&strategy=` | Quantile fan (5/25/50/75/95%) of the MCS paths |

### Control Hub Proxy (`/api/hub/*`)

Calls the FastAPI endpoints of the three pipeline services via `httpx`. Long read timeouts (8 h) so that `train-all` completes in walk-forward mode without client abort.

| Method | Path | Description |
|---------|------|-------------|
| GET | `/api/hub/catalog` | Returns the endpoint catalog for dynamic UI rendering |
| GET | `/api/hub/health` | Ping to all three services (OpenAPI JSON as marker) |
| POST | `/api/hub/call?service=&path=&method=&query=` | Generic proxy call |
| GET | `/api/hub/pipeline/plan` | Canonical step catalog + clean targets for the Full Pipeline Run panel |
| GET | `/api/hub/pipeline/status` | Current orchestrator job snapshot (polled by the UI) |
| POST | `/api/hub/pipeline/run` | Starts the full pipeline in a background thread (409 while one is running) |
| POST | `/api/hub/pipeline/stop` | Graceful stop: the current step finishes, no further step starts (forwards HPO stop) |

Service URLs configurable via environment variables:
- `DATA_SERVICE_URL` (default: `http://data-service:8001`)
- `MODEL_SERVICE_URL` (default: `http://model-service:8002`)
- `BACKTEST_SERVICE_URL` (default: `http://backtest-service:8003`)

#### Full Pipeline Run (orchestrator)

`POST /api/hub/pipeline/run` walks the canonical sequence in one background job so
that a single run produces every paper asset:

```
[clean] → ingest → [optimize] → [hpo-analysis] → train-all → [seed-sensitivity]
        → [label-analysis] → backtest/run → [bootstrap-robustness] → evaluate
        → [notebooks] → report
```

Steps in brackets are optional (toggled per run); the rest are core. The request
body selects steps, per-step params (e.g. `hpo-analysis` scope, `seed-sensitivity`
seeds/models, `bootstrap` n_paths) and clean targets:

```jsonc
{
  "steps":  { "optimize": false, "seed_sensitivity": true, "notebooks": true },
  "params": { "seed_sensitivity": { "seeds": 5, "models": "all" } },
  "clean":  { "wf_cache": true, "derived_data": true }
}
```

Clean targets deletable before the run: `wf_cache` (essential, otherwise a cache
hit skips retraining), `derived_data` (test_df, backtesting_*, mcs_data), `assets`
(all generated assets + `docs/statistics.md`), `optuna_db` (HPO history).

**Failure policy:** a **core** step failure aborts the run and skips the rest; an
**optional** step failure is recorded (`state: done_with_errors`) but the run
continues, so a single flaky extra never blocks the core paper assets. Progress is
polled via `/api/hub/pipeline/status` (per-step `pending/running/ok/failed/skipped`
with timings); `/api/hub/pipeline/stop` halts at the next step boundary. The
`notebooks` step needs `nbclient`/`nbformat`/`ipykernel` in the image and the
`./jupyter` mount; without them the step reports a clear error instead of crashing.

### Config Editor (`/api/config/*`)

Writes `config/config.yaml` with a safety net: YAML parse check → required-sections check → backup → atomic swap → reload verification. On errors during the reload step, automatic rollback from the backup.

| Method | Path | Description |
|---------|------|-------------|
| GET | `/api/config` | Current `config.yaml` as text + meta (mtime, size) |
| POST | `/api/config` | Save with body `{"content": "<YAML text>"}` |
| GET | `/api/config/backups` | List of all `.bak` files (newest first) |
| POST | `/api/config/restore` | Restore a backup: `{"name": "config.YYYYMMDD-HHMMSS.bak"}` |

Required sections for a successful save:
`data`, `features`, `portfolio`, `models`, `backtesting`, `walk_forward`, `evaluation`, `paths`, `plotting`.

### Live Log Streaming (`/ws/logs/*`)

WebSocket-based file tail. Portable alternative to the Docker socket. Works anywhere the `logs/` volume is mounted. Rotation and truncation are detected.

| Method | Path | Description |
|---------|------|-------------|
| GET | `/api/logs/files` | List of all `logs/*.log` incl. size and mtime |
| GET | `/api/logs/snapshot/{filename}?lines=` | Last N lines without WebSocket (initial load) |
| WS | `/ws/logs/{filename}?tail=` | Streams `tail` lines initially, then live updates (~300 ms polling) |

Path traversal protection: only file names within `logs/` are allowed.

---

## Security

- **Binding:** `127.0.0.1:8004:8004` in `docker-compose.yml`. The service is **not** exposed to the network, only reachable locally.
- **Write scope:** `config/config.yaml` is written with backup + rollback. The chart/asset/log adapters treat `data/`, `assets/`, `docs/`, `logs/` as read-only, but the **Full Pipeline Run** orchestrator can write to them: it deletes selected artifacts under `data/`, `assets/`, `docs/` and `models/` when a clean target is chosen, and the `notebooks` step writes figures into `assets/` and executes files under `jupyter/`. These are deliberate, explicitly triggered actions confirmed in the UI, safe under the local-only binding.
- **Path traversal:** Both the asset endpoint and the WS log endpoint validate file names against `..` and `/`.
- **Proxy semantics:** The control hub proxy restricts `service` via regex to `(data|model|backtest)` and `method` to `(GET|POST)`; free URL input is not possible.

For production use, authentication (Basic / OIDC), CSRF tokens for the writing endpoints, and rate limiting would additionally be required. In the thesis context, the local binding is sufficient.

---

## Dependencies

The service carries no heavy ML training frameworks (no TensorFlow/PyTorch). Only `[services]` + `[dashboard]` are installed (see `pyproject.toml`):

```toml
dashboard = [
    "jinja2==3.1.4",
    "python-multipart==0.0.20",
    "optuna==4.8.0",       # read-only access to the Optuna studies DB
    "nbclient==0.10.2",    # execute the jupyter/ paper-figure notebooks
    "nbformat==5.10.4",    #   (Full Pipeline Run "notebooks" step)
    "ipykernel==6.29.5",   #   kernel for nbclient
]
```

`watchfiles`/`websockets` come transitively from `uvicorn[standard]` in the
`[services]` extra. The notebooks themselves only need the core deps
(numpy/pandas/matplotlib/scipy/yfinance), which are already present, so no ML
training stack is pulled in. Image size stays modest (roughly 250 MB, vs. ~5 GB
for the Model Service with TensorFlow + PyTorch).

---

## Volumes

All non-config volumes are read-only for the visualization adapters; the Full
Pipeline Run orchestrator additionally deletes selected artifacts (clean targets)
and the `notebooks` step writes figures, hence the R/W where noted.

| Volume | Mode | Purpose |
|--------|:----:|-------|
| `./data` | R/W | Parquet artifacts (read for charts; clean deletes wf_cache/derived data) |
| `./assets` | R/W | PNG/MD assets (read for the gallery; clean/notebooks write here) |
| `./docs` | R/W | `statistics.md` (read for the evaluation panel; clean can delete it) |
| `./config` | R/W | `config.yaml` + `.bak` files |
| `./logs` | R | Service and pipeline logs (file tail) |
| `./models` | R/W | Optuna studies DB (read for best-params; clean can delete it) |
| `./jupyter` | R/W | Paper-figure notebooks, executed in place by the `notebooks` step |

---

## Development

### Local Start (without Docker)

```bash
pip install -e ".[services,dashboard]"
uvicorn services.dashboard_service.main:app --reload --host 127.0.0.1 --port 8004
```

The three pipeline services must be reachable separately for the control hub (or the service URLs must point to `localhost:8001/2/3` via `DATA_SERVICE_URL` etc.).

### With Docker Compose

```bash
docker compose up -d --build dashboard-service
# UI: http://localhost:8004/
# Swagger: http://localhost:8004/docs
```

### Adding New Chart Endpoints

1. Create a new `@router.get("/chart/...")` handler in [`data_adapters.py`](../services/dashboard_service/data_adapters.py), read the Parquet file, build the Plotly figure, and return `_fig_to_json(fig)`.
2. In the template (e.g. [`evaluation.html`](../services/dashboard_service/templates/evaluation.html)), add a `<div id="my-chart"></div>` and load it via `renderChart('my-chart', '/api/chart/my-endpoint')` from `common.js`.
3. Dark mode works automatically. The MutationObserver in `common.js` calls `Plotly.relayout()` on all charts.

### Adding New Control Hub Endpoints

Extend the catalog in [`hub_api.py`](../services/dashboard_service/hub_api.py). The frontend renders the forms dynamically from the `_CATALOG` object. New path/query parameters are automatically displayed as input fields.
