# Dashboard Service

Der **Dashboard Service** (Port 8004) ist das interaktive Control- und Visualisierungs-Frontend des Projekts. Er deckt drei Aufgaben gleichzeitig ab:

1. **Visualisierung** aller Pipeline-Artefakte (EDA, Regime-Erkennung, Backtest, Evaluation, MCS): komplett interaktiv mit Plotly.js.
2. **Control Hub**: alle FastAPI-Endpoints der drei Pipeline-Services (`data`, `model`, `backtest`) werden direkt aus der UI heraus aufgerufen.
3. **Operativer Self-Service**: Konfiguration (`config.yaml`) editieren und Live-Logs der Container streamen, ohne den Terminal-Flow zu verlassen.

Der Service ist **dev-only** konzipiert und ausschließlich an `127.0.0.1` gebunden. Er bleibt bei jedem Pipeline-Lauf unangetastet. Keinerlei Write-Zugriffe außer auf `config/config.yaml` (mit Backup/Rollback).

---

## Architektur

```
┌───────────────────────────────────────────────────────────────┐
│  Browser  ──►  Dashboard Service (:8004)                       │
│                                                                │
│                ├─ HTML / Jinja2  (8 Seiten)                    │
│                ├─ /api/*         (Parquet → Plotly-JSON)       │
│                ├─ /api/hub/*     (httpx-Proxy)  ──► :8001/2/3  │
│                ├─ /api/config/*  (YAML + Backup + Rollback)    │
│                └─ /ws/logs/*     (WebSocket File-Tail)         │
└───────────────────────────────────────────────────────────────┘
```

Zero-Build-Frontend-Stack (alles per CDN, kein npm-Toolchain nötig):

- **Tailwind CSS** (Play-CDN) — Styling
- **Plotly.js 2.35** — Interaktive Charts
- **Alpine.js 3.x** — UI-State
- **HTMX 1.9** — HTML-Partials
- **Monaco Editor** — YAML-Editor mit Syntax-Highlighting
- **marked.js** — Markdown-Rendering für Asset-Gallery

---

## Seitenstruktur

| Route | Seite | Inhalt |
|-------|-------|--------|
| `/` | **Overview** | Status-Kacheln (End-Date, WF, Fast-Mode), Pipeline-Artefakt-Grid, Abdeckungs-Map zu `statistics.md` |
| `/hub` | **Control Hub** | Dynamisch gerenderte Service-Cards aus `/api/hub/catalog`, Health-Tiles, Execute-Forms mit Spinner, JSON-Response-Viewer |
| `/eda` | **EDA** | Returns-Chart (Spalten- & Smoothing-Picker), Feature-Korrelationsmatrix, 60/40-Kapitalkurve, PNG-Gallery aus `assets/` |
| `/models` | **Modelle** | Regime-Overlay-Chart (MSM/HMM/HMM_Uni/LSTM/Transformer), Label-Konkordanz, Modell-Plots, Optuna-Heatmaps |
| `/backtest` | **Backtest** | Equity Curves, Drawdown, Rolling-Sharpe (Fenster-Slider), annualisierte Metriken, Krisen-Perf, SORR-Szenarien, Pipeline-Timing |
| `/evaluation` | **Evaluation** | Volle `statistics.md`-Abdeckung: Eval-Tabelle, Confusion/ROC/PR, Churning, Switch-Timing, Regime-Heatmap, Threshold-Sensitivity, TTR, MCS, Depletion-CI, H1/H2-Tests, Break-Even, Withdrawal-Sensitivity |
| `/config` | **Config-Editor** | Monaco-basierter YAML-Editor (Ctrl+S, Dirty-State, Backup-Drawer, Restore-Funktionalität) |
| `/logs` | **Live-Logs** | File-Tail via WebSocket, Datei-Dropdown, Regex-Filter, Autoscroll, Level-Coloring (ERROR/WARN/INFO/DEBUG) |

Alle Seiten teilen `base.html` (Sidebar-Navigation, Dark-Mode-Toggle, Build-Info).

---

## API-Endpoints

### Data-Adapter (`/api/*`)

Liefert Pipeline-Artefakte als Plotly-kompatibles JSON. Keine Neuberechnungen. Alles basiert auf den Parquet-Dateien, die von der Pipeline geschrieben wurden.

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| GET | `/api/status` | Übersicht aller Pipeline-Artefakte (Existenz, Größe, mtime) |
| GET | `/api/asset/{name}` | Liefert PNG/MD aus `assets/` aus (read-only) |
| GET | `/api/markdown/{name}` | MD-Datei als JSON (für clientseitiges Rendering mit marked.js) |
| GET | `/api/chart/returns?col=&smoothing=` | Renditen-Zeitreihe (beliebige Spalte, optionales MA-Smoothing) |
| GET | `/api/chart/feature-correlation` | Korrelationsmatrix der Modell-Features |
| GET | `/api/chart/capital-curve` | 60/40-Benchmark-Kapitalkurve |
| GET | `/api/chart/equity-curves` | Equity Curves aller Strategien (OOS) |
| GET | `/api/chart/drawdown` | Drawdown-Verlauf aller Strategien |
| GET | `/api/chart/rolling-sharpe?window=` | Rolling Sharpe mit konfigurierbarem Fenster |
| GET | `/api/chart/regime-overlay?model=` | Preis + Bear-Probability + Signal-Overlay pro Modell |
| GET | `/api/chart/mcs-quantiles?scenario=&strategy=` | Quantil-Fächer (5/25/50/75/95%) der MCS-Pfade |

### Control-Hub-Proxy (`/api/hub/*`)

Ruft per `httpx` die FastAPI-Endpoints der drei Pipeline-Services auf. Lange Read-Timeouts (8 h), damit `train-all` im Walk-Forward-Modus ohne Client-Abbruch durchläuft.

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| GET | `/api/hub/catalog` | Liefert den Endpoint-Katalog für dynamisches UI-Rendering |
| GET | `/api/hub/health` | Ping auf alle drei Services (OpenAPI-JSON als Marker) |
| POST | `/api/hub/call?service=&path=&method=&query=` | Generischer Proxy-Call |

Service-URLs via Environment-Variablen konfigurierbar:
- `DATA_SERVICE_URL` (Default: `http://data-service:8001`)
- `MODEL_SERVICE_URL` (Default: `http://model-service:8002`)
- `BACKTEST_SERVICE_URL` (Default: `http://backtest-service:8003`)

### Config-Editor (`/api/config/*`)

Schreibt `config/config.yaml` mit Sicherheitsnetz: YAML-Parse-Check → Pflicht-Sections-Check → Backup → Atomic Swap → Reload-Verifikation. Bei Fehlern im Reload-Schritt automatischer Rollback aus Backup.

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| GET | `/api/config` | Aktuelle `config.yaml` als Text + Meta (mtime, size) |
| POST | `/api/config` | Speichern mit Body `{"content": "<YAML-Text>"}` |
| GET | `/api/config/backups` | Liste aller `.bak`-Dateien (neueste zuerst) |
| POST | `/api/config/restore` | Backup zurückspielen: `{"name": "config.YYYYMMDD-HHMMSS.bak"}` |

Pflicht-Sections für erfolgreichen Save:
`data`, `features`, `portfolio`, `models`, `backtesting`, `walk_forward`, `evaluation`, `paths`, `plotting`.

### Live-Log-Streaming (`/ws/logs/*`)

WebSocket-basierter File-Tail. Portable Alternative zum Docker-Socket. Funktioniert überall, wo das `logs/`-Volume gemountet ist. Rotation und Truncation werden erkannt.

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| GET | `/api/logs/files` | Liste aller `logs/*.log` inkl. Größe und mtime |
| GET | `/api/logs/snapshot/{filename}?lines=` | Letzte N Zeilen ohne WebSocket (Initial-Load) |
| WS | `/ws/logs/{filename}?tail=` | Streamt initial `tail` Zeilen, danach Live-Updates (~300 ms-Polling) |

Pfad-Traversal-Schutz: Nur Dateinamen innerhalb `logs/` werden zugelassen.

---

## Sicherheit

- **Binding:** `127.0.0.1:8004:8004` in `docker-compose.yml`. Service ist **nicht** im Netz exponiert, nur lokal erreichbar.
- **Write-Scope:** Einziger schreibender Pfad ist `config/config.yaml` (mit Backup + Rollback). Alle anderen Volumes (`data/`, `assets/`, `logs/`, `docs/`) sind read-only für das Dashboard.
- **Path Traversal:** Sowohl der Asset-Endpoint als auch der WS-Log-Endpoint validieren Dateinamen gegen `..` und `/`.
- **Proxy-Semantik:** Der Control-Hub-Proxy schränkt `service` per Regex auf `(data|model|backtest)` und `method` auf `(GET|POST)` ein; freie URL-Eingabe ist nicht möglich.

Für einen Produktions-Einsatz wären zusätzlich Auth (Basic / OIDC), CSRF-Token für die schreibenden Endpoints sowie Rate-Limiting vorzusehen. Im Thesis-Kontext genügt die lokale Bindung.

---

## Dependencies

Der Service ist bewusst schlank — keine ML-Libraries im Container. Installiert werden nur `[services]` + `[dashboard]` (siehe `pyproject.toml`):

```toml
dashboard = [
    "jinja2==3.1.4",
    "python-multipart==0.0.20",
    "watchfiles==1.1.0",
    "websockets==14.2",
]
```

Image-Größe: ca. 220 MB (vs. ~5 GB für den Model-Service mit TensorFlow + PyTorch).

---

## Volumes

| Volume | Mode | Zweck |
|--------|:----:|-------|
| `./data` | R | Parquet-Artefakte aus Medallion |
| `./assets` | R | PNG- und MD-Assets für die Gallery |
| `./docs` | R | `statistics.md` für das Evaluation-Panel |
| `./config` | R/W | `config.yaml` + `.bak`-Dateien |
| `./logs` | R | Service- und Pipeline-Logs (File-Tail) |

---

## Entwicklung

### Lokaler Start (ohne Docker)

```bash
pip install -e ".[services,dashboard]"
uvicorn services.dashboard_service.main:app --reload --host 127.0.0.1 --port 8004
```

Die drei Pipeline-Services müssen für den Control-Hub separat erreichbar sein (oder die Service-URLs über `DATA_SERVICE_URL` etc. auf `localhost:8001/2/3` zeigen).

### Mit Docker Compose

```bash
docker compose up -d --build dashboard-service
# UI: http://localhost:8004/
# Swagger: http://localhost:8004/docs
```

### Neue Chart-Endpoints hinzufügen

1. In [`data_adapters.py`](../services/dashboard_service/data_adapters.py) neuen `@router.get("/chart/...")`-Handler anlegen, Parquet lesen, Plotly-Figure bauen, `_fig_to_json(fig)` zurückgeben.
2. Im Template (z.B. [`evaluation.html`](../services/dashboard_service/templates/evaluation.html)) einen `<div id="my-chart"></div>` ergänzen und via `renderChart('my-chart', '/api/chart/my-endpoint')` aus `common.js` laden.
3. Dark-Mode funktioniert automatisch. Der MutationObserver in `common.js` ruft `Plotly.relayout()` auf allen Charts.

### Neue Control-Hub-Endpoints

Katalog in [`hub_api.py`](../services/dashboard_service/hub_api.py) erweitern. Das Frontend rendert die Forms dynamisch aus dem `_CATALOG`-Objekt. Neue Path-/Query-Params werden automatisch als Input-Felder dargestellt.
