# Microservice-Architektur

Drei Services bilden die Pipeline ab, ein vierter stellt das interaktive Dashboard bereit. Alle Services nutzen die Business Logic unter `src/` und die Konfiguration unter `config/config.yaml`.

## Services

| Service | Port | Beschreibung |
|---------|------|-------------|
| **Data Service** | 8001 | Datenakquise (yfinance), Preprocessing, Feature Engineering, EDA |
| **Model Service** | 8002 | Training & Prediction aller 4 Regime-Switching-Modelle |
| **Backtest Service** | 8003 | Backtesting, SORR-Simulation, Monte Carlo, Reporting |
| **Dashboard Service** | 8004 | Interaktives Frontend: Visualisierung aller Artefakte, Control Hub für Pipeline-Endpoints (httpx-Proxy), YAML-Config-Editor, WebSocket-Log-Streaming. Nur lokal gebunden (`127.0.0.1:8004`). Details: [Dashboard Service](dashboard-service.md). |

## Endpunkte

### Data Service (`:8001`)

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| POST | `/data/ingest` | Download, Preprocessing, Feature Engineering, EDA-Plots + Stats |
| POST | `/data/label-analysis` | Label-Analyse von Labelalternativen (für supervised Modelle) |
| GET | `/data/features` | Feature-DataFrame als JSON |

### Model Service (`:8002`)

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| POST | `/models/train/{model_name}` | Einzelnes Modell trainieren (nur bei `walk_forward.enabled: false`) |
| POST | `/models/train-all` | Alle 4 Modelle trainieren; Single-Split oder Walk-Forward je nach Config |
| POST | `/models/optimize/{model_name}` | Optuna-Optimierung für ein Modell (Walk-Forward als innere CV) |
| POST | `/models/optimize-all` | Alle 4 Modelle sequenziell optimieren |
| GET | `/models/status` | Persistierungsstatus aller Modelle |

### Backtest Service (`:8003`)

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| POST | `/backtest/run` | Backtesting + Equity Curves + Drawdown + Rolling Sharpe + SORR + Krisen-Performance |
| POST | `/backtest/evaluate` | Evaluation-Tabelle + Monte Carlo Simulation + `statistics.md` |
| POST | `/backtest/report` | `statistics.md` erneut generieren |
| GET | `/backtest/results` | Evaluation-Tabelle als Markdown |

### Dashboard Service (`:8004`)

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| GET | `/`, `/eda`, `/models`, `/backtest`, `/evaluation`, `/hub`, `/config`, `/logs` | HTML-Seiten (Jinja2) |
| GET | `/api/status` | Pipeline-Artefakt-Übersicht |
| GET | `/api/chart/{...}` | Plotly-JSON für EDA, Backtest, Regime-Overlay, MCS-Quantile |
| GET/POST | `/api/hub/{catalog,health,call}` | Control-Hub-Proxy auf `:8001/:8002/:8003` |
| GET/POST | `/api/config`, `/api/config/backups`, `/api/config/restore` | YAML-Editor mit Backup + Rollback |
| GET | `/api/logs/files`, `/api/logs/snapshot/{file}` | Log-Listing + Initial-Tail |
| WS | `/ws/logs/{file}?tail=` | Live-Log-Streaming (File-Tail) |

Vollständige Liste in [dashboard-service.md](dashboard-service.md) und [fastapi-endpoints.md](fastapi-endpoints.md).

## Abhängigkeiten

Die Pipeline-Services müssen in folgender Reihenfolge aufgerufen werden, da sie über das gemeinsame Dateisystem kommunizieren:

```
Data Service → Model Service → Backtest Service
```

Das Dashboard konsumiert die Artefakte read-only und proxy't schreibende Calls (Training/Backtest) an die Pipeline-Services weiter. Es ist kein Pflicht-Schritt der Pipeline, sondern ein Control- und Visualisierungs-Layer darüber.

`docker-compose.yml` bildet die Abhängigkeiten über `depends_on` ab:
- `model-service` depends_on `data-service`
- `backtest-service` depends_on `model-service`
- `dashboard-service` depends_on `data-service`, `model-service`, `backtest-service`

Innerhalb des Model Service gilt eine feste Trainingsreihenfolge:
1. **MSM** (Markov-Switching) — unabhängig
2. **HMM** (Hidden Markov) — unabhängig, unsupervised Baseline
2b. HMM_Uni — univariate Ablationsvariante
3. **LSTM** — Supervised auf Pagan-Sossounov-Labels (aus `feature_engineered_data`), erstellt `test_df`
4. **Transformer** — Supervised auf Pagan-Sossounov-Labels
Bei `walk_forward.enabled: true` wird `/models/train-all` direkt aufgerufen und steuert alle Modelle über `run_walk_forward()`. Die Einzelrouten `/models/train/{model_name}` sind in diesem Modus blockiert (HTTP 400). Ein fingerprint-basierter Parquet-Cache beschleunigt wiederholte Durchläufe bei unveränderter Konfiguration.

## Shared Volumes

Alle Services kommunizieren über gemountete Host-Verzeichnisse:

| Volume | Data | Model | Backtest | Dashboard | Inhalt |
|--------|:---:|:---:|:---:|:---:|--------|
| `./data` | R/W | R/W | R | R | Parquet-Dateien (Medallion: Bronze/Silver/Gold) |
| `./models` | — | R/W | — | R | Persistierte Modelldateien (.pkl, .keras, .pt) + Optuna SQLite DB |
| `./assets` | R/W | R/W | R/W | R | Plots (PNG) und Tabellen (Markdown) |
| `./config` | R | R | R | **R/W** | `config.yaml` (Dashboard schreibt mit `.bak`-Backup + Rollback) |
| `./logs` | R/W | R/W | R/W | R | Service-Logdateien (Dashboard tailt per File-Tail) |
| `./docs` | — | — | R/W | R | `statistics.md` |

## Logging

Jeder Service schreibt in eine eigene Logdatei unter `logs/`:

- `logs/data_service.log`
- `logs/model_service.log`
- `logs/backtest_service.log`
- `logs/dashboard_service.log`

Format: `YYYY-MM-DD HH:MM:SS,ms | service_name | LEVEL | message`

Zeitzone: `Europe/Berlin` (konfiguriert via `TZ` Environment-Variable in `docker-compose.yml`)

Der Dashboard-Service stellt zusätzlich einen WebSocket unter `/ws/logs/{filename}` bereit, der jede dieser Log-Dateien live streamt (File-Tail) — äquivalent zu `docker compose logs -f`.

## Projektstruktur

```
regime-switching-daa/
├── src/                              # Shared Business Logic
│   ├── data/
│   │   ├── ingestion.py              # yfinance-Download
│   │   ├── preprocessing.py          # Portfolio-Konstruktion, Returns
│   │   ├── feature_engineering.py    # Rolling Features (Vol, SMA, Momentum)
│   │   ├── eda.py                    # Deskriptive Statistik, ADF-Tests
│   │   └── plots.py                  # EDA- und Feature-Plots
│   ├── models/
│   │   ├── common.py                 # Konstanten, validate_regime_signal(), create_sequences()
│   │   ├── msm.py                    # Markov-Switching Model
│   │   ├── hmm.py                    # Hidden Markov Model
│   │   ├── lstm.py                   # LSTM Network
│   │   ├── transformer.py            # Transformer (PositionalEncoding + Classifier)
│   │   └── plots.py                  # Regime-Plots (MSM, HMM, DL, Vergleich)
│   └── backtest/
│       ├── engine.py                 # Backtesting-Logik
│       ├── optimize.py               # Optuna Hyperparameter-Optimierung
│       ├── sorr.py                   # SORR-Simulation
│       ├── evaluation.py             # Strategie-Evaluation, Monte Carlo
│       ├── reporting.py              # statistics.md Generierung
│       └── plots.py                  # Equity Curves, SORR, MCS-Plots
│
├── services/                         # FastAPI-Services
│   ├── __init__.py
│   ├── logging_config.py             # Zentrales Logging (File + Console)
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
│   └── dashboard_service/            # Interaktives Frontend (Port 8004, dev-only)
│       ├── Dockerfile
│       ├── main.py
│       ├── routes.py                 # HTML-Seiten (Jinja2)
│       ├── data_adapters.py          # Parquet → Plotly-JSON (/api/chart/*)
│       ├── hub_api.py                # httpx-Proxy zu data/model/backtest (/api/hub/*)
│       ├── config_api.py             # YAML-Editor + Backup/Restore (/api/config/*)
│       ├── websockets.py             # File-Tail-Log-Streaming (/ws/logs/*)
│       ├── templates/                # 9 Jinja-Templates (base + 8 Pages)
│       └── static/                   # dashboard.css + common.js
│
├── docker-compose.yml
├── pyproject.toml
├── config/                           # config.yaml + config_loader.py
├── jupyter/                          # Explorative Notebooks
├── data/                             # Medallion-Architektur (bronze/silver/gold)
├── models/                           # Persistierte Modelldateien
├── assets/                           # Generierte Plots und Tabellen
├── docs/                             # Projektdokumentation
└── logs/                             # Service-Logdateien
```

## Vergleich: Microservices vs. Dashboard

| Aspekt | Microservice-Pipeline | Dashboard-UI |
|--------|----------------------|--------------|
| Ausführung | Docker + curl/HTTP | Browser-Klick (Control Hub) |
| Interaktivität | Swagger UI, JSON-Responses | Plotly-Charts, Forms, Live-Logs |
| Reproduzierbarkeit | `docker-compose up --build` | `http://localhost:8004/` |
| Business Logic | `src/` (identisch) | konsumiert Artefakte, proxy't Services |
| Konfiguration | `config/config.yaml` (identisch) | In-UI-Editor mit Validation + Rollback |
| Plot-Erzeugung | `src/*/plots.py` + `matplotlib.use("Agg")` | Plotly.js (clientseitig, interaktiv) |
| Daten-Persistierung | Parquet (Medallion) | read-only Konsum |
| Timing-Report | Logs pro Service | Live-Log-Stream via WebSocket |

Siehe auch: [Sequenzdiagramm: Microservice-Pipeline](microservice-sequence-diagram.md) für den detaillierten Ablauf eines Pipeline-Durchlaufs und [Dashboard Service](dashboard-service.md) für Architektur und Seitenstruktur des Frontends.