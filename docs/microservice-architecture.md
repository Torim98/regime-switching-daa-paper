# Microservice-Architektur

Die Pipeline kann neben der Jupyter-Notebook-Ausführung auch über drei containerisierte FastAPI-Services betrieben werden. Beide Wege nutzen dieselbe Business Logic unter `src/` und dieselbe Konfiguration unter `config/config.yaml`.

## Services

| Service | Port | Beschreibung |
|---------|------|-------------|
| **Data Service** | 8001 | Datenakquise (yfinance), Preprocessing, Feature Engineering, EDA |
| **Model Service** | 8002 | Training & Prediction aller 4 Regime-Switching-Modelle |
| **Backtest Service** | 8003 | Backtesting, SORR-Simulation, Monte Carlo, Reporting |

## Endpunkte

### Data Service (`:8001`)

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| POST | `/data/ingest` | Download, Preprocessing, Feature Engineering, EDA-Plots + Stats |
| GET | `/data/features` | Feature-DataFrame als JSON |

### Model Service (`:8002`)

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| POST | `/models/train/{model_name}` | Einzelnes Modell trainieren (`msm`, `hmm`, `lstm`, `transformer`) |
| POST | `/models/train/{model_name}` | Einzelnes Modell trainieren (nur bei `walk_forward.enabled: false`) |
| POST | `/models/train-all` | Alle 4 Modelle trainieren; Single-Split oder Walk-Forward je nach Config |
| GET | `/models/status` | Persistierungsstatus aller Modelle |

### Backtest Service (`:8003`)

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| POST | `/backtest/run` | Backtesting + Equity Curves + Drawdown + Rolling Sharpe + SORR + Krisen-Performance |
| POST | `/backtest/evaluate` | Evaluation-Tabelle + Monte Carlo Simulation + `statistics.md` |
| GET | `/backtest/results` | Evaluation-Tabelle als Markdown |

## Abhängigkeiten

Die Services müssen in folgender Reihenfolge aufgerufen werden, da sie über das gemeinsame Dateisystem kommunizieren:

```
Data Service → Model Service → Backtest Service
```

`docker-compose.yml` bildet dies über `depends_on` ab:
- `model-service` depends_on `data-service`
- `backtest-service` depends_on `model-service`

Innerhalb des Model Service gilt eine feste Trainingsreihenfolge:
1. **MSM** (Markov-Switching) — erzeugt Labels für LSTM/Transformer
2. **HMM** (Hidden Markov) — unabhängig, aber konventionell nach MSM
3. **LSTM** — benötigt MSM-Labels, erstellt `test_df`
4. **Transformer** — benötigt `test_df` aus dem LSTM-Schritt
Bei `walk_forward.enabled: true` wird `/models/train-all` direkt aufgerufen und steuert alle Modelle über `run_walk_forward()`. Die Einzelrouten `/models/train/{model_name}` sind in diesem Modus blockiert (HTTP 400). Ein fingerprint-basierter Parquet-Cache beschleunigt wiederholte Durchläufe bei unveränderter Konfiguration.

## Shared Volumes

Alle Services kommunizieren über gemountete Host-Verzeichnisse:

| Volume | Data Service | Model Service | Backtest Service | Inhalt |
|--------|:---:|:---:|:---:|--------|
| `./data` | R/W | R/W | R | Parquet-Dateien (Medallion: Bronze/Silver/Gold) |
| `./models` | — | R/W | — | Persistierte Modelldateien (.pkl, .keras, .pt) |
| `./assets` | R/W | R/W | R/W | Plots (PNG) und Tabellen (Markdown) |
| `./config` | R | R | R | `config.yaml` |
| `./logs` | R/W | R/W | R/W | Service-Logdateien |
| `./docs` | — | — | R/W | `statistics.md` |

## Logging

Jeder Service schreibt in eine eigene Logdatei unter `logs/`:

- `logs/data_service.log`
- `logs/model_service.log`
- `logs/backtest_service.log`

Format: `YYYY-MM-DD HH:MM:SS,ms | service_name | LEVEL | message`

Zeitzone: `Europe/Berlin` (konfiguriert via `TZ` Environment-Variable in `docker-compose.yml`)

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
│   └── backtest_service/
│       ├── Dockerfile
│       ├── main.py
│       └── routes.py
│
├── docker-compose.yml
├── pyproject.toml
├── config/                           # config.yaml + config_loader.py
├── jupyter/                          # Notebook-Pipeline (01–99)
├── data/                             # Medallion-Architektur (bronze/silver/gold)
├── models/                           # Persistierte Modelldateien
├── assets/                           # Generierte Plots und Tabellen
├── docs/                             # Projektdokumentation
└── logs/                             # Service-Logdateien
```

## Vergleich: Notebook vs. Microservices

| Aspekt | Notebook-Pipeline | Microservice-Pipeline |
|--------|------------------|----------------------|
| Ausführung | Jupyter / Papermill | Docker + curl/HTTP |
| Interaktivität | Inline-Plots, Zellen-Output | Swagger UI, JSON-Responses |
| Reproduzierbarkeit | `regime-switching-daa.ipynb` | `docker-compose up --build` |
| Business Logic | `src/` (import) | `src/` (identisch) |
| Konfiguration | `config/config.yaml` | `config/config.yaml` (identisch) |
| Plot-Erzeugung | `src/*/plots.py` + `IPython.display` | `src/*/plots.py` + `matplotlib.use("Agg")` |
| Daten-Persistierung | Parquet (Medallion) | Parquet (Medallion, identisch) |
| Timing-Report | `pipeline_timing.md` (Papermill) | Logs pro Service |

Siehe auch: [Sequenzdiagramm](sequence-diagram.md) für den detaillierten Ablauf eines Pipeline-Durchlaufs.