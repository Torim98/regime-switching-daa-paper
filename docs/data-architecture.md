# Datenarchitektur – Medallion-Modell (Bronze / Silver / Gold)

## Übersicht

Die Datenablage folgt einem dreistufigen Medallion-Modell.
Jede Stufe repräsentiert einen definierten Verarbeitungsgrad der Daten.

```
Yahoo Finance API
       │
       ▼
  ┌──────────┐
  │  Bronze  │  Rohdaten – unbereinigt, direkt von der Quelle
  └────┬─────┘
       │  01_data_preprocessing.ipynb  /  POST /data/ingest
       ▼
  ┌──────────┐
  │  Silver  │  Bereinigte, transformierte und feature-engineerte Daten
  └────┬─────┘
       │  02–04 Notebooks  /  POST /models/train/{name}
       ▼
  ┌──────────┐
  │   Gold   │  Ergebnisse: Backtesting, Monte-Carlo-Simulation
  └──────────┘
```

## Verzeichnisstruktur

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

## Tier-Beschreibung

### Bronze – Rohdaten

Unveränderte Marktdaten direkt aus der Yahoo-Finance-API.
Enthält alle Original-NaNs und Lücken.

| Datei | Erzeugt von | Inhalt |
|---|---|---|
| `01_raw_data.parquet` | `01_data_preprocessing.ipynb` / Data Service | Tägliche Schlusskurse aller Ticker (^GSPC, VUSTX, ^VIX, ^IRX, ^TNX) |

### Silver – Bereinigte und transformierte Daten

Forward-Fill, Dropna, Log-Renditen, Feature-Engineering und
Modell-Vorhersagen. Daten sind analyse-bereit, aber noch keine
Endergebnisse.

| Datei | Erzeugt von | Inhalt |
|---|---|---|
| `02_preprocessed_data.parquet` | `01_data_preprocessing.ipynb` / Data Service | Portfolio-Renditen, Cash-Renditen, VIX, Zinsen |
| `03_feature_engineered_data.parquet` | `02_feature_engineering.ipynb` / Data Service | Zusätzliche Features: SMA, Volatilität, Momentum, Yield Spread |
| `04_test_df_data.parquet` | `03_regime_switching_models.ipynb` / Model Service | Test-Datensatz mit Regime-Vorhersagen aller Modelle |
| `wf_cache.parquet` | `03_regime_switching_models.ipynb` / Model Service | Walk-Forward OOS-Cache mit Fingerprint-Validierung (nur bei `walk_forward.cache_enabled: true`) |

### Gold – Endergebnisse

Aggregierte Backtesting-Ergebnisse und Simulationen, die direkt
in die Auswertung und Thesis einfließen.

| Datei | Erzeugt von | Inhalt |
|---|---|---|
| `05_backtesting_results_data.parquet` | `04_backtesting.ipynb` / Backtest Service | Equity-Kurven und Renditen aller Strategien |
| `05_backtesting_transaction_costs_data.parquet` | `04_backtesting.ipynb` / Backtest Service | Transaktionskostenanalyse |
| `05_backtesting_sorr_simulation.parquet` | `04_backtesting.ipynb` / Backtest Service | Sequence-of-Returns-Risk-Simulation |
| `06_mcs_data.parquet` | `05_evaluation.ipynb` / Backtest Service | Monte-Carlo-Simulationspfade |

## Pfadverwaltung

Alle Dateipfade sind zentral in `config/config.yaml` unter `paths.files`
definiert und werden über `cfg.data_path("<key>")` aufgelöst.
Notebooks referenzieren keine hart-kodierten Pfade.

## Microservice-Zugriff

Die Docker-basierte Microservice-Architektur (siehe [microservice-architecture.md](microservice-architecture.md)) erzeugt dieselben Dateien auf denselben Pfaden. Die `data/`-Verzeichnisse werden über Docker Volume-Mounts (`./data:/app/data`) zwischen Host und Containern geteilt. Dadurch sind die Parquet-Dateien sowohl lokal als auch im Container unter identischen relativen Pfaden erreichbar.

| Service | Erzeugte Tier | Endpunkt |
|---|---|---|
| Data Service (`:8001`) | Bronze + Silver | `POST /data/ingest` |
| Model Service (`:8002`) | Silver | `POST /models/train/{model_name}` |
| Backtest Service (`:8003`) | Gold | `POST /backtest/run` |
