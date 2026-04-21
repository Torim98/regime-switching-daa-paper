# FastAPI Endpoints Dokumentation

Das Projekt `regime-switching-daa` nutzt eine Microservice-Architektur: drei containerisierte FastAPI-Services bilden die quantitative Pipeline ab, ein vierter (Dashboard) stellt das interaktive Frontend mit Control Hub, Config-Editor und Live-Logs bereit.

---

## Data Service (Port: 8001)
*Zuständig für Datenbeschaffung, Aufbereitung, Feature Engineering und explorative Datenanalyse (EDA).*

### `POST /data/ingest`
- **Beschreibung**: Startet die gesamte Daten-Pipeline. Lädt historische Marktdaten via `yfinance` herunter, führt Portfolio-Konstruktionen durch (Preprocessing) und generiert Indikatoren/Features (Volatility, SMA, Momentum). Führt zudem eine EDA (Deskriptive Statistik, ADF-Tests) durch, erstellt diverse Plots und speichert die Zwischenergebnisse (Parquet-Format) nach der Medallion-Architektur ab.

### `POST /data/label-analysis`
- **Beschreibung**: Berechnet die Konkordanz-Matrix und Switch-Statistiken für alle Regime-Labeler (MSM, HMM, Pagan-Sossounov, Peak-to-Trough, Lunde-Timmermann, NBER) auf dem `test_df`. Schreibt die Plots `label_concordance_matrix.png` und `label_timeline_comparison.png` nach `assets/` und liefert die numerischen Ergebnisse als JSON (`{status, elapsed_s, concordance, switch_stats}`). Dient der Begründung der Label-Wahl (Pagan-Sossounov) für LSTM/Transformer. Setzt voraus, dass `/data/ingest` und ein anschließender Modell-Trainingslauf bereits ausgeführt wurden (benötigt `test_df`).

### `GET /data/features`
- **Beschreibung**: Gibt den vollständig aufbereiteten Datensatz inklusive aller berechneten Features (den "Feature-Engineered" DataFrame) als JSON-Struktur (`orient="split"`) zurück. Setzt voraus, dass `/data/ingest` zuvor ausgeführt wurde.

---

## Model Service (Port: 8002)
*Zuständig für das Training, die Prädiktion und die Persistierung der vier eingesetzten Machine Learning- und Ökonometrie-Modelle.*

### `POST /models/train/{model_name}`
- **Parameter**: `model_name` (String: `msm` | `hmm` | `lstm` | `transformer`)
- **Beschreibung**: Trainiert ein einzelnes, spezifiziertes Modell. **Nur verfügbar bei `walk_forward.enabled: false`.** Im Walk-Forward-Modus gibt dieser Endpoint HTTP 400 zurück mit dem Hinweis, `/models/train-all` zu verwenden.

### `POST /models/train-all`
- **Beschreibung**: Trainiert alle vier Modelle. Bei `walk_forward.enabled: false` sequentiell im Single-Split-Modus. Bei `walk_forward.enabled: true` über die Walk-Forward-Engine (`run_walk_forward`) mit rollierenden Folds, fingerprint-basiertem Cache und OOS-Aggregation. Gibt im Walk-Forward-Modus zusätzlich `folds` und `oos_days` zurück.

### `POST /models/optimize/{model_name}`
- **Parameter**: `model_name` (String: `MSM` | `HMM` | `LSTM` | `Transformer`)
- **Beschreibung**: Führt eine Optuna-Hyperparameter-Optimierung für das angegebene Modell durch. Nutzt Walk-Forward-Splits als innere Cross-Validation. `n_trials` und `every_nth_fold` werden pro Modell aus `config.yaml` (`optimization.n_trials_per_model` bzw. `optimization.every_nth_fold_per_model`) gelesen. Keine API-Overrides, damit die Thesis-Konfiguration reproduzierbar bleibt. Erfordert `walk_forward.enabled: true`. Ergebnisse werden in `models/optuna_studies.db` persistiert. Gibt `best_sharpe`, `best_params` und `n_trials` zurück.

### `POST /models/optimize-all`
- **Parameter**: keine
- **Beschreibung**: Optimiert alle vier Modelle sequenziell (MSM → HMM → LSTM → Transformer). `n_trials` und `every_nth_fold` werden pro Modell aus `config.yaml` gelesen (Thesis-Default: 50 Trials / `every_nth_fold=2` für MSM & HMM, 30 / 5 für LSTM & Transformer). Gibt ein Dict mit `best_sharpe` und `best_params` pro Modell zurück.

### `GET /models/status`
- **Beschreibung**: Überprüft das Dateisystem und gibt für jedes der vier Modelle als Boolean (`true`/`false`) zurück, ob das jeweilige Modell bereits trainiert und erfolgreich auf der Festplatte persistiert wurde.

---

## Backtest Service (Port: 8003)
*Zuständig für die Strategie-Evaluation, Monte Carlo Simulationen und das finale Reporting.*

### `POST /backtest/run`
- **Beschreibung**: Führt das historische Backtesting durch. Im Walk-Forward-Modus wird `test_df` auf das gemeinsame OOS-Fenster beschnitten (`dropna(how="any")`). Erzeugt neben Equity Curves und Transaktionskosten auch annualisierte Metriken, Krisen-Performance-Tabelle, Rolling-Sharpe-Plot und Drawdown-Plot. Führt SORR-Simulationen für alle konfigurierten Szenarien durch.

### `POST /backtest/evaluate`
- **Beschreibung**: Evaluiert alle Strategien tiefgreifend. Führt eine Block-Bootstrap Monte Carlo Simulation durch, um die Robustheit der Strategien zu testen. Erstellt detaillierte Performance-Metriken, Boxplots, Quantil-Visualisierungen und MCS-Pfade. Stößt am Ende automatisiert die Report-Erstellung an.

### `POST /backtest/report`
- **Beschreibung**: Sammelt alle generierten Tabellen, Metriken und Markdown-Schnipsel und kombiniert sie in einen finalen Statistik-Report (üblicherweise `statistics.md`), der im `assets/` oder `docs/` Verzeichnis zur Verfügung gestellt wird.

### `GET /backtest/results`
- **Beschreibung**: Gibt die Ergebnisse der Strategie-Evaluation (die finale Performance-Tabelle) als reinen Text/Markdown im JSON-Format zurück, um die Resultate über die API abfragen zu können. Setzt eine erfolgreiche Ausführung von `/backtest/evaluate` voraus.

---

## Dashboard Service (Port: 8004)
*Interaktives Frontend mit Visualisierung, Control Hub, Config-Editor und Live-Log-Streaming. Dev-only an `127.0.0.1` gebunden. Ausführliche Beschreibung: [Dashboard Service](dashboard-service.md).*

### HTML-Seiten (Jinja2)

| Pfad | Seite |
|------|-------|
| `GET /` | Overview (Status-Kacheln, Artefakt-Grid, Coverage-Map) |
| `GET /hub` | Control Hub (alle Pipeline-Endpoints per Klick) |
| `GET /eda` | EDA-Charts + PNG-Gallery |
| `GET /models` | Regime-Erkennung, Label-Konkordanz, Optuna-Heatmaps |
| `GET /backtest` | Equity Curves, Drawdown, Rolling Sharpe, SORR, Krisen-Perf |
| `GET /evaluation` | Vollständige `statistics.md`-Abdeckung (MCS, H1/H2, Break-Even, ...) |
| `GET /config` | Monaco-YAML-Editor für `config.yaml` |
| `GET /logs` | Live-Log-Stream (WebSocket) |

### Data-Adapter — Parquet → Plotly-JSON

Alle Chart-Endpoints liefern Plotly-kompatibles JSON, das clientseitig mit Plotly.js gerendert wird. Keine Neuberechnungen, alle Werte stammen aus den Parquet-Artefakten der Pipeline.

#### `GET /api/status`
- **Beschreibung**: Übersicht aller Pipeline-Artefakte (Existenz, Größe in MB, mtime) plus Meta-Info (End-Date, Walk-Forward-Flag, Fast-Mode-Flag). Basis für die Status-Kacheln und das Artefakt-Grid auf der Overview-Seite.

#### `GET /api/asset/{name}` / `GET /api/markdown/{name}`
- **Beschreibung**: Read-only-Auslieferung eines PNG- oder MD-Assets aus `assets/`. Path-Traversal-Schutz integriert. `GET /api/markdown/{name}` liefert MD-Inhalte als JSON-Payload für clientseitiges Rendering mit marked.js.

#### `GET /api/chart/returns`
- **Parameter**: `col` (str, default: `Returns`), `smoothing` (int 0–252, default: 0)
- **Beschreibung**: Zeitreihe einer beliebigen Spalte aus `feature_engineered_data.parquet` mit optionalem Moving-Average-Smoothing.

#### `GET /api/chart/feature-correlation`
- **Beschreibung**: Korrelationsmatrix (Pearson) über die in `features.model_features` konfigurierten Spalten.

#### `GET /api/chart/capital-curve`
- **Beschreibung**: 60/40-Benchmark-Kapitalkurve (kumulierte Returns).

#### `GET /api/chart/equity-curves`
- **Beschreibung**: Equity Curves aller Strategien aus `backtesting_results.parquet`. Setzt `/backtest/run` voraus.

#### `GET /api/chart/drawdown`
- **Beschreibung**: Drawdown-Verlauf aller Strategien.

#### `GET /api/chart/rolling-sharpe`
- **Parameter**: `window` (int 21–1260, default: 252)
- **Beschreibung**: Rolling Sharpe Ratio mit konfigurierbarem Fenster.

#### `GET /api/chart/regime-overlay`
- **Parameter**: `model` (str: `MSM` | `HMM` | `LSTM` | `Transformer`, default: `MSM`)
- **Beschreibung**: 60/40-Kurs mit Bear-Probability und Bear-Signal-Bändern (rote Shapes) überlagert. Zeigt, wann und wo das ausgewählte Modell Bärenmarkt-Phasen erkannt hat.

#### `GET /api/chart/mcs-quantiles`
- **Parameter**: `scenario` (str, default: `Standard`), `strategy` (str, default: `Transformer`)
- **Beschreibung**: Quantil-Fächer (5 / 25 / 50 / 75 / 95 %) der Monte-Carlo-Simulation-Pfade. Setzt `/backtest/evaluate` voraus.

### Control-Hub-Proxy

Ruft die Pipeline-Services per `httpx` auf. Read-Timeout: 8 h (für Walk-Forward-Train-All). Service-URLs über Env-Vars konfigurierbar (`DATA_SERVICE_URL`, `MODEL_SERVICE_URL`, `BACKTEST_SERVICE_URL`).

#### `GET /api/hub/catalog`
- **Beschreibung**: Liefert den kompletten Endpoint-Katalog (Service, Pfad, Methode, Label, Beschreibung, Parameter-Schema, Danger-Flag) für das dynamische UI-Rendering der Control-Hub-Seite.

#### `GET /api/hub/health`
- **Beschreibung**: Ping-Check auf alle drei Pipeline-Services (`/openapi.json` als Marker). Liefert `{up, status, url}` je Service für die Health-Tiles.

#### `POST /api/hub/call`
- **Parameter**: `service` (`data` | `model` | `backtest`), `path` (z.B. `/data/ingest`), `method` (`GET` | `POST`), `query` (optional: JSON-String mit Query-Params)
- **Beschreibung**: Generischer Proxy-Call. Nutzt die UI, um beliebige Endpoints der Pipeline-Services auszulösen. Response: `{status_code, ok, body}` — bei Non-JSON-Responses wird der Body als `{"text": ...}` verpackt.

### Config-Editor

Sicherheitsnetz beim Schreiben: (1) YAML-Parse → (2) Pflicht-Sections-Check → (3) `.bak`-Backup → (4) Atomic Swap via tempfile → (5) `PipelineConfig()`-Reload-Verifikation → (6) Rollback aus Backup bei Reload-Fehler.

#### `GET /api/config`
- **Beschreibung**: Liefert die aktuelle `config.yaml` als reinen Text plus Meta (Pfad, mtime, Size).

#### `POST /api/config`
- **Body**: `{"content": "<gesamter YAML-Text>"}`
- **Beschreibung**: Speichert den übergebenen YAML-Text nach bestandener Validierung. Pflicht-Sections: `data`, `features`, `portfolio`, `models`, `backtesting`, `walk_forward`, `evaluation`, `paths`, `plotting`. Response: `{status, backup, bytes_written, reloaded}`.

#### `GET /api/config/backups`
- **Beschreibung**: Liste aller `.bak`-Dateien im `config/`-Ordner, sortiert nach mtime (neueste zuerst).

#### `POST /api/config/restore`
- **Body**: `{"name": "config.YYYYMMDD-HHMMSS.bak"}`
- **Beschreibung**: Spielt eine bestimmte Backup-Datei als aktive `config.yaml` zurück. Der vorherige Zustand wird zusätzlich als `*.pre-restore.bak` gesichert.

### Live-Log-Streaming

WebSocket-basierter File-Tail auf `logs/*.log`. Rotation und Truncation werden erkannt und mit einer System-Zeile (`[dashboard] Datei truncated — resume from 0`) signalisiert.

#### `GET /api/logs/files`
- **Beschreibung**: Liste aller verfügbaren `logs/*.log`-Dateien inklusive Größe (KB) und mtime.

#### `GET /api/logs/snapshot/{filename}`
- **Parameter**: `lines` (int 1–10000, default: 500)
- **Beschreibung**: Liefert die letzten N Zeilen der angegebenen Log-Datei ohne WebSocket (für Initial-Load oder Snapshots). Path-Traversal-Schutz aktiv.

#### `WS /ws/logs/{filename}`
- **Parameter**: `tail` (int, default: 200)
- **Beschreibung**: Sendet zunächst die letzten `tail` Zeilen, dann via ~300 ms-Polling alle neuen Zeilen als Text-Frames. Bei Datei-Truncation wird ab Position 0 neu gelesen. Bei `WebSocketDisconnect` wird die Verbindung sauber geschlossen.