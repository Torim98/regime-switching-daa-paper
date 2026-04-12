# FastAPI Endpoints Dokumentation

Das Projekt `regime-switching-daa` nutzt eine Microservice-Architektur, die in drei containerisierte FastAPI-Services unterteilt ist. Jeder Service übernimmt einen spezifischen Teil der quantitativen Pipeline.

---

## Data Service (Port: 8001)
*Zuständig für Datenbeschaffung, Aufbereitung, Feature Engineering und explorative Datenanalyse (EDA).*

### `POST /data/ingest`
- **Beschreibung**: Startet die gesamte Daten-Pipeline. Lädt historische Marktdaten via `yfinance` herunter, führt Portfolio-Konstruktionen durch (Preprocessing) und generiert Indikatoren/Features (Volatility, SMA, Momentum). Führt zudem eine EDA (Deskriptive Statistik, ADF-Tests) durch, erstellt diverse Plots und speichert die Zwischenergebnisse (Parquet-Format) nach der Medallion-Architektur ab.

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
- **Parameter**: `model_name` (String: `MSM` | `HMM` | `LSTM` | `Transformer`), `n_trials` (int, default: 50), `every_nth_fold` (int, default: 2)
- **Beschreibung**: Führt eine Optuna-Hyperparameter-Optimierung für das angegebene Modell durch. Nutzt Walk-Forward-Splits als innere Cross-Validation. Erfordert `walk_forward.enabled: true`. Ergebnisse werden in `models/optuna_studies.db` persistiert. Gibt `best_sharpe`, `best_params` und `n_trials` zurück.

### `POST /models/optimize-all`
- **Parameter**: `n_trials` (int, default: 50), `every_nth_fold` (int, default: 2)
- **Beschreibung**: Optimiert alle vier Modelle sequenziell (MSM → HMM → LSTM → Transformer). Gleiche Funktionalität wie `/models/optimize/{model_name}`, aber für alle Modelle in einem Aufruf. Gibt ein Dict mit `best_sharpe` und `best_params` pro Modell zurück.

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