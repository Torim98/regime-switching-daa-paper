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
- **Beschreibung**: Trainiert ein einzelnes, spezifiziertes Modell auf den Feature-Daten, generiert die entsprechenden Regime-Wahrscheinlichkeiten und Handelssignale, validiert diese und persistiert das trainierte Modell. Erstellt zudem modellspezifische Evaluierungsplots. *(Achtung: `lstm` und `transformer` benötigen die Signale des `msm` als Labels).*

### `POST /models/train-all`
- **Beschreibung**: Trainiert alle vier Modelle (Markov-Switching Model, Hidden Markov Model, LSTM und Transformer) automatisiert in der korrekten, sequenziellen Reihenfolge (MSM zuerst), sodass die Pipeline-Abhängigkeiten gewahrt bleiben.

### `GET /models/status`
- **Beschreibung**: Überprüft das Dateisystem und gibt für jedes der vier Modelle als Boolean (`true`/`false`) zurück, ob das jeweilige Modell bereits trainiert und erfolgreich auf der Festplatte persistiert wurde.

---

## Backtest Service (Port: 8003)
*Zuständig für die Strategie-Evaluation, Monte Carlo Simulationen und das finale Reporting.*

### `POST /backtest/run`
- **Beschreibung**: Führt das historische Backtesting der durch die Modelle generierten Signale durch. Berechnet Transaktionskosten, generiert Performance Summaries und Equity Curves. Führt zusätzlich die "Sequence of Returns Risk" (SORR) Simulationen auf Basis definierter Entnahme-Szenarien durch und generiert entsprechende Visualisierungen.

### `POST /backtest/evaluate`
- **Beschreibung**: Evaluiert alle Strategien tiefgreifend. Führt eine Block-Bootstrap Monte Carlo Simulation durch, um die Robustheit der Strategien zu testen. Erstellt detaillierte Performance-Metriken, Boxplots, Quantil-Visualisierungen und MCS-Pfade. Stößt am Ende automatisiert die Report-Erstellung an.

### `POST /backtest/report`
- **Beschreibung**: Sammelt alle generierten Tabellen, Metriken und Markdown-Schnipsel und kombiniert sie in einen finalen Statistik-Report (üblicherweise `statistics.md`), der im `assets/` oder `docs/` Verzeichnis zur Verfügung gestellt wird.

### `GET /backtest/results`
- **Beschreibung**: Gibt die Ergebnisse der Strategie-Evaluation (die finale Performance-Tabelle) als reinen Text/Markdown im JSON-Format zurück, um die Resultate über die API abfragen zu können. Setzt eine erfolgreiche Ausführung von `/backtest/evaluate` voraus.
