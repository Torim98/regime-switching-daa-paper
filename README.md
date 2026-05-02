# Dynamische Asset-Allokation mittels Regime-Switching-Modellen

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dieses Repository enthält den Code und die Analysen meiner Master-Thesis zum Thema: 
**"Dynamische Asset-Allokation mittels Regime-Switching-Modellen: Ein Vergleich ökonometrischer Modelle und moderner Machine-Learning-Verfahren zur Reduktion von Maximum Drawdowns"**.

## Ziel der Arbeit

Das Kernziel dieser Arbeit ist der **systematische Vergleich** zwischen zwei Paradigmen der Finanzmarktanalyse zur Identifikation von Marktregimes: klassischen **ökonometrischen Modellen** und modernen **Machine-Learning-Verfahren**. 

Statische Anlagestrategien leiden in Zeiten zunehmender Marktvolatilität und komplexer Krisenzyklen (wie der Dotcom-Blase, der Finanzkrise 2008 oder der Zinswende 2022) besonders unter dem **Sequence of Returns Risk (SORR)**, das Risiko, dass Markteinbrüche zu Beginn einer Entnahmephase das Kapital irreversibel schädigen. Dieses Projekt untersucht, wie eine **Dynamische Asset-Allokation (DAA)**, gestützt auf eine **automatisierte Regime-Erkennung** (Bulle vs. Bär) genutzt werden kann, um durch rechtzeitige Umschichtung in den Geldmarkt (Cash) das Portfolio-Risiko zu glätten und die Kapitalsicherheit im Ruhestand massiv zu erhöhen.

### Die Untersuchungsschwerpunkte sind:

1.  **Modell-Vergleich:** Evaluierung der Vorhersagegüte von statistisch fundierten ökonometrischen Modellen (z. B. Markov-Switching) gegenüber hochflexiblen Machine-Learning-Architekturen (z. B. LSTM-Netzwerken, Transformer) unter Einbeziehung von Makro-Indikatoren (VIX, Yield Spreads,...).
2.  **Risikominimierung:** Quantifizierung des Mehrwerts dieser Modelle zur signifikanten **Reduktion von Maximum Drawdowns**. Es wird geprüft, ob die Modelle in der Lage sind, rechtzeitig Signale zur Umschichtung von Risiko-Assets (Aktien) in sichere Häfen (Geldmarkt/Bonds) zu generieren.
3.  **SORR-Prävention:** Die praktische Anwendung fokussiert sich auf die Minderung des **Sequence of Returns Risk (SORR)**. Hierbei soll bewiesen werden, dass eine regimebasierte Steuerung das Pfadrisiko des Vermögensaufbaus glättet und somit die Kapitalsicherheit, insbesondere in der kritischen Phase kurz vor oder zu Beginn der Entnahmephase (Renteneintritt), massiv erhöht.

Durch diesen Vergleich soll aufgezeigt werden, ob die höhere Komplexität moderner KI-Verfahren gegenüber bewährter Ökonometrie einen ökonomisch messbaren Vorteil in der risikoadjustierten Performance liefert.

---

## Methodik & Modelle

In diesem Projekt werden zwei verschiedene Ansätze zur Regime-Erkennung verglichen:

1. **Ökonometrische Modelle:** Diese Modelle basieren auf der Annahme, dass Finanzmärkte stochastischen Prozessen folgen und Regimes als verborgene Zustände (Latent States) mathematisch modelliert werden können.
*   **Markov-Switching-Modelle (MSM):** Ein klassisches Regressionsverfahren, bei dem Parameter (wie Mittelwert und Varianz der Rendite) zwischen Zuständen springen. Die Wechselwahrscheinlichkeiten werden über eine Übergangsmatrix berechnet.
*   **Hidden-Markov-Modelle (HMM):** Ein Unsupervised-Learning-Ansatz aus der Statistik. Das HMM identifiziert Cluster in den Datenverteilungen, um Phasen hoher und niedriger Volatilität voneinander zu trennen, ohne dass vorab gelabelte Daten nötig sind.
2. **Moderne Machine-Learning-Verfahren:** Dieser Ansatz nutzt die Fähigkeit von künstlichen neuronalen Netzen, hochkomplexe, nicht-lineare Zusammenhänge in großen Datenmengen zu identifizieren, ohne explizite statistische Verteilungsannahmen vorauszusetzen.
*   **LSTM-Netzwerke (Long Short-Term Memory):** Eine spezialisierte Form von Recurrent Neural Networks (RNN), die über ein "Gedächtnis" für zeitliche Abhängigkeiten verfügen. In dieser Arbeit wird das LSTM in einem **Supervised-Learning-Setting** eingesetzt: Es lernt, die durch die ökonometrischen Modelle identifizierten Regime-Wechsel unter Berücksichtigung von Zeitreihen-Fenstern (Sequenzen) vorherzusagen.
*   **Transformer-Netzwerk (Multi-Head Self-Attention):** Eine Attention-basierte Architektur, die im Gegensatz zu rekurrenten Netzwerken **alle Zeitschritte einer Sequenz parallel** verarbeiten kann. Durch den Multi-Head Self-Attention-Mechanismus lernt das Modell, welche historischen Zeitpunkte innerhalb eines Fensters die stärkste Relevanz für die aktuelle Regime-Klassifikation besitzen. Ein Positional Encoding bewahrt dabei die zeitliche Ordnung der Inputdaten. Der Transformer wird im **Supervised-Setting** (trainiert auf Pagan-Sossounov-Labels, siehe Label-Analyse in `01a_label_analysis`) eingesetzt und dient dem Test der Hypothese H2: Ob Attention-basierte Architekturen eine höhere Vorhersagegüte als ökonometrische Modelle und rekurrente Netze erreichen.

---

## Technologie-Stack

Für die Umsetzung der Forschungsumgebung wurde ein moderner Data-Science-Stack gewählt, der Stabilität mit hoher Rechenleistung kombiniert:

*   **Programmiersprache:** Python 3.10+
*   **Datenquellen:** Yahoo Finance API (`yfinance`)
*   **Datenverarbeitung:** `Pandas`, `NumPy`, `PyArrow` (Parquet-Engine)
*   **Ökonometrie & Statistik:** `Statsmodels` (Markov-Regression), `hmmlearn` (Hidden Markov Models), `SciPy`
*   **Machine Learning:** `TensorFlow` / `Keras` (LSTM-Architekturen), `PyTorch` (Transformer), `Scikit-Learn`
*   **Hyperparameter-Optimierung:** `Optuna` (Bayessche Optimierung via TPE)
*   **Reporting:** `Matplotlib` (Visualisierung), `Seaborn` (Heatmaps), `Tabulate` (Markdown-Export)
*   **Microservices:** `FastAPI`, `Uvicorn`, `Docker` / `Docker Compose`

---

## Architektur

Das Projekt bietet zwei gleichwertige Ausführungswege, die dieselbe Business Logic (`src/`), Konfiguration (`config/config.yaml`) und Datenpersistierung (Medallion-Architektur) nutzen.

### Notebook-Pipeline (Forschung & Exploration)

Ein modulares Pipeline-Design mit spezialisierten Jupyter Notebooks. Ein zentrales Master-Notebook (`regime-switching-daa.ipynb`) orchestriert die Ausführung der einzelnen Module via Papermill in der korrekten Reihenfolge.

### Microservice-Architektur (Reproduzierbarkeit & Deployment)

Vier containerisierte FastAPI-Services bilden die gesamte Pipeline und das Frontend ab:

| Service | Port | Beschreibung |
|---------|------|-------------|
| **Data Service** | 8001 | Datenakquise, Preprocessing, Feature Engineering, EDA |
| **Model Service** | 8002 | Training & Prediction (MSM, HMM, LSTM, Transformer) |
| **Backtest Service** | 8003 | Backtesting, SORR, Monte Carlo Simulation, Reporting |
| **Dashboard Service** | 8004 | Interaktives UI: EDA/Backtest/Evaluation-Visualisierung, Control Hub für alle Pipeline-Endpoints, YAML-Config-Editor, Live-Log-Streaming |

Die Pipeline-Services kommunizieren über gemeinsame Dateisystem-Volumes; der Dashboard-Service liest dieselben Artefakte read-only und ruft die anderen Services per `httpx`-Proxy auf. Orchestrierung über `docker-compose`.

**Weiterführende Dokumentationen:**
* [Microservice-Architektur & Setup](docs/microservice-architecture.md)
* [Dashboard Service (UI & Control Hub)](docs/dashboard-service.md)
* [Sequenzdiagramm: Microservice-Pipeline](docs/microservice-sequence-diagram.md)
* [Sequenzdiagramm: Jupyter-Pipeline](docs/jupyter-sequence-diagram.md)
* [API Endpoints & Routen](docs/fastapi-endpoints.md)

> Da alle Services auf FastAPI basieren, steht nach dem Start (via `docker-compose up`) für jeden Service auch eine interaktive **Swagger UI** zur Verfügung:
> * Data Service: [http://localhost:8001/docs](http://localhost:8001/docs)
> * Model Service: [http://localhost:8002/docs](http://localhost:8002/docs)
> * Backtest Service: [http://localhost:8003/docs](http://localhost:8003/docs)
> * Dashboard Service: [http://localhost:8004/docs](http://localhost:8004/docs)
>
> Das **interaktive Frontend** selbst läuft unter [http://localhost:8004/](http://localhost:8004/) (nur an `127.0.0.1` gebunden, dev-only). Es löst alle Pipeline-Schritte per Klick aus, visualisiert sämtliche Artefakte mit Plotly, erlaubt das Editieren der `config.yaml` mit automatischem Backup/Rollback und streamt Container-Logs via WebSocket — äquivalent zu `docker compose logs -f`.

### Dashboard (Screenshots)

| Overview | Control Hub |
|:---:|:---:|
| ![Overview](assets/screenshots/dashboard_overview.png) | ![Control Hub](assets/screenshots/dashboard_control-hub.png) |
| Status-Kacheln, Pipeline-Artefakte, Coverage-Map | Alle Endpoints per Klick auslösen, Health-Tiles, JSON-Viewer |

| Evaluation | Config-Editor |
|:---:|:---:|
| ![Evaluation](assets/screenshots/dashboard_evaluation.png) | ![Config-Editor](assets/screenshots/dashboard_config.png) |
| Interaktive Plotly-Charts, MCS-Quantile, Hypothesentests | Monaco-YAML-Editor mit Backup/Restore, Dirty-State |

---

## Engineering-Konzepte

Hinter der Pipeline stehen fortgeschrittene Konzepte der Software-Entwicklung und Finanzmathematik, um die Validität der Ergebnisse sicherzustellen:

### Daten-Persistierung & Entkopplung
Um Notebooks voneinander zu entkoppeln und den Arbeitsspeicher effizient zu nutzen, werden Zwischenergebnisse im **Apache Parquet-Format** gespeichert. Parquet bietet gegenüber CSV eine höhere Performance und erhält die Integrität der Datentypen (insb. Zeitstempel), was für die Zeitreihenanalyse essentiell ist. Die Datenablage folgt einem **[Medallion-Modell](./docs/data-architecture.md)** (Bronze → Silver → Gold) zur klaren Trennung von Rohdaten, bereinigten Zwischenergebnissen und finalen Analyseergebnissen.

### Shared Business Logic
Die gesamte Fachlogik ist in wiederverwendbaren Python-Modulen unter `src/` gekapselt. Sowohl die Jupyter Notebooks als auch die FastAPI-Services importieren aus denselben Modulen, was Konsistenz zwischen beiden Ausführungswegen garantiert.

### Modell-Persistierung & Caching
Trainierte Modelle (MSM, HMM, LSTM, Transformer) werden im Ordner `models/` zwischengespeichert. Dies ermöglicht es, das rechenintensive Training zu überspringen und stattdessen vortrainierte Modelle zu laden. Das Verhalten wird über `model_persistence.enabled` in der `config.yaml` gesteuert. Ist die Option aktiviert und existieren die Modelldateien, wird das Training automatisch übersprungen. Andernfalls wird normal trainiert und das Ergebnis für zukünftige Läufe gespeichert.

### Walk-Forward-Validierung
Für die robuste Out-of-Sample-Evaluation steht ein konfigurierbares Walk-Forward-Framework zur Verfügung (`walk_forward.enabled: true` in `config.yaml`). Anstelle eines einzelnen 80/20-Splits werden rollierende Folds generiert (10 Jahre Training, 12 Monate Test, 12 Monate Step), in denen jedes Modell pro Fold neu trainiert wird. Die OOS-Vorhersagen aller Folds werden zu einer durchgehenden Serie aggregiert. Ein fingerprint-basierter Parquet-Cache verhindert unnötiges Re-Training bei unveränderter Konfiguration. Im Walk-Forward-Modus ist die Modell-Persistierung deaktiviert, da jeder Fold ein eigenes Modell erzeugt.

### Vermeidung von Look-ahead Bias
Ein kritischer Aspekt im Backtesting ist die Vermeidung von Informationslecks aus der Zukunft. Alle generierten Handelssignale werden systematisch um einen Zeitschritt ($T+1$) verschoben. Entscheidungen werden somit ausschließlich auf Basis der zum Handelszeitpunkt verfügbaren historischen Informationen getroffen.

### Data-Driven Automation (Dynamic Matching)
Das Framework ist **vollständig dynamisch** aufgebaut. Ein spezialisierter Such-Algorithmus identifiziert neue Modell-Outputs automatisch anhand eines definierten Namensschemas (`Modell_Signal`). Dadurch können neue Modell-Architekturen integriert werden, ohne den Code für das Backtesting, die Evaluation oder das Reporting manuell anpassen zu müssen. Siehe [How to add a ML Model](docs/how-to-add-ml-model.md).

### Hyperparameter-Optimierung (Optuna)
Alle Modellparameter werden systematisch mittels Bayesscher Optimierung (Optuna, TPE-Sampler) gesucht. Die Optimierung nutzt die Walk-Forward-Splits als innere Cross-Validation. Optuna sieht ausschließlich OOS-Metriken (medianer Sharpe Ratio), sodass kein Look-Ahead-Bias durch die Parametersuche entsteht. Die aktuellen Config-Defaults werden als Baseline-Trial (#0) eingespeist. Ergebnisse werden in einer SQLite-Datenbank unter `models/optuna_studies.db` persistiert, sodass Optimierungsläufe fortgesetzt werden können. Sensitivitäts-Heatmaps und Parameter-Importance-Plots werden automatisch unter `assets/` abgelegt.

### Realitätsnahe Kostensimulation
Die Simulation berücksichtigt reale Marktreibungen:
*   **Transaktionskosten:** Jede Umschichtung zwischen Portfolio und Cash wird mit einer Gebühr (0,1%) belegt.
*   **Liquiditätsgebühren:** Bei Entnahmen in investierten Marktphasen werden zusätzliche Verkaufsgebühren simuliert, während Entnahmen aus Cash-Beständen spesenfrei erfolgen.

### Automated Reporting (Live-Docs)
Die Datei `statistics.md` wird am Ende jedes Pipeline-Durchlaufs neu generiert. Hierbei werden Markdown-Tabellen und PNG-Assets direkt in das Dokument eingebettet, was eine lückenlose und stets aktuelle Dokumentation der Forschungsergebnisse ermöglicht.

### Interaktives Control Hub & Visualisierung
Der **Dashboard Service** (`:8004`) stellt ein modernes Zero-Build-Frontend (Tailwind + Plotly + HTMX + Alpine.js + Monaco) bereit. Es deckt vier Bereiche ab: (1) interaktive Visualisierung aller Pipeline-Artefakte inkl. vollständiger `statistics.md`-Abdeckung, (2) ein **Control Hub**, der alle FastAPI-Endpoints der drei Pipeline-Services per Klick aufruft (via `httpx`-Proxy mit langen Read-Timeouts für Walk-Forward-Läufe), (3) ein **YAML-Config-Editor** mit Server-Side-Validation, atomarem Swap und automatischem Rollback bei Reload-Fehlern, sowie (4) ein **Live-Log-Streaming** per WebSocket-File-Tail. Der Service bleibt an `127.0.0.1` gebunden (dev-only) und verändert außer `config/config.yaml` (mit `.bak`-Backup) nichts an den Pipeline-Artefakten.

---

## Quickstart

### Option A: Jupyter Notebooks (Forschung)

```bash
git clone https://github.com/Torim98/regime-switching-daa.git
cd regime-switching-daa

# Virtuelle Umgebung erstellen und aktivieren:
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Abhängigkeiten installieren:
pip install -e .

# Master-Notebook ausführen:
jupyter notebook jupyter/regime-switching-daa.ipynb
```

### Option B: Docker Compose (ein Befehl)

```bash
git clone https://github.com/Torim98/regime-switching-daa.git
cd regime-switching-daa
docker-compose up --build -d

# Variante 1 — Pipeline per curl ausführen:
curl -X POST http://localhost:8001/data/ingest
curl -X POST http://localhost:8002/models/optimize-all    # Optional: Hyperparameter-Optimierung
curl -X POST http://localhost:8002/models/train-all
curl -X POST http://localhost:8003/backtest/run
curl -X POST http://localhost:8003/backtest/evaluate

# Variante 2 — Interaktives Dashboard im Browser (empfohlen):
#   http://localhost:8004/        ← EDA, Backtest, Evaluation, MCS, Config-Editor, Live-Logs
#   http://localhost:8004/hub     ← Alle Pipeline-Endpoints per Klick
#
# Swagger UIs: http://localhost:8001/docs, :8002/docs, :8003/docs, :8004/docs
```

---

## Die Research-Pipeline (Modularer Aufbau)

Das Projekt ist als vollautomatisierte Pipeline konzipiert. Jedes Modul baut auf den persistierten Daten des Vorgängers auf:

1.  **`00_dependencies`**: Initialisierung der Forschungsumgebung.
2.  **`01_data_preprocessing`**: Download (YFinance) und Bereinigung von Multi-Asset-Daten (Aktien, Bonds, Cash).
3.  **`01a_label_analysis`** *(manuell, optional)*: Vergleich alternativer Regime-Labeler (Pagan-Sossounov, Peak-to-Trough, Lunde-Timmermann, NBER) gegen MSM/HMM. Erzeugt Konkordanz-Matrix und Switch-Statistiken. Dient als Begründung der Label-Wahl (Pagan-Sossounov) für LSTM und Transformer. Nicht Teil der automatischen Pipeline.
4.  **`02_feature_engineering`**: Berechnung technischer und makroökonomischer Indikatoren.
5.  **`03a_hyperparameter_optimization`** *(manuell)*: Bayessche Hyperparameter-Optimierung via Optuna. Nutzt Walk-Forward als innere CV. Nicht Teil der automatischen Pipeline. Wird einmalig vor dem finalen Durchlauf ausgeführt.
6.  **`03_regime_switching_models`**: Training der Regime-Switching-Modelle. Bei `walk_forward.enabled: false` klassischer 80/20-Split mit optionaler Modell-Persistierung. Bei `walk_forward.enabled: true` rollierende Walk-Forward-Validierung über alle Folds mit OOS-Caching.
7.  **`04_backtesting`**: Simulation realer Investitionsszenarien inkl. variabler Entnahmen und Transaktionskosten.
8.  **`05_evaluation`**: Stress-Tests mittels Block-Bootstrap zur statistischen Validierung der Ergebnisse.
9.  **`99_statistics_md`**: Automatisierte Zusammenführung aller Ergebnisse in die Dokumentation.

---

## Aktuelle Ergebnisse (Live-Update)

Die folgenden Grafiken werden automatisch generiert und repräsentieren den aktuellen Stand der Backtesting-Simulation auf dem S&P 500 / Long-Bond (60/40) Portfolio.

### 1. Performance-Vergleich (Equity Curves)
Vergleich der kumulierten Rendite zwischen der statischen "Buy & Hold"-Strategie und den aktiven Regime-Switching-Modellen.

![Equity Curves](./assets/equity_curves.png)

### 2. Regime-Erkennung im Detail
Visualisierung der berechneten Wahrscheinlichkeiten für ein Bärenmarkt-Regime über den Testzeitraum.

![Regime Comparison](./assets/regime_comparison.png)

### 3. Drawdown-Verlauf
Zeitlicher Verlauf der Drawdowns aller Strategien. Zeigt, wo Regime-Switching Krisen-Drawdowns reduziert.

![Drawdown](./assets/drawdown.png)

### 4. Zusammenfassung der Kennzahlen
| Kategorie | Kennzahlen | Relevanz für die Thesis |
| :--- | :--- | :--- |
| **Risikoschutz (SORR)** | **Max Drawdown, Calmar Ratio** | Die primäre Zielgröße. Misst die Effektivität der Verlustvermeidung und das Verhältnis von Rendite zu maximalem Drawdown. |
| **Risikoadjustierung** | **Sortino Ratio, Sharpe Ratio** | Bewertet, ob die Modelle eine Überrendite pro Risikoeinheit liefern. Die Sortino Ratio ist hierbei zentral, da sie gezielt das Abwärtsrisiko (Downside) betrachtet. |
| **Wachstumsdynamik** | **Total Return, CAGR (p.a.)** | Zeigt, ob die Modelle trotz der defensiven Ausrichtung in Bärenmärkten langfristig in der Lage sind, den Markt (Buy & Hold) zu schlagen. |
| **Modellstabilität** | **Regime-Wechsel, Volatilität** | Evaluiert die praktische Umsetzbarkeit. Eine hohe Anzahl an Wechseln ("Churning") deutet auf Instabilität und hohe Transaktionskosten hin. |

### 5. Risikoprofil (SORR Stress-Test)
Simulation einer Entnahmephase: Wie lange reicht das Kapital unter Berücksichtigung von Marktschocks und monatlichen Rentenzahlungen?

![SORR Standard](./assets/sorr_sim_standard.png)

### 6. Statistische Signifikanz (Monte-Carlo-Simulation)
Um die statistische Signifikanz zu prüfen, wurden 10.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.

![MCS Boxplots Standard](./assets/mcs_boxplot_standard.png)

Detaillierte statistische Auswertungen, Tabellen und Einzelauswertungen: **[statistics.md](./docs/statistics.md)**

---

## Projektstruktur

```
regime-switching-daa/
├── assets/          Generierte Grafiken und Statistiken (PNG, Markdown)
├── config/          Zentrale Konfiguration (config.yaml, config_loader.py)
├── data/            Medallion-Architektur (bronze/ silver/ gold/)
├── docs/            Projektdokumentation
├── jupyter/         Jupyter Notebooks (Pipeline 00–99)
├── logs/            Log-Dateien (Notebook-Pipeline + Services)
├── models/          Persistierte Modelldateien (.pkl, .keras, .pt) + Optuna DB
├── services/        FastAPI Microservices
│   ├── data_service/
│   ├── model_service/
│   ├── backtest_service/
│   └── dashboard_service/   Interaktives Frontend (UI, Control Hub, Config-Editor, Live-Logs)
├── src/             Shared Business Logic
│   ├── data/        Ingestion, Preprocessing, Feature Engineering, EDA, Plots
│   ├── models/      MSM, HMM, LSTM, Transformer, Plots
│   └── backtest/    Engine, Walk-Forward, Optimize, SORR, Evaluation, Reporting, Plots
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Dokumentation

| Dokument | Beschreibung |
|----------|-------------|
| [Repository Index](docs/index.md) | Navigations-Hub: kategorisierte Auflistung aller Repo-Dateien |
| [Data Architecture](docs/data-architecture.md) | Medallion-Modell (Bronze/Silver/Gold) |
| [Microservice Architecture](docs/microservice-architecture.md) | Services, Endpunkte, Volumes, Logging |
| [Dashboard Service](docs/dashboard-service.md) | UI-Seitenstruktur, Control Hub, Config-Editor, WebSocket-Log-Streaming, Security |
| [Sequence Diagram: Microservices](docs/microservice-sequence-diagram.md) | Mermaid-Sequenzdiagramm der Microservice-Pipeline |
| [Sequence Diagram: Jupyter](docs/jupyter-sequence-diagram.md) | Mermaid-Sequenzdiagramm der Notebook-Pipeline |
| [How to Add a ML Model](docs/how-to-add-ml-model.md) | Integrations-Anleitung für neue Modelle |
| [Statistics (Live)](docs/statistics.md) | Auto-generierte Ergebnisse und Tabellen |
| [FastAPI Endpoints](docs/fastapi-endpoints.md) | API-Routen und Parameter aller vier Services |

---

## Reproduzierbarkeit

Beide Pipelines (Notebook und Docker) erzeugen identische Ergebnisse für deterministische Modelle (MSM, HMM). LSTM und Transformer weichen durch nicht-deterministisches Training (zufällige Gewichtsinitialisierung, Batch-Shuffling) zwischen Läufen leicht ab — die Abweichungen kaskadieren in Backtesting, SORR und Monte Carlo Simulation.

---

## Limitations

Bestimmte Einflussfaktoren wurden bewusst aus dem Scope der Thesis ausgegrenzt. Da alle Modelle und die Benchmark identischen Rahmenbedingungen unterliegen, bleibt der relative Vergleich davon unberührt. Details und Begründungen: **[Limitations & Scope Boundaries](docs/limitations.md)**

---

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Details findest du in der Datei [LICENSE](./LICENSE).

**Autor:** Tom Maurer B.Sc.<br>
**Betreuer:** Prof. Dr. Christian Müller-Kett<br>
**Akademischer Kontext:** Master-Thesis Projekt im Bereich Quantitative Finance / Data Science.
