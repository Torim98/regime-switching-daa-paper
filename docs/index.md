# Repository Index

Navigations-Hub für das Repository: kategorisierte Auflistung aller versionierten Dateien mit einzeiliger Beschreibung. Pfade sind relativ zum Repo-Root.

> Artefakte, die bei Pipeline-Läufen entstehen (`data/`, `models/`, `logs/`), werden nur
> unter ihrem *Verzeichnis* gelistet. Die konkreten Dateinamen folgen der Pipeline-Stage
> und werden zur Laufzeit erzeugt.

---

## 1. Projekt-Meta & Dokumentation

| Datei | Beschreibung |
|-------|-------------|
| [README.md](../README.md) | Hauptübersicht: Motivation, Methodik, Architektur, Quickstart, Ergebnisse |
| [LICENSE](../LICENSE) | MIT-Lizenz |
| [docs/index.md](./index.md) | Dieses Dokument; Repository-Index |
| [docs/data-architecture.md](./data-architecture.md) | Medallion-Datenarchitektur (Bronze / Silver / Gold) |
| [docs/microservice-architecture.md](./microservice-architecture.md) | FastAPI-Services, Volumes, Logging, Deployment |
| [docs/microservice-sequence-diagram.md](./microservice-sequence-diagram.md) | Mermaid-Sequenzdiagramm der Microservice-Pipeline |
| [docs/dashboard-service.md](./dashboard-service.md) | Dashboard: Seitenstruktur, Control Hub, Config-Editor, WebSocket-Logs |
| [docs/fastapi-endpoints.md](./fastapi-endpoints.md) | API-Routen und Parameter aller vier Services |
| [docs/how-to-add-ml-model.md](./how-to-add-ml-model.md) | Schritt-für-Schritt-Integration eines neuen Modells |
| [docs/statistics.md](./statistics.md) | Auto-generierter Master-Report (Ergebnisse & Tabellen) |
| [docs/limitations.md](./limitations.md) | Bewusst ausgeklammerte Einflussfaktoren und Scope-Grenzen |

### GitHub-Meta

| Datei | Beschreibung |
|-------|-------------|
| [.github/ISSUE_TEMPLATE/bug_report.md](../.github/ISSUE_TEMPLATE/bug_report.md) | Issue-Template für Bug-Reports |
| [.github/ISSUE_TEMPLATE/feature_request.md](../.github/ISSUE_TEMPLATE/feature_request.md) | Issue-Template für Feature-Requests |
| [.github/ISSUE_TEMPLATE/task.md](../.github/ISSUE_TEMPLATE/task.md) | Issue-Template für Tasks |
| [.github/PULL_REQUEST_TEMPLATE.md](../.github/PULL_REQUEST_TEMPLATE.md) | Pull-Request-Template |

---

## 2. Konfiguration & Build

| Datei | Beschreibung |
|-------|-------------|
| [pyproject.toml](../pyproject.toml) | Python-Projekt-Metadaten, Abhängigkeiten, Packaging |
| [docker-compose.yml](../docker-compose.yml) | Orchestrierung der vier FastAPI-Services + Volumes |
| [.dockerignore](../.dockerignore) | Ausschlüsse für Docker-Build-Kontexte |
| [.editorconfig](../.editorconfig) | Editor-übergreifende Formatierungs-Defaults |
| [.gitattributes](../.gitattributes) | Git-Attribut-Regeln (u. a. Jupyter-Notebook-Diffs, LFS) |
| [.gitignore](../.gitignore) | Git-Ausschlüsse für Artefakte, Cache, Secrets |
| [config/config.yaml](../config/config.yaml) | Zentrale YAML-Konfiguration (Daten, Modelle, Backtest, Evaluation) |
| [config/config_loader.py](../config/config_loader.py) | Pydantic-basierter Loader + Validierung für `config.yaml` |

---

## 3. Shared Business Logic (`src/`)

Projekt-weit geteilter Python-Code. Die Microservices importieren aus `src/`.

| Datei | Beschreibung |
|-------|-------------|
| [src/\_\_init\_\_.py](../src/__init__.py) | Package-Marker |

### 3.1 Daten-Pipeline (`src/data/`)

| Datei | Beschreibung |
|-------|-------------|
| [src/data/\_\_init\_\_.py](../src/data/__init__.py) | Package-Marker |
| [src/data/ingestion.py](../src/data/ingestion.py) | Yahoo-Finance-Download und Rohdaten-Persistierung (Bronze Layer) |
| [src/data/preprocessing.py](../src/data/preprocessing.py) | Portfolio-Konstruktion, Renditeberechnung, Datenbereinigung |
| [src/data/feature_engineering.py](../src/data/feature_engineering.py) | Rolling-Window-Features für Regime-Erkennung |
| [src/data/eda.py](../src/data/eda.py) | Deskriptive Statistik, ADF-Stationaritätstests |
| [src/data/plots.py](../src/data/plots.py) | EDA- und Preprocessing-Plots (Volatilitäts-Cluster, Drawdowns etc.) |

### 3.2 Regime-Labeling (`src/data/labels/`)

| Datei | Beschreibung |
|-------|-------------|
| [src/data/labels/\_\_init\_\_.py](../src/data/labels/__init__.py) | Package-Marker |
| [src/data/labels/peak_to_trough.py](../src/data/labels/peak_to_trough.py) | Klassische 20%-Regel für Bullen-/Bären-Phasen |
| [src/data/labels/pagan_sossounov.py](../src/data/labels/pagan_sossounov.py) | Pagan-Sossounov (2003) Bull/Bear-Market-Labeling |
| [src/data/labels/lunde_timmermann.py](../src/data/labels/lunde_timmermann.py) | Lunde-Timmermann (2004) Duration-Dependence-Labeling |
| [src/data/labels/nber.py](../src/data/labels/nber.py) | NBER-Rezessionsdaten via FRED (USREC-Serie) |
| [src/data/labels/concordance.py](../src/data/labels/concordance.py) | Concordance-Analyse und Timeline-Visualisierung der Label-Schemata |
| [src/data/labels/resolver.py](../src/data/labels/resolver.py) | Zentrale Auflösung der Supervised-Label-Quelle aus der Config |
| [src/data/labels/test_labels.py](../src/data/labels/test_labels.py) | Unit-Tests für Label-Schemata |

### 3.3 Modelle (`src/models/`)

| Datei | Beschreibung |
|-------|-------------|
| [src/models/\_\_init\_\_.py](../src/models/__init__.py) | Package-Marker |
| [src/models/common.py](../src/models/common.py) | Gemeinsame Modell-Hilfsfunktionen und Konstanten |
| [src/models/msm.py](../src/models/msm.py) | Markov-Switching-Modell (statsmodels) |
| [src/models/hmm.py](../src/models/hmm.py) | Hidden Markov Model (hmmlearn, Gaussian-Emissions) |
| [src/models/lstm.py](../src/models/lstm.py) | LSTM-Netzwerk; Supervised Regime Classification (TF/Keras) |
| [src/models/transformer.py](../src/models/transformer.py) | Transformer; Multi-Head Self-Attention Regime-Detection (PyTorch) |
| [src/models/plots.py](../src/models/plots.py) | Regime-Visualisierung je Modell |

### 3.4 Backtesting & Evaluation (`src/backtest/`)

| Datei | Beschreibung |
|-------|-------------|
| [src/backtest/\_\_init\_\_.py](../src/backtest/__init__.py) | Package-Marker |
| [src/backtest/engine.py](../src/backtest/engine.py) | Backtesting-Engine mit Transaktionskosten |
| [src/backtest/walk_forward.py](../src/backtest/walk_forward.py) | Walk-Forward-Splitter und OOS-Helper |
| [src/backtest/parallel.py](../src/backtest/parallel.py) | Parallele Fold-Ausführung (joblib) für CPU-bound Modelle |
| [src/backtest/optimize.py](../src/backtest/optimize.py) | Optuna Hyperparameter-Optimierung mit Walk-Forward-CV |
| [src/backtest/evaluation.py](../src/backtest/evaluation.py) | Performance-Metriken, MCS, H1/H2-Tests, Churning, Time-to-Recovery |
| [src/backtest/sorr.py](../src/backtest/sorr.py) | Sequence-of-Returns-Risk Simulation der Entnahmephase |
| [src/backtest/reporting.py](../src/backtest/reporting.py) | Generierung des `statistics.md` Master-Reports |
| [src/backtest/plots.py](../src/backtest/plots.py) | Backtest- und SORR-Visualisierungen |

---

## 4. Microservices (`services/`)

Vier containerisierte FastAPI-Services + gemeinsame Infrastruktur.

### 4.1 Gemeinsame Infrastruktur

| Datei | Beschreibung |
|-------|-------------|
| [services/\_\_init\_\_.py](../services/__init__.py) | Package-Marker |
| [services/logging_config.py](../services/logging_config.py) | Zentrales Logging-Setup für alle Services |
| [services/warnings_config.py](../services/warnings_config.py) | Globale Warning-Suppression (statsmodels, Keras, TF) |

### 4.2 Data Service (Port 8001)

| Datei | Beschreibung |
|-------|-------------|
| [services/data_service/\_\_init\_\_.py](../services/data_service/__init__.py) | Package-Marker |
| [services/data_service/main.py](../services/data_service/main.py) | FastAPI-App-Entry: Ingestion, Preprocessing, Feature-Eng., EDA |
| [services/data_service/routes.py](../services/data_service/routes.py) | HTTP-Routen des Data Service |
| [services/data_service/Dockerfile](../services/data_service/Dockerfile) | Container-Image-Definition Data Service |

### 4.3 Model Service (Port 8002)

| Datei | Beschreibung |
|-------|-------------|
| [services/model_service/\_\_init\_\_.py](../services/model_service/__init__.py) | Package-Marker |
| [services/model_service/main.py](../services/model_service/main.py) | FastAPI-App-Entry: Training & Inference (MSM/HMM/LSTM/Transformer) |
| [services/model_service/routes.py](../services/model_service/routes.py) | HTTP-Routen des Model Service |
| [services/model_service/Dockerfile](../services/model_service/Dockerfile) | Container-Image-Definition Model Service |
| [services/model_service/tests/test_walk_forward_snapshot.py](../services/model_service/tests/test_walk_forward_snapshot.py) | Snapshot-Test der Walk-Forward-Ergebnisse |

### 4.4 Backtest Service (Port 8003)

| Datei | Beschreibung |
|-------|-------------|
| [services/backtest_service/\_\_init\_\_.py](../services/backtest_service/__init__.py) | Package-Marker |
| [services/backtest_service/main.py](../services/backtest_service/main.py) | FastAPI-App-Entry: Backtesting, SORR, MCS, Reporting |
| [services/backtest_service/routes.py](../services/backtest_service/routes.py) | HTTP-Routen des Backtest Service |
| [services/backtest_service/Dockerfile](../services/backtest_service/Dockerfile) | Container-Image-Definition Backtest Service |

### 4.5 Dashboard Service (Port 8004)

Interaktives UI, Control Hub, Config-Editor, Live-Log-Streaming.

| Datei | Beschreibung |
|-------|-------------|
| [services/dashboard_service/\_\_init\_\_.py](../services/dashboard_service/__init__.py) | Package-Marker |
| [services/dashboard_service/main.py](../services/dashboard_service/main.py) | FastAPI-App-Entry: UI + Control Hub |
| [services/dashboard_service/routes.py](../services/dashboard_service/routes.py) | HTML-Seiten-Routen |
| [services/dashboard_service/data_adapters.py](../services/dashboard_service/data_adapters.py) | Parquet/MD → Plotly-JSON-Adapter für die UI |
| [services/dashboard_service/config_api.py](../services/dashboard_service/config_api.py) | Config-Editor-API (Read/Write `config.yaml`) |
| [services/dashboard_service/hub_api.py](../services/dashboard_service/hub_api.py) | Control-Hub-Proxy zu data/model/backtest-Services via httpx |
| [services/dashboard_service/websockets.py](../services/dashboard_service/websockets.py) | WebSocket-Tailing der `logs/*.log`-Dateien |
| [services/dashboard_service/Dockerfile](../services/dashboard_service/Dockerfile) | Container-Image-Definition Dashboard Service |

#### Templates

| Datei | Beschreibung |
|-------|-------------|
| [services/dashboard_service/templates/base.html](../services/dashboard_service/templates/base.html) | Basis-Layout (Header, Sidebar, Footer) |
| [services/dashboard_service/templates/index.html](../services/dashboard_service/templates/index.html) | Startseite/Übersicht |
| [services/dashboard_service/templates/hub.html](../services/dashboard_service/templates/hub.html) | Control Hub zum Auslösen von Pipeline-Stages |
| [services/dashboard_service/templates/config_editor.html](../services/dashboard_service/templates/config_editor.html) | YAML-Config-Editor |
| [services/dashboard_service/templates/eda.html](../services/dashboard_service/templates/eda.html) | EDA-Visualisierungen |
| [services/dashboard_service/templates/models.html](../services/dashboard_service/templates/models.html) | Modell-Vergleich (Regime-Klassifikation, Metriken) |
| [services/dashboard_service/templates/backtest.html](../services/dashboard_service/templates/backtest.html) | Backtest- und SORR-Ergebnisse |
| [services/dashboard_service/templates/evaluation.html](../services/dashboard_service/templates/evaluation.html) | Evaluation: Confusion-Matrizen, ROC, H1/H2, MCS |
| [services/dashboard_service/templates/logs.html](../services/dashboard_service/templates/logs.html) | Live-Log-Viewer (WebSocket-gestreamt) |

#### Static Assets

| Datei | Beschreibung |
|-------|-------------|
| [services/dashboard_service/static/css/dashboard.css](../services/dashboard_service/static/css/dashboard.css) | Dashboard-Styles |
| [services/dashboard_service/static/js/common.js](../services/dashboard_service/static/js/common.js) | Geteilte Client-seitige Logik (Plotly-Renderer, Fetch-Helfer) |

---

## 5. Explorative Jupyter Notebooks (`jupyter/`)

| Notebook | Beschreibung |
|----------|-------------|
| [jupyter/Asymmetric_correlation_Ang_Chen.ipynb](../jupyter/Asymmetric_correlation_Ang_Chen.ipynb) | Asymmetrische Korrelation nach Ang & Chen |
| [jupyter/Concept_matrix_venn.ipynb](../jupyter/Concept_matrix_venn.ipynb) | Konzept-Venn-Diagramm für die Thesis |
| [jupyter/S&P500_NBER-recessions.ipynb](../jupyter/S&P500_NBER-recessions.ipynb) | S&P-500-Verlauf mit NBER-Rezessionsperioden |
| [jupyter/SORR.ipynb](../jupyter/SORR.ipynb) | Explorative SORR-Illustration |

---

## 6. Daten-Artefakte (`data/` — Medallion)

Details siehe [docs/data-architecture.md](./data-architecture.md).

| Layer | Inhalt |
|-------|--------|
| [data/bronze/](../data/bronze/) | Rohdaten — `01_raw_data.parquet` (Yahoo-Finance-Download) |
| [data/silver/](../data/silver/) | Bereinigte & feature-engineerte Daten (`02_preprocessed`, `03_feature_engineered`, `04_test_df`, `wf_cache`) |
| [data/gold/](../data/gold/) | Ergebnisartefakte für Reporting (Backtest-Results, Transaktionskosten, SORR-Simulation, MCS) |

---

## 7. Modelle (`models/`)

Artefakte aus dem Training — versioniert via Git LFS bzw. zur Laufzeit erzeugt.

| Datei / Pattern | Beschreibung |
|-----------------|-------------|
| `models/optuna_studies.db` | Persistente Optuna-Studies (SQLite) |
| `models/msm_regime_model.pkl` | Trainiertes MSM-Gewicht |
| `models/hmm_regime_model.pkl` | Trainiertes HMM-Gewicht |
| `models/hmm_scaler.pkl` | Feature-Scaler zum HMM |
| `models/lstm_regime_model.keras` | Trainiertes LSTM-Gewicht |
| `models/tlstm_scaler.pkl` | Feature-Scaler zum LSTM-Modell |
| `models/transformer_regime_model.pt` | Trainiertes Transformer-Gewicht |
| `models/transformer_scaler.pkl` | Feature-Scaler zum Transformer |

---

## 8. Logs (`logs/`)

Laufzeit-Logs; werden bei jeder Pipeline-/Service-Ausführung neu geschrieben.

| Datei-Pattern | Quelle |
|---------------|--------|
| `logs/data_service.log` | FastAPI Data Service |
| `logs/model_service.log` | FastAPI Model Service |
| `logs/backtest_service.log` | FastAPI Backtest Service |
| `logs/dashboard_service.log` | FastAPI Dashboard Service |

---

## 9. Assets (`assets/`)

Von den Notebooks bzw. Services generierte Plots und Markdown-Tabellen.
Jedes Artefakt wird in `docs/statistics.md` und/oder im Dashboard eingebettet.

### 9.1 EDA & Preprocessing

| Datei | Beschreibung |
|-------|-------------|
| [assets/SORR_schema.png](../assets/SORR_schema.png) | SORR-Schema |
| [assets/eda_descriptive_stats.md](../assets/eda_descriptive_stats.md) | Deskriptive Statistik der Inputs |
| [assets/eda_adf_tests.md](../assets/eda_adf_tests.md) | ADF-Stationaritätstests |
| [assets/eda_historical_drawdowns.png](../assets/eda_historical_drawdowns.png) | Historische Drawdowns des Aktien-Portfolios |
| [assets/eda_volatility_clusters.png](../assets/eda_volatility_clusters.png) | Volatilitäts-Cluster (ARCH-Effekte) |
| [assets/feature_correlation_matrix.png](../assets/feature_correlation_matrix.png) | Feature-Korrelationsmatrix (Plot) |
| [assets/feature_correlation_table.md](../assets/feature_correlation_table.md) | Feature-Korrelationsmatrix (Tabelle) |
| [assets/concept_matrix_venn.png](../assets/concept_matrix_venn.png) | Konzept-Venn-Diagramm (Thesis) |
| [assets/asymmetric_correlation_ang_chen.png](../assets/asymmetric_correlation_ang_chen.png) | Asymmetrische Korrelation (Ang & Chen) |
| [assets/data_quality_report.md](../assets/data_quality_report.md) | Data-Quality-Report |

### 9.2 Label-Analyse

| Datei | Beschreibung |
|-------|-------------|
| [assets/label_timeline_comparison.png](../assets/label_timeline_comparison.png) | Timeline-Vergleich Label-Schemata |
| [assets/label_concordance_matrix.png](../assets/label_concordance_matrix.png) | Concordance-Matrix der Label-Schemata |
| [assets/label_kappa_matrix.png](../assets/label_kappa_matrix.png) | Cohen's-κ-Matrix der Label-Schemata |

### 9.3 Modell-Visualisierungen

| Datei | Beschreibung |
|-------|-------------|
| [assets/walk_forward_schema.png](../assets/walk_forward_schema.png) | Walk-Forward Schema |
| [assets/hmm_regimes.png](../assets/hmm_regimes.png) | Von HMM identifizierte Regimes |
| [assets/hmm_uni_regimes.png](../assets/hmm_uni_regimes.png) | Von HMM (univariat) identifizierte Regimes |
| [assets/msm_regimes.png](../assets/msm_regimes.png) | Von MSM identifizierte Regimes |
| [assets/lstm_model.png](../assets/lstm_model.png) | LSTM-Architekturskizze |
| [assets/transformer_model.png](../assets/transformer_model.png) | Transformer-Architekturskizze |
| [assets/regime_comparison.png](../assets/regime_comparison.png) | Modell-übergreifender Regime-Vergleich |
| [assets/regime_probability_heatmap.png](../assets/regime_probability_heatmap.png) | Regime-Wahrscheinlichkeiten als Heatmap |

### 9.4 Optuna-Hyperparameter-Optimierung

Je Modell vier Standard-Plots (History / Importance / Contour / Slice):

| Datei | Beschreibung |
|-------|-------------|
| [assets/optuna_MSM_history.png](../assets/optuna_MSM_history.png) | MSM — Optuna Trial-History |
| [assets/optuna_MSM_importance.png](../assets/optuna_MSM_importance.png) | MSM — Parameter-Importance |
| [assets/optuna_MSM_slice.png](../assets/optuna_MSM_slice.png) | MSM — Parameter-Slice |
| [assets/optuna_HMM_history.png](../assets/optuna_HMM_history.png) | HMM — Optuna Trial-History |
| [assets/optuna_HMM_importance.png](../assets/optuna_HMM_importance.png) | HMM — Parameter-Importance |
| [assets/optuna_HMM_contour.png](../assets/optuna_HMM_contour.png) | HMM — Parameter-Contour |
| [assets/optuna_HMM_slice.png](../assets/optuna_HMM_slice.png) | HMM — Parameter-Slice |
| [assets/optuna_LSTM_history.png](../assets/optuna_LSTM_history.png) | LSTM — Optuna Trial-History |
| [assets/optuna_LSTM_importance.png](../assets/optuna_LSTM_importance.png) | LSTM — Parameter-Importance |
| [assets/optuna_LSTM_contour.png](../assets/optuna_LSTM_contour.png) | LSTM — Parameter-Contour |
| [assets/optuna_LSTM_slice.png](../assets/optuna_LSTM_slice.png) | LSTM — Parameter-Slice |
| [assets/optuna_Transformer_history.png](../assets/optuna_Transformer_history.png) | Transformer — Optuna Trial-History |
| [assets/optuna_Transformer_importance.png](../assets/optuna_Transformer_importance.png) | Transformer — Parameter-Importance |
| [assets/optuna_Transformer_contour.png](../assets/optuna_Transformer_contour.png) | Transformer — Parameter-Contour |
| [assets/optuna_Transformer_slice.png](../assets/optuna_Transformer_slice.png) | Transformer — Parameter-Slice |
| [assets/optuna_importance_values.json](../assets/optuna_importance_values.json) | fANOVA-Importance-Cache (Dashboard liest daraus, um 1:1 mit PNG zu matchen. fANOVA ist stochastisch und würde bei Re-Computation abweichen) |

### 9.5 Backtest & Performance

| Datei | Beschreibung |
|-------|-------------|
| [assets/equity_curves.png](../assets/equity_curves.png) | Equity-Kurven aller Strategien |
| [assets/capital_curve.png](../assets/capital_curve.png) | Kapitalkurve (aggregiert) |
| [assets/drawdown.png](../assets/drawdown.png) | Drawdown-Pfade |
| [assets/rolling_sharpe.png](../assets/rolling_sharpe.png) | Rolling Sharpe Ratio |
| [assets/transaction_costs.png](../assets/transaction_costs.png) | Transaktionskosten je Strategie |
| [assets/annualized_metrics.md](../assets/annualized_metrics.md) | Annualisierte Performance-Metriken |
| [assets/performance_summary.md](../assets/performance_summary.md) | Gesamt-Performance-Zusammenfassung |
| [assets/crisis_performance.md](../assets/crisis_performance.md) | Performance während Krisenphasen |
| [assets/break_even_costs.md](../assets/break_even_costs.md) | Break-Even-Transaktionskosten (Tabelle) |
| [assets/break_even_costs.png](../assets/break_even_costs.png) | Break-Even-Transaktionskosten (Plot) |

### 9.6 Evaluation & Klassifikation

| Datei | Beschreibung |
|-------|-------------|
| [assets/classification_metrics.md](../assets/classification_metrics.md) | Klassifikations-Metriken je Modell |
| [assets/confusion_matrices.png](../assets/confusion_matrices.png) | Confusion-Matrizen der Modelle |
| [assets/roc_curves.png](../assets/roc_curves.png) | ROC-Kurven |
| [assets/pr_curves.png](../assets/pr_curves.png) | Precision-Recall-Kurven |
| [assets/evaluation_table.md](../assets/evaluation_table.md) | Gesamt-Evaluationstabelle |
| [assets/churning_stats.md](../assets/churning_stats.md) | Whipsaw- / Churning-Statistiken |
| [assets/switch_timing.md](../assets/switch_timing.md) | Timing der Regime-Wechsel |

#### Threshold-Sensitivität

| Datei | Beschreibung |
|-------|-------------|
| [assets/threshold_sensitivity_MSM.md](../assets/threshold_sensitivity_MSM.md) | MSM — Threshold-Sensitivität |
| [assets/threshold_sensitivity_HMM.md](../assets/threshold_sensitivity_HMM.md) | HMM — Threshold-Sensitivität |
| [assets/threshold_sensitivity_HMM_Uni.md](../assets/threshold_sensitivity_HMM_Uni.md) | HMM (univariat) — Threshold-Sensitivität |
| [assets/threshold_sensitivity_LSTM.md](../assets/threshold_sensitivity_LSTM.md) | LSTM — Threshold-Sensitivität |
| [assets/threshold_sensitivity_Transformer.md](../assets/threshold_sensitivity_Transformer.md) | Transformer — Threshold-Sensitivität |

#### Time-to-Recovery

| Datei | Beschreibung |
|-------|-------------|
| [assets/time_to_recovery_Buy_Hold.md](../assets/time_to_recovery_Buy_Hold.md) | Buy-&-Hold — Time-to-Recovery |
| [assets/time_to_recovery_MSM.md](../assets/time_to_recovery_MSM.md) | MSM — Time-to-Recovery |
| [assets/time_to_recovery_HMM.md](../assets/time_to_recovery_HMM.md) | HMM — Time-to-Recovery |
| [assets/time_to_recovery_HMM_Uni.md](../assets/time_to_recovery_HMM_Uni.md) | HMM (univariat) — Time-to-Recovery |
| [assets/time_to_recovery_LSTM.md](../assets/time_to_recovery_LSTM.md) | LSTM — Time-to-Recovery |
| [assets/time_to_recovery_Transformer.md](../assets/time_to_recovery_Transformer.md) | Transformer — Time-to-Recovery |

### 9.7 Hypothesentests

| Datei | Beschreibung |
|-------|-------------|
| [assets/h1_drawdown_test.md](../assets/h1_drawdown_test.md) | H1 — Drawdown-Reduktion (signifikant?) |
| [assets/mcs_h1_mdd_forest.png](../assets/mcs_h1_mdd_forest.png) | H1 - Forest-Plot H1/MDD |
| [assets/h2_transformer_test.md](../assets/h2_transformer_test.md) | H2 — Transformer-Überlegenheit (signifikant?) |
| [assets/mcs_h2_endkapital_forest.png](../assets/mcs_h2_endkapital_forest.png) | H2 — Forest-Plot H2/Endkapital |

### 9.8 SORR- und Monte-Carlo-Simulation

| Datei | Beschreibung |
|-------|-------------|
| [assets/sorr_summary.md](../assets/sorr_summary.md) | SORR-Ergebnisübersicht |
| [assets/sorr_sim_standard.png](../assets/sorr_sim_standard.png) | SORR-Simulation — Standard-Entnahme |
| [assets/sorr_sim_aggressive.png](../assets/sorr_sim_aggressive.png) | SORR-Simulation — aggressive Entnahme |
| [assets/sorr_sim_low_capital.png](../assets/sorr_sim_low_capital.png) | SORR-Simulation — niedriges Startkapital |
| [assets/withdrawal_sensitivity.md](../assets/withdrawal_sensitivity.md) | Sensitivität gegenüber Entnahmeraten |
| [assets/depletion_rate_ci.md](../assets/depletion_rate_ci.md) | Depletion-Rate mit Konfidenzintervall |
| [assets/mcs_depletion_rate_forest.png](../assets/mcs_depletion_rate_forest.png) | Forest-Plot Depletion Rate
| [assets/mcs_summary.md](../assets/mcs_summary.md) | Monte-Carlo-Simulation — Zusammenfassung |
| [assets/mcs_paths.png](../assets/mcs_paths.png) | MCS-Pfade |
| [assets/mcs_quantiles.png](../assets/mcs_quantiles.png) | MCS-Quantile |
| [assets/mcs_boxplot_standard.png](../assets/mcs_boxplot_standard.png) | MCS-Boxplot — Standard-Entnahme |
| [assets/mcs_boxplot_aggressive.png](../assets/mcs_boxplot_aggressive.png) | MCS-Boxplot — aggressive Entnahme |
| [assets/mcs_boxplot_low_capital.png](../assets/mcs_boxplot_low_capital.png) | MCS-Boxplot — niedriges Startkapital |
| [assets/mcs_violin_standard.png](../assets/mcs_violin_standard.png) | MCS-Violin — Standard-Entnahme |
| [assets/mcs_violin_aggressive.png](../assets/mcs_violin_aggressive.png) | MCS-Violin — aggressive Entnahme |
| [assets/mcs_violin_low_capital.png](../assets/mcs_violin_low_capital.png) | MCS-Violin — niedriges Startkapital |
| [assets/risk_return_positioning.png](../assets/risk_return_positioning.png) | Risiko-Rendite-Positionierung |

### 9.9 Dashboard-Screenshots

| Datei | Beschreibung |
|-------|-------------|
| [assets/screenshots/dashboard_overview.png](../assets/screenshots/dashboard_overview.png) | Dashboard — Übersichtsseite |
| [assets/screenshots/dashboard_control-hub.png](../assets/screenshots/dashboard_control-hub.png) | Dashboard — Control Hub |
| [assets/screenshots/dashboard_config.png](../assets/screenshots/dashboard_config.png) | Dashboard — Config-Editor |
| [assets/screenshots/dashboard_evaluation.png](../assets/screenshots/dashboard_evaluation.png) | Dashboard — Evaluation-Seite |

---

## Pflege

Dieser Index wird manuell gepflegt. Bei Strukturänderungen bitte hier nachziehen
(oder künftig per Skript aus `git ls-files` generieren; siehe Issue #7,
*Additional Notes*).