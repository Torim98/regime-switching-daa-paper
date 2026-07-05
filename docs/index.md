# Repository Index

Navigation hub for the repository: categorized listing of all versioned files with a one-line description. Paths are relative to the repo root.

> Artifacts created during pipeline runs (`data/`, `models/`, `logs/`) are listed only
> by their *directory*. The concrete file names follow the pipeline stage and are
> created at runtime.

---

## 1. Project Meta & Documentation

| File | Description |
|-------|-------------|
| [README.md](../README.md) | Main overview: motivation, methodology, architecture, quickstart, results |
| [LICENSE](../LICENSE) | MIT license |
| [docs/index.md](./index.md) | This document; repository index |
| [docs/data-architecture.md](./data-architecture.md) | Medallion data architecture (Bronze / Silver / Gold) |
| [docs/microservice-architecture.md](./microservice-architecture.md) | FastAPI services, volumes, logging, deployment |
| [docs/microservice-sequence-diagram.md](./microservice-sequence-diagram.md) | Mermaid sequence diagram of the microservice pipeline |
| [docs/dashboard-service.md](./dashboard-service.md) | Dashboard: page structure, control hub, config editor, WebSocket logs |
| [docs/fastapi-endpoints.md](./fastapi-endpoints.md) | API routes and parameters of all four services |
| [docs/how-to-add-ml-model.md](./how-to-add-ml-model.md) | Step-by-step integration of a new model |
| [docs/hyperparameter-optimization.md](./hyperparameter-optimization.md) | Optuna HPO: objective, search spaces, samplers, selection vs. evaluation (Issue #5) |
| [docs/statistics.md](./statistics.md) | Auto-generated master report (results & tables) |
| [docs/limitations.md](./limitations.md) | Deliberately excluded influencing factors and scope boundaries |

### GitHub Meta

| File | Description |
|-------|-------------|
| [.github/ISSUE_TEMPLATE/bug_report.md](../.github/ISSUE_TEMPLATE/bug_report.md) | Issue template for bug reports |
| [.github/ISSUE_TEMPLATE/feature_request.md](../.github/ISSUE_TEMPLATE/feature_request.md) | Issue template for feature requests |
| [.github/ISSUE_TEMPLATE/task.md](../.github/ISSUE_TEMPLATE/task.md) | Issue template for tasks |
| [.github/PULL_REQUEST_TEMPLATE.md](../.github/PULL_REQUEST_TEMPLATE.md) | Pull request template |

---

## 2. Configuration & Build

| File | Description |
|-------|-------------|
| [pyproject.toml](../pyproject.toml) | Python project metadata, dependencies, packaging |
| [docker-compose.yml](../docker-compose.yml) | Orchestration of the four FastAPI services + volumes |
| [.dockerignore](../.dockerignore) | Exclusions for Docker build contexts |
| [.editorconfig](../.editorconfig) | Cross-editor formatting defaults |
| [.gitattributes](../.gitattributes) | Git attribute rules (incl. Jupyter notebook diffs, LFS) |
| [.gitignore](../.gitignore) | Git exclusions for artifacts, cache, secrets |
| [config/config.yaml](../config/config.yaml) | Central YAML configuration (data, models, backtest, evaluation) |
| [config/config_loader.py](../config/config_loader.py) | Pydantic-based loader + validation for `config.yaml` |

---

## 3. Shared Business Logic (`src/`)

Project-wide shared Python code. The microservices import from `src/`.

| File | Description |
|-------|-------------|
| [src/\_\_init\_\_.py](../src/__init__.py) | Package marker |

### 3.1 Data Pipeline (`src/data/`)

| File | Description |
|-------|-------------|
| [src/data/\_\_init\_\_.py](../src/data/__init__.py) | Package marker |
| [src/data/ingestion.py](../src/data/ingestion.py) | Yahoo Finance download and raw-data persistence (Bronze layer) |
| [src/data/preprocessing.py](../src/data/preprocessing.py) | Portfolio construction, return calculation, data cleaning |
| [src/data/feature_engineering.py](../src/data/feature_engineering.py) | Rolling-window features for regime detection |
| [src/data/eda.py](../src/data/eda.py) | Descriptive statistics, ADF stationarity tests |
| [src/data/plots.py](../src/data/plots.py) | EDA and preprocessing plots (volatility clusters, drawdowns, etc.) |

### 3.2 Regime Labeling (`src/data/labels/`)

| File | Description |
|-------|-------------|
| [src/data/labels/\_\_init\_\_.py](../src/data/labels/__init__.py) | Package marker |
| [src/data/labels/peak_to_trough.py](../src/data/labels/peak_to_trough.py) | Classic 20% rule for bull/bear phases |
| [src/data/labels/pagan_sossounov.py](../src/data/labels/pagan_sossounov.py) | Pagan-Sossounov (2003) bull/bear market labeling |
| [src/data/labels/lunde_timmermann.py](../src/data/labels/lunde_timmermann.py) | Lunde-Timmermann (2004) duration-dependence labeling |
| [src/data/labels/nber.py](../src/data/labels/nber.py) | NBER recession data via FRED (USREC series) |
| [src/data/labels/concordance.py](../src/data/labels/concordance.py) | Concordance analysis and timeline visualization of the label schemes |
| [src/data/labels/resolver.py](../src/data/labels/resolver.py) | Central resolution of the supervised label source from the config |
| [src/data/labels/test_labels.py](../src/data/labels/test_labels.py) | Unit tests for label schemes |

### 3.3 Models (`src/models/`)

| File | Description |
|-------|-------------|
| [src/models/\_\_init\_\_.py](../src/models/__init__.py) | Package marker |
| [src/models/common.py](../src/models/common.py) | Shared model helper functions and constants |
| [src/models/msm.py](../src/models/msm.py) | Markov-switching model (statsmodels) |
| [src/models/hmm.py](../src/models/hmm.py) | Hidden Markov model (hmmlearn, Gaussian emissions) |
| [src/models/lstm.py](../src/models/lstm.py) | LSTM network; supervised regime classification (TF/Keras) |
| [src/models/transformer.py](../src/models/transformer.py) | Transformer; multi-head self-attention regime detection (PyTorch) |
| [src/models/plots.py](../src/models/plots.py) | Regime visualization per model |

### 3.4 Backtesting & Evaluation (`src/backtest/`)

| File | Description |
|-------|-------------|
| [src/backtest/\_\_init\_\_.py](../src/backtest/__init__.py) | Package marker |
| [src/backtest/engine.py](../src/backtest/engine.py) | Backtesting engine with transaction costs |
| [src/backtest/walk_forward.py](../src/backtest/walk_forward.py) | Walk-forward splitter and OOS helpers |
| [src/backtest/bear_coverage.py](../src/backtest/bear_coverage.py) | Per-fold bear-market coverage diagnostic of the walk-forward windows (Issue #8) |
| [src/backtest/parallel.py](../src/backtest/parallel.py) | Parallel fold execution (joblib) for CPU-bound models |
| [src/backtest/optimize.py](../src/backtest/optimize.py) | Optuna hyperparameter optimization with walk-forward CV (grid/TPE, pooled-OOS objective) |
| [src/backtest/hpo_analysis.py](../src/backtest/hpo_analysis.py) | Post-HPO analysis: convergence review, objective sensitivity, DSR, PBO, multi-seed reeval, best-param transfer (Issue #5) |
| [src/backtest/evaluation.py](../src/backtest/evaluation.py) | Performance metrics, MCS, H1/H2 tests, churning, time-to-recovery |
| [src/backtest/sorr.py](../src/backtest/sorr.py) | Sequence-of-returns-risk simulation of the withdrawal phase |
| [src/backtest/reporting.py](../src/backtest/reporting.py) | Generation of the `statistics.md` master report |
| [src/backtest/plots.py](../src/backtest/plots.py) | Backtest and SORR visualizations |

---

## 4. Microservices (`services/`)

Four containerized FastAPI services + shared infrastructure.

### 4.1 Shared Infrastructure

| File | Description |
|-------|-------------|
| [services/\_\_init\_\_.py](../services/__init__.py) | Package marker |
| [services/logging_config.py](../services/logging_config.py) | Central logging setup for all services |
| [services/warnings_config.py](../services/warnings_config.py) | Global warning suppression (statsmodels, Keras, TF) |

### 4.2 Data Service (Port 8001)

| File | Description |
|-------|-------------|
| [services/data_service/\_\_init\_\_.py](../services/data_service/__init__.py) | Package marker |
| [services/data_service/main.py](../services/data_service/main.py) | FastAPI app entry: ingestion, preprocessing, feature engineering, EDA |
| [services/data_service/routes.py](../services/data_service/routes.py) | HTTP routes of the Data Service |
| [services/data_service/Dockerfile](../services/data_service/Dockerfile) | Container image definition, Data Service |

### 4.3 Model Service (Port 8002)

| File | Description |
|-------|-------------|
| [services/model_service/\_\_init\_\_.py](../services/model_service/__init__.py) | Package marker |
| [services/model_service/main.py](../services/model_service/main.py) | FastAPI app entry: training & inference (MSM/HMM/LSTM/Transformer) |
| [services/model_service/routes.py](../services/model_service/routes.py) | HTTP routes of the Model Service |
| [services/model_service/Dockerfile](../services/model_service/Dockerfile) | Container image definition, Model Service |
| [services/model_service/tests/test_walk_forward_snapshot.py](../services/model_service/tests/test_walk_forward_snapshot.py) | Snapshot test of the walk-forward results |

### 4.4 Backtest Service (Port 8003)

| File | Description |
|-------|-------------|
| [services/backtest_service/\_\_init\_\_.py](../services/backtest_service/__init__.py) | Package marker |
| [services/backtest_service/main.py](../services/backtest_service/main.py) | FastAPI app entry: backtesting, SORR, MCS, reporting |
| [services/backtest_service/routes.py](../services/backtest_service/routes.py) | HTTP routes of the Backtest Service |
| [services/backtest_service/Dockerfile](../services/backtest_service/Dockerfile) | Container image definition, Backtest Service |

### 4.5 Dashboard Service (Port 8004)

Interactive UI, control hub, config editor, live log streaming.

| File | Description |
|-------|-------------|
| [services/dashboard_service/\_\_init\_\_.py](../services/dashboard_service/__init__.py) | Package marker |
| [services/dashboard_service/main.py](../services/dashboard_service/main.py) | FastAPI app entry: UI + control hub |
| [services/dashboard_service/routes.py](../services/dashboard_service/routes.py) | HTML page routes |
| [services/dashboard_service/data_adapters.py](../services/dashboard_service/data_adapters.py) | Parquet/MD to Plotly JSON adapters for the UI |
| [services/dashboard_service/config_api.py](../services/dashboard_service/config_api.py) | Config editor API (read/write `config.yaml`) |
| [services/dashboard_service/hub_api.py](../services/dashboard_service/hub_api.py) | Control hub proxy to the data/model/backtest services via httpx |
| [services/dashboard_service/websockets.py](../services/dashboard_service/websockets.py) | WebSocket tailing of the `logs/*.log` files |
| [services/dashboard_service/Dockerfile](../services/dashboard_service/Dockerfile) | Container image definition, Dashboard Service |

#### Templates

| File | Description |
|-------|-------------|
| [services/dashboard_service/templates/base.html](../services/dashboard_service/templates/base.html) | Base layout (header, sidebar, footer) |
| [services/dashboard_service/templates/index.html](../services/dashboard_service/templates/index.html) | Landing page/overview |
| [services/dashboard_service/templates/hub.html](../services/dashboard_service/templates/hub.html) | Control hub for triggering pipeline stages |
| [services/dashboard_service/templates/config_editor.html](../services/dashboard_service/templates/config_editor.html) | YAML config editor |
| [services/dashboard_service/templates/eda.html](../services/dashboard_service/templates/eda.html) | EDA visualizations |
| [services/dashboard_service/templates/models.html](../services/dashboard_service/templates/models.html) | Model comparison (regime classification, metrics) |
| [services/dashboard_service/templates/backtest.html](../services/dashboard_service/templates/backtest.html) | Backtest and SORR results |
| [services/dashboard_service/templates/evaluation.html](../services/dashboard_service/templates/evaluation.html) | Evaluation: confusion matrices, ROC, H1/H2, MCS |
| [services/dashboard_service/templates/logs.html](../services/dashboard_service/templates/logs.html) | Live log viewer (WebSocket-streamed) |

#### Static Assets

| File | Description |
|-------|-------------|
| [services/dashboard_service/static/css/dashboard.css](../services/dashboard_service/static/css/dashboard.css) | Dashboard styles |
| [services/dashboard_service/static/js/common.js](../services/dashboard_service/static/js/common.js) | Shared client-side logic (Plotly renderer, fetch helpers) |

---

## 5. Exploratory Jupyter Notebooks (`jupyter/`)

| Notebook | Description |
|----------|-------------|
| [jupyter/Asymmetric_correlation_Ang_Chen.ipynb](../jupyter/Asymmetric_correlation_Ang_Chen.ipynb) | Asymmetric correlation following Ang & Chen |
| [jupyter/Concept_matrix_venn.ipynb](../jupyter/Concept_matrix_venn.ipynb) | Concept Venn diagram for the thesis |
| [jupyter/S&P500_NBER-recessions.ipynb](../jupyter/S&P500_NBER-recessions.ipynb) | S&P 500 history with NBER recession periods |
| [jupyter/SORR.ipynb](../jupyter/SORR.ipynb) | Exploratory SORR illustration |

---

## 6. Data Artifacts (`data/`, Medallion)

For details, see [docs/data-architecture.md](./data-architecture.md).

| Layer | Content |
|-------|--------|
| [data/bronze/](../data/bronze/) | Raw data: `01_raw_data.parquet` (Yahoo Finance download) |
| [data/silver/](../data/silver/) | Cleaned & feature-engineered data (`02_preprocessed`, `03_feature_engineered`, `04_test_df`, `wf_cache`) |
| [data/gold/](../data/gold/) | Result artifacts for reporting (backtest results, transaction costs, SORR simulation, MCS) |

---

## 7. Models (`models/`)

Training artifacts, versioned via Git LFS or created at runtime.

| File / Pattern | Description |
|-----------------|-------------|
| `models/optuna_studies.db` | Persistent Optuna studies (SQLite) |
| `models/msm_regime_model.pkl` | Trained MSM weights |
| `models/hmm_regime_model.pkl` | Trained HMM weights |
| `models/hmm_scaler.pkl` | Feature scaler for the HMM |
| `models/lstm_regime_model.keras` | Trained LSTM weights |
| `models/tlstm_scaler.pkl` | Feature scaler for the LSTM model |
| `models/transformer_regime_model.pt` | Trained Transformer weights |
| `models/transformer_scaler.pkl` | Feature scaler for the Transformer |

---

## 8. Logs (`logs/`)

Runtime logs; rewritten on every pipeline/service execution.

| File Pattern | Source |
|---------------|--------|
| `logs/data_service.log` | FastAPI Data Service |
| `logs/model_service.log` | FastAPI Model Service |
| `logs/backtest_service.log` | FastAPI Backtest Service |
| `logs/dashboard_service.log` | FastAPI Dashboard Service |

---

## 9. Assets (`assets/`)

Plots and Markdown tables generated by the notebooks and services.
Each artifact is embedded in `docs/statistics.md` and/or in the dashboard.

### 9.1 EDA & Preprocessing

| File | Description |
|-------|-------------|
| [assets/SORR_schema.png](../assets/SORR_schema.png) | SORR schema |
| [assets/eda_descriptive_stats.md](../assets/eda_descriptive_stats.md) | Descriptive statistics of the inputs |
| [assets/eda_adf_tests.md](../assets/eda_adf_tests.md) | ADF stationarity tests |
| [assets/eda_historical_drawdowns.png](../assets/eda_historical_drawdowns.png) | Historical drawdowns of the equity portfolio |
| [assets/eda_volatility_clusters.png](../assets/eda_volatility_clusters.png) | Volatility clusters (ARCH effects) |
| [assets/feature_correlation_matrix.png](../assets/feature_correlation_matrix.png) | Feature correlation matrix (plot) |
| [assets/feature_correlation_table.md](../assets/feature_correlation_table.md) | Feature correlation matrix (table) |
| [assets/concept_matrix_venn.png](../assets/concept_matrix_venn.png) | Concept Venn diagram (thesis) |
| [assets/asymmetric_correlation_ang_chen.png](../assets/asymmetric_correlation_ang_chen.png) | Asymmetric correlation (Ang & Chen) |
| [assets/data_quality_report.md](../assets/data_quality_report.md) | Data quality report |

### 9.2 Label Analysis

| File | Description |
|-------|-------------|
| [assets/label_timeline_comparison.png](../assets/label_timeline_comparison.png) | Timeline comparison of the label schemes |
| [assets/label_concordance_matrix.png](../assets/label_concordance_matrix.png) | Concordance matrix of the label schemes |
| [assets/label_kappa_matrix.png](../assets/label_kappa_matrix.png) | Cohen's kappa matrix of the label schemes |

### 9.3 Model Visualizations

| File | Description |
|-------|-------------|
| [assets/walk_forward_schema.png](../assets/walk_forward_schema.png) | Walk-forward schema |
| [assets/hmm_regimes.png](../assets/hmm_regimes.png) | Regimes identified by the HMM |
| [assets/hmm_uni_regimes.png](../assets/hmm_uni_regimes.png) | Regimes identified by the HMM (univariate) |
| [assets/msm_regimes.png](../assets/msm_regimes.png) | Regimes identified by the MSM |
| [assets/lstm_model.png](../assets/lstm_model.png) | LSTM architecture sketch |
| [assets/transformer_model.png](../assets/transformer_model.png) | Transformer architecture sketch |
| [assets/regime_comparison.png](../assets/regime_comparison.png) | Cross-model regime comparison |
| [assets/regime_probability_heatmap.png](../assets/regime_probability_heatmap.png) | Regime probabilities as heatmap |

### 9.4 Optuna Hyperparameter Optimization

Four standard plots per model (history / importance / contour / slice):

| File | Description |
|-------|-------------|
| [assets/optuna_MSM_history.png](../assets/optuna_MSM_history.png) | MSM: Optuna trial history |
| [assets/optuna_MSM_importance.png](../assets/optuna_MSM_importance.png) | MSM: parameter importance |
| [assets/optuna_MSM_slice.png](../assets/optuna_MSM_slice.png) | MSM: parameter slice |
| [assets/optuna_HMM_history.png](../assets/optuna_HMM_history.png) | HMM: Optuna trial history |
| [assets/optuna_HMM_importance.png](../assets/optuna_HMM_importance.png) | HMM: parameter importance |
| [assets/optuna_HMM_contour.png](../assets/optuna_HMM_contour.png) | HMM: parameter contour |
| [assets/optuna_HMM_slice.png](../assets/optuna_HMM_slice.png) | HMM: parameter slice |
| [assets/optuna_LSTM_history.png](../assets/optuna_LSTM_history.png) | LSTM: Optuna trial history |
| [assets/optuna_LSTM_importance.png](../assets/optuna_LSTM_importance.png) | LSTM: parameter importance |
| [assets/optuna_LSTM_contour.png](../assets/optuna_LSTM_contour.png) | LSTM: parameter contour |
| [assets/optuna_LSTM_slice.png](../assets/optuna_LSTM_slice.png) | LSTM: parameter slice |
| [assets/optuna_Transformer_history.png](../assets/optuna_Transformer_history.png) | Transformer: Optuna trial history |
| [assets/optuna_Transformer_importance.png](../assets/optuna_Transformer_importance.png) | Transformer: parameter importance |
| [assets/optuna_Transformer_contour.png](../assets/optuna_Transformer_contour.png) | Transformer: parameter contour |
| [assets/optuna_Transformer_slice.png](../assets/optuna_Transformer_slice.png) | Transformer: parameter slice |
| [assets/optuna_importance_values.json](../assets/optuna_importance_values.json) | fANOVA importance cache (the dashboard reads from it to match the PNG exactly; fANOVA is stochastic and would deviate on re-computation) |
| [assets/optuna_best_params.md](../assets/optuna_best_params.md) | Best hyperparameters per study with secondary risk metrics of the best trial |

#### Post-HPO Analysis (Issue #5)

Generated by [src/backtest/hpo_analysis.py](../src/backtest/hpo_analysis.py) (route `POST /models/hpo-analysis`); embedded in `statistics.md` (section G) and the Models dashboard page. See [docs/hyperparameter-optimization.md](./hyperparameter-optimization.md).

| File | Description |
|-------|-------------|
| [assets/hpo_convergence.md](../assets/hpo_convergence.md) | Convergence and edge-of-range review (best value/trial, fANOVA importance, bound flags) |
| [assets/objective_sensitivity.md](../assets/objective_sensitivity.md) | Best config under each candidate metric, valued across all metrics |
| [assets/hpo_dsr.md](../assets/hpo_dsr.md) | Deflated Sharpe Ratio (multiple-testing adjusted) |
| [assets/hpo_pbo.md](../assets/hpo_pbo.md) | Probability of Backtest Overfitting (CSCV) |
| [assets/hpo_multiseed.md](../assets/hpo_multiseed.md) | Multi-seed re-evaluation of the top DL configs |

### 9.5 Backtest & Performance

| File | Description |
|-------|-------------|
| [assets/equity_curves.png](../assets/equity_curves.png) | Equity curves of all strategies |
| [assets/capital_curve.png](../assets/capital_curve.png) | Capital curve (aggregated) |
| [assets/drawdown.png](../assets/drawdown.png) | Drawdown paths |
| [assets/rolling_sharpe.png](../assets/rolling_sharpe.png) | Rolling Sharpe ratio |
| [assets/transaction_costs.png](../assets/transaction_costs.png) | Transaction costs per strategy |
| [assets/annualized_metrics.md](../assets/annualized_metrics.md) | Annualized performance metrics |
| [assets/performance_summary.md](../assets/performance_summary.md) | Overall performance summary |
| [assets/crisis_performance.md](../assets/crisis_performance.md) | Performance during crisis periods |
| [assets/break_even_costs.md](../assets/break_even_costs.md) | Break-even transaction costs (table) |
| [assets/break_even_costs.png](../assets/break_even_costs.png) | Break-even transaction costs (plot) |

### 9.6 Evaluation & Classification

| File | Description |
|-------|-------------|
| [assets/classification_metrics.md](../assets/classification_metrics.md) | Classification metrics per model |
| [assets/confusion_matrices.png](../assets/confusion_matrices.png) | Confusion matrices of the models |
| [assets/roc_curves.png](../assets/roc_curves.png) | ROC curves |
| [assets/pr_curves.png](../assets/pr_curves.png) | Precision-recall curves |
| [assets/evaluation_table.md](../assets/evaluation_table.md) | Overall evaluation table |
| [assets/churning_stats.md](../assets/churning_stats.md) | Whipsaw / churning statistics |
| [assets/bear_coverage.md](../assets/bear_coverage.md) | Per-fold OOS bear-market coverage of the walk-forward windows (Issue #8) |
| [assets/switch_timing.md](../assets/switch_timing.md) | Timing of the regime switches |

#### Threshold Sensitivity

| File | Description |
|-------|-------------|
| [assets/threshold_sensitivity_MSM.md](../assets/threshold_sensitivity_MSM.md) | MSM: threshold sensitivity |
| [assets/threshold_sensitivity_HMM.md](../assets/threshold_sensitivity_HMM.md) | HMM: threshold sensitivity |
| [assets/threshold_sensitivity_HMM_Uni.md](../assets/threshold_sensitivity_HMM_Uni.md) | HMM (univariate): threshold sensitivity |
| [assets/threshold_sensitivity_LSTM.md](../assets/threshold_sensitivity_LSTM.md) | LSTM: threshold sensitivity |
| [assets/threshold_sensitivity_Transformer.md](../assets/threshold_sensitivity_Transformer.md) | Transformer: threshold sensitivity |

#### Time-to-Recovery

| File | Description |
|-------|-------------|
| [assets/time_to_recovery_Buy_Hold.md](../assets/time_to_recovery_Buy_Hold.md) | Buy & hold: time-to-recovery |
| [assets/time_to_recovery_MSM.md](../assets/time_to_recovery_MSM.md) | MSM: time-to-recovery |
| [assets/time_to_recovery_HMM.md](../assets/time_to_recovery_HMM.md) | HMM: time-to-recovery |
| [assets/time_to_recovery_HMM_Uni.md](../assets/time_to_recovery_HMM_Uni.md) | HMM (univariate): time-to-recovery |
| [assets/time_to_recovery_LSTM.md](../assets/time_to_recovery_LSTM.md) | LSTM: time-to-recovery |
| [assets/time_to_recovery_Transformer.md](../assets/time_to_recovery_Transformer.md) | Transformer: time-to-recovery |

### 9.7 Hypothesis Tests

| File | Description |
|-------|-------------|
| [assets/h1_drawdown_test.md](../assets/h1_drawdown_test.md) | H1: drawdown reduction (significant?) |
| [assets/mcs_h1_mdd_forest.png](../assets/mcs_h1_mdd_forest.png) | H1: forest plot H1/MDD |
| [assets/h2_transformer_test.md](../assets/h2_transformer_test.md) | H2: Transformer superiority (significant?) |
| [assets/mcs_h2_final_capital_forest.png](../assets/mcs_h2_final_capital_forest.png) | H2: forest plot H2/terminal capital |

### 9.8 SORR and Monte Carlo Simulation

| File | Description |
|-------|-------------|
| [assets/sorr_summary.md](../assets/sorr_summary.md) | SORR results overview |
| [assets/sorr_sim_standard.png](../assets/sorr_sim_standard.png) | SORR simulation: standard withdrawal |
| [assets/sorr_sim_aggressive.png](../assets/sorr_sim_aggressive.png) | SORR simulation: aggressive withdrawal |
| [assets/sorr_sim_low_capital.png](../assets/sorr_sim_low_capital.png) | SORR simulation: low initial capital |
| [assets/withdrawal_sensitivity.md](../assets/withdrawal_sensitivity.md) | Sensitivity to withdrawal rates |
| [assets/depletion_rate_ci.md](../assets/depletion_rate_ci.md) | Depletion rate with confidence interval |
| [assets/bootstrap_robustness.md](../assets/bootstrap_robustness.md) | Block vs. stationary bootstrap robustness comparison (Issue #7) |
| [assets/mcs_depletion_rate_forest.png](../assets/mcs_depletion_rate_forest.png) | Forest plot, depletion rate
| [assets/mcs_summary.md](../assets/mcs_summary.md) | Monte Carlo simulation: summary |
| [assets/mcs_paths.png](../assets/mcs_paths.png) | MCS paths |
| [assets/mcs_quantiles.png](../assets/mcs_quantiles.png) | MCS quantiles |
| [assets/mcs_boxplot_standard.png](../assets/mcs_boxplot_standard.png) | MCS boxplot: standard withdrawal |
| [assets/mcs_boxplot_aggressive.png](../assets/mcs_boxplot_aggressive.png) | MCS boxplot: aggressive withdrawal |
| [assets/mcs_boxplot_low_capital.png](../assets/mcs_boxplot_low_capital.png) | MCS boxplot: low initial capital |
| [assets/mcs_violin_standard.png](../assets/mcs_violin_standard.png) | MCS violin plot: standard withdrawal |
| [assets/mcs_violin_aggressive.png](../assets/mcs_violin_aggressive.png) | MCS violin plot: aggressive withdrawal |
| [assets/mcs_violin_low_capital.png](../assets/mcs_violin_low_capital.png) | MCS violin plot: low initial capital |
| [assets/risk_return_positioning.png](../assets/risk_return_positioning.png) | Risk-return positioning |

### 9.9 Dashboard Screenshots

| File | Description |
|-------|-------------|
| [assets/screenshots/dashboard_overview.png](../assets/screenshots/dashboard_overview.png) | Dashboard: overview page |
| [assets/screenshots/dashboard_control-hub.png](../assets/screenshots/dashboard_control-hub.png) | Dashboard: control hub |
| [assets/screenshots/dashboard_config.png](../assets/screenshots/dashboard_config.png) | Dashboard: config editor |
| [assets/screenshots/dashboard_evaluation.png](../assets/screenshots/dashboard_evaluation.png) | Dashboard: evaluation page |

---

## Maintenance

This index is maintained manually. Please update it when the structure changes
(or generate it from `git ls-files` via script in the future; see Issue #7,
*Additional Notes*).
