
# Detailed Statistical Evaluation & Research Results

This page documents the numerical and graphical results of the research pipeline. All evaluations are based on the dataset up to yesterday (2026-07-05) and are updated automatically.

---

## 1. Executive Summary: Performance & Risk
A direct comparison of the core metrics over the entire **out-of-sample test period**.

| Strategy    | Final Wealth   | Total Return   | Max Drawdown   |
|:------------|:---------------|:---------------|:---------------|
| Buy_Hold    | 2,097,913 €    | +319.58%       | -35.08%        |
| MSM         | 1,552,020 €    | +210.40%       | -12.79%        |
| HMM         | 2,199,262 €    | +339.85%       | -22.92%        |
| HMM_Uni     | 1,541,837 €    | +208.37%       | -12.79%        |
| LSTM        | 1,973,851 €    | +294.77%       | -31.72%        |
| Transformer | 1,964,834 €    | +292.97%       | -27.71%        |

> **Key point:** Compare the **max drawdown** of the active strategies with the buy-and-hold benchmark. The objective of this work is a significant reduction of this value to mitigate SORR.

---

## 2. Data Basis & Baseline Portfolio
The analysis is based on a global multi-asset approach.

### Exploratory Data Analysis (EDA)
**Descriptive statistics of the base time series:**
| Time Series   |   Mean (daily) |   Std. Dev. (daily) |     Min |     Max |   Skewness |   Kurtosis |
|:--------------|---------------:|--------------------:|--------:|--------:|-----------:|-----------:|
| Returns_GSPC  |       0.00033  |            0.011354 | -0.1277 |  0.1096 |    -0.3642 |    10.8578 |
| Returns_VUSTX |       0.000213 |            0.007253 | -0.0605 |  0.0654 |    -0.031  |     4.5144 |
| Returns       |       0.000283 |            0.006883 | -0.0662 |  0.0584 |    -0.2803 |     7.6584 |
| VIX           |      19.4547   |            7.74392  |  9.14   | 82.69   |     2.2073 |     8.7245 |
| TNX_10Y       |       4.24471  |            1.9274   |  0.499  |  9.09   |     0.326  |    -0.6296 |
| IRX_3M        |       2.71399  |            2.19752  | -0.105  |  7.99   |     0.192  |    -1.251  |

**Stationarity check (augmented Dickey-Fuller test):**
| Time Series   |   ADF Statistic |    p-Value |   Crit. Value (5%) | Stationary?   |
|:--------------|----------------:|-----------:|-------------------:|:--------------|
| Returns_GSPC  |        -17.556  | 4.1193e-30 |            -2.8619 | Yes           |
| Returns_VUSTX |        -18.7273 | 2.0315e-30 |            -2.8619 | Yes           |
| Returns       |        -21.0026 | 0          |            -2.8619 | Yes           |
| VIX           |         -7.3102 | 1.2703e-10 |            -2.8619 | Yes           |
| TNX_10Y       |         -2.3445 | 0.15801    |            -2.8619 | No            |
| IRX_3M        |         -2.3484 | 0.15684    |            -2.8619 | No            |

**Volatility clusters and autocorrelation (heteroskedasticity):**
![Volatility Clusters](../assets/eda_volatility_clusters.png)

### Feature Correlation
Pearson correlation matrix of the six model features to check for multicollinearity.

![Feature Correlation Matrix](../assets/feature_correlation_matrix.png)

### SORR Context: Historical Drawdowns
The most extreme loss phases of the 60/40 portfolio, shown as motivation for active capital protection.
![Historical Drawdowns](../assets/eda_historical_drawdowns.png)

### 60/40 Portfolio Capital Curve
The figure shows the cumulative performance of the static reference portfolio (60% equities / 40% bonds).

![Capital Curve](../assets/capital_curve.png)

*   **Data source:** S&P 500 (`^GSPC`) and Vanguard Long-Term Treasury (`VUSTX`).
*   **Reproducibility:** The cleaned dataset incl. all features is stored at: `data/02_feature_engineered_data.parquet`.

---

## 3. Regime Detection of the Individual Models
This section visualizes the identification results of the model categories (statistics, clustering, deep learning).

### A. Markov-Switching Models (Econometrics)
Identification of bull and bear regimes using a univariate two-regime Markov-switching model based on S&P 500 returns.
![Markov Models](../assets/msm_regimes.png)

### B. Hidden Markov Model (Unsupervised Clustering)
![HMM Regimes](../assets/hmm_regimes.png)

### C. Univariate HMM (Ablation, Issue #3)
Robustness check for the MSM-vs-HMM architecture comparison: identical
input space to the MSM (60/40 returns only). Isolates the architectural effect
(clustering vs. Markov-switching regression) from the information contribution of the
extended features (VIX, yield spread).
![HMM Uni Regimes](../assets/hmm_uni_regimes.png)

### D. LSTM Network (Deep Learning)
Prediction of market phases by the neural network (trained on Pagan-Sossounov labels).
![LSTM Model](../assets/lstm_model.png)

### E. Transformer Network (Attention-Based Regime Detection)
Classification of market regimes using a Transformer encoder with multi-head self-attention and positional encoding. In contrast to recurrent architectures (LSTM), the Transformer processes all time steps of a sequence in parallel and learns via the attention mechanism which historical data points are most relevant for the current regime classification. Trained in a supervised setting on Pagan-Sossounov labels.
![Transformer Model](../assets/transformer_model.png)

### F. Global Regime Comparison
Detailed comparison of the probabilities and hard signals of all models.
![Regime Comparison](../assets/regime_comparison.png)

### G. Hyperparameter Optimization (Optuna)
Search over the hyperparameter space of all models using walk-forward validation as inner CV. The objective is the configured risk metric (default: Martin ratio = CAGR / Ulcer index) on the **pooled** OOS return series across all HPO folds. Econometric models are searched exhaustively via GridSampler, the DL models via a multivariate TPESampler. Selection runs on the development folds only (`tune_until`); the holdout folds stay selection-free for the final walk-forward run. The values reported here were adopted into `config.yaml` for the final run.

# Optuna: Best Hyperparameters

_Generated at 2026-07-05 10:38:33_  
Optimization metric: **martin (pooled OOS)**

## Overview

| Model | Best Score | ✓ Complete | ✗ Pruned | Total |
|:---|---:|---:|---:|---:|
| **MSM** | 1.6508 | 36 | 0 | 36 |
| **HMM** | 1.2037 | 108 | 0 | 108 |
| **HMM_Uni** | 1.6384 | 36 | 0 | 36 |
| **LSTM** | 2.6426 | 194 | 0 | 203 |

### MSM: Best Score `1.6508`

| Parameter | Value |
|:---|---:|
| `threshold` | `0.175` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.8513 |
| sortino | 1.0119 |
| calmar | 0.6465 |
| martin | 1.6508 |
| ulcer | 3.1627 |
| max_drawdown | -0.0808 |
| cagr | 0.0522 |

### HMM: Best Score `1.2037`

| Parameter | Value |
|:---|---:|
| `covariance_type` | `tied` |
| `threshold` | `0.875` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.7243 |
| sortino | 0.8796 |
| calmar | 0.4153 |
| martin | 1.2037 |
| ulcer | 3.9259 |
| max_drawdown | -0.1138 |
| cagr | 0.0473 |

### HMM_Uni: Best Score `1.6384`

| Parameter | Value |
|:---|---:|
| `threshold` | `0.175` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.8420 |
| sortino | 0.9987 |
| calmar | 0.6752 |
| martin | 1.6384 |
| ulcer | 3.1419 |
| max_drawdown | -0.0762 |
| cagr | 0.0515 |

### LSTM: Best Score `2.6426`

| Parameter | Value |
|:---|---:|
| `window_size` | `250` |
| `units_l1` | `16` |
| `units_l2` | `32` |
| `batch_size` | `32` |
| `learning_rate` | `1.329e-05` |
| `dropout` | `0.45` |
| `threshold` | `0.15` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.9785 |
| sortino | 0.9191 |
| calmar | 0.6183 |
| martin | 2.6426 |
| ulcer | 2.0269 |
| max_drawdown | -0.0866 |
| cagr | 0.0536 |


**Diagnostic plots per model** (optimization history, parameter importance, slice, contour):

| Model | History | Importance | Slice | Contour |
|:---|:---|:---|:---|:---|
| MSM         | ![](../assets/optuna_MSM_history.png)         | ![](../assets/optuna_MSM_importance.png)         | ![](../assets/optuna_MSM_slice.png)         | n/a ¹                                         |
| HMM         | ![](../assets/optuna_HMM_history.png)         | ![](../assets/optuna_HMM_importance.png)         | ![](../assets/optuna_HMM_slice.png)         | ![](../assets/optuna_HMM_contour.png)         |
| HMM_Uni     | ![](../assets/optuna_HMM_Uni_history.png)     | ![](../assets/optuna_HMM_Uni_importance.png)     | ![](../assets/optuna_HMM_Uni_slice.png)     | n/a ¹                                         |
| LSTM        | ![](../assets/optuna_LSTM_history.png)        | ![](../assets/optuna_LSTM_importance.png)        | ![](../assets/optuna_LSTM_slice.png)        | ![](../assets/optuna_LSTM_contour.png)        |
| Transformer | ![](../assets/optuna_Transformer_history.png) | ![](../assets/optuna_Transformer_importance.png) | ![](../assets/optuna_Transformer_slice.png) | ![](../assets/optuna_Transformer_contour.png) |

¹ MSM and HMM_Uni have only one hyperparameter (`threshold`) in the search space. The contour plot would be degenerate and is omitted.

#### G.1 Convergence & Edge-of-Range Review
Best value and trial, fANOVA importance, and a flag for optima sitting within one grid step of a search bound (indicating the range may be too narrow).

# HPO Convergence & Edge-of-Range Review

_Generated at 2026-07-05 09:39:21_

| model       | metric   |   best_value |   best_trial |   conv_frac |   n_complete |   n_pruned | top_importance                                      | edge_flags                                                                                                                                                         |
|:------------|:---------|-------------:|-------------:|------------:|-------------:|-----------:|:----------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MSM         | martin   |       1.6508 |           19 |        0.54 |           36 |          0 |                                                     | -                                                                                                                                                                  |
| HMM         | martin   |       1.2037 |            4 |        0.04 |          108 |          0 | covariance_type=1.00                                | -                                                                                                                                                                  |
| HMM_Uni     | martin   |       1.6384 |           19 |        0.54 |           36 |          0 |                                                     | -                                                                                                                                                                  |
| LSTM        | martin   |       2.6426 |          129 |        0.66 |          187 |          0 | units_l1=0.26, threshold=0.20, dropout=0.17         | window_size=250 near UPPER bound 250; learning_rate=1.3292520737887431e-05 near LOWER bound 1e-05; threshold=0.15000000000000002 near LOWER bound 0.1              |
| Transformer | martin   |       1.3506 |           14 |        0.35 |           39 |          0 | window_size=0.38, dmodel_nheads=0.32, n_layers=0.07 | window_size=250 near UPPER bound 250; n_layers=3 near UPPER bound 4; learning_rate=0.0003589128083678785 near LOWER bound 1e-05; dropout=0.45 near UPPER bound 0.5 |


#### G.2 Objective Sensitivity
Which config would have been selected under each candidate metric, valued across all metrics. `same_as_objective = True` throughout means the model choice is robust to the objective; divergences quantify the trade-off.

# Objective Sensitivity of the Selected Hyperparameters

_Generated at 2026-07-05 09:39:22_  
Best config under each candidate metric, valued across all metrics (from the search trials' logged OOS metrics; no retraining). `same_as_objective` marks configs identical to the actual objective's pick.

## MSM (objective: martin, 36 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |
| sharpe          |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |
| sortino         |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |
| calmar          |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |
| ulcer           |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |
| max_drawdown    |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |

Selected configs:
- best under **martin**: threshold=0.175

## HMM (objective: martin, 108 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |       4 | True                |   1.2037 |   0.7243 |    0.8796 |   0.4153 |  3.9259 |        -0.1138 | 0.0473 |
| sharpe          |       4 | True                |   1.2037 |   0.7243 |    0.8796 |   0.4153 |  3.9259 |        -0.1138 | 0.0473 |
| sortino         |       4 | True                |   1.2037 |   0.7243 |    0.8796 |   0.4153 |  3.9259 |        -0.1138 | 0.0473 |
| calmar          |       4 | True                |   1.2037 |   0.7243 |    0.8796 |   0.4153 |  3.9259 |        -0.1138 | 0.0473 |
| ulcer           |      66 | False               |   0.921  |   0.5636 |    0.6061 |   0.3386 |  3.4975 |        -0.0951 | 0.0322 |
| max_drawdown    |      22 | False               |   0.9395 |   0.5714 |    0.6151 |   0.3455 |  3.4985 |        -0.0951 | 0.0329 |

Selected configs:
- best under **martin**: covariance_type=tied, threshold=0.875
- best under **ulcer**: covariance_type=tied, threshold=0.125
- best under **max_drawdown**: covariance_type=tied, threshold=0.15

## HMM_Uni (objective: martin, 36 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      19 | True                |   1.6384 |   0.842  |    0.9987 |   0.6752 |  3.1419 |        -0.0762 | 0.0515 |
| sharpe          |      19 | True                |   1.6384 |   0.842  |    0.9987 |   0.6752 |  3.1419 |        -0.0762 | 0.0515 |
| sortino         |      19 | True                |   1.6384 |   0.842  |    0.9987 |   0.6752 |  3.1419 |        -0.0762 | 0.0515 |
| calmar          |      19 | True                |   1.6384 |   0.842  |    0.9987 |   0.6752 |  3.1419 |        -0.0762 | 0.0515 |
| ulcer           |      17 | False               |   1.5245 |   0.7938 |    0.9291 |   0.578  |  3.1279 |        -0.0825 | 0.0477 |
| max_drawdown    |      19 | True                |   1.6384 |   0.842  |    0.9987 |   0.6752 |  3.1419 |        -0.0762 | 0.0515 |

Selected configs:
- best under **martin**: threshold=0.175
- best under **ulcer**: threshold=0.15

## LSTM (objective: martin, 187 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |     129 | True                |   2.6426 |   0.9785 |    0.9191 |   0.6183 |  2.0269 |        -0.0866 | 0.0536 |
| sharpe          |     129 | True                |   2.6426 |   0.9785 |    0.9191 |   0.6183 |  2.0269 |        -0.0866 | 0.0536 |
| sortino         |      99 | False               |   1.8536 |   0.8679 |    1.1122 |   0.3773 |  3.8388 |        -0.1886 | 0.0712 |
| calmar          |     148 | False               |   2.4283 |   0.8726 |    1.0941 |   0.6963 |  2.5232 |        -0.088  | 0.0613 |
| ulcer           |     160 | False               |   2.3163 |   0.8811 |    0.7852 |   0.5187 |  1.948  |        -0.087  | 0.0451 |
| max_drawdown    |     181 | False               |   1.7748 |   0.8285 |    0.6782 |   0.5021 |  2.0816 |        -0.0736 | 0.0369 |

Selected configs:
- best under **martin**: window_size=250, units_l1=16, units_l2=32, batch_size=32, learning_rate=1.3292520737887431e-05, dropout=0.45, threshold=0.15000000000000002
- best under **sortino**: window_size=250, units_l1=32, units_l2=256, batch_size=128, learning_rate=4.471806844913538e-05, dropout=0.45, threshold=0.30000000000000004
- best under **calmar**: window_size=250, units_l1=16, units_l2=256, batch_size=128, learning_rate=4.076527319051499e-05, dropout=0.4, threshold=0.15000000000000002
- best under **ulcer**: window_size=210, units_l1=16, units_l2=64, batch_size=128, learning_rate=2.0149507145662965e-05, dropout=0.55, threshold=0.1
- best under **max_drawdown**: window_size=230, units_l1=16, units_l2=32, batch_size=64, learning_rate=1.0355058430277764e-05, dropout=0.4, threshold=0.2

## Transformer (objective: martin, 39 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      14 | True                |   1.3506 |   0.7152 |    0.9173 |   0.2847 |  4.0869 |        -0.1939 | 0.0552 |
| sharpe          |      14 | True                |   1.3506 |   0.7152 |    0.9173 |   0.2847 |  4.0869 |        -0.1939 | 0.0552 |
| sortino         |      14 | True                |   1.3506 |   0.7152 |    0.9173 |   0.2847 |  4.0869 |        -0.1939 | 0.0552 |
| calmar          |      14 | True                |   1.3506 |   0.7152 |    0.9173 |   0.2847 |  4.0869 |        -0.1939 | 0.0552 |
| ulcer           |      14 | True                |   1.3506 |   0.7152 |    0.9173 |   0.2847 |  4.0869 |        -0.1939 | 0.0552 |
| max_drawdown    |      11 | False               |   0.4869 |   0.6015 |    0.5703 |   0.2006 |  6.3571 |        -0.1543 | 0.031  |

Selected configs:
- best under **martin**: window_size=250, dmodel_nheads=16-2, n_layers=3, dim_feedforward=32, batch_size=64, learning_rate=0.0003589128083678785, dropout=0.45, threshold=0.7000000000000001
- best under **max_drawdown**: window_size=40, dmodel_nheads=32-2, n_layers=4, dim_feedforward=32, batch_size=32, learning_rate=2.015647705936502e-05, dropout=0.35000000000000003, threshold=0.1


#### G.3 Deflated Sharpe Ratio & Backtest Overfitting
DSR deflates the selected config's Sharpe for the number of tested configs; PBO (CSCV) estimates the probability that the in-sample-best config underperforms out-of-sample.

# Deflated Sharpe Ratio (multiple-testing adjusted)

_Generated at 2026-07-04 11:52:05_

| model   |   n_trials |   sr_ann_best |   sr_star_ann |    dsr | significant_5pct   |
|:--------|-----------:|--------------:|--------------:|-------:|:-------------------|
| MSM     |         36 |        0.8513 |        0.3085 | 0.9843 | True               |
| HMM     |        108 |        0.7243 |        0.4648 | 0.8478 | False              |
| HMM_Uni |         36 |        0.842  |        0.3216 | 0.9804 | True               |

DSR = P(true Sharpe > 0) after deflating the best trial's Sharpe for the number of tested configs. `significant_5pct` = DSR > 0.95.


# Probability of Backtest Overfitting (CSCV)

_Generated at 2026-07-04 11:54:21_

| model   |   folds |   configs |   pbo |
|:--------|--------:|----------:|------:|
| MSM     |      16 |         5 | 0.873 |
| HMM     |      16 |         5 | 0.083 |
| HMM_Uni |      16 |         5 | 0.909 |

PBO over the top-5 configs per model (per-fold Sharpe matrix, CSCV). Lower is better; PBO > 0.5 flags overfitting.


#### G.4 Multi-Seed Robustness (DL)
Top DL configs re-evaluated over several global seeds; mean/std of every metric quantify seed sensitivity of the deep-learning ranking.

# Multi-Seed Re-Evaluation (top-5, 5 seeds)

_Generated at 2026-07-04 11:54:21_

_No DL studies available._


### H. Label Concordance (Selection of the Training Labels)
Comparison of the regime labelers (MSM, HMM, Pagan-Sossounov, Peak-to-Trough, Lunde-Timmermann, NBER) to justify the label choice for the supervised models. Pagan-Sossounov was chosen as the training target for LSTM and Transformer due to its high concordance with NBER recession periods.

![Label Concordance](../assets/label_concordance_matrix.png)
![Label Cohen's κ](../assets/label_kappa_matrix.png)
![Label Timeline](../assets/label_timeline_comparison.png)

---

## 4. Backtesting & Strategy Evaluation
The economic application of the regime signals via dynamic reallocation into the money market.

### Walk-Forward Schema
Rolling train/test windows over the entire study period. Each row corresponds to one fold; the blue bar marks the training window, the orange bar the OOS test window. The strict chronological separation prevents look-ahead bias.

![Walk-Forward Schema](../assets/walk_forward_schema.png)

### OOS Bear-Market Coverage (Issue #8)
Per-fold Pagan-Sossounov bear-market exposure of the walk-forward windows: the share of bear days, the number of overlapping bear phases, and whether a complete bear phase (peak and trough) falls inside the window, for the OOS test window and the training window. This diagnostic documents the fold-granularity limitation (few OOS windows contain a full crisis) discussed in `docs/limitations.md`, Section 5. It is complemented by the pooled-OOS objective from Issue #5, which lets crisis periods enter the optimization signal in proportion to their length instead of being averaged away per fold.

|   Fold | Test Start   | Test End   |   Test Bear % |   Test Bear Phases | Test Full Bear Phase   |   Train Bear % |   Train Bear Phases | Train Full Bear Phase   |
|-------:|:-------------|:-----------|--------------:|-------------------:|:-----------------------|---------------:|--------------------:|:------------------------|
|      1 | 2000-10-16   | 2001-10-15 |         100   |                  1 | No                     |            1.2 |                   2 | Yes                     |
|      2 | 2001-10-16   | 2002-10-15 |          76.6 |                  1 | No                     |           11   |                   1 | No                      |
|      3 | 2002-10-16   | 2003-10-15 |           0   |                  0 | No                     |           18.7 |                   1 | Yes                     |
|      4 | 2003-10-16   | 2004-10-15 |           0   |                  0 | No                     |           18.7 |                   1 | Yes                     |
|      5 | 2004-10-18   | 2005-10-14 |           0   |                  0 | No                     |           18.7 |                   1 | Yes                     |
|      6 | 2005-10-17   | 2006-10-13 |           0   |                  0 | No                     |           18.7 |                   1 | Yes                     |
|      7 | 2006-10-16   | 2007-10-15 |           0   |                  0 | No                     |           18.7 |                   1 | Yes                     |
|      8 | 2007-10-16   | 2008-10-15 |          95.3 |                  1 | No                     |           18.7 |                   1 | Yes                     |
|      9 | 2008-10-16   | 2009-10-15 |          38.9 |                  1 | No                     |           28.3 |                   2 | Yes                     |
|     10 | 2009-10-16   | 2010-10-15 |           0   |                  0 | No                     |           32.2 |                   2 | Yes                     |
|     11 | 2010-10-18   | 2011-10-14 |           0   |                  0 | No                     |           31   |                   2 | Yes                     |
|     12 | 2011-10-17   | 2012-10-15 |           0   |                  0 | No                     |           21.1 |                   2 | Yes                     |
|     13 | 2012-10-16   | 2013-10-15 |           0   |                  0 | No                     |           13.5 |                   1 | Yes                     |
|     14 | 2013-10-16   | 2014-10-15 |           0   |                  0 | No                     |           13.5 |                   1 | Yes                     |
|     15 | 2014-10-16   | 2015-10-15 |           0   |                  0 | No                     |           13.5 |                   1 | Yes                     |
|     16 | 2015-10-16   | 2016-10-14 |           0   |                  0 | No                     |           13.5 |                   1 | Yes                     |
|     17 | 2016-10-17   | 2017-10-13 |           0   |                  0 | No                     |           13.5 |                   1 | Yes                     |
|     18 | 2017-10-16   | 2018-10-15 |           0   |                  0 | No                     |           13.5 |                   1 | Yes                     |
|     19 | 2018-10-16   | 2019-10-15 |           0   |                  0 | No                     |            3.9 |                   1 | No                      |
|     20 | 2019-10-16   | 2020-10-15 |           0   |                  0 | No                     |            0   |                   0 | No                      |
|     21 | 2020-10-16   | 2021-10-15 |           0   |                  0 | No                     |            0   |                   0 | No                      |
|     22 | 2021-10-18   | 2022-10-14 |          80.5 |                  1 | Yes                    |            0   |                   0 | No                      |
|     23 | 2022-10-17   | 2023-10-13 |           0   |                  0 | No                     |            8   |                   1 | Yes                     |
|     24 | 2023-10-16   | 2024-10-15 |           0   |                  0 | No                     |            8   |                   1 | Yes                     |
|     25 | 2024-10-16   | 2025-10-15 |           0   |                  0 | No                     |            8   |                   1 | Yes                     |
|     26 | 2025-10-16   | 2026-07-02 |          11.8 |                  1 | Yes                    |            8   |                   1 | Yes                     |

Across the 26 walk-forward folds, 2 OOS test windows contain at least one complete Pagan-Sossounov bear phase (peak and trough inside the 12-month window), while 6 folds overlap at least one bear phase and 20 folds carry no bear day at all.

Because the 3-month minimum-phase filter and the 12-month fold length rarely coincide, most crisis exposure enters the folds as partial (window-truncated) bear phases rather than as fully contained episodes, whereas every training window (10 years) spans several complete bear phases.

A bear run that is still open at the global data boundary cannot be confirmed complete, so the classification is conservative for any fold whose window reaches the end of the sample.


### Equity Curves in Comparison
![Equity Curves](../assets/equity_curves.png)

### Annualized Performance Metrics
Normalized metrics (CAGR, Sharpe, Sortino, Calmar) for comparison across evaluation periods of different lengths.

| Strategy    | CAGR   | Ann. Volatility   |   Sharpe Ratio |   Sortino Ratio | Max Drawdown   |   Calmar Ratio |   OOS Days |   OOS Years |
|:------------|:-------|:------------------|---------------:|----------------:|:---------------|---------------:|-----------:|------------:|
| Buy_Hold    | +5.75% | 11.17%            |          0.557 |           0.727 | -35.08%        |          0.164 |       6464 |        25.7 |
| MSM         | +4.52% | 6.04%             |          0.761 |           0.849 | -12.79%        |          0.353 |       6464 |        25.7 |
| HMM         | +5.95% | 7.60%             |          0.798 |           0.96  | -22.92%        |          0.259 |       6464 |        25.7 |
| HMM_Uni     | +4.49% | 6.04%             |          0.757 |           0.845 | -12.79%        |          0.351 |       6464 |        25.7 |
| LSTM        | +5.50% | 10.77%            |          0.551 |           0.679 | -31.72%        |          0.173 |       6464 |        25.7 |
| Transformer | +5.48% | 9.51%             |          0.609 |           0.711 | -27.71%        |          0.198 |       6464 |        25.7 |

### Classification Metrics (vs. NBER Recessions as Ground Truth)
Comparison of the models as binary recession classifiers (precision, recall, F1).

| Model       |   Precision |   Recall |    F1 |   TN |   FP |   FN |   TP |
|:------------|------------:|---------:|------:|-----:|-----:|-----:|-----:|
| MSM         |       0.226 |    0.923 | 0.364 | 4031 | 1848 |   45 |  541 |
| HMM         |       0.313 |    0.787 | 0.448 | 4867 | 1012 |  125 |  461 |
| HMM_Uni     |       0.227 |    0.923 | 0.364 | 4033 | 1846 |   45 |  541 |
| LSTM        |       0.175 |    0.206 | 0.19  | 5310 |  569 |  465 |  121 |
| Transformer |       0.109 |    0.28  | 0.157 | 4543 | 1336 |  422 |  164 |

![Confusion Matrices](../assets/confusion_matrices.png)

**ROC and precision-recall curves** (threshold-independent comparison via `*_Prob`):

![ROC Curves](../assets/roc_curves.png)
![PR Curves](../assets/pr_curves.png)

### Signal Churning & Whipsaw Analysis
Quantification of the switching frequency and the share of very short regime phases ("whipsaws").

| Model       |   Signal Switches |   Whipsaws (<5d) | Whipsaw Share   |   Mean Phase (Days) |   Median Phase (Days) | Cumul. Costs   |
|:------------|------------------:|-----------------:|:----------------|--------------------:|----------------------:|:---------------|
| MSM         |               239 |                1 | 0.4%            |                26.9 |                     8 | 23.90%         |
| HMM         |                81 |                0 | 0.0%            |                78.8 |                    11 | 8.10%          |
| HMM_Uni     |               239 |                1 | 0.4%            |                26.9 |                     7 | 23.90%         |
| LSTM        |                12 |                0 | 0.0%            |               497.3 |                    62 | 1.20%          |
| Transformer |                32 |                0 | 0.0%            |               195.9 |                    11 | 3.20%          |

### Regime Probability Heatmap
Bear probabilities of all models over time.

![Regime Probability Heatmap](../assets/regime_probability_heatmap.png)

### Threshold Sensitivity
Variation of the decision threshold per model. Shows how robust final wealth, max drawdown, and the number of regime switches are to a modified bull/bear classification boundary (thesis ch. 4.1, smoothing).

**MSM**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.25 | 1,529,649 €    | -12.13%        |        301 |
|        0.3  | 1,353,819 €    | -16.68%        |        313 |
|        0.35 | 1,364,321 €    | -20.19%        |        319 |
|        0.4  | 1,380,008 €    | -21.55%        |        313 |
|        0.5  | 1,465,175 €    | -25.39%        |        325 |

**HMM**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.4  | 1,150,407 €    | -20.94%        |         97 |
|        0.45 | 1,054,819 €    | -25.13%        |         83 |
|        0.5  | 1,030,688 €    | -26.82%        |         87 |
|        0.55 | 1,033,512 €    | -28.12%        |         83 |
|        0.6  | 945,582 €      | -33.98%        |         97 |

**HMM_Uni**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.4  | 1,353,759 €    | -24.79%        |        315 |
|        0.45 | 1,376,439 €    | -24.41%        |        323 |
|        0.5  | 1,476,063 €    | -26.16%        |        339 |
|        0.55 | 1,483,473 €    | -23.24%        |        337 |
|        0.6  | 1,666,713 €    | -26.81%        |        341 |

**LSTM**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|         0.2 | 2,026,749 €    | -31.39%        |         10 |
|         0.3 | 1,973,021 €    | -33.26%        |         10 |
|         0.4 | 2,001,228 €    | -32.93%        |         10 |
|         0.5 | 1,990,114 €    | -33.34%        |         10 |

**Transformer**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.3  | 2,014,595 €    | -27.71%        |         44 |
|        0.4  | 1,932,064 €    | -27.71%        |         52 |
|        0.45 | 2,017,848 €    | -27.71%        |         46 |
|        0.5  | 2,002,345 €    | -27.71%        |         48 |
|        0.6  | 2,058,587 €    | -27.71%        |         52 |

### Time-to-Recovery
All drawdown phases beyond the minimum depth (per `extended.ttr_min_dd`) with peak, trough, and recovery date as well as duration in trading days. An open (not yet recovered) phase is marked as "open" in the recovery field.

**Buy_Hold**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                      49 |                      43 |          92 |
| 2001-02-02 | 2002-07-23 | 2004-03-05 | -24.04%  |                     536 |                     591 |        1127 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                      63 |                     178 |         241 |
| 2007-11-01 | 2009-03-09 | 2011-04-28 | -34.97%  |                     494 |                     780 |        1274 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                      14 |                      67 |          81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                      33 |                     120 |         153 |
| 2015-03-23 | 2015-08-25 | 2016-04-13 | -8.39%   |                     155 |                     232 |         387 |
| 2016-08-01 | 2016-11-14 | 2017-04-17 | -5.64%   |                     105 |                     154 |         259 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                      10 |                     197 |         207 |
| 2018-08-30 | 2018-12-24 | 2019-03-21 | -11.45%  |                     116 |                      87 |         203 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                      26 |                      82 |         108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                      57 |                      39 |          96 |
| 2021-12-28 | 2022-10-14 | 2024-11-29 | -27.55%  |                     290 |                     777 |        1067 |
| 2024-12-09 | 2025-04-08 | 2025-07-03 | -12.22%  |                     120 |                      86 |         206 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                      29 |                      21 |          50 |

**MSM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2001-02-02 | 2003-05-19 | 2003-12-18 | -8.03%   |                     836 |                     213 |        1049 |
| 2004-03-08 | 2004-05-10 | 2004-10-29 | -5.81%   |                      63 |                     172 |         235 |
| 2007-06-05 | 2009-09-02 | 2010-05-03 | -7.32%   |                     820 |                     243 |        1063 |
| 2010-05-04 | 2010-07-02 | 2010-09-08 | -5.03%   |                      59 |                      68 |         127 |
| 2013-05-22 | 2013-08-21 | 2013-11-29 | -5.70%   |                      91 |                     100 |         191 |
| 2015-04-16 | 2015-09-28 | 2016-04-13 | -7.19%   |                     165 |                     198 |         363 |
| 2016-08-01 | 2016-11-14 | 2017-04-18 | -5.76%   |                     105 |                     155 |         260 |
| 2018-01-29 | 2019-01-14 | 2019-04-26 | -8.35%   |                     350 |                     102 |         452 |
| 2020-09-03 | 2021-02-26 | 2021-07-07 | -6.15%   |                     176 |                     131 |         307 |
| 2021-11-10 | 2022-03-14 | 2023-06-13 | -10.67%  |                     124 |                     456 |         580 |
| 2024-04-01 | 2025-01-27 | open       | -12.35%  |                     301 |                     nan |         nan |

**HMM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2002-03-07 | 2002-06-24 | 2004-01-21 | -7.38%   |                     109 |                     576 |         685 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                      63 |                     178 |         241 |
| 2008-05-20 | 2008-07-28 | 2009-08-21 | -7.57%   |                      69 |                     389 |         458 |
| 2010-05-04 | 2010-07-02 | 2010-09-14 | -5.69%   |                      59 |                      74 |         133 |
| 2011-07-25 | 2011-11-01 | 2012-04-26 | -7.07%   |                      99 |                     177 |         276 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                      33 |                     120 |         153 |
| 2015-03-23 | 2015-09-28 | 2016-06-02 | -9.07%   |                     189 |                     248 |         437 |
| 2016-08-01 | 2016-11-14 | 2017-04-17 | -5.64%   |                     105 |                     154 |         259 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                      10 |                     197 |         207 |
| 2018-08-30 | 2018-12-24 | 2019-03-21 | -11.45%  |                     116 |                      87 |         203 |
| 2020-02-21 | 2020-03-18 | 2020-04-09 | -9.31%   |                      26 |                      22 |          48 |
| 2021-01-26 | 2021-03-04 | 2021-05-07 | -5.96%   |                      37 |                      64 |         101 |
| 2021-11-10 | 2023-10-27 | 2024-09-16 | -22.68%  |                     716 |                     325 |        1041 |
| 2024-12-09 | 2025-04-08 | 2025-06-24 | -12.22%  |                     120 |                      77 |         197 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                      29 |                      21 |          50 |

**HMM_Uni**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2001-02-02 | 2003-05-19 | 2003-12-18 | -8.03%   |                     836 |                     213 |        1049 |
| 2004-03-08 | 2004-05-10 | 2004-10-29 | -5.81%   |                      63 |                     172 |         235 |
| 2007-06-05 | 2010-02-10 | 2010-09-08 | -7.31%   |                     981 |                     210 |        1191 |
| 2013-05-22 | 2013-08-21 | 2013-11-29 | -5.70%   |                      91 |                     100 |         191 |
| 2015-04-16 | 2015-09-28 | 2016-04-13 | -7.19%   |                     165 |                     198 |         363 |
| 2016-08-01 | 2016-11-14 | 2017-04-18 | -5.76%   |                     105 |                     155 |         260 |
| 2018-01-29 | 2019-01-14 | 2019-04-24 | -8.35%   |                     350 |                     100 |         450 |
| 2020-09-03 | 2021-02-26 | 2021-07-07 | -6.15%   |                     176 |                     131 |         307 |
| 2021-11-10 | 2022-03-14 | 2023-06-15 | -10.72%  |                     124 |                     458 |         582 |
| 2024-04-01 | 2025-01-27 | open       | -12.35%  |                     301 |                     nan |         nan |

**LSTM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                      49 |                      43 |          92 |
| 2001-02-02 | 2002-07-23 | 2004-03-05 | -24.04%  |                     536 |                     591 |        1127 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                      63 |                     178 |         241 |
| 2007-12-07 | 2009-03-09 | 2010-11-04 | -31.53%  |                     458 |                     605 |        1063 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                      14 |                      67 |          81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                      33 |                     120 |         153 |
| 2015-03-23 | 2015-08-25 | 2016-06-30 | -8.39%   |                     155 |                     310 |         465 |
| 2016-08-01 | 2016-11-14 | 2017-04-17 | -5.64%   |                     105 |                     154 |         259 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                      10 |                     197 |         207 |
| 2018-08-30 | 2018-12-24 | 2019-03-21 | -11.45%  |                     116 |                      87 |         203 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                      26 |                      82 |         108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                      57 |                      39 |          96 |
| 2021-12-28 | 2022-10-14 | 2024-12-02 | -27.55%  |                     290 |                     780 |        1070 |
| 2024-12-09 | 2025-04-08 | 2025-07-03 | -12.22%  |                     120 |                      86 |         206 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                      29 |                      21 |          50 |

**Transformer**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                      49 |                      43 |          92 |
| 2001-02-02 | 2001-09-21 | 2004-12-15 | -17.28%  |                     231 |                    1181 |        1412 |
| 2007-11-01 | 2009-03-09 | 2009-08-26 | -21.54%  |                     494 |                     170 |         664 |
| 2010-05-04 | 2010-07-02 | 2010-09-13 | -5.36%   |                      59 |                      73 |         132 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                      14 |                      67 |          81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                      33 |                     120 |         153 |
| 2015-03-23 | 2015-08-25 | 2017-06-01 | -8.39%   |                     155 |                     646 |         801 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                      10 |                     197 |         207 |
| 2018-08-30 | 2018-12-24 | 2019-03-21 | -11.45%  |                     116 |                      87 |         203 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                      26 |                      82 |         108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                      57 |                      39 |          96 |
| 2021-12-28 | 2022-10-14 | 2024-11-29 | -27.55%  |                     290 |                     777 |        1067 |
| 2024-12-09 | 2025-04-08 | 2025-07-03 | -12.22%  |                     120 |                      86 |         206 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                      29 |                      21 |          50 |

### Crisis Performance
Return and max drawdown during historical crisis periods: the central evidence for the tail-risk protection of the regime-switching models.

| Crisis                              | ('Return', 'Buy_Hold')   | ('Return', 'HMM')   | ('Return', 'HMM_Uni')   | ('Return', 'LSTM')   | ('Return', 'MSM')   | ('Return', 'Transformer')   | ('Max Drawdown', 'Buy_Hold')   | ('Max Drawdown', 'HMM')   | ('Max Drawdown', 'HMM_Uni')   | ('Max Drawdown', 'LSTM')   | ('Max Drawdown', 'MSM')   | ('Max Drawdown', 'Transformer')   |
|:------------------------------------|:-------------------------|:--------------------|:------------------------|:---------------------|:--------------------|:----------------------------|:-------------------------------|:--------------------------|:------------------------------|:---------------------------|:--------------------------|:----------------------------------|
| COVID Crash (2020-02 to 2020-03)    | -8.55%                   | -3.14%              | +0.73%                  | -8.55%               | +0.73%              | -8.55%                      | -18.53%                        | -9.55%                    | -1.81%                        | -18.53%                    | -1.81%                    | -18.53%                           |
| Dot-Com (2000-03 to 2002-10)        | -15.77%                  | -2.81%              | -3.81%                  | -15.77%              | -3.81%              | -9.82%                      | -24.81%                        | -7.90%                    | -8.18%                        | -24.81%                    | -8.18%                    | -18.12%                           |
| EU Debt Crisis (2011-07 to 2011-11) | +4.10%                   | -5.52%              | +2.78%                  | +4.10%               | +2.78%              | +4.10%                      | -7.24%                         | -7.71%                    | -4.33%                        | -7.24%                     | -4.33%                    | -7.24%                            |
| GFC (2007-10 to 2009-03)            | -26.99%                  | -3.89%              | -3.09%                  | -22.61%              | -3.09%              | -11.14%                     | -35.08%                        | -7.89%                    | -4.14%                        | -31.72%                    | -4.14%                    | -21.67%                           |
| Rate Hikes (2022-01 to 2022-10)     | -24.20%                  | -16.15%             | -6.87%                  | -24.20%              | -6.87%              | -24.20%                     | -26.98%                        | -16.63%                   | -8.45%                        | -26.98%                    | -8.45%                    | -26.98%                           |

### Switch Timing Relative to the Drawdown Peak
Time lag between the model's first bear signal and the drawdown trough of the buy-and-hold portfolio per crisis. Positive = model reacted early, negative = too late.

| Crisis   | Model       | DD Trough   | First Bear Signal   |   Lead (Days) |
|:---------|:------------|:------------|:--------------------|--------------:|
| GFC      | MSM         | 2009-03-09  | 2007-10-01          |           525 |
| COVID    | MSM         | 2020-03-18  | 2020-02-24          |            23 |
| 2022     | MSM         | 2022-10-14  | 2022-01-05          |           282 |
| GFC      | HMM         | 2009-03-09  | 2007-10-24          |           502 |
| COVID    | HMM         | 2020-03-18  | 2020-02-03          |            44 |
| 2022     | HMM         | 2022-10-14  | 2022-01-24          |           263 |
| GFC      | HMM_Uni     | 2009-03-09  | 2007-10-01          |           525 |
| COVID    | HMM_Uni     | 2020-03-18  | 2020-02-24          |            23 |
| 2022     | HMM_Uni     | 2022-10-14  | 2022-01-05          |           282 |
| GFC      | LSTM        | 2009-03-09  | 2007-10-01          |           525 |
| COVID    | LSTM        | 2020-03-18  |                     |           nan |
| 2022     | LSTM        | 2022-10-14  |                     |           nan |
| GFC      | Transformer | 2009-03-09  | 2007-10-01          |           525 |
| COVID    | Transformer | 2020-03-18  |                     |           nan |
| 2022     | Transformer | 2022-10-14  |                     |           nan |

### Drawdown Profile
![Drawdown](../assets/drawdown.png)

### Rolling Sharpe Ratio
Time-varying, risk-adjusted return comparison over a rolling 252-day window.

![Rolling Sharpe](../assets/rolling_sharpe.png)

### Comprehensive Metrics Matrix
Detailed statistical analysis including risk-adjusted metrics (Sharpe, Sortino, Calmar).

| Strategy    | Total Return   | CAGR (p.a.)   | Volatility   | Max Drawdown   |   Sharpe Ratio |   Sortino Ratio |   Calmar Ratio |   Regime Switches | Total Costs (Fees)   |   Ulcer Index |
|:------------|:---------------|:--------------|:-------------|:---------------|---------------:|----------------:|---------------:|------------------:|:---------------------|--------------:|
| Buy Hold    | 319.66%        | 5.74%         | 11.17%       | -35.08%        |           0.56 |            0.73 |           0.16 |                 0 | 0.00%                |          9.07 |
| MSM         | 210.46%        | 4.51%         | 6.04%        | -12.79%        |           0.76 |            0.85 |           0.35 |               239 | 24.00%               |          4.45 |
| HMM         | 339.94%        | 5.93%         | 7.60%        | -22.92%        |           0.8  |            0.96 |           0.26 |                81 | 8.20%                |          5.43 |
| HMM Uni     | 208.43%        | 4.48%         | 6.04%        | -12.79%        |           0.76 |            0.84 |           0.35 |               239 | 24.00%               |          4.5  |
| LSTM        | 294.84%        | 5.49%         | 10.77%       | -31.72%        |           0.55 |            0.68 |           0.17 |                12 | 1.20%                |          8.47 |
| Transformer | 293.04%        | 5.47%         | 9.51%        | -27.71%        |           0.61 |            0.71 |           0.2  |                32 | 3.20%                |          7.43 |

### Transaction Costs

This figure shows the cumulative transaction costs over time. Steep increases indicate unstable regime switches ("churning").

![Transaction Costs](../assets/transaction_costs.png)

Stress test: Sequence of Returns Risk (SORR)
In addition, the survival time of the capital was simulated in a withdrawal phase (retirement scenario).

### SORR Simulation: Comparison of the Withdrawal Scenarios

This table compares different stress scenarios (standard, aggressive, low capital).

|                                | Terminal Capital   | Status            |
|:-------------------------------|:-------------------|:------------------|
| ('Standard', 'Buy Hold')       | 0.00 €             | Depleted (2026)   |
| ('Standard', 'MSM')            | 83,076.74 €        | Capital preserved |
| ('Standard', 'HMM')            | 171,847.31 €       | Capital preserved |
| ('Standard', 'HMM Uni')        | 81,548.05 €        | Capital preserved |
| ('Standard', 'LSTM')           | 0.00 €             | Depleted (2024)   |
| ('Standard', 'Transformer')    | 0.00 €             | Depleted (2025)   |
| ('Aggressive', 'Buy Hold')     | 0.00 €             | Depleted (2011)   |
| ('Aggressive', 'MSM')          | 0.00 €             | Depleted (2013)   |
| ('Aggressive', 'HMM')          | 0.00 €             | Depleted (2013)   |
| ('Aggressive', 'HMM Uni')      | 0.00 €             | Depleted (2013)   |
| ('Aggressive', 'LSTM')         | 0.00 €             | Depleted (2011)   |
| ('Aggressive', 'Transformer')  | 0.00 €             | Depleted (2011)   |
| ('Low_Capital', 'Buy Hold')    | 0.00 €             | Depleted (2015)   |
| ('Low_Capital', 'MSM')         | 0.00 €             | Depleted (2017)   |
| ('Low_Capital', 'HMM')         | 0.00 €             | Depleted (2017)   |
| ('Low_Capital', 'HMM Uni')     | 0.00 €             | Depleted (2017)   |
| ('Low_Capital', 'LSTM')        | 0.00 €             | Depleted (2014)   |
| ('Low_Capital', 'Transformer') | 0.00 €             | Depleted (2014)   |

Capital development of the different scenarios:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Stationary Bootstrap Robustness Check

To assess statistical significance, 10,000 artificial market paths were simulated via stationary bootstrap (Politis & Romano 1994).
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin Probability   | Median Terminal Capital   |
|:-------------------------------|:-------------------|:--------------------------|
| ('Low_Capital', 'HMM')         | 0.01%              | 207,896.30 €              |
| ('Aggressive', 'HMM')          | 0.74%              | 235,949.61 €              |
| ('Standard', 'HMM')            | 0.00%              | 485,283.95 €              |
| ('Aggressive', 'MSM')          | 0.87%              | 171,422.78 €              |
| ('Low_Capital', 'Buy Hold')    | 0.55%              | 204,973.28 €              |
| ('Standard', 'LSTM')           | 0.01%              | 455,137.45 €              |
| ('Standard', 'Transformer')    | 0.00%              | 456,262.76 €              |
| ('Aggressive', 'HMM Uni')      | 0.86%              | 170,252.61 €              |
| ('Standard', 'Buy Hold')       | 0.01%              | 473,380.85 €              |
| ('Standard', 'MSM')            | 0.00%              | 399,032.11 €              |
| ('Aggressive', 'Transformer')  | 3.23%              | 217,136.79 €              |
| ('Low_Capital', 'MSM')         | 0.01%              | 163,842.75 €              |
| ('Aggressive', 'LSTM')         | 5.27%              | 215,175.67 €              |
| ('Aggressive', 'Buy Hold')     | 5.03%              | 227,946.92 €              |
| ('Standard', 'HMM Uni')        | 0.00%              | 397,306.37 €              |
| ('Low_Capital', 'HMM Uni')     | 0.01%              | 162,246.59 €              |
| ('Low_Capital', 'LSTM')        | 0.58%              | 194,619.62 €              |
| ('Low_Capital', 'Transformer') | 0.29%              | 193,522.14 €              |

Distribution of the terminal capital values:

![MCS Boxplots Standard](../assets/mcs_boxplot_standard.png)
![MCS Boxplots Aggressive](../assets/mcs_boxplot_aggressive.png)
![MCS Boxplots Low Capital](../assets/mcs_boxplot_low_capital.png)

Probability corridors:

The shaded areas show the 5% to 95% confidence interval of the capital development.
![MCS Quantiles](../assets/mcs_quantiles.png)

### Depletion Rate with 95% Confidence Interval
Wilson CI for the ruin probability (P[terminal capital ≤ 0]) per scenario × strategy.

|                                | Depletion Rate   | 95% CI Lower   | 95% CI Upper   | n_ruin / n_paths   |
|:-------------------------------|:-----------------|:---------------|:---------------|:-------------------|
| ('Standard', 'Buy_Hold')       | 0.01%            | 0.00%          | 0.06%          | 1/10000            |
| ('Standard', 'MSM')            | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Standard', 'HMM')            | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Standard', 'HMM_Uni')        | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Standard', 'LSTM')           | 0.01%            | 0.00%          | 0.06%          | 1/10000            |
| ('Standard', 'Transformer')    | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Aggressive', 'Buy_Hold')     | 5.03%            | 4.62%          | 5.48%          | 503/10000          |
| ('Aggressive', 'MSM')          | 0.87%            | 0.71%          | 1.07%          | 87/10000           |
| ('Aggressive', 'HMM')          | 0.74%            | 0.59%          | 0.93%          | 74/10000           |
| ('Aggressive', 'HMM_Uni')      | 0.86%            | 0.70%          | 1.06%          | 86/10000           |
| ('Aggressive', 'LSTM')         | 5.27%            | 4.85%          | 5.73%          | 527/10000          |
| ('Aggressive', 'Transformer')  | 3.23%            | 2.90%          | 3.59%          | 323/10000          |
| ('Low_Capital', 'Buy_Hold')    | 0.55%            | 0.42%          | 0.72%          | 55/10000           |
| ('Low_Capital', 'MSM')         | 0.01%            | 0.00%          | 0.06%          | 1/10000            |
| ('Low_Capital', 'HMM')         | 0.01%            | 0.00%          | 0.06%          | 1/10000            |
| ('Low_Capital', 'HMM_Uni')     | 0.01%            | 0.00%          | 0.06%          | 1/10000            |
| ('Low_Capital', 'LSTM')        | 0.58%            | 0.45%          | 0.75%          | 58/10000           |
| ('Low_Capital', 'Transformer') | 0.29%            | 0.20%          | 0.42%          | 29/10000           |

### Hypothesis Tests (Paired Wilcoxon, α = 0.05)
**H1: Regime switching reduces MaxDD vs. buy and hold:**

| Model       | Median MaxDD (Model)   | Median MaxDD (B&H)   | Δ Median   |   Wilcoxon p | H1 (α=0.05)   |
|:------------|:-----------------------|:---------------------|:-----------|-------------:|:--------------|
| MSM         | -66.88%                | -59.34%              | -7.55 pp   |     1        | rejected      |
| HMM         | -55.87%                | -59.34%              | +3.46 pp   |     2.64e-38 | confirmed     |
| HMM_Uni     | -67.17%                | -59.34%              | -7.84 pp   |     1        | rejected      |
| LSTM        | -60.95%                | -59.34%              | -1.62 pp   |     1        | rejected      |
| Transformer | -60.11%                | -59.34%              | -0.77 pp   |     0.907    | rejected      |

**H2: The Transformer dominates econometrics and LSTM in terminal wealth:**

| Comparison           | Median Transformer   | Median MSM   | Δ Median   |   Wilcoxon p | H2 (α=0.05)   | Median HMM   | Median LSTM   |
|:---------------------|:---------------------|:-------------|:-----------|-------------:|:--------------|:-------------|:--------------|
| Transformer vs. MSM  | 217,137 €            | 171,423 €    | +45,714 €  |    9.91e-170 | confirmed     | nan          | nan           |
| Transformer vs. HMM  | 217,137 €            | nan          | -18,813 €  |    1         | rejected      | 235,950 €    | nan           |
| Transformer vs. LSTM | 217,137 €            | nan          | +1,961 €   |    0.914     | rejected      | nan          | 215,176 €     |

### Break-Even Transaction Costs
At what cost rate (in basis points per reallocation) does active switching lose its return advantage over buy and hold?

| Model       |   Final @10bps |   B&H Final |   Break-Even (bps) |
|:------------|---------------:|------------:|-------------------:|
| MSM         |          3.104 |       4.196 |                  0 |
| HMM         |          4.399 |       4.196 |                 20 |
| HMM_Uni     |          3.084 |       4.196 |                  0 |
| LSTM        |          3.948 |       4.196 |                  0 |
| Transformer |          3.93  |       4.196 |                  0 |

![Break-Even Analysis](../assets/break_even_costs.png)

### Withdrawal Rate Sensitivity (3.5% / 4% / 5%)
Robustness of the SORR results under varying annual withdrawals.

| Strategy    | ('Terminal Capital', '3.5%')   | ('Terminal Capital', '4.0%')   | ('Terminal Capital', '5.0%')   | ('Status', '3.5%')   | ('Status', '4.0%')   | ('Status', '5.0%')   |
|:------------|:-------------------------------|:-------------------------------|:-------------------------------|:---------------------|:---------------------|:---------------------|
| Buy_Hold    | 866,713 €                      | 690,770 €                      | 338,886 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| HMM         | 1,016,777 €                    | 847,791 €                      | 509,819 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| HMM_Uni     | 690,123 €                      | 568,408 €                      | 324,978 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| LSTM        | 783,709 €                      | 613,635 €                      | 273,488 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| MSM         | 695,258 €                      | 572,822 €                      | 327,949 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| Transformer | 789,762 €                      | 621,841 €                      | 286,000 €                      | Capital preserved    | Capital preserved    | Capital preserved    |

---

## Research Notes & Methodology
- **Cash component:** On a "bear" signal, the strategy reallocates into the current money market rate (**^IRX**).
- **Look-ahead bias prevention:** All signals are shifted by one day for the backtesting (`shift(1)`) to simulate real trading conditions.
- **Feature set:** The models use returns, volatility (20d), SMA distance, momentum, VIX, and yield spread.
- **Cost simulation:** A flat fee of 10 basis points (0.1%) is charged per reallocation.
- **SORR specifics:** For withdrawals during "bull" phases, an additional liquidity fee of 0.1% is charged on the withdrawal amount (asset sales). In "bear" phases (cash), this fee does not apply.

---

## Model Persistence

Model persistence status for this pipeline run:

- **Persistence:** ENABLED
- **Model directory:** `../models`

| Model | File | Status |
|:---|:---|:---|
| MSM | `msm_regime_model.pkl` | Newly trained |
| HMM | `hmm_regime_model.pkl` | Newly trained |
| LSTM | `lstm_regime_model.keras` | Newly trained |
| TRANSFORMER | `transformer_regime_model.pt` | Newly trained |

> **Note:** With persistence enabled, pre-trained models are loaded from `../models` if the files exist. Otherwise, training runs normally and the result is stored for future runs. When hyperparameters change, the corresponding model files must be deleted.

---

**Last updated:** 2026-07-05 10:56<br>
**End date:** `2026-07-05`<br>
**Fast mode status at runtime:** FALSE (Full Run)<br>
**Walk-forward validation:** ENABLED (mode: rolling, train: 10y, test: 12m, step: 12m)<br>
**Model persistence:** ENABLED<br>
*Generated by the Backtest Service (reporting).*
