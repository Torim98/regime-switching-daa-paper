
# Detailed Statistical Evaluation & Research Results

This page documents the numerical and graphical results of the research pipeline. All evaluations are based on the dataset up to yesterday (2026-07-09) and are updated automatically.

---

## 1. Executive Summary: Performance & Risk
A direct comparison of the core metrics over the entire **out-of-sample test period**.

| Strategy    | Final Wealth   | Total Return   | Max Drawdown   |
|:------------|:---------------|:---------------|:---------------|
| Buy_Hold    | 2,097,913 €    | +319.58%       | -35.08%        |
| MSM         | 1,632,308 €    | +226.46%       | -10.91%        |
| HMM         | 1,211,580 €    | +142.32%       | -23.55%        |
| HMM_Uni     | 1,619,457 €    | +223.89%       | -11.01%        |
| LSTM        | 2,338,545 €    | +367.71%       | -27.71%        |
| Transformer | 2,472,399 €    | +394.48%       | -27.71%        |

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

_Generated at 2026-07-09 10:00:33_

| model       | metric   |   best_value |   best_trial |   conv_frac |   n_complete |   n_pruned | top_importance                                         | edge_flags                                                                                                                                            |
|:------------|:---------|-------------:|-------------:|------------:|-------------:|-----------:|:-------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|
| MSM         | martin   |       1.6508 |           19 |        0.54 |           36 |          0 |                                                        | -                                                                                                                                                     |
| HMM         | martin   |       1.2037 |            4 |        0.04 |          108 |          0 | covariance_type=1.00                                   | -                                                                                                                                                     |
| HMM_Uni     | martin   |       1.6384 |           19 |        0.54 |           36 |          0 |                                                        | -                                                                                                                                                     |
| LSTM        | martin   |       2.6426 |          129 |        0.64 |          194 |          0 | units_l1=0.27, threshold=0.21, dropout=0.15            | window_size=250 near UPPER bound 250; learning_rate=1.3292520737887431e-05 near LOWER bound 1e-05; threshold=0.15000000000000002 near LOWER bound 0.1 |
| Transformer | martin   |       1.3728 |           42 |        0.22 |          190 |          0 | learning_rate=0.50, dmodel_nheads=0.18, threshold=0.12 | n_layers=3 near UPPER bound 4; threshold=0.15000000000000002 near LOWER bound 0.1                                                                     |


#### G.2 Objective Sensitivity
Which config would have been selected under each candidate metric, valued across all metrics. `same_as_objective = True` throughout means the model choice is robust to the objective; divergences quantify the trade-off.

# Objective Sensitivity of the Selected Hyperparameters

_Generated at 2026-07-09 10:00:34_  
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

## LSTM (objective: martin, 194 trials)

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

## Transformer (objective: martin, 190 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      42 | True                |   1.3728 |   0.7082 |    0.8823 |   0.319  |  4.6002 |        -0.1979 | 0.0632 |
| sharpe          |      14 | False               |   1.3506 |   0.7152 |    0.9173 |   0.2847 |  4.0869 |        -0.1939 | 0.0552 |
| sortino         |      84 | False               |   0.8467 |   0.6852 |    0.9236 |   0.2245 |  6.8603 |        -0.2587 | 0.0581 |
| calmar          |      42 | True                |   1.3728 |   0.7082 |    0.8823 |   0.319  |  4.6002 |        -0.1979 | 0.0632 |
| ulcer           |      14 | False               |   1.3506 |   0.7152 |    0.9173 |   0.2847 |  4.0869 |        -0.1939 | 0.0552 |
| max_drawdown    |      51 | False               |   0.2719 |   0.4477 |    0.4366 |   0.1581 |  8.8183 |        -0.1517 | 0.024  |

Selected configs:
- best under **martin**: window_size=80, dmodel_nheads=128-4, n_layers=3, dim_feedforward=128, batch_size=128, learning_rate=0.002661901888489057, dropout=0.15000000000000002, threshold=0.15000000000000002
- best under **sharpe**: window_size=250, dmodel_nheads=16-2, n_layers=3, dim_feedforward=32, batch_size=64, learning_rate=0.0003589128083678785, dropout=0.45, threshold=0.7000000000000001
- best under **sortino**: window_size=220, dmodel_nheads=32-2, n_layers=4, dim_feedforward=32, batch_size=128, learning_rate=0.0028818323079306337, dropout=0.30000000000000004, threshold=0.6
- best under **max_drawdown**: window_size=40, dmodel_nheads=32-2, n_layers=4, dim_feedforward=32, batch_size=32, learning_rate=2.015647705936502e-05, dropout=0.35000000000000003, threshold=0.1


#### G.3 Deflated Sharpe Ratio & Backtest Overfitting
DSR deflates the selected config's Sharpe for the number of tested configs; PBO (CSCV) estimates the probability that the in-sample-best config underperforms out-of-sample.

# Deflated Sharpe Ratio (multiple-testing adjusted)

_Generated at 2026-07-09 10:09:36_

| model       |   n_trials |   sr_ann_best |   sr_star_ann |    dsr | significant_5pct   |
|:------------|-----------:|--------------:|--------------:|-------:|:-------------------|
| MSM         |         36 |        0.8513 |        0.3085 | 0.9843 | True               |
| HMM         |        108 |        0.7243 |        0.4648 | 0.8478 | False              |
| HMM_Uni     |         36 |        0.842  |        0.3216 | 0.9804 | True               |
| LSTM        |        194 |        0.8604 |        0.378  | 0.9717 | True               |
| Transformer |        190 |        0.3075 |        0.3105 | 0.4952 | False              |

DSR = P(true Sharpe > 0) after deflating the best trial's Sharpe for the number of tested configs. `significant_5pct` = DSR > 0.95.


# Probability of Backtest Overfitting (CSCV)

_Generated at 2026-07-09 11:23:27_

| model       |   folds |   configs |   pbo |
|:------------|--------:|----------:|------:|
| MSM         |      16 |        15 | 0.444 |
| HMM         |      16 |        15 | 0.456 |
| HMM_Uni     |      16 |        15 | 0.452 |
| LSTM        |      16 |        15 | 0.933 |
| Transformer |      16 |        15 | 0.821 |

PBO over the top-15 configs per model (per-fold Sharpe matrix, CSCV). Lower is better; PBO > 0.5 flags overfitting.


#### G.4 Multi-Seed Robustness (DL)
Top DL configs re-evaluated over several global seeds; mean/std of every metric quantify seed sensitivity of the deep-learning ranking.

# Multi-Seed Re-Evaluation (top-5, 5 seeds)

_Generated at 2026-07-09 13:09:06_

### LSTM

|   rank |   trial |   martin_hpo |   martin_mean |   martin_std |   sharpe_mean |   sharpe_std |   sortino_mean |   sortino_std |   calmar_mean |   calmar_std |   ulcer_mean |   ulcer_std |   max_drawdown_mean |   max_drawdown_std |   cagr_mean |   cagr_std |
|-------:|--------:|-------------:|--------------:|-------------:|--------------:|-------------:|---------------:|--------------:|--------------:|-------------:|-------------:|------------:|--------------------:|-------------------:|------------:|-----------:|
|      1 |     129 |       2.6426 |        1.7805 |       0.6827 |        0.8067 |       0.1466 |         0.7741 |        0.1398 |        0.4014 |       0.1623 |       3.0809 |      1.2704 |             -0.1323 |             0.0445 |      0.0471 |     0.0096 |
|      2 |     148 |       2.4283 |        0.9103 |       0.3632 |        0.6192 |       0.0843 |         0.7727 |        0.1456 |        0.2343 |       0.0555 |       6.1313 |      1.1146 |             -0.2243 |             0.0219 |      0.0518 |     0.0091 |
|      3 |     136 |       2.3832 |        1.0713 |       0.4096 |        0.6349 |       0.0819 |         0.7827 |        0.1286 |        0.2453 |       0.0669 |       5.5296 |      1.4117 |             -0.2314 |             0.0536 |      0.0538 |     0.0068 |
|      4 |     160 |       2.3163 |        1.5211 |       0.8075 |        0.7027 |       0.13   |         0.7237 |        0.0774 |        0.3201 |       0.1432 |       3.5809 |      1.5409 |             -0.157  |             0.0578 |      0.0421 |     0.0048 |
|      5 |      93 |       2.3038 |        0.9095 |       0.2615 |        0.6057 |       0.0651 |         0.7946 |        0.0931 |        0.2233 |       0.0469 |       6.0744 |      1.1286 |             -0.2415 |             0.0426 |      0.0525 |     0.0066 |

### Transformer

|   rank |   trial |   martin_hpo |   martin_mean |   martin_std |   sharpe_mean |   sharpe_std |   sortino_mean |   sortino_std |   calmar_mean |   calmar_std |   ulcer_mean |   ulcer_std |   max_drawdown_mean |   max_drawdown_std |   cagr_mean |   cagr_std |
|-------:|--------:|-------------:|--------------:|-------------:|--------------:|-------------:|---------------:|--------------:|--------------:|-------------:|-------------:|------------:|--------------------:|-------------------:|------------:|-----------:|
|      1 |      42 |       1.3728 |        0.3547 |       0.2716 |        0.3406 |       0.15   |         0.4084 |        0.1883 |        0.1049 |       0.0645 |       9.6546 |      3.5233 |             -0.2623 |             0.0455 |      0.0251 |     0.0126 |
|      2 |      14 |       1.3506 |        0.5671 |       0.129  |        0.521  |       0.0455 |         0.6504 |        0.0559 |        0.1605 |       0.0241 |       7.6211 |      1.0643 |             -0.263  |             0.0148 |      0.0419 |     0.0047 |
|      3 |      29 |       1.0487 |        0.4931 |       0.1196 |        0.4508 |       0.0509 |         0.5396 |        0.0698 |        0.1408 |       0.0225 |       8.207  |      1.38   |             -0.2776 |             0.0105 |      0.0389 |     0.0052 |
|      4 |      60 |       1.04   |        0.3547 |       0.2716 |        0.3406 |       0.15   |         0.4084 |        0.1883 |        0.1049 |       0.0645 |       9.6546 |      3.5233 |             -0.2623 |             0.0455 |      0.0251 |     0.0126 |
|      5 |     160 |       0.8924 |        0.7106 |       0.1239 |        0.5319 |       0.0575 |         0.6722 |        0.0873 |        0.191  |       0.0236 |       7.0401 |      0.6032 |             -0.2585 |             0.0039 |      0.0493 |     0.0056 |


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
| MSM         | +4.72% | 6.02%             |          0.796 |           0.895 | -10.91%        |          0.433 |       6464 |        25.7 |
| HMM         | +3.51% | 7.45%             |          0.501 |           0.46  | -23.55%        |          0.149 |       6464 |        25.7 |
| HMM_Uni     | +4.69% | 6.03%             |          0.79  |           0.889 | -11.01%        |          0.426 |       6464 |        25.7 |
| LSTM        | +6.20% | 10.46%            |          0.627 |           0.763 | -27.71%        |          0.224 |       6464 |        25.7 |
| Transformer | +6.43% | 9.89%             |          0.68  |           0.855 | -27.71%        |          0.232 |       6464 |        25.7 |

### Classification Metrics (vs. NBER Recessions as Ground Truth)
Comparison of the models as binary recession classifiers (precision, recall, F1).

| Model       |   Precision |   Recall |    F1 |   TN |   FP |   FN |   TP |
|:------------|------------:|---------:|------:|-----:|-----:|-----:|-----:|
| MSM         |       0.228 |    0.922 | 0.366 | 4054 | 1825 |   46 |  540 |
| HMM         |       0.128 |    0.614 | 0.211 | 3419 | 2460 |  226 |  360 |
| HMM_Uni     |       0.229 |    0.922 | 0.366 | 4058 | 1821 |   46 |  540 |
| LSTM        |       0.187 |    0.333 | 0.239 | 5029 |  850 |  391 |  195 |
| Transformer |       0.238 |    0.336 | 0.279 | 5248 |  631 |  389 |  197 |

![Confusion Matrices](../assets/confusion_matrices.png)

**ROC and precision-recall curves** (threshold-independent comparison via `*_Prob`):

![ROC Curves](../assets/roc_curves.png)
![PR Curves](../assets/pr_curves.png)

### Signal Churning & Whipsaw Analysis
Quantification of the switching frequency and the share of very short regime phases ("whipsaws").

| Model       |   Signal Switches |   Whipsaws (<5d) | Whipsaw Share   |   Mean Phase (Days) |   Median Phase (Days) | Cumul. Costs   |
|:------------|------------------:|-----------------:|:----------------|--------------------:|----------------------:|:---------------|
| MSM         |               323 |              159 | 49.1%           |                20   |                     5 | 32.30%         |
| HMM         |                31 |                2 | 6.2%            |               202   |                    94 | 3.10%          |
| HMM_Uni     |               317 |              153 | 48.1%           |                20.3 |                     5 | 31.70%         |
| LSTM        |                12 |                0 | 0.0%            |               497.3 |                    97 | 1.20%          |
| Transformer |                14 |                4 | 26.7%           |               431   |                    36 | 1.40%          |

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
|        0.4  | 1,151,748 €    | -23.55%        |         37 |
|        0.45 | 1,147,318 €    | -23.55%        |         37 |
|        0.5  | 1,149,909 €    | -23.55%        |         37 |
|        0.55 | 1,137,764 €    | -23.55%        |         37 |
|        0.6  | 1,160,435 €    | -23.55%        |         35 |

**HMM_Uni**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.4  | 1,353,759 €    | -24.79%        |        315 |
|        0.45 | 1,369,795 €    | -24.41%        |        325 |
|        0.5  | 1,492,706 €    | -25.57%        |        331 |
|        0.55 | 1,472,022 €    | -23.24%        |        331 |
|        0.6  | 1,676,047 €    | -26.67%        |        337 |

**LSTM**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|         0.2 | 2,261,693 €    | -27.71%        |          8 |
|         0.3 | 2,170,140 €    | -27.71%        |          6 |
|         0.4 | 2,104,408 €    | -27.71%        |         12 |
|         0.5 | 2,163,209 €    | -27.71%        |         12 |

**Transformer**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.3  | 2,014,848 €    | -27.71%        |         46 |
|        0.4  | 2,130,331 €    | -27.71%        |         30 |
|        0.45 | 2,191,293 €    | -27.71%        |         22 |
|        0.5  | 2,113,893 €    | -27.71%        |         22 |
|        0.6  | 2,403,758 €    | -27.71%        |         16 |

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
| 2001-02-02 | 2003-05-20 | 2003-09-18 | -5.82%   |                     837 |                     121 |         958 |
| 2006-01-17 | 2006-07-18 | 2006-09-20 | -5.08%   |                     182 |                      64 |         246 |
| 2007-06-05 | 2009-11-04 | 2010-04-21 | -7.68%   |                     883 |                     168 |        1051 |
| 2010-05-04 | 2010-07-02 | 2010-09-13 | -5.64%   |                      59 |                      73 |         132 |
| 2013-05-22 | 2013-08-21 | 2013-11-29 | -5.70%   |                      91 |                     100 |         191 |
| 2015-02-26 | 2015-09-28 | 2016-03-16 | -7.59%   |                     214 |                     170 |         384 |
| 2016-08-01 | 2016-11-14 | 2017-04-18 | -5.76%   |                     105 |                     155 |         260 |
| 2018-01-29 | 2018-05-02 | 2018-08-27 | -5.92%   |                      93 |                     117 |         210 |
| 2018-08-30 | 2019-01-14 | 2019-03-21 | -5.95%   |                     137 |                      66 |         203 |
| 2020-09-03 | 2021-02-26 | 2021-07-02 | -6.85%   |                     176 |                     126 |         302 |
| 2021-11-10 | 2022-07-14 | 2023-12-13 | -10.56%  |                     246 |                     517 |         763 |
| 2024-09-03 | 2025-06-20 | 2025-10-02 | -7.99%   |                     290 |                     104 |         394 |
| 2025-10-29 | 2026-07-02 | open       | -5.57%   |                     246 |                     nan |         nan |

**HMM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2001-11-14 | 2002-07-23 | 2003-06-16 | -18.68%  |                     251 |                     328 |         579 |
| 2003-06-17 | 2003-08-05 | 2003-12-04 | -7.21%   |                      49 |                     121 |         170 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                      63 |                     178 |         241 |
| 2008-05-20 | 2008-10-10 | 2014-02-28 | -23.28%  |                     143 |                    1967 |        2110 |
| 2014-09-02 | 2016-11-14 | 2017-06-02 | -7.83%   |                     804 |                     200 |        1004 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                      10 |                     197 |         207 |
| 2018-08-30 | 2018-12-24 | 2019-06-07 | -10.21%  |                     116 |                     165 |         281 |
| 2020-02-21 | 2020-03-12 | 2020-08-05 | -14.02%  |                      20 |                     146 |         166 |
| 2021-12-28 | 2022-06-16 | open       | -23.28%  |                     170 |                     nan |         nan |

**HMM_Uni**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2001-02-02 | 2003-05-20 | 2003-09-18 | -5.82%   |                     837 |                     121 |         958 |
| 2006-01-17 | 2006-07-18 | 2006-09-20 | -5.08%   |                     182 |                      64 |         246 |
| 2007-06-05 | 2009-11-04 | 2010-04-14 | -7.06%   |                     883 |                     161 |        1044 |
| 2010-05-04 | 2010-07-02 | 2010-09-13 | -5.64%   |                      59 |                      73 |         132 |
| 2013-05-22 | 2013-08-21 | 2013-11-29 | -5.70%   |                      91 |                     100 |         191 |
| 2015-02-26 | 2015-09-28 | 2016-03-29 | -7.59%   |                     214 |                     183 |         397 |
| 2016-08-01 | 2016-11-14 | 2017-04-18 | -5.76%   |                     105 |                     155 |         260 |
| 2018-01-29 | 2018-05-02 | 2018-08-27 | -5.92%   |                      93 |                     117 |         210 |
| 2018-08-30 | 2019-01-14 | 2019-03-21 | -5.95%   |                     137 |                      66 |         203 |
| 2020-09-03 | 2021-02-26 | 2021-07-02 | -6.85%   |                     176 |                     126 |         302 |
| 2021-11-10 | 2022-07-14 | 2023-12-26 | -10.56%  |                     246 |                     530 |         776 |
| 2024-09-03 | 2025-06-20 | 2025-10-02 | -7.99%   |                     290 |                     104 |         394 |
| 2025-10-29 | 2026-07-02 | open       | -5.57%   |                     246 |                     nan |         nan |

**LSTM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                      49 |                      43 |          92 |
| 2001-02-02 | 2002-07-23 | 2004-03-05 | -24.04%  |                     536 |                     591 |        1127 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                      63 |                     178 |         241 |
| 2008-10-14 | 2008-10-27 | 2008-11-04 | -8.61%   |                      13 |                       8 |          21 |
| 2008-11-05 | 2008-11-20 | 2008-12-16 | -10.78%  |                      15 |                      26 |          41 |
| 2009-01-05 | 2009-03-09 | 2009-08-03 | -18.97%  |                      63 |                     147 |         210 |
| 2010-05-04 | 2010-07-02 | 2010-09-13 | -5.36%   |                      59 |                      73 |         132 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                      14 |                      67 |          81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                      33 |                     120 |         153 |
| 2015-03-23 | 2015-08-25 | 2016-07-08 | -8.39%   |                     155 |                     318 |         473 |
| 2016-08-01 | 2016-11-14 | 2017-04-17 | -5.64%   |                     105 |                     154 |         259 |
| 2018-01-29 | 2018-12-24 | 2019-06-07 | -13.63%  |                     329 |                     165 |         494 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                      26 |                      82 |         108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                      57 |                      39 |          96 |
| 2021-12-28 | 2022-10-14 | 2024-11-29 | -27.55%  |                     290 |                     777 |        1067 |
| 2024-12-09 | 2025-04-08 | 2025-07-03 | -12.22%  |                     120 |                      86 |         206 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                      29 |                      21 |          50 |

**Transformer**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                      49 |                      43 |          92 |
| 2001-02-02 | 2002-07-23 | 2004-01-21 | -22.89%  |                     536 |                     547 |        1083 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                      63 |                     178 |         241 |
| 2007-11-01 | 2008-09-24 | 2009-08-03 | -13.79%  |                     328 |                     313 |         641 |
| 2010-05-04 | 2010-07-02 | 2010-09-13 | -5.36%   |                      59 |                      73 |         132 |
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

### Crisis Performance
Return and max drawdown during historical crisis periods: the central evidence for the tail-risk protection of the regime-switching models.

| Crisis                              | ('Return', 'Buy_Hold')   | ('Return', 'HMM')   | ('Return', 'HMM_Uni')   | ('Return', 'LSTM')   | ('Return', 'MSM')   | ('Return', 'Transformer')   | ('Max Drawdown', 'Buy_Hold')   | ('Max Drawdown', 'HMM')   | ('Max Drawdown', 'HMM_Uni')   | ('Max Drawdown', 'LSTM')   | ('Max Drawdown', 'MSM')   | ('Max Drawdown', 'Transformer')   |
|:------------------------------------|:-------------------------|:--------------------|:------------------------|:---------------------|:--------------------|:----------------------------|:-------------------------------|:--------------------------|:------------------------------|:---------------------------|:--------------------------|:----------------------------------|
| COVID Crash (2020-02 to 2020-03)    | -8.55%                   | -8.16%              | +0.73%                  | -8.55%               | +0.73%              | -8.55%                      | -18.53%                        | -14.25%                   | -1.81%                        | -18.53%                    | -1.81%                    | -18.53%                           |
| Dot-Com (2000-03 to 2002-10)        | -15.77%                  | -3.58%              | -2.45%                  | -15.77%              | -2.45%              | -14.49%                     | -24.81%                        | -19.02%                   | -6.22%                        | -24.81%                    | -6.22%                    | -23.67%                           |
| EU Debt Crisis (2011-07 to 2011-11) | +4.10%                   | +0.01%              | +2.32%                  | +4.10%               | +2.32%              | +4.10%                      | -7.24%                         | 0.00%                     | -4.74%                        | -7.24%                     | -4.74%                    | -7.24%                            |
| GFC (2007-10 to 2009-03)            | -26.99%                  | -19.62%             | -2.99%                  | -1.25%               | -2.99%              | -5.87%                      | -35.08%                        | -23.55%                   | -4.04%                        | -19.46%                    | -4.04%                    | -13.93%                           |
| Rate Hikes (2022-01 to 2022-10)     | -24.20%                  | -19.10%             | -6.33%                  | -24.20%              | -6.33%              | -24.20%                     | -26.98%                        | -22.67%                   | -8.34%                        | -26.98%                    | -8.34%                    | -26.98%                           |

### Switch Timing Relative to the Drawdown Peak
Time lag between the model's first bear signal and the drawdown trough of the buy-and-hold portfolio per crisis. Positive = model reacted early, negative = too late.

| Crisis                              | Model       | DD Trough   | First Bear Signal   |   Lead (Days) |
|:------------------------------------|:------------|:------------|:--------------------|--------------:|
| Dot-Com (2000-03 to 2002-10)        | MSM         | 2002-07-23  | 2000-10-16          |           645 |
| GFC (2007-10 to 2009-03)            | MSM         | 2009-03-09  | 2007-10-01          |           525 |
| EU Debt Crisis (2011-07 to 2011-11) | MSM         | 2011-08-08  | 2011-08-04          |             4 |
| COVID Crash (2020-02 to 2020-03)    | MSM         | 2020-03-18  | 2020-02-24          |            23 |
| Rate Hikes (2022-01 to 2022-10)     | MSM         | 2022-10-14  | 2022-01-05          |           282 |
| Dot-Com (2000-03 to 2002-10)        | HMM         | 2002-07-23  | 2000-10-16          |           645 |
| GFC (2007-10 to 2009-03)            | HMM         | 2009-03-09  | 2007-10-16          |           510 |
| EU Debt Crisis (2011-07 to 2011-11) | HMM         | 2011-08-08  | 2011-07-01          |            38 |
| COVID Crash (2020-02 to 2020-03)    | HMM         | 2020-03-18  | 2020-03-13          |             5 |
| Rate Hikes (2022-01 to 2022-10)     | HMM         | 2022-10-14  | 2022-07-12          |            94 |
| Dot-Com (2000-03 to 2002-10)        | HMM_Uni     | 2002-07-23  | 2000-10-16          |           645 |
| GFC (2007-10 to 2009-03)            | HMM_Uni     | 2009-03-09  | 2007-10-01          |           525 |
| EU Debt Crisis (2011-07 to 2011-11) | HMM_Uni     | 2011-08-08  | 2011-08-04          |             4 |
| COVID Crash (2020-02 to 2020-03)    | HMM_Uni     | 2020-03-18  | 2020-02-24          |            23 |
| Rate Hikes (2022-01 to 2022-10)     | HMM_Uni     | 2022-10-14  | 2022-01-05          |           282 |
| Dot-Com (2000-03 to 2002-10)        | LSTM        | 2002-07-23  |                     |           nan |
| GFC (2007-10 to 2009-03)            | LSTM        | 2009-03-09  | 2007-10-01          |           525 |
| EU Debt Crisis (2011-07 to 2011-11) | LSTM        | 2011-08-08  |                     |           nan |
| COVID Crash (2020-02 to 2020-03)    | LSTM        | 2020-03-18  |                     |           nan |
| Rate Hikes (2022-01 to 2022-10)     | LSTM        | 2022-10-14  |                     |           nan |
| Dot-Com (2000-03 to 2002-10)        | Transformer | 2002-07-23  | 2001-10-16          |           280 |
| GFC (2007-10 to 2009-03)            | Transformer | 2009-03-09  | 2007-10-01          |           525 |
| EU Debt Crisis (2011-07 to 2011-11) | Transformer | 2011-08-08  |                     |           nan |
| COVID Crash (2020-02 to 2020-03)    | Transformer | 2020-03-18  |                     |           nan |
| Rate Hikes (2022-01 to 2022-10)     | Transformer | 2022-10-14  |                     |           nan |

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
| MSM         | 226.52%        | 4.71%         | 6.02%        | -10.91%        |           0.8  |            0.9  |           0.43 |               323 | 32.40%               |          3.75 |
| HMM         | 142.36%        | 3.50%         | 7.45%        | -23.55%        |           0.5  |            0.46 |           0.15 |                31 | 3.20%                |          9.06 |
| HMM Uni     | 223.95%        | 4.68%         | 6.03%        | -11.01%        |           0.79 |            0.89 |           0.42 |               317 | 31.80%               |          3.77 |
| LSTM        | 367.80%        | 6.19%         | 10.46%       | -27.71%        |           0.63 |            0.76 |           0.22 |                12 | 1.20%                |          7.4  |
| Transformer | 394.57%        | 6.42%         | 9.89%        | -27.71%        |           0.68 |            0.86 |           0.23 |                14 | 1.40%                |          7.42 |

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
| ('Standard', 'MSM')            | 137,169.42 €       | Capital preserved |
| ('Standard', 'HMM')            | 0.00 €             | Depleted (2024)   |
| ('Standard', 'HMM Uni')        | 120,369.11 €       | Capital preserved |
| ('Standard', 'LSTM')           | 180,990.27 €       | Capital preserved |
| ('Standard', 'Transformer')    | 182,753.99 €       | Capital preserved |
| ('Aggressive', 'Buy Hold')     | 0.00 €             | Depleted (2011)   |
| ('Aggressive', 'MSM')          | 0.00 €             | Depleted (2013)   |
| ('Aggressive', 'HMM')          | 0.00 €             | Depleted (2013)   |
| ('Aggressive', 'HMM Uni')      | 0.00 €             | Depleted (2013)   |
| ('Aggressive', 'LSTM')         | 0.00 €             | Depleted (2012)   |
| ('Aggressive', 'Transformer')  | 0.00 €             | Depleted (2012)   |
| ('Low_Capital', 'Buy Hold')    | 0.00 €             | Depleted (2015)   |
| ('Low_Capital', 'MSM')         | 0.00 €             | Depleted (2018)   |
| ('Low_Capital', 'HMM')         | 0.00 €             | Depleted (2016)   |
| ('Low_Capital', 'HMM Uni')     | 0.00 €             | Depleted (2017)   |
| ('Low_Capital', 'LSTM')        | 0.00 €             | Depleted (2017)   |
| ('Low_Capital', 'Transformer') | 0.00 €             | Depleted (2016)   |

Capital development of the different scenarios:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Stationary Bootstrap Robustness Check

To assess statistical significance, 10,000 artificial market paths were simulated via stationary bootstrap (Politis & Romano 1994).
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin Probability   | Median Terminal Capital   |
|:-------------------------------|:-------------------|:--------------------------|
| ('Standard', 'MSM')            | 0.00%              | 409,322.01 €              |
| ('Standard', 'Buy Hold')       | 0.01%              | 470,625.12 €              |
| ('Standard', 'HMM')            | 0.00%              | 350,335.42 €              |
| ('Standard', 'LSTM')           | 0.00%              | 497,391.40 €              |
| ('Low_Capital', 'Buy Hold')    | 0.63%              | 201,384.39 €              |
| ('Low_Capital', 'MSM')         | 0.00%              | 168,961.18 €              |
| ('Aggressive', 'Transformer')  | 2.44%              | 262,221.37 €              |
| ('Standard', 'Transformer')    | 0.00%              | 515,039.54 €              |
| ('Aggressive', 'LSTM')         | 2.67%              | 246,885.86 €              |
| ('Aggressive', 'Buy Hold')     | 5.27%              | 228,313.50 €              |
| ('Aggressive', 'MSM')          | 0.57%              | 179,895.13 €              |
| ('Aggressive', 'HMM')          | 6.32%              | 134,900.91 €              |
| ('Standard', 'HMM Uni')        | 0.00%              | 408,366.15 €              |
| ('Aggressive', 'HMM Uni')      | 0.57%              | 178,844.60 €              |
| ('Low_Capital', 'HMM')         | 0.50%              | 138,610.48 €              |
| ('Low_Capital', 'HMM Uni')     | 0.00%              | 168,334.93 €              |
| ('Low_Capital', 'LSTM')        | 0.18%              | 214,959.56 €              |
| ('Low_Capital', 'Transformer') | 0.08%              | 225,005.29 €              |

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
| ('Standard', 'LSTM')           | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Standard', 'Transformer')    | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Aggressive', 'Buy_Hold')     | 5.27%            | 4.85%          | 5.73%          | 527/10000          |
| ('Aggressive', 'MSM')          | 0.57%            | 0.44%          | 0.74%          | 57/10000           |
| ('Aggressive', 'HMM')          | 6.32%            | 5.86%          | 6.81%          | 632/10000          |
| ('Aggressive', 'HMM_Uni')      | 0.57%            | 0.44%          | 0.74%          | 57/10000           |
| ('Aggressive', 'LSTM')         | 2.67%            | 2.37%          | 3.00%          | 267/10000          |
| ('Aggressive', 'Transformer')  | 2.44%            | 2.16%          | 2.76%          | 244/10000          |
| ('Low_Capital', 'Buy_Hold')    | 0.63%            | 0.49%          | 0.81%          | 63/10000           |
| ('Low_Capital', 'MSM')         | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Low_Capital', 'HMM')         | 0.50%            | 0.38%          | 0.66%          | 50/10000           |
| ('Low_Capital', 'HMM_Uni')     | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Low_Capital', 'LSTM')        | 0.18%            | 0.11%          | 0.28%          | 18/10000           |
| ('Low_Capital', 'Transformer') | 0.08%            | 0.04%          | 0.16%          | 8/10000            |

### Hypothesis Tests (Paired Wilcoxon, α = 0.05)
**H1: Regime switching reduces MaxDD vs. buy and hold:**

| Model       | Median MaxDD (Model)   | Median MaxDD (B&H)   | Δ Median   |   Wilcoxon p | H1 (α=0.05)   |
|:------------|:-----------------------|:---------------------|:-----------|-------------:|:--------------|
| MSM         | -65.24%                | -59.45%              | -5.79 pp   |    1         | rejected      |
| HMM         | -74.04%                | -59.45%              | -14.59 pp  |    1         | rejected      |
| HMM_Uni     | -65.49%                | -59.45%              | -6.04 pp   |    1         | rejected      |
| LSTM        | -55.59%                | -59.45%              | +3.86 pp   |    2.76e-171 | confirmed     |
| Transformer | -53.09%                | -59.45%              | +6.36 pp   |    0         | confirmed     |

**H2: The Transformer dominates econometrics and LSTM in terminal wealth:**

| Comparison              | Median Transformer   | Median MSM   | Δ Median   |   Wilcoxon p | H2 (α=0.05)   | Median HMM   | Median HMM_Uni   | Median LSTM   |
|:------------------------|:---------------------|:-------------|:-----------|-------------:|:--------------|:-------------|:-----------------|:--------------|
| Transformer vs. MSM     | 262,221 €            | 179,895 €    | +82,326 €  |     0        | confirmed     | nan          | nan              | nan           |
| Transformer vs. HMM     | 262,221 €            | nan          | +127,320 € |     0        | confirmed     | 134,901 €    | nan              | nan           |
| Transformer vs. HMM_Uni | 262,221 €            | nan          | +83,377 €  |     0        | confirmed     | nan          | 178,845 €        | nan           |
| Transformer vs. LSTM    | 262,221 €            | nan          | +15,336 €  |     6.56e-55 | confirmed     | nan          | nan              | 246,886 €     |

### Break-Even Transaction Costs
At what cost rate (in basis points per reallocation) does active switching lose its return advantage over buy and hold?

| Model       |   Final @10bps |   B&H Final |   Break-Even (bps) |
|:------------|---------------:|------------:|-------------------:|
| MSM         |          3.265 |       4.196 |                  5 |
| HMM         |          2.423 |       4.196 |                  0 |
| HMM_Uni     |          3.239 |       4.196 |                  5 |
| LSTM        |          4.677 |       4.196 |                150 |
| Transformer |          4.945 |       4.196 |                150 |

![Break-Even Analysis](../assets/break_even_costs.png)

### Withdrawal Rate Sensitivity (3.5% / 4% / 5%)
Robustness of the SORR results under varying annual withdrawals.

| Strategy    | ('Terminal Capital', '3.5%')   | ('Terminal Capital', '4.0%')   | ('Terminal Capital', '5.0%')   | ('Status', '3.5%')   | ('Status', '4.0%')   | ('Status', '5.0%')   |
|:------------|:-------------------------------|:-------------------------------|:-------------------------------|:---------------------|:---------------------|:---------------------|
| Buy_Hold    | 866,713 €                      | 690,770 €                      | 338,886 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| HMM         | 471,779 €                      | 366,060 €                      | 154,623 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| HMM_Uni     | 745,116 €                      | 620,167 €                      | 370,268 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| LSTM        | 1,080,155 €                    | 900,322 €                      | 540,656 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| MSM         | 760,272 €                      | 635,652 €                      | 386,411 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| Transformer | 1,136,967 €                    | 946,125 €                      | 564,439 €                      | Capital preserved    | Capital preserved    | Capital preserved    |

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

**Last updated:** 2026-07-09 14:02<br>
**End date:** `2026-07-09`<br>
**Fast mode status at runtime:** FALSE (Full Run)<br>
**Walk-forward validation:** ENABLED (mode: rolling, train: 10y, test: 12m, step: 12m)<br>
**Model persistence:** ENABLED<br>
*Generated by the Backtest Service (reporting).*
