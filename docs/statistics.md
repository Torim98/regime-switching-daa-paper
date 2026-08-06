
# Detailed Statistical Evaluation & Research Results

This page documents the numerical and graphical results of the research pipeline. All evaluations are based on the **frozen dataset** from **1990-01-02** to **2026-07-31** (thesis freeze).

---

## 1. Executive Summary: Performance & Risk
A direct comparison of the core metrics over the entire **out-of-sample test period**.

| Strategy    | Final Wealth   | Total Return   | Max Drawdown   |
|:------------|:---------------|:---------------|:---------------|
| Buy_Hold    | 2,070,561 €    | +314.11%       | -35.08%        |
| MSM         | 1,590,052 €    | +218.01%       | -10.91%        |
| HMM         | 1,057,241 €    | +111.45%       | -17.02%        |
| HMM_Uni     | 1,577,535 €    | +215.51%       | -11.01%        |
| LSTM        | 1,840,170 €    | +268.03%       | -21.52%        |
| Transformer | 2,101,990 €    | +320.40%       | -29.45%        |

> **Key point:** Compare the **max drawdown** of the active strategies with the buy-and-hold benchmark. The objective of this work is a significant reduction of this value to mitigate SORR.

---

## 2. Data Basis & Baseline Portfolio
The analysis is based on a global multi-asset approach.

### Exploratory Data Analysis (EDA)
**Descriptive statistics of the base time series:**
| Time Series   |   Mean (daily) |   Std. Dev. (daily) |     Min |     Max |   Skewness |   Kurtosis |
|:--------------|---------------:|--------------------:|--------:|--------:|-----------:|-----------:|
| Returns_GSPC  |       0.00033  |            0.011348 | -0.1277 |  0.1096 |    -0.364  |    10.8616 |
| Returns_VUSTX |       0.000209 |            0.007249 | -0.0605 |  0.0654 |    -0.0304 |     4.5173 |
| Returns       |       0.000281 |            0.006881 | -0.0662 |  0.0584 |    -0.2805 |     7.6516 |
| VIX           |      19.4499   |            7.73648  |  9.14   | 82.69   |     2.2107 |     8.7497 |
| TNX_10Y       |       4.24552  |            1.92538  |  0.499  |  9.09   |     0.3251 |    -0.6254 |
| IRX_3M        |       2.71619  |            2.19564  | -0.105  |  7.99   |     0.1893 |    -1.2495 |

**Stationarity check (augmented Dickey-Fuller test):**
| Time Series   |   ADF Statistic |    p-Value |   Crit. Value (5%) | Stationary?   |
|:--------------|----------------:|-----------:|-------------------:|:--------------|
| Returns_GSPC  |        -17.5642 | 4.0821e-30 |            -2.8619 | Yes           |
| Returns_VUSTX |        -18.7155 | 2.0338e-30 |            -2.8619 | Yes           |
| Returns       |        -21.0141 | 0          |            -2.8619 | Yes           |
| VIX           |         -7.3175 | 1.2184e-10 |            -2.8619 | Yes           |
| TNX_10Y       |         -2.3356 | 0.16076    |            -2.8619 | No            |
| IRX_3M        |         -2.3494 | 0.15654    |            -2.8619 | No            |

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

_Generated at 2026-08-03 14:31:37_  
Optimization metric: **martin (pooled OOS)**

## Overview

| Model | Best Score | ✓ Complete | ✗ Pruned | Total |
|:---|---:|---:|---:|---:|
| **MSM** | 1.5402 | 36 | 0 | 36 |
| **HMM** | 0.8231 | 108 | 0 | 108 |
| **HMM_Uni** | 1.5296 | 36 | 0 | 36 |
| **LSTM** | 2.3342 | 300 | 0 | 302 |
| **Transformer** | 1.0331 | 400 | 0 | 401 |

### MSM: Best Score `1.5402`

| Parameter | Value |
|:---|---:|
| `threshold` | `0.175` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.8473 |
| sortino | 1.0053 |
| calmar | 0.6179 |
| martin | 1.5402 |
| ulcer | 3.3506 |
| max_drawdown | -0.0835 |
| cagr | 0.0516 |

### HMM: Best Score `0.8231`

| Parameter | Value |
|:---|---:|
| `covariance_type` | `diag` |
| `threshold` | `0.975` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.5510 |
| sortino | 0.5753 |
| calmar | 0.3050 |
| martin | 0.8231 |
| ulcer | 3.3470 |
| max_drawdown | -0.0903 |
| cagr | 0.0275 |

### HMM_Uni: Best Score `1.5296`

| Parameter | Value |
|:---|---:|
| `threshold` | `0.175` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.8391 |
| sortino | 0.9954 |
| calmar | 0.6334 |
| martin | 1.5296 |
| ulcer | 3.3440 |
| max_drawdown | -0.0808 |
| cagr | 0.0511 |

### LSTM: Best Score `2.3342`

| Parameter | Value |
|:---|---:|
| `window_size` | `140` |
| `units_l1` | `16` |
| `units_l2` | `32` |
| `batch_size` | `128` |
| `learning_rate` | `2.008e-05` |
| `dropout` | `0.5` |
| `threshold` | `0.35` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.9020 |
| sortino | 0.7655 |
| calmar | 0.5638 |
| martin | 2.3342 |
| ulcer | 1.7773 |
| max_drawdown | -0.0736 |
| cagr | 0.0415 |

### Transformer: Best Score `1.0331`

| Parameter | Value |
|:---|---:|
| `window_size` | `230` |
| `dmodel_nheads` | `16-2` |
| `n_layers` | `2` |
| `dim_feedforward` | `256` |
| `batch_size` | `64` |
| `learning_rate` | `0.003203` |
| `dropout` | `0.25` |
| `threshold` | `0.5` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.6100 |
| sortino | 0.7195 |
| calmar | 0.2133 |
| martin | 1.0331 |
| ulcer | 5.2343 |
| max_drawdown | -0.2535 |
| cagr | 0.0541 |


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

_Generated at 2026-08-03 14:44:41_

| model       | metric   |   best_value |   best_trial |   conv_frac |   n_complete |   n_pruned | top_importance                                       | edge_flags                                                 |
|:------------|:---------|-------------:|-------------:|------------:|-------------:|-----------:|:-----------------------------------------------------|:-----------------------------------------------------------|
| MSM         | martin   |       1.5402 |           23 |        0.66 |           36 |          0 | threshold=1.00                                       | -                                                          |
| HMM         | martin   |       0.8231 |          103 |        0.96 |          108 |          0 | covariance_type=0.98, threshold=0.02                 | threshold=0.975 near UPPER bound 0.975                     |
| HMM_Uni     | martin   |       1.5296 |           23 |        0.66 |           36 |          0 | threshold=1.00                                       | -                                                          |
| LSTM        | martin   |       2.3342 |          191 |        0.63 |          300 |          0 | learning_rate=0.30, threshold=0.28, dropout=0.17     | learning_rate=2.007996207208503e-05 near LOWER bound 1e-05 |
| Transformer | martin   |       1.0331 |          298 |        0.74 |          400 |          0 | window_size=0.48, learning_rate=0.18, threshold=0.12 | n_layers=2 near LOWER bound 1                              |


#### G.2 Objective Sensitivity
Which config would have been selected under each candidate metric, valued across all metrics. `same_as_objective = True` throughout means the model choice is robust to the objective; divergences quantify the trade-off.

# Objective Sensitivity of the Selected Hyperparameters

_Generated at 2026-08-03 14:44:41_  
Best config under each candidate metric, valued across all metrics (from the search trials' logged OOS metrics; no retraining). `same_as_objective` marks configs identical to the actual objective's pick.

## MSM (objective: martin, 36 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |
| sharpe          |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |
| sortino         |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |
| calmar          |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |
| ulcer           |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |
| max_drawdown    |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |

Selected configs:
- best under **martin**: threshold=0.175

## HMM (objective: martin, 108 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |
| sharpe          |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |
| sortino         |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |
| calmar          |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |
| ulcer           |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |
| max_drawdown    |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |

Selected configs:
- best under **martin**: covariance_type=diag, threshold=0.975

## HMM_Uni (objective: martin, 36 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      23 | True                |   1.5296 |   0.8391 |    0.9954 |   0.6334 |  3.344  |        -0.0808 | 0.0511 |
| sharpe          |      23 | True                |   1.5296 |   0.8391 |    0.9954 |   0.6334 |  3.344  |        -0.0808 | 0.0511 |
| sortino         |      23 | True                |   1.5296 |   0.8391 |    0.9954 |   0.6334 |  3.344  |        -0.0808 | 0.0511 |
| calmar          |      23 | True                |   1.5296 |   0.8391 |    0.9954 |   0.6334 |  3.344  |        -0.0808 | 0.0511 |
| ulcer           |      12 | False               |   1.3523 |   0.7702 |    0.8698 |   0.4296 |  3.2777 |        -0.1032 | 0.0443 |
| max_drawdown    |      23 | True                |   1.5296 |   0.8391 |    0.9954 |   0.6334 |  3.344  |        -0.0808 | 0.0511 |

Selected configs:
- best under **martin**: threshold=0.175
- best under **ulcer**: threshold=0.1

## LSTM (objective: martin, 300 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |     191 | True                |   2.3342 |   0.902  |    0.7655 |   0.5638 |  1.7773 |        -0.0736 | 0.0415 |
| sharpe          |     191 | True                |   2.3342 |   0.902  |    0.7655 |   0.5638 |  1.7773 |        -0.0736 | 0.0415 |
| sortino         |     248 | False               |   1.3234 |   0.8061 |    1.0478 |   0.295  |  5.0452 |        -0.2263 | 0.0668 |
| calmar          |     191 | True                |   2.3342 |   0.902  |    0.7655 |   0.5638 |  1.7773 |        -0.0736 | 0.0415 |
| ulcer           |     191 | True                |   2.3342 |   0.902  |    0.7655 |   0.5638 |  1.7773 |        -0.0736 | 0.0415 |
| max_drawdown    |     191 | True                |   2.3342 |   0.902  |    0.7655 |   0.5638 |  1.7773 |        -0.0736 | 0.0415 |

Selected configs:
- best under **martin**: window_size=140, units_l1=16, units_l2=32, batch_size=128, learning_rate=2.007996207208503e-05, dropout=0.5, threshold=0.35
- best under **sortino**: window_size=140, units_l1=32, units_l2=256, batch_size=64, learning_rate=0.00010450700015167667, dropout=0.5, threshold=0.45000000000000007

## Transformer (objective: martin, 400 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |     298 | True                |   1.0331 |   0.61   |    0.7195 |   0.2133 |  5.2343 |        -0.2535 | 0.0541 |
| sharpe          |     300 | False               |   0.5833 |   0.6814 |    0.9056 |   0.1758 |  9.6727 |        -0.321  | 0.0564 |
| sortino         |     300 | False               |   0.5833 |   0.6814 |    0.9056 |   0.1758 |  9.6727 |        -0.321  | 0.0564 |
| calmar          |      86 | False               |   1.0191 |   0.5766 |    0.6604 |   0.2418 |  4.6971 |        -0.1979 | 0.0479 |
| ulcer           |     327 | False               |   0.9252 |   0.5938 |    0.7122 |   0.2245 |  4.5029 |        -0.1855 | 0.0417 |
| max_drawdown    |     156 | False               |   0.3567 |   0.4903 |    0.4401 |   0.1893 |  6.1963 |        -0.1167 | 0.0221 |

Selected configs:
- best under **martin**: window_size=230, dmodel_nheads=16-2, n_layers=2, dim_feedforward=256, batch_size=64, learning_rate=0.003203170398725006, dropout=0.25, threshold=0.5
- best under **sharpe**: window_size=170, dmodel_nheads=16-2, n_layers=2, dim_feedforward=256, batch_size=64, learning_rate=0.005797912630385614, dropout=0.15000000000000002, threshold=0.4
- best under **calmar**: window_size=100, dmodel_nheads=32-4, n_layers=3, dim_feedforward=32, batch_size=32, learning_rate=6.3198947237583e-05, dropout=0.4, threshold=0.15000000000000002
- best under **ulcer**: window_size=230, dmodel_nheads=128-8, n_layers=2, dim_feedforward=256, batch_size=64, learning_rate=0.003682334936836272, dropout=0.35000000000000003, threshold=0.45000000000000007
- best under **max_drawdown**: window_size=50, dmodel_nheads=32-4, n_layers=2, dim_feedforward=64, batch_size=32, learning_rate=1.9311434907966562e-05, dropout=0.45, threshold=0.1


#### G.3 Deflated Sharpe Ratio & Backtest Overfitting
DSR deflates the selected config's Sharpe for the number of tested configs; PBO (CSCV) estimates the probability that the in-sample-best config underperforms out-of-sample.

# Deflated Sharpe Ratio (multiple-testing adjusted)

_Generated at 2026-08-03 14:48:33_

| model       |   n_trials |   sr_ann_best |   sr_star_ann |    dsr | significant_5pct   |
|:------------|-----------:|--------------:|--------------:|-------:|:-------------------|
| MSM         |         36 |        0.8473 |        0.3111 | 0.9832 | True               |
| HMM         |        108 |        0.551  |        0.284  | 0.8551 | False              |
| HMM_Uni     |         36 |        0.8391 |        0.3274 | 0.9787 | True               |
| LSTM        |        300 |        0.7153 |        0.3409 | 0.932  | False              |
| Transformer |        400 |        0.3248 |        0.2524 | 0.6136 | False              |

DSR = P(true Sharpe > 0) after deflating the best trial's Sharpe for the number of tested configs. `significant_5pct` = DSR > 0.95.


# Probability of Backtest Overfitting (CSCV)

_Generated at 2026-08-03 16:02:19_

| model       |   folds |   configs |   pbo |
|:------------|--------:|----------:|------:|
| MSM         |      16 |        15 | 0.464 |
| HMM         |      16 |        15 | 0.972 |
| HMM_Uni     |      16 |        15 | 0.357 |
| LSTM        |      16 |        15 | 0     |
| Transformer |      16 |        15 | 0.464 |

PBO over the top-15 configs per model (per-fold Sharpe matrix, CSCV). Lower is better; PBO > 0.5 flags overfitting.


#### G.4 Multi-Seed Robustness (DL)
Top DL configs re-evaluated over several global seeds; mean/std of every metric quantify seed sensitivity of the deep-learning ranking.

# Multi-Seed Re-Evaluation (top-5, 5 seeds)

_Generated at 2026-08-03 17:42:22_

### LSTM

|   rank |   trial |   martin_hpo |   martin_mean |   martin_std |   sharpe_mean |   sharpe_std |   sortino_mean |   sortino_std |   calmar_mean |   calmar_std |   ulcer_mean |   ulcer_std |   max_drawdown_mean |   max_drawdown_std |   cagr_mean |   cagr_std |
|-------:|--------:|-------------:|--------------:|-------------:|--------------:|-------------:|---------------:|--------------:|--------------:|-------------:|-------------:|------------:|--------------------:|-------------------:|------------:|-----------:|
|      1 |     191 |       2.3342 |        0.9288 |       0.2334 |        0.6395 |       0.0994 |         0.6627 |        0.0952 |        0.2397 |       0.0611 |       4.7267 |      1.6514 |             -0.1842 |             0.0661 |      0.0402 |     0.0068 |
|      2 |     189 |       2.0067 |        0.8818 |       0.2121 |        0.5575 |       0.0448 |         0.6585 |        0.0386 |        0.2059 |       0.0331 |       5.5479 |      0.9059 |             -0.232  |             0.0218 |      0.0471 |     0.0031 |
|      3 |       3 |       1.7795 |        1.0667 |       0.2898 |        0.6447 |       0.0692 |         0.7894 |        0.0972 |        0.2416 |       0.0487 |       5.552  |      1.0494 |             -0.2378 |             0.0265 |      0.0562 |     0.0063 |
|      4 |     258 |       1.7229 |        1.3319 |       0.191  |        0.6894 |       0.0492 |         0.8236 |        0.0532 |        0.2828 |       0.0361 |       4.317  |      0.3176 |             -0.2024 |             0.009  |      0.0569 |     0.0054 |
|      5 |     293 |       1.716  |        1.3771 |       0.4857 |        0.6784 |       0.1091 |         0.7928 |        0.0714 |        0.3143 |       0.1225 |       4.4434 |      1.4179 |             -0.1948 |             0.0548 |      0.0546 |     0.0043 |

### Transformer

|   rank |   trial |   martin_hpo |   martin_mean |   martin_std |   sharpe_mean |   sharpe_std |   sortino_mean |   sortino_std |   calmar_mean |   calmar_std |   ulcer_mean |   ulcer_std |   max_drawdown_mean |   max_drawdown_std |   cagr_mean |   cagr_std |
|-------:|--------:|-------------:|--------------:|-------------:|--------------:|-------------:|---------------:|--------------:|--------------:|-------------:|-------------:|------------:|--------------------:|-------------------:|------------:|-----------:|
|      1 |     298 |       1.0331 |        0.5881 |       0.109  |        0.5111 |       0.049  |         0.6396 |        0.077  |        0.1628 |       0.0206 |       7.428  |      0.6316 |             -0.2648 |             0.0084 |      0.0432 |     0.006  |
|      2 |      86 |       1.0191 |        0.3043 |       0.1337 |        0.4173 |       0.0906 |         0.4869 |        0.1157 |        0.1048 |       0.0367 |      11.6941 |      3.1721 |             -0.3128 |             0.0443 |      0.0316 |     0.0091 |
|      3 |     327 |       0.9252 |        0.3144 |       0.1568 |        0.3565 |       0.1106 |         0.433  |        0.1493 |        0.1007 |       0.0408 |       9.2788 |      2.0327 |             -0.2806 |             0.0575 |      0.0265 |     0.009  |
|      4 |     208 |       0.8651 |        0.4735 |       0.0335 |        0.4356 |       0.029  |         0.538  |        0.0474 |        0.129  |       0.0096 |       8.3417 |      0.5254 |             -0.3065 |             0.0219 |      0.0394 |     0.0028 |
|      5 |     381 |       0.8115 |        0.6172 |       0.155  |        0.4823 |       0.0705 |         0.5849 |        0.0946 |        0.1656 |       0.036  |       7.4126 |      0.7705 |             -0.2731 |             0.0179 |      0.0446 |     0.0073 |


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
|     26 | 2025-10-16   | 2026-07-31 |          20.7 |                  1 | Yes                    |            8   |                   1 | Yes                     |

Across the 26 walk-forward folds, 2 OOS test windows contain at least one complete Pagan-Sossounov bear phase (peak and trough inside the 12-month window), while 6 folds overlap at least one bear phase and 20 folds carry no bear day at all.

Because the 3-month minimum-phase filter and the 12-month fold length rarely coincide, most crisis exposure enters the folds as partial (window-truncated) bear phases rather than as fully contained episodes, whereas every training window (10 years) spans several complete bear phases.

A bear run that is still open at the global data boundary cannot be confirmed complete, so the classification is conservative for any fold whose window reaches the end of the sample.


### Equity Curves in Comparison
![Equity Curves](../assets/equity_curves.png)

### Annualized Performance Metrics
Normalized metrics (CAGR, Sharpe, Sortino, Calmar) for comparison across evaluation periods of different lengths.

| Strategy    | CAGR   | Ann. Volatility   |   Sharpe Ratio |   Sortino Ratio | Max Drawdown   |   Calmar Ratio |   OOS Days |   OOS Years |
|:------------|:-------|:------------------|---------------:|----------------:|:---------------|---------------:|-----------:|------------:|
| Buy_Hold    | +5.68% | 11.16%            |          0.551 |           0.719 | -35.08%        |          0.162 |       6484 |        25.7 |
| MSM         | +4.60% | 6.03%             |          0.776 |           0.872 | -10.91%        |          0.422 |       6484 |        25.7 |
| HMM         | +2.95% | 5.50%             |          0.557 |           0.573 | -17.02%        |          0.174 |       6484 |        25.7 |
| HMM_Uni     | +4.57% | 6.04%             |          0.77  |           0.866 | -11.01%        |          0.415 |       6484 |        25.7 |
| LSTM        | +5.20% | 7.46%             |          0.716 |           0.675 | -21.52%        |          0.241 |       6484 |        25.7 |
| Transformer | +5.74% | 9.51%             |          0.635 |           0.793 | -29.45%        |          0.195 |       6484 |        25.7 |

### Classification Metrics (vs. NBER Recessions as Ground Truth)
Comparison of the models as binary recession classifiers (precision, recall, F1).

| Model       |   Precision |   Recall |    F1 |   TN |   FP |   FN |   TP |
|:------------|------------:|---------:|------:|-----:|-----:|-----:|-----:|
| MSM         |       0.228 |    0.922 | 0.366 | 4071 | 1828 |   46 |  540 |
| HMM         |       0.2   |    0.985 | 0.332 | 3590 | 2309 |    9 |  577 |
| HMM_Uni     |       0.228 |    0.922 | 0.366 | 4075 | 1824 |   46 |  540 |
| LSTM        |       0.195 |    0.927 | 0.322 | 3660 | 2239 |   43 |  543 |
| Transformer |       0.311 |    0.497 | 0.382 | 5253 |  646 |  295 |  291 |

![Confusion Matrices](../assets/confusion_matrices.png)

**ROC and precision-recall curves** (threshold-independent comparison via `*_Prob`):

![ROC Curves](../assets/roc_curves.png)
![PR Curves](../assets/pr_curves.png)

### Signal Churning & Whipsaw Analysis
Quantification of the switching frequency and the share of very short regime phases ("whipsaws").

| Model       |   Signal Switches |   Whipsaws (<5d) | Whipsaw Share   |   Mean Phase (Days) |   Median Phase (Days) | Cumul. Costs   |
|:------------|------------------:|-----------------:|:----------------|--------------------:|----------------------:|:---------------|
| MSM         |               324 |              159 | 48.9%           |                20   |                     5 | 32.40%         |
| HMM         |               129 |               52 | 40.0%           |                49.9 |                     6 | 12.90%         |
| HMM_Uni     |               318 |              153 | 48.0%           |                20.3 |                     5 | 31.80%         |
| LSTM        |                23 |                1 | 4.2%            |               270.2 |                    52 | 2.30%          |
| Transformer |                37 |               10 | 26.3%           |               170.7 |                    14 | 3.70%          |

### Regime Probability Heatmap
Bear probabilities of all models over time.

![Regime Probability Heatmap](../assets/regime_probability_heatmap.png)

### Threshold Sensitivity
Variation of the decision threshold per model. Shows how robust final wealth, max drawdown, and the number of regime switches are to a modified bull/bear classification boundary (thesis ch. 4.1, smoothing).

**MSM**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.25 | 1,490,046 €    | -12.13%        |        302 |
|        0.3  | 1,320,764 €    | -16.68%        |        314 |
|        0.35 | 1,331,009 €    | -20.19%        |        320 |
|        0.4  | 1,346,313 €    | -21.55%        |        314 |
|        0.5  | 1,429,408 €    | -25.39%        |        326 |

**HMM**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.4  | 944,272 €      | -12.03%        |        125 |
|        0.45 | 963,679 €      | -11.35%        |        119 |
|        0.5  | 979,029 €      | -12.07%        |        115 |
|        0.55 | 997,182 €      | -12.51%        |        105 |
|        0.6  | 979,313 €      | -12.72%        |        105 |

**HMM_Uni**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.4  | 1,320,705 €    | -24.79%        |        316 |
|        0.45 | 1,336,353 €    | -24.41%        |        326 |
|        0.5  | 1,456,267 €    | -25.57%        |        332 |
|        0.55 | 1,436,089 €    | -23.24%        |        332 |
|        0.6  | 1,635,135 €    | -26.67%        |        338 |

**LSTM**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|         0.2 | 1,654,664 €    | -18.54%        |         27 |
|         0.3 | 1,815,461 €    | -24.95%        |         17 |
|         0.4 | 1,972,713 €    | -26.16%        |         41 |
|         0.5 | 2,036,734 €    | -27.99%        |         12 |

**Transformer**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.3  | 1,834,933 €    | -27.71%        |         85 |
|        0.4  | 2,145,664 €    | -28.54%        |         45 |
|        0.45 | 2,138,206 €    | -30.09%        |         45 |
|        0.5  | 2,101,990 €    | -29.45%        |         37 |
|        0.6  | 1,783,528 €    | -27.71%        |         34 |

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
| 2025-10-29 | 2026-07-30 | open       | -8.03%   |                     274 |                     nan |         nan |

**HMM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2002-01-11 | 2002-06-13 | 2004-02-11 | -8.25%   |                     153 |                     608 |         761 |
| 2004-03-08 | 2004-08-12 | 2004-12-13 | -7.74%   |                     157 |                     123 |         280 |
| 2015-03-23 | 2016-02-03 | 2017-05-24 | -8.71%   |                     317 |                     476 |         793 |
| 2018-01-29 | 2018-12-20 | 2019-06-07 | -9.06%   |                     325 |                     169 |         494 |
| 2020-02-21 | 2021-03-19 | 2021-05-06 | -5.11%   |                     392 |                      48 |         440 |
| 2021-09-03 | 2021-10-11 | 2021-11-05 | -5.28%   |                      38 |                      25 |          63 |
| 2021-11-10 | 2023-10-04 | open       | -16.61%  |                     693 |                     nan |         nan |

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
| 2025-10-29 | 2026-07-30 | open       | -8.03%   |                     274 |                     nan |         nan |

**LSTM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2010-05-04 | 2010-08-30 | 2010-11-02 | -5.36%   |                     118 |                      64 |         182 |
| 2011-07-25 | 2011-08-08 | 2012-07-02 | -6.59%   |                      14 |                     329 |         343 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                      33 |                     120 |         153 |
| 2015-03-23 | 2016-05-19 | 2017-08-31 | -10.99%  |                     423 |                     469 |         892 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                      10 |                     197 |         207 |
| 2018-08-30 | 2018-12-24 | 2019-07-02 | -11.45%  |                     116 |                     190 |         306 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                      26 |                      82 |         108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                      57 |                      39 |          96 |
| 2021-12-28 | 2023-10-27 | 2024-07-16 | -21.35%  |                     668 |                     263 |         931 |
| 2024-12-09 | 2025-04-08 | 2025-07-03 | -12.22%  |                     120 |                      86 |         206 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                      29 |                      21 |          50 |

**Transformer**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                      49 |                      43 |          92 |
| 2001-02-02 | 2002-07-23 | 2004-12-09 | -25.89%  |                     536 |                     870 |        1406 |
| 2007-11-01 | 2008-03-10 | 2009-08-03 | -8.44%   |                     130 |                     511 |         641 |
| 2010-05-04 | 2010-07-02 | 2010-09-13 | -5.36%   |                      59 |                      73 |         132 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                      14 |                      67 |          81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                      33 |                     120 |         153 |
| 2015-03-23 | 2015-08-25 | 2017-05-19 | -8.39%   |                     155 |                     633 |         788 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                      10 |                     197 |         207 |
| 2018-08-30 | 2018-12-24 | 2019-03-21 | -11.45%  |                     116 |                      87 |         203 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                      26 |                      82 |         108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                      57 |                      39 |          96 |
| 2021-12-28 | 2023-10-27 | 2025-09-11 | -29.30%  |                     668 |                     685 |        1353 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                      29 |                      21 |          50 |

### Crisis Performance
Return and max drawdown during historical crisis periods: the central evidence for the tail-risk protection of the regime-switching models.

| Crisis                              | ('Return', 'Buy_Hold')   | ('Return', 'HMM')   | ('Return', 'HMM_Uni')   | ('Return', 'LSTM')   | ('Return', 'MSM')   | ('Return', 'Transformer')   | ('Max Drawdown', 'Buy_Hold')   | ('Max Drawdown', 'HMM')   | ('Max Drawdown', 'HMM_Uni')   | ('Max Drawdown', 'LSTM')   | ('Max Drawdown', 'MSM')   | ('Max Drawdown', 'Transformer')   |
|:------------------------------------|:-------------------------|:--------------------|:------------------------|:---------------------|:--------------------|:----------------------------|:-------------------------------|:--------------------------|:------------------------------|:---------------------------|:--------------------------|:----------------------------------|
| COVID Crash (2020-02 to 2020-03)    | -8.55%                   | +0.73%              | +0.73%                  | -8.55%               | +0.73%              | -8.55%                      | -18.53%                        | -1.81%                    | -1.81%                        | -18.53%                    | -1.81%                    | -18.53%                           |
| Dot-Com (2000-03 to 2002-10)        | -15.77%                  | -1.60%              | -2.45%                  | +6.04%               | -2.45%              | -17.82%                     | -24.81%                        | -8.40%                    | -6.22%                        | -0.08%                     | -6.22%                    | -26.65%                           |
| EU Debt Crisis (2011-07 to 2011-11) | +4.10%                   | +0.01%              | +2.32%                  | -5.30%               | +2.32%              | +4.10%                      | -7.24%                         | 0.00%                     | -4.74%                        | -7.24%                     | -4.74%                    | -7.24%                            |
| GFC (2007-10 to 2009-03)            | -26.99%                  | -1.39%              | -2.99%                  | +0.49%               | -2.99%              | -4.91%                      | -35.08%                        | -2.80%                    | -4.04%                        | -3.58%                     | -4.04%                    | -8.59%                            |
| Rate Hikes (2022-01 to 2022-10)     | -24.20%                  | -7.66%              | -6.33%                  | -14.71%              | -6.33%              | -24.20%                     | -26.98%                        | -8.74%                    | -8.34%                        | -18.29%                    | -8.34%                    | -26.98%                           |

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
| GFC (2007-10 to 2009-03)            | HMM         | 2009-03-09  | 2007-10-19          |           507 |
| EU Debt Crisis (2011-07 to 2011-11) | HMM         | 2011-08-08  | 2011-07-01          |            38 |
| COVID Crash (2020-02 to 2020-03)    | HMM         | 2020-03-18  | 2020-02-24          |            23 |
| Rate Hikes (2022-01 to 2022-10)     | HMM         | 2022-10-14  | 2022-01-18          |           269 |
| Dot-Com (2000-03 to 2002-10)        | HMM_Uni     | 2002-07-23  | 2000-10-16          |           645 |
| GFC (2007-10 to 2009-03)            | HMM_Uni     | 2009-03-09  | 2007-10-01          |           525 |
| EU Debt Crisis (2011-07 to 2011-11) | HMM_Uni     | 2011-08-08  | 2011-08-04          |             4 |
| COVID Crash (2020-02 to 2020-03)    | HMM_Uni     | 2020-03-18  | 2020-02-24          |            23 |
| Rate Hikes (2022-01 to 2022-10)     | HMM_Uni     | 2022-10-14  | 2022-01-05          |           282 |
| Dot-Com (2000-03 to 2002-10)        | LSTM        | 2002-07-23  | 2000-10-16          |           645 |
| GFC (2007-10 to 2009-03)            | LSTM        | 2009-03-09  | 2007-10-01          |           525 |
| EU Debt Crisis (2011-07 to 2011-11) | LSTM        | 2011-08-08  | 2011-08-23          |           -15 |
| COVID Crash (2020-02 to 2020-03)    | LSTM        | 2020-03-18  |                     |           nan |
| Rate Hikes (2022-01 to 2022-10)     | LSTM        | 2022-10-14  | 2022-06-07          |           129 |
| Dot-Com (2000-03 to 2002-10)        | Transformer | 2002-07-23  | 2001-10-16          |           280 |
| GFC (2007-10 to 2009-03)            | Transformer | 2009-03-09  | 2008-06-09          |           273 |
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
| Buy Hold    | 314.19%        | 5.67%         | 11.16%       | -35.08%        |           0.55 |            0.72 |           0.16 |                 0 | 0.00%                |          9.05 |
| MSM         | 218.07%        | 4.59%         | 6.03%        | -10.91%        |           0.78 |            0.87 |           0.42 |               324 | 32.50%               |          3.77 |
| HMM         | 111.49%        | 2.95%         | 5.50%        | -17.02%        |           0.56 |            0.57 |           0.17 |               129 | 13.00%               |          5.76 |
| HMM Uni     | 215.57%        | 4.56%         | 6.04%        | -11.01%        |           0.77 |            0.87 |           0.41 |               318 | 31.90%               |          3.78 |
| LSTM        | 268.10%        | 5.18%         | 7.46%        | -21.52%        |           0.72 |            0.67 |           0.24 |                23 | 2.40%                |          5.07 |
| Transformer | 320.48%        | 5.73%         | 9.51%        | -29.45%        |           0.63 |            0.79 |           0.19 |                37 | 3.70%                |          8.36 |

### Transaction Costs

This figure shows the cumulative transaction costs over time. Steep increases indicate unstable regime switches ("churning").

![Transaction Costs](../assets/transaction_costs.png)

Stress test: Sequence of Returns Risk (SORR)
In addition, the survival time of the capital was simulated in a withdrawal phase (retirement scenario).

### SORR Simulation: Comparison of the Withdrawal Scenarios

This table compares different stress scenarios (standard, aggressive, low capital).

|                                | Terminal Capital   | Status            |
|:-------------------------------|:-------------------|:------------------|
| ('Standard', 'Buy Hold')       | 681,767.31 €       | Capital preserved |
| ('Standard', 'MSM')            | 619,194.78 €       | Capital preserved |
| ('Standard', 'HMM')            | 284,202.45 €       | Capital preserved |
| ('Standard', 'HMM Uni')        | 604,112.20 €       | Capital preserved |
| ('Standard', 'LSTM')           | 633,884.33 €       | Capital preserved |
| ('Standard', 'Transformer')    | 741,139.49 €       | Capital preserved |
| ('Aggressive', 'Buy Hold')     | 0.00 €             | Depleted (2026)   |
| ('Aggressive', 'MSM')          | 133,615.86 €       | Capital preserved |
| ('Aggressive', 'HMM')          | 0.00 €             | Depleted (2023)   |
| ('Aggressive', 'HMM Uni')      | 117,251.72 €       | Capital preserved |
| ('Aggressive', 'LSTM')         | 30,567.83 €        | Capital preserved |
| ('Aggressive', 'Transformer')  | 60,515.80 €        | Capital preserved |
| ('Low_Capital', 'Buy Hold')    | 200,682.66 €       | Capital preserved |
| ('Low_Capital', 'MSM')         | 225,843.19 €       | Capital preserved |
| ('Low_Capital', 'HMM')         | 54,535.72 €        | Capital preserved |
| ('Low_Capital', 'HMM Uni')     | 216,409.17 €       | Capital preserved |
| ('Low_Capital', 'LSTM')        | 199,335.65 €       | Capital preserved |
| ('Low_Capital', 'Transformer') | 240,496.59 €       | Capital preserved |

Capital development of the different scenarios:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Stationary Bootstrap Robustness Check

To assess statistical significance, 10,000 artificial market paths were simulated via stationary bootstrap (Politis & Romano 1994).
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin Probability   | Median Terminal Capital   |
|:-------------------------------|:-------------------|:--------------------------|
| ('Low_Capital', 'MSM')         | 11.82%             | 193,526.66 €              |
| ('Aggressive', 'Buy Hold')     | 33.04%             | 270,635.72 €              |
| ('Low_Capital', 'Buy Hold')    | 14.06%             | 390,859.94 €              |
| ('Aggressive', 'LSTM')         | 34.75%             | 149,329.38 €              |
| ('Standard', 'MSM')            | 0.57%              | 643,117.31 €              |
| ('Aggressive', 'HMM Uni')      | 49.73%             | 1,657.38 €                |
| ('Standard', 'HMM Uni')        | 0.60%              | 634,034.61 €              |
| ('Standard', 'LSTM')           | 0.70%              | 856,293.29 €              |
| ('Standard', 'Buy Hold')       | 3.30%              | 1,040,834.53 €            |
| ('Aggressive', 'MSM')          | 48.97%             | 9,308.23 €                |
| ('Low_Capital', 'HMM')         | 51.99%             | 0.00 €                    |
| ('Standard', 'Transformer')    | 1.55%              | 1,067,051.62 €            |
| ('Standard', 'HMM')            | 8.88%              | 230,547.04 €              |
| ('Aggressive', 'Transformer')  | 29.75%             | 290,526.23 €              |
| ('Aggressive', 'HMM')          | 89.54%             | 0.00 €                    |
| ('Low_Capital', 'HMM Uni')     | 12.46%             | 188,832.71 €              |
| ('Low_Capital', 'LSTM')        | 8.36%              | 299,773.98 €              |
| ('Low_Capital', 'Transformer') | 9.48%              | 403,422.33 €              |

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
| ('Low_Capital', 'MSM')         | 11.82%           | 11.20%         | 12.47%         | 1182/10000         |
| ('Aggressive', 'Buy_Hold')     | 33.04%           | 32.12%         | 33.97%         | 3304/10000         |
| ('Low_Capital', 'Buy_Hold')    | 14.06%           | 13.39%         | 14.76%         | 1406/10000         |
| ('Aggressive', 'LSTM')         | 34.75%           | 33.82%         | 35.69%         | 3475/10000         |
| ('Standard', 'MSM')            | 0.57%            | 0.44%          | 0.74%          | 57/10000           |
| ('Aggressive', 'HMM_Uni')      | 49.73%           | 48.75%         | 50.71%         | 4973/10000         |
| ('Standard', 'HMM_Uni')        | 0.60%            | 0.47%          | 0.77%          | 60/10000           |
| ('Standard', 'LSTM')           | 0.70%            | 0.55%          | 0.88%          | 70/10000           |
| ('Standard', 'Buy_Hold')       | 3.30%            | 2.97%          | 3.67%          | 330/10000          |
| ('Aggressive', 'MSM')          | 48.97%           | 47.99%         | 49.95%         | 4897/10000         |
| ('Low_Capital', 'HMM')         | 51.99%           | 51.01%         | 52.97%         | 5199/10000         |
| ('Standard', 'Transformer')    | 1.55%            | 1.33%          | 1.81%          | 155/10000          |
| ('Standard', 'HMM')            | 8.88%            | 8.34%          | 9.45%          | 888/10000          |
| ('Aggressive', 'Transformer')  | 29.75%           | 28.86%         | 30.65%         | 2975/10000         |
| ('Aggressive', 'HMM')          | 89.54%           | 88.92%         | 90.12%         | 8954/10000         |
| ('Low_Capital', 'HMM_Uni')     | 12.46%           | 11.83%         | 13.12%         | 1246/10000         |
| ('Low_Capital', 'LSTM')        | 8.36%            | 7.83%          | 8.92%          | 836/10000          |
| ('Low_Capital', 'Transformer') | 9.48%            | 8.92%          | 10.07%         | 948/10000          |

### Hypothesis Tests (Paired Wilcoxon, α = 0.05)
**H1: Regime switching reduces MaxDD vs. buy and hold:**

| Model       | Median MaxDD (Model)   | Median MaxDD (B&H)   | Δ Median   |   Wilcoxon p | H1 (α=0.05)   |
|:------------|:-----------------------|:---------------------|:-----------|-------------:|:--------------|
| MSM         | -98.27%                | -62.00%              | -36.27 pp  |     1        | rejected      |
| HMM         | -100.00%               | -62.00%              | -38.00 pp  |     1        | rejected      |
| HMM_Uni     | -99.70%                | -62.00%              | -37.70 pp  |     1        | rejected      |
| LSTM        | -74.09%                | -62.00%              | -12.08 pp  |     1        | rejected      |
| Transformer | -57.55%                | -62.00%              | +4.45 pp   |     1.32e-83 | confirmed     |

**H2: The Transformer dominates econometrics and LSTM in terminal wealth:**

| Comparison              | Median Transformer   | Median MSM   | Δ Median   |   Wilcoxon p | H2 (α=0.05)   | Median HMM   | Median HMM_Uni   | Median LSTM   |
|:------------------------|:---------------------|:-------------|:-----------|-------------:|:--------------|:-------------|:-----------------|:--------------|
| Transformer vs. MSM     | 290,526 €            | 9,308 €      | +281,218 € |            0 | confirmed     | nan          | nan              | nan           |
| Transformer vs. HMM     | 290,526 €            | nan          | +290,526 € |            0 | confirmed     | 0 €          | nan              | nan           |
| Transformer vs. HMM_Uni | 290,526 €            | nan          | +288,869 € |            0 | confirmed     | nan          | 1,657 €          | nan           |
| Transformer vs. LSTM    | 290,526 €            | nan          | +141,197 € |            0 | confirmed     | nan          | nan              | 149,329 €     |

### Break-Even Transaction Costs
At what cost rate (in basis points per reallocation) does active switching lose its return advantage over buy and hold?

| Model       |   Final @10bps |   B&H Final |   Break-Even (bps) |
|:------------|---------------:|------------:|-------------------:|
| MSM         |          3.18  |       4.141 |                  5 |
| HMM         |          2.114 |       4.141 |                  0 |
| HMM_Uni     |          3.155 |       4.141 |                  5 |
| LSTM        |          3.68  |       4.141 |                  0 |
| Transformer |          4.204 |       4.141 |                 20 |

![Break-Even Analysis](../assets/break_even_costs.png)

### Withdrawal Rate Sensitivity (3.5% / 4% / 5%)
Robustness of the SORR results under varying annual withdrawals.

| Strategy    | ('Terminal Capital', '3.5%')   | ('Terminal Capital', '4.0%')   | ('Terminal Capital', '5.0%')   | ('Status', '3.5%')   | ('Status', '4.0%')   | ('Status', '5.0%')   |
|:------------|:-------------------------------|:-------------------------------|:-------------------------------|:---------------------|:---------------------|:---------------------|
| Buy_Hold    | 855,415 €                      | 681,767 €                      | 334,471 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| HMM         | 380,857 €                      | 284,202 €                      | 90,893 €                       | Capital preserved    | Capital preserved    | Capital preserved    |
| HMM_Uni     | 725,827 €                      | 604,112 €                      | 360,682 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| LSTM        | 784,713 €                      | 633,884 €                      | 332,226 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| MSM         | 740,590 €                      | 619,195 €                      | 376,405 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| Transformer | 911,295 €                      | 741,139 €                      | 400,828 €                      | Capital preserved    | Capital preserved    | Capital preserved    |

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

**Last updated:** 2026-08-06 07:31<br>
**End date:** `2026-07-31`<br>
**Fast mode status at runtime:** FALSE (Full Run)<br>
**Walk-forward validation:** ENABLED (mode: rolling, train: 10y, test: 12m, step: 12m)<br>
**Model persistence:** ENABLED<br>
*Generated by the Backtest Service (reporting).*
