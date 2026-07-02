
# Detailed Statistical Evaluation & Research Results

This page documents the numerical and graphical results of the research pipeline. All evaluations are based on the dataset up to yesterday (2026-07-02) and are updated automatically.

---

## 1. Executive Summary: Performance & Risk
A direct comparison of the core metrics over the entire **out-of-sample test period**.

| Strategy    | Final Wealth   | Total Return   | Max Drawdown   |
|:------------|:---------------|:---------------|:---------------|
| Buy_Hold    | 2,100,062 €    | +320.01%       | -35.08%        |
| MSM         | 1,490,753 €    | +198.15%       | -28.59%        |
| HMM         | 1,128,987 €    | +125.80%       | -18.92%        |
| HMM_Uni     | 1,428,851 €    | +185.77%       | -20.84%        |
| LSTM        | 2,073,088 €    | +314.62%       | -28.07%        |
| Transformer | 2,077,888 €    | +315.58%       | -27.71%        |

> **Key point:** Compare the **max drawdown** of the active strategies with the buy-and-hold benchmark. The objective of this work is a significant reduction of this value to mitigate SORR.

---

## 2. Data Basis & Baseline Portfolio
The analysis is based on a global multi-asset approach.

### Exploratory Data Analysis (EDA)
**Descriptive statistics of the base time series:**
| Time Series   |   Mean (daily) |   Std. Dev. (daily) |     Min |     Max |   Skewness |   Kurtosis |
|:--------------|---------------:|--------------------:|--------:|--------:|-----------:|-----------:|
| Returns_GSPC  |       0.00033  |            0.011355 | -0.1277 |  0.1096 |    -0.3642 |    10.8563 |
| Returns_VUSTX |       0.000213 |            0.007253 | -0.0605 |  0.0654 |    -0.0311 |     4.5138 |
| Returns       |       0.000283 |            0.006883 | -0.0662 |  0.0584 |    -0.2803 |     7.6573 |
| VIX           |      19.4551   |            7.74426  |  9.14   | 82.69   |     2.2071 |     8.7233 |
| TNX_10Y       |       4.2447   |            1.92751  |  0.499  |  9.09   |     0.326  |    -0.6298 |
| IRX_3M        |       2.71389  |            2.19761  | -0.105  |  7.99   |     0.1921 |    -1.2511 |

**Stationarity check (augmented Dickey-Fuller test):**
| Time Series   |   ADF Statistic |    p-Value |   Crit. Value (5%) | Stationary?   |
|:--------------|----------------:|-----------:|-------------------:|:--------------|
| Returns_GSPC  |        -17.5551 | 4.1236e-30 |            -2.8619 | Yes           |
| Returns_VUSTX |        -18.7254 | 2.0318e-30 |            -2.8619 | Yes           |
| Returns       |        -21.0027 | 0          |            -2.8619 | Yes           |
| VIX           |         -7.3109 | 1.2654e-10 |            -2.8619 | Yes           |
| TNX_10Y       |         -2.3444 | 0.15805    |            -2.8619 | No            |
| IRX_3M        |         -2.3483 | 0.15686    |            -2.8619 | No            |

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
Bayesian search over the hyperparameter space of all four models using walk-forward validation as inner CV. The optimization target is the median OOS Sharpe ratio over the subsampled folds; pruned trials use the median pruner. The values reported here were adopted 1:1 into `config.yaml` and used for the final walk-forward run.

# Optuna — Beste Hyperparameter

_Generiert am 2026-04-21 22:01:55_  
Optimierungs-Metrik: **Sharpe (Median OOS)**

## Übersicht

| Modell | Best Score | ✓ Complete | ✗ Pruned | Total |
|:---|---:|---:|---:|---:|
| **MSM** | 0.9308 | 23 | 27 | 50 |
| **HMM** | 1.2843 | 50 | 0 | 50 |
| **LSTM** | 1.4595 | 16 | 14 | 30 |
| **Transformer** | 1.0530 | 19 | 11 | 30 |

### MSM — Best Score `0.9308`

| Parameter | Wert |
|:---|---:|
| `threshold` | `0.7` |

### HMM — Best Score `1.2843`

| Parameter | Wert |
|:---|---:|
| `covariance_type` | `tied` |
| `threshold` | `0.35` |

### LSTM — Best Score `1.4595`

| Parameter | Wert |
|:---|---:|
| `window_size` | `120` |
| `units_l1` | `32` |
| `units_l2` | `64` |
| `learning_rate` | `1.053e-04` |
| `dropout` | `0.4` |
| `epochs` | `40` |
| `threshold` | `0.3` |

### Transformer — Best Score `1.0530`

| Parameter | Wert |
|:---|---:|
| `d_model` | `32` |
| `n_heads` | `4` |
| `n_layers` | `3` |
| `dim_feedforward` | `128` |
| `learning_rate` | `3.282e-05` |
| `dropout` | `0.1` |
| `epochs` | `40` |
| `window_size` | `40` |
| `threshold` | `0.55` |


**Diagnostic plots per model** (optimization history, parameter importance, slice, contour):

| Model | History | Importance | Slice | Contour |
|:---|:---|:---|:---|:---|
| MSM         | ![](../assets/optuna_MSM_history.png)         | ![](../assets/optuna_MSM_importance.png)         | ![](../assets/optuna_MSM_slice.png)         | n/a ¹                                         |
| HMM         | ![](../assets/optuna_HMM_history.png)         | ![](../assets/optuna_HMM_importance.png)         | ![](../assets/optuna_HMM_slice.png)         | ![](../assets/optuna_HMM_contour.png)         |
| HMM_Uni     | ![](../assets/optuna_HMM_Uni_history.png)     | ![](../assets/optuna_HMM_Uni_importance.png)     | ![](../assets/optuna_HMM_Uni_slice.png)     | n/a ¹                                         |
| LSTM        | ![](../assets/optuna_LSTM_history.png)        | ![](../assets/optuna_LSTM_importance.png)        | ![](../assets/optuna_LSTM_slice.png)        | ![](../assets/optuna_LSTM_contour.png)        |
| Transformer | ![](../assets/optuna_Transformer_history.png) | ![](../assets/optuna_Transformer_importance.png) | ![](../assets/optuna_Transformer_slice.png) | ![](../assets/optuna_Transformer_contour.png) |

¹ MSM and HMM_Uni have only one hyperparameter (`threshold`) in the search space. The contour plot would be degenerate and is omitted.

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

### Equity Curves in Comparison
![Equity Curves](../assets/equity_curves.png)

### Annualized Performance Metrics
Normalized metrics (CAGR, Sharpe, Sortino, Calmar) for comparison across evaluation periods of different lengths.

| Strategy    | CAGR   | Ann. Volatility   |   Sharpe Ratio |   Sortino Ratio | Max Drawdown   |   Calmar Ratio |   OOS Days |   OOS Years |
|:------------|:-------|:------------------|---------------:|----------------:|:---------------|---------------:|-----------:|------------:|
| Buy_Hold    | +5.76% | 11.17%            |          0.557 |           0.727 | -35.08%        |          0.164 |       6463 |        25.6 |
| MSM         | +4.35% | 7.30%             |          0.62  |           0.769 | -28.59%        |          0.152 |       6463 |        25.6 |
| HMM         | +3.23% | 6.78%             |          0.503 |           0.527 | -18.92%        |          0.171 |       6463 |        25.6 |
| HMM_Uni     | +4.18% | 6.49%             |          0.664 |           0.764 | -20.84%        |          0.201 |       6463 |        25.6 |
| LSTM        | +5.70% | 10.24%            |          0.593 |           0.74  | -28.07%        |          0.203 |       6463 |        25.6 |
| Transformer | +5.71% | 10.49%            |          0.582 |           0.736 | -27.71%        |          0.206 |       6463 |        25.6 |

### Classification Metrics (vs. NBER Recessions as Ground Truth)
Comparison of the models as binary recession classifiers (precision, recall, F1).

| Model       |   Precision |   Recall |    F1 |   TN |   FP |   FN |   TP |
|:------------|------------:|---------:|------:|-----:|-----:|-----:|-----:|
| MSM         |       0.3   |    0.765 | 0.431 | 4832 | 1046 |  138 |  448 |
| HMM         |       0.193 |    0.84  | 0.314 | 3818 | 2060 |   94 |  492 |
| HMM_Uni     |       0.256 |    0.882 | 0.397 | 4376 | 1502 |   69 |  517 |
| LSTM        |       0.358 |    0.377 | 0.367 | 5481 |  397 |  365 |  221 |
| Transformer |       0.324 |    0.324 | 0.324 | 5482 |  396 |  396 |  190 |

![Confusion Matrices](../assets/confusion_matrices.png)

**ROC and precision-recall curves** (threshold-independent comparison via `*_Prob`):

![ROC Curves](../assets/roc_curves.png)
![PR Curves](../assets/pr_curves.png)

### Signal Churning & Whipsaw Analysis
Quantification of the switching frequency and the share of very short regime phases ("whipsaws").

| Model       |   Signal Switches |   Whipsaws (<5d) | Whipsaw Share   |   Mean Phase (Days) |   Median Phase (Days) | Cumul. Costs   |
|:------------|------------------:|-----------------:|:----------------|--------------------:|----------------------:|:---------------|
| MSM         |               369 |              208 | 56.2%           |                17.5 |                     4 | 36.90%         |
| HMM         |               141 |               64 | 45.1%           |                45.5 |                     6 | 14.10%         |
| HMM_Uni     |               311 |              160 | 51.3%           |                20.7 |                     4 | 31.10%         |
| LSTM        |                18 |                0 | 0.0%            |               340.2 |                    59 | 1.80%          |
| Transformer |                54 |               27 | 49.1%           |               117.5 |                     5 | 5.40%          |

### Regime Probability Heatmap
Bear probabilities of all models over time.

![Regime Probability Heatmap](../assets/regime_probability_heatmap.png)

### Threshold Sensitivity
Variation of the decision threshold per model. Shows how robust final wealth, max drawdown, and the number of regime switches are to a modified bull/bear classification boundary (thesis ch. 4.1, smoothing).

**MSM**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.25 | 1,531,210 €    | -12.13%        |        301 |
|        0.3  | 1,355,205 €    | -16.68%        |        313 |
|        0.35 | 1,365,720 €    | -20.19%        |        319 |
|        0.4  | 1,381,425 €    | -21.55%        |        313 |
|        0.5  | 1,466,683 €    | -25.39%        |        325 |

**HMM**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.4  | 1,151,590 €    | -20.94%        |         97 |
|        0.45 | 1,055,903 €    | -25.13%        |         83 |
|        0.5  | 1,031,745 €    | -26.82%        |         87 |
|        0.55 | 1,034,573 €    | -28.12%        |         83 |
|        0.6  | 946,552 €      | -33.98%        |         97 |

**HMM_Uni**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.4  | 1,355,149 €    | -24.79%        |        315 |
|        0.45 | 1,377,857 €    | -24.41%        |        323 |
|        0.5  | 1,477,583 €    | -26.16%        |        339 |
|        0.55 | 1,484,996 €    | -23.24%        |        337 |
|        0.6  | 1,673,848 €    | -26.81%        |        341 |

**LSTM**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|         0.2 | 2,077,541 €    | -28.07%        |         18 |
|         0.3 | 2,073,088 €    | -28.07%        |         18 |
|         0.4 | 2,164,983 €    | -28.08%        |         20 |
|         0.5 | 2,136,351 €    | -27.98%        |         20 |

**Transformer**

|   Threshold | Final Wealth   | Max Drawdown   |   Switches |
|------------:|:---------------|:---------------|-----------:|
|        0.3  | 1,424,697 €    | -32.24%        |        106 |
|        0.4  | 1,869,738 €    | -27.71%        |         84 |
|        0.45 | 1,925,907 €    | -27.71%        |         74 |
|        0.5  | 1,982,421 €    | -27.71%        |         66 |
|        0.6  | 2,015,762 €    | -27.71%        |         56 |

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
| 2001-02-02 | 2003-02-12 | 2007-04-20 | -27.85%  |                     740 |                    1528 |        2268 |
| 2007-06-05 | 2009-08-06 | 2010-11-04 | -15.37%  |                     793 |                     455 |        1248 |
| 2011-07-25 | 2011-10-03 | 2012-01-17 | -7.98%   |                      70 |                     106 |         176 |
| 2015-02-26 | 2016-01-20 | 2016-03-18 | -6.77%   |                     328 |                      58 |         386 |
| 2016-08-01 | 2016-11-14 | 2017-04-18 | -5.84%   |                     105 |                     155 |         260 |
| 2018-01-29 | 2018-04-25 | 2018-08-20 | -5.80%   |                      86 |                     117 |         203 |
| 2018-08-30 | 2019-01-14 | 2019-06-07 | -9.46%   |                     137 |                     144 |         281 |
| 2020-09-03 | 2021-03-24 | 2021-08-30 | -10.20%  |                     202 |                     159 |         361 |
| 2021-12-28 | 2022-04-06 | 2023-04-06 | -10.67%  |                      99 |                     365 |         464 |
| 2023-07-20 | 2023-10-19 | 2024-06-05 | -10.89%  |                      91 |                     230 |         321 |
| 2024-12-09 | 2025-01-13 | 2025-09-05 | -6.46%   |                      35 |                     235 |         270 |
| 2025-10-29 | 2026-03-19 | 2026-05-29 | -6.25%   |                     141 |                      71 |         212 |

**HMM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2002-03-07 | 2002-06-13 | 2003-06-16 | -6.63%   |                      98 |                     368 |         466 |
| 2003-06-17 | 2003-07-21 | 2003-12-31 | -5.16%   |                      34 |                     163 |         197 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                      63 |                     178 |         241 |
| 2008-05-20 | 2008-07-28 | 2009-11-16 | -8.46%   |                      69 |                     476 |         545 |
| 2010-05-04 | 2010-07-16 | 2010-11-04 | -7.14%   |                      73 |                     111 |         184 |
| 2011-07-25 | 2011-12-14 | 2012-06-29 | -7.85%   |                     142 |                     198 |         340 |
| 2013-05-22 | 2013-06-24 | 2014-10-21 | -5.37%   |                      33 |                     484 |         517 |
| 2015-03-23 | 2015-09-28 | 2016-06-03 | -9.82%   |                     189 |                     249 |         438 |
| 2018-01-29 | 2021-03-18 | 2021-11-05 | -13.45%  |                    1144 |                     232 |        1376 |
| 2021-11-10 | 2023-10-27 | 2024-06-12 | -18.52%  |                     716 |                     229 |         945 |
| 2024-12-09 | 2025-04-08 | 2025-08-04 | -12.80%  |                     120 |                     118 |         238 |
| 2025-10-29 | 2026-03-27 | 2026-05-06 | -6.95%   |                     149 |                      40 |         189 |

**HMM_Uni**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2001-02-02 | 2003-01-27 | 2006-01-03 | -19.80%  |                     724 |                    1072 |        1796 |
| 2007-06-05 | 2009-06-23 | 2010-09-20 | -11.77%  |                     749 |                     454 |        1203 |
| 2011-07-25 | 2011-10-03 | 2012-01-17 | -7.53%   |                      70 |                     106 |         176 |
| 2013-05-22 | 2013-08-21 | 2013-11-29 | -5.70%   |                      91 |                     100 |         191 |
| 2015-02-26 | 2016-01-20 | 2016-04-13 | -7.68%   |                     328 |                      84 |         412 |
| 2016-08-01 | 2016-11-14 | 2017-04-18 | -5.76%   |                     105 |                     155 |         260 |
| 2018-01-29 | 2018-04-25 | 2018-08-27 | -6.34%   |                      86 |                     124 |         210 |
| 2018-08-30 | 2019-01-14 | 2019-03-29 | -6.93%   |                     137 |                      74 |         211 |
| 2020-09-03 | 2021-03-25 | 2021-07-07 | -6.73%   |                     203 |                     104 |         307 |
| 2021-11-10 | 2022-05-19 | 2023-03-30 | -8.22%   |                     190 |                     315 |         505 |
| 2023-07-03 | 2023-09-22 | 2023-12-26 | -5.50%   |                      81 |                      95 |         176 |
| 2024-07-17 | 2025-05-06 | open       | -10.72%  |                     293 |                     nan |         nan |

**LSTM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                      49 |                      43 |          92 |
| 2001-02-02 | 2002-07-23 | 2004-03-05 | -24.04%  |                     536 |                     591 |        1127 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                      63 |                     178 |         241 |
| 2008-05-20 | 2009-03-09 | 2010-04-14 | -27.82%  |                     293 |                     401 |         694 |
| 2010-05-04 | 2010-07-02 | 2010-09-13 | -5.36%   |                      59 |                      73 |         132 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                      14 |                      67 |          81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                      33 |                     120 |         153 |
| 2015-03-23 | 2015-08-25 | 2016-06-07 | -8.39%   |                     155 |                     287 |         442 |
| 2016-08-01 | 2016-11-14 | 2017-04-17 | -5.64%   |                     105 |                     154 |         259 |
| 2018-01-29 | 2018-12-24 | 2019-04-05 | -12.21%  |                     329 |                     102 |         431 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                      26 |                      82 |         108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                      57 |                      39 |          96 |
| 2021-12-28 | 2022-10-14 | 2024-11-29 | -27.55%  |                     290 |                     777 |        1067 |
| 2024-12-09 | 2025-04-08 | 2025-07-03 | -12.22%  |                     120 |                      86 |         206 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                      29 |                      21 |          50 |

**Transformer**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown Duration (d) |   Recovery Duration (d) |   Total (d) |
|:-----------|:-----------|:-----------|:---------|------------------------:|------------------------:|------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                      49 |                      43 |          92 |
| 2001-02-02 | 2002-07-23 | 2004-03-05 | -24.04%  |                     536 |                     591 |        1127 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                      63 |                     178 |         241 |
| 2007-07-20 | 2009-03-09 | 2009-09-16 | -23.26%  |                     598 |                     191 |         789 |
| 2010-05-04 | 2010-07-02 | 2010-09-28 | -5.36%   |                      59 |                      88 |         147 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                      14 |                      67 |          81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                      33 |                     120 |         153 |
| 2015-03-23 | 2015-12-03 | 2017-06-02 | -10.35%  |                     255 |                     547 |         802 |
| 2018-01-29 | 2018-02-08 | 2018-08-29 | -6.93%   |                      10 |                     202 |         212 |
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
| COVID Crash (2020-02 to 2020-03)    | -8.55%                   | -6.60%              | +0.73%                  | -8.55%               | +5.87%              | -8.55%                      | -18.53%                        | -6.72%                    | -1.81%                        | -18.53%                    | -1.81%                    | -18.53%                           |
| Dot-Com (2000-03 to 2002-10)        | -15.77%                  | -0.95%              | -13.20%                 | -15.77%              | -23.94%             | -15.77%                     | -24.81%                        | -7.15%                    | -16.37%                       | -24.81%                    | -26.53%                   | -24.81%                           |
| EU Debt Crisis (2011-07 to 2011-11) | +4.10%                   | -7.09%              | -1.37%                  | +4.10%               | -1.84%              | +5.61%                      | -7.24%                         | -8.17%                    | -8.17%                        | -7.24%                     | -8.61%                    | -7.24%                            |
| GFC (2007-10 to 2009-03)            | -26.99%                  | -4.06%              | -8.55%                  | -17.21%              | -11.72%             | -10.59%                     | -35.08%                        | -8.78%                    | -9.39%                        | -28.07%                    | -13.30%                   | -20.50%                           |
| Rate Hikes (2022-01 to 2022-10)     | -24.20%                  | -12.94%             | -4.32%                  | -24.20%              | -4.59%              | -24.20%                     | -26.98%                        | -14.56%                   | -7.13%                        | -26.98%                    | -9.96%                    | -26.98%                           |

### Switch Timing Relative to the Drawdown Peak
Time lag between the model's first bear signal and the drawdown trough of the buy-and-hold portfolio per crisis. Positive = model reacted early, negative = too late.

| Crisis   | Model       | DD Trough   | First Bear Signal   |   Lead (Days) |
|:---------|:------------|:------------|:--------------------|--------------:|
| GFC      | MSM         | 2009-03-09  | 2007-10-01          |           525 |
| COVID    | MSM         | 2020-03-18  | 2020-02-24          |            23 |
| 2022     | MSM         | 2022-10-14  | 2022-01-05          |           282 |
| GFC      | HMM         | 2009-03-09  | 2007-10-22          |           504 |
| COVID    | HMM         | 2020-03-18  | 2020-02-03          |            44 |
| 2022     | HMM         | 2022-10-14  | 2022-01-21          |           266 |
| GFC      | HMM_Uni     | 2009-03-09  | 2007-10-01          |           525 |
| COVID    | HMM_Uni     | 2020-03-18  | 2020-02-24          |            23 |
| 2022     | HMM_Uni     | 2022-10-14  | 2022-01-05          |           282 |
| GFC      | LSTM        | 2009-03-09  | 2007-10-01          |           525 |
| COVID    | LSTM        | 2020-03-18  |                     |           nan |
| 2022     | LSTM        | 2022-10-14  |                     |           nan |
| GFC      | Transformer | 2009-03-09  | 2008-01-08          |           426 |
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
| Buy Hold    | 320.09%        | 5.74%         | 11.17%       | -35.08%        |           0.56 |            0.73 |           0.16 |                 0 | 0.00%                |          9.07 |
| MSM         | 198.21%        | 4.34%         | 7.30%        | -28.59%        |           0.62 |            0.77 |           0.15 |               369 | 37.00%               |          9.8  |
| HMM         | 125.84%        | 3.22%         | 6.78%        | -18.92%        |           0.5  |            0.53 |           0.17 |               141 | 14.20%               |          5.23 |
| HMM Uni     | 185.82%        | 4.17%         | 6.49%        | -20.84%        |           0.66 |            0.76 |           0.2  |               311 | 31.20%               |          6.69 |
| LSTM        | 314.70%        | 5.69%         | 10.24%       | -28.07%        |           0.59 |            0.74 |           0.2  |                18 | 1.80%                |          7.9  |
| Transformer | 315.66%        | 5.70%         | 10.49%       | -27.71%        |           0.58 |            0.74 |           0.21 |                54 | 5.40%                |          7.94 |

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
| ('Standard', 'MSM')            | 0.00 €             | Depleted (2017)   |
| ('Standard', 'HMM')            | 0.00 €             | Depleted (2024)   |
| ('Standard', 'HMM Uni')        | 0.00 €             | Depleted (2022)   |
| ('Standard', 'LSTM')           | 0.00 €             | Depleted (2026)   |
| ('Standard', 'Transformer')    | 15,117.44 €        | Capital preserved |
| ('Aggressive', 'Buy Hold')     | 0.00 €             | Depleted (2011)   |
| ('Aggressive', 'MSM')          | 0.00 €             | Depleted (2009)   |
| ('Aggressive', 'HMM')          | 0.00 €             | Depleted (2013)   |
| ('Aggressive', 'HMM Uni')      | 0.00 €             | Depleted (2011)   |
| ('Aggressive', 'LSTM')         | 0.00 €             | Depleted (2011)   |
| ('Aggressive', 'Transformer')  | 0.00 €             | Depleted (2011)   |
| ('Low_Capital', 'Buy Hold')    | 0.00 €             | Depleted (2015)   |
| ('Low_Capital', 'MSM')         | 0.00 €             | Depleted (2011)   |
| ('Low_Capital', 'HMM')         | 0.00 €             | Depleted (2016)   |
| ('Low_Capital', 'HMM Uni')     | 0.00 €             | Depleted (2013)   |
| ('Low_Capital', 'LSTM')        | 0.00 €             | Depleted (2015)   |
| ('Low_Capital', 'Transformer') | 0.00 €             | Depleted (2015)   |

Capital development of the different scenarios:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness Check

To assess statistical significance, 10,000 artificial market paths were simulated via block bootstrap.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin Probability   | Median Terminal Capital   |
|:-------------------------------|:-------------------|:--------------------------|
| ('Standard', 'HMM')            | 0.00%              | 334,366.67 €              |
| ('Aggressive', 'LSTM')         | 4.50%              | 225,927.81 €              |
| ('Standard', 'LSTM')           | 0.00%              | 469,590.62 €              |
| ('Low_Capital', 'HMM')         | 0.12%              | 129,406.62 €              |
| ('Low_Capital', 'Buy Hold')    | 0.90%              | 200,414.37 €              |
| ('Aggressive', 'Transformer')  | 4.40%              | 222,758.70 €              |
| ('Standard', 'Buy Hold')       | 0.02%              | 471,862.39 €              |
| ('Standard', 'MSM')            | 0.00%              | 393,350.92 €              |
| ('Aggressive', 'Buy Hold')     | 5.73%              | 227,732.31 €              |
| ('Standard', 'Transformer')    | 0.01%              | 464,921.13 €              |
| ('Aggressive', 'HMM Uni')      | 1.88%              | 158,086.97 €              |
| ('Low_Capital', 'MSM')         | 0.08%              | 159,542.78 €              |
| ('Standard', 'HMM Uni')        | 0.00%              | 383,999.36 €              |
| ('Aggressive', 'MSM')          | 3.25%              | 165,348.20 €              |
| ('Aggressive', 'HMM')          | 5.36%              | 121,219.91 €              |
| ('Low_Capital', 'HMM Uni')     | 0.03%              | 153,572.06 €              |
| ('Low_Capital', 'LSTM')        | 0.38%              | 196,097.75 €              |
| ('Low_Capital', 'Transformer') | 0.49%              | 198,513.01 €              |

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
| ('Standard', 'Buy_Hold')       | 0.02%            | 0.01%          | 0.07%          | 2/10000            |
| ('Standard', 'MSM')            | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Standard', 'HMM')            | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Standard', 'HMM_Uni')        | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Standard', 'LSTM')           | 0.00%            | 0.00%          | 0.04%          | 0/10000            |
| ('Standard', 'Transformer')    | 0.01%            | 0.00%          | 0.06%          | 1/10000            |
| ('Aggressive', 'Buy_Hold')     | 5.73%            | 5.29%          | 6.20%          | 573/10000          |
| ('Aggressive', 'MSM')          | 3.25%            | 2.92%          | 3.62%          | 325/10000          |
| ('Aggressive', 'HMM')          | 5.36%            | 4.94%          | 5.82%          | 536/10000          |
| ('Aggressive', 'HMM_Uni')      | 1.88%            | 1.63%          | 2.17%          | 188/10000          |
| ('Aggressive', 'LSTM')         | 4.50%            | 4.11%          | 4.92%          | 450/10000          |
| ('Aggressive', 'Transformer')  | 4.40%            | 4.02%          | 4.82%          | 440/10000          |
| ('Low_Capital', 'Buy_Hold')    | 0.90%            | 0.73%          | 1.10%          | 90/10000           |
| ('Low_Capital', 'MSM')         | 0.08%            | 0.04%          | 0.16%          | 8/10000            |
| ('Low_Capital', 'HMM')         | 0.12%            | 0.07%          | 0.21%          | 12/10000           |
| ('Low_Capital', 'HMM_Uni')     | 0.03%            | 0.01%          | 0.09%          | 3/10000            |
| ('Low_Capital', 'LSTM')        | 0.38%            | 0.28%          | 0.52%          | 38/10000           |
| ('Low_Capital', 'Transformer') | 0.49%            | 0.37%          | 0.65%          | 49/10000           |

### Hypothesis Tests (Paired Wilcoxon, α = 0.05)
**H1: Regime switching reduces MaxDD vs. buy and hold:**

| Model       | Median MaxDD (Model)   | Median MaxDD (B&H)   | Δ Median   |   Wilcoxon p | H1 (α=0.05)   |
|:------------|:-----------------------|:---------------------|:-----------|-------------:|:--------------|
| MSM         | -68.71%                | -59.46%              | -9.25 pp   |        1     | rejected      |
| HMM         | -76.54%                | -59.46%              | -17.08 pp  |        1     | rejected      |
| HMM_Uni     | -69.66%                | -59.46%              | -10.20 pp  |        1     | rejected      |
| LSTM        | -59.32%                | -59.46%              | +0.13 pp   |        0.36  | rejected      |
| Transformer | -59.81%                | -59.46%              | -0.35 pp   |        0.765 | rejected      |

**H2: The Transformer dominates econometrics and LSTM in terminal wealth:**

| Comparison           | Median Transformer   | Median MSM   | Δ Median   |   Wilcoxon p | H2 (α=0.05)   | Median HMM   | Median LSTM   |
|:---------------------|:---------------------|:-------------|:-----------|-------------:|:--------------|:-------------|:--------------|
| Transformer vs. MSM  | 222,759 €            | 165,348 €    | +57,411 €  |    5.47e-196 | confirmed     | nan          | nan           |
| Transformer vs. HMM  | 222,759 €            | nan          | +101,539 € |    0         | confirmed     | 121,220 €    | nan           |
| Transformer vs. LSTM | 222,759 €            | nan          | -3,169 €   |    0.854     | rejected      | nan          | 225,928 €     |

### Break-Even Transaction Costs
At what cost rate (in basis points per reallocation) does active switching lose its return advantage over buy and hold?

| Model       |   Final @10bps |   B&H Final |   Break-Even (bps) |
|:------------|---------------:|------------:|-------------------:|
| MSM         |          2.982 |         4.2 |                  5 |
| HMM         |          2.258 |         4.2 |                  0 |
| HMM_Uni     |          2.858 |         4.2 |                  0 |
| LSTM        |          4.146 |         4.2 |                  5 |
| Transformer |          4.156 |         4.2 |                 10 |

![Break-Even Analysis](../assets/break_even_costs.png)

### Withdrawal Rate Sensitivity (3.5% / 4% / 5%)
Robustness of the SORR results under varying annual withdrawals.

| Strategy    | ('Terminal Capital', '3.5%')   | ('Terminal Capital', '4.0%')   | ('Terminal Capital', '5.0%')   | ('Status', '3.5%')   | ('Status', '4.0%')   | ('Status', '5.0%')   |
|:------------|:-------------------------------|:-------------------------------|:-------------------------------|:---------------------|:---------------------|:---------------------|
| Buy_Hold    | 867,601 €                      | 691,478 €                      | 339,233 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| HMM         | 422,752 €                      | 321,831 €                      | 119,989 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| HMM_Uni     | 515,236 €                      | 384,681 €                      | 123,571 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| LSTM        | 858,236 €                      | 684,630 €                      | 337,418 €                      | Capital preserved    | Capital preserved    | Capital preserved    |
| MSM         | 425,496 €                      | 273,276 €                      | 0 €                            | Capital preserved    | Capital preserved    | Depleted (2025)      |
| Transformer | 874,769 €                      | 702,838 €                      | 358,978 €                      | Capital preserved    | Capital preserved    | Capital preserved    |

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

**Last updated:** 2026-07-02 13:51<br>
**End date:** `2026-07-02`<br>
**Fast mode status at runtime:** FALSE (Full Run)<br>
**Walk-forward validation:** ENABLED (mode: rolling, train: 10y, test: 12m, step: 12m)<br>
**Model persistence:** ENABLED<br>
*Generated by the Backtest Service (reporting).*
