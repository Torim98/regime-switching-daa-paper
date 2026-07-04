# Probability of Backtest Overfitting (CSCV)

_Generated at 2026-07-04 11:54:21_

| model   |   folds |   configs |   pbo |
|:--------|--------:|----------:|------:|
| MSM     |      16 |         5 | 0.873 |
| HMM     |      16 |         5 | 0.083 |
| HMM_Uni |      16 |         5 | 0.909 |

PBO over the top-5 configs per model (per-fold Sharpe matrix, CSCV). Lower is better; PBO > 0.5 flags overfitting.
