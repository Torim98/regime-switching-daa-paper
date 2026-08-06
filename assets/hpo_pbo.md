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
