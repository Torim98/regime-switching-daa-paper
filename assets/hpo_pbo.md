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
