# Probability of Backtest Overfitting (CSCV)

_Generated at 2026-08-08 10:13:32_

| model       |   folds |   configs |   pbo |
|:------------|--------:|----------:|------:|
| MSM         |      16 |        15 | 0.488 |
| HMM         |      16 |        15 | 0.972 |
| HMM_Uni     |      16 |        15 | 0.484 |
| LSTM        |      16 |        15 | 0.159 |
| Transformer |      16 |        15 | 0.091 |

PBO over the top-15 configs per model (per-fold Sharpe matrix, CSCV). Lower is better; PBO > 0.5 flags overfitting.
