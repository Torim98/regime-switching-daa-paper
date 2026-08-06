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
