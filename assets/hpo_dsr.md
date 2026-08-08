# Deflated Sharpe Ratio (multiple-testing adjusted)

_Generated at 2026-08-08 08:41:21_

| model       |   n_trials |   sr_ann_best |   sr_star_ann |    dsr | significant_5pct   |
|:------------|-----------:|--------------:|--------------:|-------:|:-------------------|
| MSM         |         36 |        0.8473 |        0.3051 | 0.9841 | True               |
| HMM         |        108 |        0.551  |        0.2763 | 0.8619 | False              |
| HMM_Uni     |         36 |        0.8391 |        0.3206 | 0.98   | True               |
| LSTM        |        300 |        0.7908 |        0.2834 | 0.9785 | True               |
| Transformer |        400 |        0.9737 |        0.304  | 0.9959 | True               |

DSR = P(true Sharpe > 0) after deflating the best trial's Sharpe for the number of tested configs. `significant_5pct` = DSR > 0.95.
