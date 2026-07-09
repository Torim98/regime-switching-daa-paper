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
