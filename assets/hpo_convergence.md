# HPO Convergence & Edge-of-Range Review

_Generated at 2026-07-04 13:59:29_

| model       | metric   |   best_value |   best_trial |   conv_frac |   n_complete |   n_pruned | top_importance                              | edge_flags                                                                                                                          | status   |
|:------------|:---------|-------------:|-------------:|------------:|-------------:|-----------:|:--------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|:---------|
| MSM         | martin   |       1.6508 |           19 |        0.54 |           36 |          0 |                                             | -                                                                                                                                   | nan      |
| HMM         | martin   |       1.2037 |            4 |        0.04 |          108 |          0 | covariance_type=1.00                        | -                                                                                                                                   | nan      |
| HMM_Uni     | martin   |       1.6384 |           19 |        0.54 |           36 |          0 |                                             | -                                                                                                                                   | nan      |
| LSTM        | martin   |       1.885  |           55 |        0.95 |           57 |          0 | dropout=0.21, units_l1=0.20, threshold=0.18 | window_size=250 near UPPER bound 250; learning_rate=1.630044194460958e-05 near LOWER bound 1e-05; dropout=0.55 near UPPER bound 0.6 | nan      |
| Transformer | nan      |     nan      |          nan |      nan    |          nan |        nan | nan                                         | nan                                                                                                                                 | no study |
