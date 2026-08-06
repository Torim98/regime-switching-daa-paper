# HPO Convergence & Edge-of-Range Review

_Generated at 2026-08-03 14:44:41_

| model       | metric   |   best_value |   best_trial |   conv_frac |   n_complete |   n_pruned | top_importance                                       | edge_flags                                                 |
|:------------|:---------|-------------:|-------------:|------------:|-------------:|-----------:|:-----------------------------------------------------|:-----------------------------------------------------------|
| MSM         | martin   |       1.5402 |           23 |        0.66 |           36 |          0 | threshold=1.00                                       | -                                                          |
| HMM         | martin   |       0.8231 |          103 |        0.96 |          108 |          0 | covariance_type=0.98, threshold=0.02                 | threshold=0.975 near UPPER bound 0.975                     |
| HMM_Uni     | martin   |       1.5296 |           23 |        0.66 |           36 |          0 | threshold=1.00                                       | -                                                          |
| LSTM        | martin   |       2.3342 |          191 |        0.63 |          300 |          0 | learning_rate=0.30, threshold=0.28, dropout=0.17     | learning_rate=2.007996207208503e-05 near LOWER bound 1e-05 |
| Transformer | martin   |       1.0331 |          298 |        0.74 |          400 |          0 | window_size=0.48, learning_rate=0.18, threshold=0.12 | n_layers=2 near LOWER bound 1                              |
