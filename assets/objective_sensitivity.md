# Objective Sensitivity of the Selected Hyperparameters

_Generated at 2026-08-03 14:44:41_  
Best config under each candidate metric, valued across all metrics (from the search trials' logged OOS metrics; no retraining). `same_as_objective` marks configs identical to the actual objective's pick.

## MSM (objective: martin, 36 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |
| sharpe          |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |
| sortino         |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |
| calmar          |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |
| ulcer           |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |
| max_drawdown    |      23 | True                |   1.5402 |   0.8473 |    1.0053 |   0.6179 |  3.3506 |        -0.0835 | 0.0516 |

Selected configs:
- best under **martin**: threshold=0.175

## HMM (objective: martin, 108 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |
| sharpe          |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |
| sortino         |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |
| calmar          |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |
| ulcer           |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |
| max_drawdown    |     103 | True                |   0.8231 |    0.551 |    0.5753 |    0.305 |   3.347 |        -0.0903 | 0.0275 |

Selected configs:
- best under **martin**: covariance_type=diag, threshold=0.975

## HMM_Uni (objective: martin, 36 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      23 | True                |   1.5296 |   0.8391 |    0.9954 |   0.6334 |  3.344  |        -0.0808 | 0.0511 |
| sharpe          |      23 | True                |   1.5296 |   0.8391 |    0.9954 |   0.6334 |  3.344  |        -0.0808 | 0.0511 |
| sortino         |      23 | True                |   1.5296 |   0.8391 |    0.9954 |   0.6334 |  3.344  |        -0.0808 | 0.0511 |
| calmar          |      23 | True                |   1.5296 |   0.8391 |    0.9954 |   0.6334 |  3.344  |        -0.0808 | 0.0511 |
| ulcer           |      12 | False               |   1.3523 |   0.7702 |    0.8698 |   0.4296 |  3.2777 |        -0.1032 | 0.0443 |
| max_drawdown    |      23 | True                |   1.5296 |   0.8391 |    0.9954 |   0.6334 |  3.344  |        -0.0808 | 0.0511 |

Selected configs:
- best under **martin**: threshold=0.175
- best under **ulcer**: threshold=0.1

## LSTM (objective: martin, 300 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |     191 | True                |   2.3342 |   0.902  |    0.7655 |   0.5638 |  1.7773 |        -0.0736 | 0.0415 |
| sharpe          |     191 | True                |   2.3342 |   0.902  |    0.7655 |   0.5638 |  1.7773 |        -0.0736 | 0.0415 |
| sortino         |     248 | False               |   1.3234 |   0.8061 |    1.0478 |   0.295  |  5.0452 |        -0.2263 | 0.0668 |
| calmar          |     191 | True                |   2.3342 |   0.902  |    0.7655 |   0.5638 |  1.7773 |        -0.0736 | 0.0415 |
| ulcer           |     191 | True                |   2.3342 |   0.902  |    0.7655 |   0.5638 |  1.7773 |        -0.0736 | 0.0415 |
| max_drawdown    |     191 | True                |   2.3342 |   0.902  |    0.7655 |   0.5638 |  1.7773 |        -0.0736 | 0.0415 |

Selected configs:
- best under **martin**: window_size=140, units_l1=16, units_l2=32, batch_size=128, learning_rate=2.007996207208503e-05, dropout=0.5, threshold=0.35
- best under **sortino**: window_size=140, units_l1=32, units_l2=256, batch_size=64, learning_rate=0.00010450700015167667, dropout=0.5, threshold=0.45000000000000007

## Transformer (objective: martin, 400 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |     298 | True                |   1.0331 |   0.61   |    0.7195 |   0.2133 |  5.2343 |        -0.2535 | 0.0541 |
| sharpe          |     300 | False               |   0.5833 |   0.6814 |    0.9056 |   0.1758 |  9.6727 |        -0.321  | 0.0564 |
| sortino         |     300 | False               |   0.5833 |   0.6814 |    0.9056 |   0.1758 |  9.6727 |        -0.321  | 0.0564 |
| calmar          |      86 | False               |   1.0191 |   0.5766 |    0.6604 |   0.2418 |  4.6971 |        -0.1979 | 0.0479 |
| ulcer           |     327 | False               |   0.9252 |   0.5938 |    0.7122 |   0.2245 |  4.5029 |        -0.1855 | 0.0417 |
| max_drawdown    |     156 | False               |   0.3567 |   0.4903 |    0.4401 |   0.1893 |  6.1963 |        -0.1167 | 0.0221 |

Selected configs:
- best under **martin**: window_size=230, dmodel_nheads=16-2, n_layers=2, dim_feedforward=256, batch_size=64, learning_rate=0.003203170398725006, dropout=0.25, threshold=0.5
- best under **sharpe**: window_size=170, dmodel_nheads=16-2, n_layers=2, dim_feedforward=256, batch_size=64, learning_rate=0.005797912630385614, dropout=0.15000000000000002, threshold=0.4
- best under **calmar**: window_size=100, dmodel_nheads=32-4, n_layers=3, dim_feedforward=32, batch_size=32, learning_rate=6.3198947237583e-05, dropout=0.4, threshold=0.15000000000000002
- best under **ulcer**: window_size=230, dmodel_nheads=128-8, n_layers=2, dim_feedforward=256, batch_size=64, learning_rate=0.003682334936836272, dropout=0.35000000000000003, threshold=0.45000000000000007
- best under **max_drawdown**: window_size=50, dmodel_nheads=32-4, n_layers=2, dim_feedforward=64, batch_size=32, learning_rate=1.9311434907966562e-05, dropout=0.45, threshold=0.1
