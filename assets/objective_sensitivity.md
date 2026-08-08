# Objective Sensitivity of the Selected Hyperparameters

_Generated at 2026-08-08 08:37:28_  
Best config under each candidate metric, valued across all metrics (from the search trials' logged OOS metrics; no retraining). `same_as_objective` marks configs identical to the actual objective's pick.

## MSM (objective: martin, 36 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      23 | True                |   1.6366 |   0.8784 |    1.0486 |   0.6579 |  3.2758 |        -0.0815 | 0.0536 |
| sharpe          |      23 | True                |   1.6366 |   0.8784 |    1.0486 |   0.6579 |  3.2758 |        -0.0815 | 0.0536 |
| sortino         |      23 | True                |   1.6366 |   0.8784 |    1.0486 |   0.6579 |  3.2758 |        -0.0815 | 0.0536 |
| calmar          |      23 | True                |   1.6366 |   0.8784 |    1.0486 |   0.6579 |  3.2758 |        -0.0815 | 0.0536 |
| ulcer           |      23 | True                |   1.6366 |   0.8784 |    1.0486 |   0.6579 |  3.2758 |        -0.0815 | 0.0536 |
| max_drawdown    |      23 | True                |   1.6366 |   0.8784 |    1.0486 |   0.6579 |  3.2758 |        -0.0815 | 0.0536 |

Selected configs:
- best under **martin**: threshold=0.175

## HMM (objective: martin, 108 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |     103 | True                |   0.8884 |   0.5772 |    0.6061 |   0.3276 |  3.2562 |        -0.0883 | 0.0289 |
| sharpe          |     103 | True                |   0.8884 |   0.5772 |    0.6061 |   0.3276 |  3.2562 |        -0.0883 | 0.0289 |
| sortino         |     103 | True                |   0.8884 |   0.5772 |    0.6061 |   0.3276 |  3.2562 |        -0.0883 | 0.0289 |
| calmar          |     103 | True                |   0.8884 |   0.5772 |    0.6061 |   0.3276 |  3.2562 |        -0.0883 | 0.0289 |
| ulcer           |     103 | True                |   0.8884 |   0.5772 |    0.6061 |   0.3276 |  3.2562 |        -0.0883 | 0.0289 |
| max_drawdown    |     103 | True                |   0.8884 |   0.5772 |    0.6061 |   0.3276 |  3.2562 |        -0.0883 | 0.0289 |

Selected configs:
- best under **martin**: covariance_type=diag, threshold=0.975

## HMM_Uni (objective: martin, 36 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      23 | True                |   1.6261 |   0.8702 |    1.0387 |   0.6703 |  3.2691 |        -0.0793 | 0.0532 |
| sharpe          |      23 | True                |   1.6261 |   0.8702 |    1.0387 |   0.6703 |  3.2691 |        -0.0793 | 0.0532 |
| sortino         |      23 | True                |   1.6261 |   0.8702 |    1.0387 |   0.6703 |  3.2691 |        -0.0793 | 0.0532 |
| calmar          |      23 | True                |   1.6261 |   0.8702 |    1.0387 |   0.6703 |  3.2691 |        -0.0793 | 0.0532 |
| ulcer           |      12 | False               |   1.4351 |   0.7999 |    0.9088 |   0.4525 |  3.2139 |        -0.1019 | 0.0461 |
| max_drawdown    |      23 | True                |   1.6261 |   0.8702 |    1.0387 |   0.6703 |  3.2691 |        -0.0793 | 0.0532 |

Selected configs:
- best under **martin**: threshold=0.175
- best under **ulcer**: threshold=0.1

## LSTM (objective: martin, 300 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      98 | True                |   2.3114 |   0.8823 |    1.0285 |   0.341  |  2.8716 |        -0.1946 | 0.0664 |
| sharpe          |      98 | True                |   2.3114 |   0.8823 |    1.0285 |   0.341  |  2.8716 |        -0.1946 | 0.0664 |
| sortino         |     256 | False               |   1.7195 |   0.8217 |    1.1045 |   0.348  |  3.8486 |        -0.1902 | 0.0662 |
| calmar          |     167 | False               |   1.8084 |   0.8056 |    1.0125 |   0.3731 |  4.0151 |        -0.1946 | 0.0726 |
| ulcer           |      98 | True                |   2.3114 |   0.8823 |    1.0285 |   0.341  |  2.8716 |        -0.1946 | 0.0664 |
| max_drawdown    |     222 | False               |   1.503  |   0.7961 |    0.8874 |   0.2944 |  3.5731 |        -0.1824 | 0.0537 |

Selected configs:
- best under **martin**: window_size=100, units_l1=32, units_l2=256, batch_size=64, learning_rate=2.4402813755083274e-05, dropout=0.25, threshold=0.30000000000000004
- best under **sortino**: window_size=220, units_l1=128, units_l2=32, batch_size=32, learning_rate=1.2305735878442937e-05, dropout=0.0, threshold=0.2
- best under **calmar**: window_size=120, units_l1=128, units_l2=32, batch_size=32, learning_rate=1.2474927184485425e-05, dropout=0.30000000000000004, threshold=0.30000000000000004
- best under **max_drawdown**: window_size=170, units_l1=32, units_l2=128, batch_size=32, learning_rate=1.6583991689712137e-05, dropout=0.0, threshold=0.15000000000000002

## Transformer (objective: martin, 400 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |     367 | True                |   2.5278 |   0.9823 |    1.0888 |   0.6987 |  2.3539 |        -0.0852 | 0.0595 |
| sharpe          |     367 | True                |   2.5278 |   0.9823 |    1.0888 |   0.6987 |  2.3539 |        -0.0852 | 0.0595 |
| sortino         |     367 | True                |   2.5278 |   0.9823 |    1.0888 |   0.6987 |  2.3539 |        -0.0852 | 0.0595 |
| calmar          |     367 | True                |   2.5278 |   0.9823 |    1.0888 |   0.6987 |  2.3539 |        -0.0852 | 0.0595 |
| ulcer           |     367 | True                |   2.5278 |   0.9823 |    1.0888 |   0.6987 |  2.3539 |        -0.0852 | 0.0595 |
| max_drawdown    |     367 | True                |   2.5278 |   0.9823 |    1.0888 |   0.6987 |  2.3539 |        -0.0852 | 0.0595 |

Selected configs:
- best under **martin**: window_size=220, dmodel_nheads=128-8, n_layers=4, dim_feedforward=32, batch_size=128, learning_rate=1.1146705922954242e-05, dropout=0.05, threshold=0.1
