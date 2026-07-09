# Objective Sensitivity of the Selected Hyperparameters

_Generated at 2026-07-09 10:00:34_  
Best config under each candidate metric, valued across all metrics (from the search trials' logged OOS metrics; no retraining). `same_as_objective` marks configs identical to the actual objective's pick.

## MSM (objective: martin, 36 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |
| sharpe          |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |
| sortino         |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |
| calmar          |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |
| ulcer           |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |
| max_drawdown    |      19 | True                |   1.6508 |   0.8513 |    1.0119 |   0.6465 |  3.1627 |        -0.0808 | 0.0522 |

Selected configs:
- best under **martin**: threshold=0.175

## HMM (objective: martin, 108 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |       4 | True                |   1.2037 |   0.7243 |    0.8796 |   0.4153 |  3.9259 |        -0.1138 | 0.0473 |
| sharpe          |       4 | True                |   1.2037 |   0.7243 |    0.8796 |   0.4153 |  3.9259 |        -0.1138 | 0.0473 |
| sortino         |       4 | True                |   1.2037 |   0.7243 |    0.8796 |   0.4153 |  3.9259 |        -0.1138 | 0.0473 |
| calmar          |       4 | True                |   1.2037 |   0.7243 |    0.8796 |   0.4153 |  3.9259 |        -0.1138 | 0.0473 |
| ulcer           |      66 | False               |   0.921  |   0.5636 |    0.6061 |   0.3386 |  3.4975 |        -0.0951 | 0.0322 |
| max_drawdown    |      22 | False               |   0.9395 |   0.5714 |    0.6151 |   0.3455 |  3.4985 |        -0.0951 | 0.0329 |

Selected configs:
- best under **martin**: covariance_type=tied, threshold=0.875
- best under **ulcer**: covariance_type=tied, threshold=0.125
- best under **max_drawdown**: covariance_type=tied, threshold=0.15

## HMM_Uni (objective: martin, 36 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      19 | True                |   1.6384 |   0.842  |    0.9987 |   0.6752 |  3.1419 |        -0.0762 | 0.0515 |
| sharpe          |      19 | True                |   1.6384 |   0.842  |    0.9987 |   0.6752 |  3.1419 |        -0.0762 | 0.0515 |
| sortino         |      19 | True                |   1.6384 |   0.842  |    0.9987 |   0.6752 |  3.1419 |        -0.0762 | 0.0515 |
| calmar          |      19 | True                |   1.6384 |   0.842  |    0.9987 |   0.6752 |  3.1419 |        -0.0762 | 0.0515 |
| ulcer           |      17 | False               |   1.5245 |   0.7938 |    0.9291 |   0.578  |  3.1279 |        -0.0825 | 0.0477 |
| max_drawdown    |      19 | True                |   1.6384 |   0.842  |    0.9987 |   0.6752 |  3.1419 |        -0.0762 | 0.0515 |

Selected configs:
- best under **martin**: threshold=0.175
- best under **ulcer**: threshold=0.15

## LSTM (objective: martin, 194 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |     129 | True                |   2.6426 |   0.9785 |    0.9191 |   0.6183 |  2.0269 |        -0.0866 | 0.0536 |
| sharpe          |     129 | True                |   2.6426 |   0.9785 |    0.9191 |   0.6183 |  2.0269 |        -0.0866 | 0.0536 |
| sortino         |      99 | False               |   1.8536 |   0.8679 |    1.1122 |   0.3773 |  3.8388 |        -0.1886 | 0.0712 |
| calmar          |     148 | False               |   2.4283 |   0.8726 |    1.0941 |   0.6963 |  2.5232 |        -0.088  | 0.0613 |
| ulcer           |     160 | False               |   2.3163 |   0.8811 |    0.7852 |   0.5187 |  1.948  |        -0.087  | 0.0451 |
| max_drawdown    |     181 | False               |   1.7748 |   0.8285 |    0.6782 |   0.5021 |  2.0816 |        -0.0736 | 0.0369 |

Selected configs:
- best under **martin**: window_size=250, units_l1=16, units_l2=32, batch_size=32, learning_rate=1.3292520737887431e-05, dropout=0.45, threshold=0.15000000000000002
- best under **sortino**: window_size=250, units_l1=32, units_l2=256, batch_size=128, learning_rate=4.471806844913538e-05, dropout=0.45, threshold=0.30000000000000004
- best under **calmar**: window_size=250, units_l1=16, units_l2=256, batch_size=128, learning_rate=4.076527319051499e-05, dropout=0.4, threshold=0.15000000000000002
- best under **ulcer**: window_size=210, units_l1=16, units_l2=64, batch_size=128, learning_rate=2.0149507145662965e-05, dropout=0.55, threshold=0.1
- best under **max_drawdown**: window_size=230, units_l1=16, units_l2=32, batch_size=64, learning_rate=1.0355058430277764e-05, dropout=0.4, threshold=0.2

## Transformer (objective: martin, 190 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      42 | True                |   1.3728 |   0.7082 |    0.8823 |   0.319  |  4.6002 |        -0.1979 | 0.0632 |
| sharpe          |      14 | False               |   1.3506 |   0.7152 |    0.9173 |   0.2847 |  4.0869 |        -0.1939 | 0.0552 |
| sortino         |      84 | False               |   0.8467 |   0.6852 |    0.9236 |   0.2245 |  6.8603 |        -0.2587 | 0.0581 |
| calmar          |      42 | True                |   1.3728 |   0.7082 |    0.8823 |   0.319  |  4.6002 |        -0.1979 | 0.0632 |
| ulcer           |      14 | False               |   1.3506 |   0.7152 |    0.9173 |   0.2847 |  4.0869 |        -0.1939 | 0.0552 |
| max_drawdown    |      51 | False               |   0.2719 |   0.4477 |    0.4366 |   0.1581 |  8.8183 |        -0.1517 | 0.024  |

Selected configs:
- best under **martin**: window_size=80, dmodel_nheads=128-4, n_layers=3, dim_feedforward=128, batch_size=128, learning_rate=0.002661901888489057, dropout=0.15000000000000002, threshold=0.15000000000000002
- best under **sharpe**: window_size=250, dmodel_nheads=16-2, n_layers=3, dim_feedforward=32, batch_size=64, learning_rate=0.0003589128083678785, dropout=0.45, threshold=0.7000000000000001
- best under **sortino**: window_size=220, dmodel_nheads=32-2, n_layers=4, dim_feedforward=32, batch_size=128, learning_rate=0.0028818323079306337, dropout=0.30000000000000004, threshold=0.6
- best under **max_drawdown**: window_size=40, dmodel_nheads=32-2, n_layers=4, dim_feedforward=32, batch_size=32, learning_rate=2.015647705936502e-05, dropout=0.35000000000000003, threshold=0.1
