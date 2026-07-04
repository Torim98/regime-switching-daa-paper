# Objective Sensitivity of the Selected Hyperparameters

_Generated at 2026-07-04 13:59:29_  
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

## LSTM (objective: martin, 57 trials)

| optimized_for   |   trial | same_as_objective   |   martin |   sharpe |   sortino |   calmar |   ulcer |   max_drawdown |   cagr |
|:----------------|--------:|:--------------------|---------:|---------:|----------:|---------:|--------:|---------------:|-------:|
| martin          |      55 | True                |   1.885  |   0.7214 |    0.7761 |   0.2932 |  3.0792 |        -0.1979 | 0.058  |
| sharpe          |      54 | False               |   1.2498 |   0.7813 |    0.9017 |   0.2878 |  4.2568 |        -0.1849 | 0.0532 |
| sortino         |      39 | False               |   1.5072 |   0.7463 |    0.9151 |   0.3407 |  4.4745 |        -0.1979 | 0.0674 |
| calmar          |      39 | False               |   1.5072 |   0.7463 |    0.9151 |   0.3407 |  4.4745 |        -0.1979 | 0.0674 |
| ulcer           |      55 | True                |   1.885  |   0.7214 |    0.7761 |   0.2932 |  3.0792 |        -0.1979 | 0.058  |
| max_drawdown    |      54 | False               |   1.2498 |   0.7813 |    0.9017 |   0.2878 |  4.2568 |        -0.1849 | 0.0532 |

Selected configs:
- best under **martin**: window_size=250, units_l1=64, units_l2=32, batch_size=32, learning_rate=1.630044194460958e-05, dropout=0.55, threshold=0.2
- best under **sharpe**: window_size=230, units_l1=64, units_l2=32, batch_size=32, learning_rate=1.6386746906216097e-05, dropout=0.55, threshold=0.15000000000000002
- best under **sortino**: window_size=110, units_l1=64, units_l2=32, batch_size=64, learning_rate=0.00020531693199466844, dropout=0.55, threshold=0.35
