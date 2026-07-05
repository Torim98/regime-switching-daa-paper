# Optuna: Best Hyperparameters

_Generated at 2026-07-05 10:38:33_  
Optimization metric: **martin (pooled OOS)**

## Overview

| Model | Best Score | ✓ Complete | ✗ Pruned | Total |
|:---|---:|---:|---:|---:|
| **MSM** | 1.6508 | 36 | 0 | 36 |
| **HMM** | 1.2037 | 108 | 0 | 108 |
| **HMM_Uni** | 1.6384 | 36 | 0 | 36 |
| **LSTM** | 2.6426 | 194 | 0 | 203 |

### MSM: Best Score `1.6508`

| Parameter | Value |
|:---|---:|
| `threshold` | `0.175` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.8513 |
| sortino | 1.0119 |
| calmar | 0.6465 |
| martin | 1.6508 |
| ulcer | 3.1627 |
| max_drawdown | -0.0808 |
| cagr | 0.0522 |

### HMM: Best Score `1.2037`

| Parameter | Value |
|:---|---:|
| `covariance_type` | `tied` |
| `threshold` | `0.875` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.7243 |
| sortino | 0.8796 |
| calmar | 0.4153 |
| martin | 1.2037 |
| ulcer | 3.9259 |
| max_drawdown | -0.1138 |
| cagr | 0.0473 |

### HMM_Uni: Best Score `1.6384`

| Parameter | Value |
|:---|---:|
| `threshold` | `0.175` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.8420 |
| sortino | 0.9987 |
| calmar | 0.6752 |
| martin | 1.6384 |
| ulcer | 3.1419 |
| max_drawdown | -0.0762 |
| cagr | 0.0515 |

### LSTM: Best Score `2.6426`

| Parameter | Value |
|:---|---:|
| `window_size` | `250` |
| `units_l1` | `16` |
| `units_l2` | `32` |
| `batch_size` | `32` |
| `learning_rate` | `1.329e-05` |
| `dropout` | `0.45` |
| `threshold` | `0.15` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.9785 |
| sortino | 0.9191 |
| calmar | 0.6183 |
| martin | 2.6426 |
| ulcer | 2.0269 |
| max_drawdown | -0.0866 |
| cagr | 0.0536 |
