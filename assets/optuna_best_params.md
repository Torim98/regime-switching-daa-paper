# Optuna: Best Hyperparameters

_Generated at 2026-08-08 08:37:18_  
Optimization metric: **martin (pooled OOS)**

## Overview

| Model | Best Score | ✓ Complete | ✗ Pruned | Total |
|:---|---:|---:|---:|---:|
| **MSM** | 1.6366 | 36 | 0 | 36 |
| **HMM** | 0.8884 | 108 | 0 | 108 |
| **HMM_Uni** | 1.6261 | 36 | 0 | 36 |
| **LSTM** | 2.3114 | 300 | 0 | 301 |
| **Transformer** | 2.5278 | 400 | 0 | 400 |

### MSM: Best Score `1.6366`

| Parameter | Value |
|:---|---:|
| `threshold` | `0.175` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.8784 |
| sortino | 1.0486 |
| calmar | 0.6579 |
| martin | 1.6366 |
| ulcer | 3.2758 |
| max_drawdown | -0.0815 |
| cagr | 0.0536 |

### HMM: Best Score `0.8884`

| Parameter | Value |
|:---|---:|
| `covariance_type` | `diag` |
| `threshold` | `0.975` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.5772 |
| sortino | 0.6061 |
| calmar | 0.3276 |
| martin | 0.8884 |
| ulcer | 3.2562 |
| max_drawdown | -0.0883 |
| cagr | 0.0289 |

### HMM_Uni: Best Score `1.6261`

| Parameter | Value |
|:---|---:|
| `threshold` | `0.175` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.8702 |
| sortino | 1.0387 |
| calmar | 0.6703 |
| martin | 1.6261 |
| ulcer | 3.2691 |
| max_drawdown | -0.0793 |
| cagr | 0.0532 |

### LSTM: Best Score `2.3114`

| Parameter | Value |
|:---|---:|
| `window_size` | `100` |
| `units_l1` | `32` |
| `units_l2` | `256` |
| `batch_size` | `64` |
| `learning_rate` | `2.440e-05` |
| `dropout` | `0.25` |
| `threshold` | `0.3` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.8823 |
| sortino | 1.0285 |
| calmar | 0.3410 |
| martin | 2.3114 |
| ulcer | 2.8716 |
| max_drawdown | -0.1946 |
| cagr | 0.0664 |

### Transformer: Best Score `2.5278`

| Parameter | Value |
|:---|---:|
| `window_size` | `220` |
| `dmodel_nheads` | `128-8` |
| `n_layers` | `4` |
| `dim_feedforward` | `32` |
| `batch_size` | `128` |
| `learning_rate` | `1.115e-05` |
| `dropout` | `0.05` |
| `threshold` | `0.1` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.9823 |
| sortino | 1.0888 |
| calmar | 0.6987 |
| martin | 2.5278 |
| ulcer | 2.3539 |
| max_drawdown | -0.0852 |
| cagr | 0.0595 |
