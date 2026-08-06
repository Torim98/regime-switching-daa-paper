# Optuna: Best Hyperparameters

_Generated at 2026-08-03 14:31:37_  
Optimization metric: **martin (pooled OOS)**

## Overview

| Model | Best Score | ✓ Complete | ✗ Pruned | Total |
|:---|---:|---:|---:|---:|
| **MSM** | 1.5402 | 36 | 0 | 36 |
| **HMM** | 0.8231 | 108 | 0 | 108 |
| **HMM_Uni** | 1.5296 | 36 | 0 | 36 |
| **LSTM** | 2.3342 | 300 | 0 | 302 |
| **Transformer** | 1.0331 | 400 | 0 | 401 |

### MSM: Best Score `1.5402`

| Parameter | Value |
|:---|---:|
| `threshold` | `0.175` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.8473 |
| sortino | 1.0053 |
| calmar | 0.6179 |
| martin | 1.5402 |
| ulcer | 3.3506 |
| max_drawdown | -0.0835 |
| cagr | 0.0516 |

### HMM: Best Score `0.8231`

| Parameter | Value |
|:---|---:|
| `covariance_type` | `diag` |
| `threshold` | `0.975` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.5510 |
| sortino | 0.5753 |
| calmar | 0.3050 |
| martin | 0.8231 |
| ulcer | 3.3470 |
| max_drawdown | -0.0903 |
| cagr | 0.0275 |

### HMM_Uni: Best Score `1.5296`

| Parameter | Value |
|:---|---:|
| `threshold` | `0.175` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.8391 |
| sortino | 0.9954 |
| calmar | 0.6334 |
| martin | 1.5296 |
| ulcer | 3.3440 |
| max_drawdown | -0.0808 |
| cagr | 0.0511 |

### LSTM: Best Score `2.3342`

| Parameter | Value |
|:---|---:|
| `window_size` | `140` |
| `units_l1` | `16` |
| `units_l2` | `32` |
| `batch_size` | `128` |
| `learning_rate` | `2.008e-05` |
| `dropout` | `0.5` |
| `threshold` | `0.35` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.9020 |
| sortino | 0.7655 |
| calmar | 0.5638 |
| martin | 2.3342 |
| ulcer | 1.7773 |
| max_drawdown | -0.0736 |
| cagr | 0.0415 |

### Transformer: Best Score `1.0331`

| Parameter | Value |
|:---|---:|
| `window_size` | `230` |
| `dmodel_nheads` | `16-2` |
| `n_layers` | `2` |
| `dim_feedforward` | `256` |
| `batch_size` | `64` |
| `learning_rate` | `0.003203` |
| `dropout` | `0.25` |
| `threshold` | `0.5` |

Secondary metrics of the best trial (pooled OOS):

| Metric | Value |
|:---|---:|
| sharpe | 0.6100 |
| sortino | 0.7195 |
| calmar | 0.2133 |
| martin | 1.0331 |
| ulcer | 5.2343 |
| max_drawdown | -0.2535 |
| cagr | 0.0541 |
