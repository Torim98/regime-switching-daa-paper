# Optuna — Beste Hyperparameter

_Generiert am 2026-04-19 19:52:21_  
Optimierungs-Metrik: **Sharpe (Median OOS)**

## Übersicht

| Modell | Best Score | ✓ Complete | ✂ Pruned | Total |
|:---|---:|---:|---:|---:|
| **MSM** | 1.3918 | 28 | 34 | 83 |
| **HMM** | 2.0117 | 35 | 15 | 59 |
| **LSTM** | 1.3393 | 30 | 20 | 53 |
| **Transformer** | 1.3537 | 22 | 28 | 54 |

### MSM — Best Score `1.3918`

| Parameter | Wert |
|:---|---:|
| `k_regimes` | `2` |
| `threshold` | `0.35` |

### HMM — Best Score `2.0117`

| Parameter | Wert |
|:---|---:|
| `n_components` | `2` |
| `covariance_type` | `full` |
| `threshold` | `0.5` |

### LSTM — Best Score `1.3393`

| Parameter | Wert |
|:---|---:|
| `window_size` | `120` |
| `units_l1` | `16` |
| `units_l2` | `128` |
| `learning_rate` | `1.254e-04` |
| `dropout` | `0.1` |
| `epochs` | `40` |
| `threshold` | `0.3` |

### Transformer — Best Score `1.3537`

| Parameter | Wert |
|:---|---:|
| `d_model` | `128` |
| `n_heads` | `8` |
| `n_layers` | `3` |
| `dim_feedforward` | `64` |
| `learning_rate` | `3.148e-05` |
| `dropout` | `0.25` |
| `epochs` | `20` |
| `window_size` | `120` |
| `threshold` | `0.45` |
