# Optuna — Beste Hyperparameter

_Generiert am 2026-04-21 08:53:02_  
Optimierungs-Metrik: **Sharpe (Median OOS)**

## Übersicht

| Modell | Best Score | ✓ Complete | ✗ Pruned | Total |
|:---|---:|---:|---:|---:|
| **MSM** | 0.9308 | 20 | 30 | 50 |
| **HMM** | 1.2843 | 50 | 0 | 50 |
| **LSTM** | 0.9707 | 27 | 3 | 30 |
| **Transformer** | 0.9343 | 20 | 10 | 30 |

### MSM — Best Score `0.9308`

| Parameter | Wert |
|:---|---:|
| `threshold` | `0.35` |

### HMM — Best Score `1.2843`

| Parameter | Wert |
|:---|---:|
| `covariance_type` | `tied` |
| `threshold` | `0.45` |

### LSTM — Best Score `0.9707`

| Parameter | Wert |
|:---|---:|
| `window_size` | `110` |
| `units_l1` | `32` |
| `units_l2` | `64` |
| `learning_rate` | `9.209e-04` |
| `dropout` | `0.2` |
| `epochs` | `45` |
| `threshold` | `0.5` |

### Transformer — Best Score `0.9343`

| Parameter | Wert |
|:---|---:|
| `d_model` | `64` |
| `n_heads` | `8` |
| `n_layers` | `3` |
| `dim_feedforward` | `128` |
| `learning_rate` | `3.176e-04` |
| `dropout` | `0.25` |
| `epochs` | `60` |
| `window_size` | `20` |
| `threshold` | `0.45` |
