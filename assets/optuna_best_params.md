# Optuna — Beste Hyperparameter

_Generiert am 2026-04-19 19:42:45_

Optimierungs-Metrik: **Sharpe (Median OOS)**

| Modell | Best Score | Trials |
|:---|---:|---:|
| MSM | 1.3918 | 82 |
| HMM | 2.0117 | 58 |
| LSTM | 1.3393 | 52 |
| Transformer | 1.3537 | 53 |

## MSM

- Best Score: **1.3918**
- Trials: 82

```yaml
k_regimes: 2
threshold: 0.35
```

## HMM

- Best Score: **2.0117**
- Trials: 58

```yaml
covariance_type: full
n_components: 2
threshold: 0.5
```

## LSTM

- Best Score: **1.3393**
- Trials: 52

```yaml
dropout: 0.1
epochs: 40
learning_rate: 0.00012536386810872734
threshold: 0.3
units_l1: 16
units_l2: 128
window_size: 120
```

## Transformer

- Best Score: **1.3537**
- Trials: 53

```yaml
d_model: 128
dim_feedforward: 64
dropout: 0.25
epochs: 20
learning_rate: 3.1476320331974455e-05
n_heads: 8
n_layers: 3
threshold: 0.45
window_size: 120
```
