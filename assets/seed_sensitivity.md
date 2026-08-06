# Seed Sensitivity of the Production Config (5 seeds)

Each model is re-run on the production hyperparameters and the full walk-forward fold set, varying only its random source: the EM initialization (random_state) for HMM and HMM_Uni, the global RNG (weight init, batch shuffle, dropout) for LSTM and Transformer. MSM is deterministic and shown as the zero-variance control. Metrics are the pooled OOS values (CV = std / |mean|).

## Summary (headline metrics)

| Model | CAGR mean | CAGR std | CAGR CV | Martin mean | Martin std | Martin CV | Verdict |
|---|---|---|---|---|---|---|---|
| MSM | 0.0441 | 0.0000 | 0.000 | 1.1472 | 0.0000 | 0.000 | stable |
| HMM | 0.0336 | 0.0008 | 0.023 | 0.6977 | 0.0224 | 0.032 | moderate |
| HMM_Uni | 0.0433 | 0.0001 | 0.003 | 1.1186 | 0.0031 | 0.003 | stable |
| LSTM | 0.0495 | 0.0049 | 0.098 | 0.8023 | 0.1680 | 0.209 | unstable |
| Transformer | 0.0492 | 0.0045 | 0.092 | 0.5350 | 0.0970 | 0.181 | unstable |

## Per-model detail

### MSM (1 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0441 | 0.0000 | 0.0441 | 0.0441 | 0.000 |
| martin | 1.1472 | 0.0000 | 1.1472 | 1.1472 | 0.000 |
| sharpe | 0.7450 | 0.0000 | 0.7450 | 0.7450 | 0.000 |
| sortino | 0.8325 | 0.0000 | 0.8325 | 0.8325 | 0.000 |
| calmar | 0.4003 | 0.0000 | 0.4003 | 0.4003 | 0.000 |
| ulcer | 3.8419 | 0.0000 | 3.8419 | 3.8419 | 0.000 |
| max_drawdown | -0.1101 | 0.0000 | -0.1101 | -0.1101 | 0.000 |

### HMM (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0336 | 0.0008 | 0.0322 | 0.0342 | 0.023 |
| martin | 0.6977 | 0.0224 | 0.6583 | 0.7201 | 0.032 |
| sharpe | 0.6235 | 0.0143 | 0.5982 | 0.6352 | 0.023 |
| sortino | 0.6506 | 0.0142 | 0.6257 | 0.6613 | 0.022 |
| calmar | 0.2285 | 0.0033 | 0.2220 | 0.2306 | 0.015 |
| ulcer | 4.8171 | 0.0481 | 4.7445 | 4.8948 | 0.010 |
| max_drawdown | -0.1470 | 0.0015 | -0.1483 | -0.1452 | 0.010 |

### HMM_Uni (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0433 | 0.0001 | 0.0431 | 0.0433 | 0.003 |
| martin | 1.1186 | 0.0031 | 1.1125 | 1.1202 | 0.003 |
| sharpe | 0.7330 | 0.0018 | 0.7294 | 0.7338 | 0.002 |
| sortino | 0.8176 | 0.0021 | 0.8135 | 0.8187 | 0.003 |
| calmar | 0.3895 | 0.0010 | 0.3874 | 0.3901 | 0.003 |
| ulcer | 3.8698 | 0.0003 | 3.8697 | 3.8704 | 0.000 |
| max_drawdown | -0.1111 | 0.0000 | -0.1111 | -0.1111 | 0.000 |

### LSTM (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0495 | 0.0049 | 0.0451 | 0.0587 | 0.098 |
| martin | 0.8023 | 0.1680 | 0.6199 | 1.0422 | 0.209 |
| sharpe | 0.6172 | 0.0637 | 0.5110 | 0.6888 | 0.103 |
| sortino | 0.6506 | 0.0788 | 0.5746 | 0.7944 | 0.121 |
| calmar | 0.2063 | 0.0317 | 0.1725 | 0.2536 | 0.154 |
| ulcer | 6.3917 | 1.1623 | 4.4721 | 7.6066 | 0.182 |
| max_drawdown | -0.2431 | 0.0252 | -0.2732 | -0.2002 | 0.104 |

### Transformer (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0492 | 0.0045 | 0.0411 | 0.0548 | 0.092 |
| martin | 0.5350 | 0.0970 | 0.4179 | 0.6614 | 0.181 |
| sharpe | 0.5336 | 0.0434 | 0.4556 | 0.5837 | 0.081 |
| sortino | 0.6535 | 0.0655 | 0.5405 | 0.7419 | 0.100 |
| calmar | 0.1629 | 0.0191 | 0.1443 | 0.1923 | 0.117 |
| ulcer | 9.3554 | 1.0096 | 7.9945 | 10.3554 | 0.108 |
| max_drawdown | -0.3033 | 0.0228 | -0.3389 | -0.2852 | 0.075 |

Reading the verdict: 'stable' (headline CV < 0.02) means a single seed is representative; 'moderate' (< 0.10) warrants reporting a seed band; 'unstable' (>= 0.10) calls for a seed-averaged ensemble (DL) or best-of-k by log-likelihood (HMM) as the headline result.
