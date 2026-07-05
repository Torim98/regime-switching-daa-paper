# Seed Sensitivity of the Production Config (5 seeds)

Each model is re-run on the production hyperparameters and the full walk-forward fold set, varying only its random source: the EM initialization (random_state) for HMM and HMM_Uni, the global RNG (weight init, batch shuffle, dropout) for LSTM and Transformer. MSM is deterministic and shown as the zero-variance control. Metrics are the pooled OOS values (CV = std / |mean|).

## Summary (headline metrics)

| Model | CAGR mean | CAGR std | CAGR CV | Martin mean | Martin std | Martin CV | Verdict |
|---|---|---|---|---|---|---|---|
| MSM | 0.0469 | 0.0000 | 0.000 | 1.2883 | 0.0000 | 0.000 | stable |
| HMM | 0.0451 | 0.0046 | 0.102 | 0.6351 | 0.1522 | 0.240 | unstable |
| HMM_Uni | 0.0454 | 0.0011 | 0.024 | 1.2329 | 0.0385 | 0.031 | moderate |
| LSTM | 0.0442 | 0.0050 | 0.113 | 0.4898 | 0.0567 | 0.116 | unstable |
| Transformer | 0.0519 | 0.0030 | 0.058 | 0.6114 | 0.0708 | 0.116 | moderate |

## Per-model detail

### MSM (1 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0469 | 0.0000 | 0.0469 | 0.0469 | 0.000 |
| martin | 1.2883 | 0.0000 | 1.2883 | 1.2883 | 0.000 |
| sharpe | 0.7858 | 0.0000 | 0.7858 | 0.7858 | 0.000 |
| sortino | 0.8820 | 0.0000 | 0.8820 | 0.8820 | 0.000 |
| calmar | 0.4256 | 0.0000 | 0.4256 | 0.4256 | 0.000 |
| ulcer | 3.6375 | 0.0000 | 3.6375 | 3.6375 | 0.000 |
| max_drawdown | -0.1101 | 0.0000 | -0.1101 | -0.1101 | 0.000 |

### HMM (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0451 | 0.0046 | 0.0363 | 0.0497 | 0.102 |
| martin | 0.6351 | 0.1522 | 0.4378 | 0.7927 | 0.240 |
| sharpe | 0.5838 | 0.0704 | 0.4680 | 0.6759 | 0.121 |
| sortino | 0.6910 | 0.0728 | 0.5999 | 0.8027 | 0.105 |
| calmar | 0.1665 | 0.0284 | 0.1266 | 0.1981 | 0.171 |
| ulcer | 7.4340 | 1.5515 | 6.2310 | 10.1023 | 0.209 |
| max_drawdown | -0.2769 | 0.0470 | -0.3706 | -0.2508 | 0.170 |

### HMM_Uni (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0454 | 0.0011 | 0.0442 | 0.0475 | 0.024 |
| martin | 1.2329 | 0.0385 | 1.1919 | 1.3061 | 0.031 |
| sharpe | 0.7642 | 0.0156 | 0.7466 | 0.7935 | 0.021 |
| sortino | 0.8561 | 0.0187 | 0.8356 | 0.8914 | 0.022 |
| calmar | 0.4095 | 0.0112 | 0.3981 | 0.4310 | 0.027 |
| ulcer | 3.6844 | 0.0266 | 3.6334 | 3.7117 | 0.007 |
| max_drawdown | -0.1109 | 0.0004 | -0.1111 | -0.1101 | 0.004 |

### LSTM (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0442 | 0.0050 | 0.0390 | 0.0514 | 0.113 |
| martin | 0.4898 | 0.0567 | 0.4290 | 0.5658 | 0.116 |
| sharpe | 0.6215 | 0.0515 | 0.5364 | 0.6757 | 0.083 |
| sortino | 0.5577 | 0.0645 | 0.4869 | 0.6579 | 0.116 |
| calmar | 0.1552 | 0.0175 | 0.1368 | 0.1805 | 0.113 |
| ulcer | 9.0388 | 0.2408 | 8.8071 | 9.4844 | 0.027 |
| max_drawdown | -0.2850 | 0.0000 | -0.2850 | -0.2850 | 0.000 |

### Transformer (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0519 | 0.0030 | 0.0472 | 0.0562 | 0.058 |
| martin | 0.6114 | 0.0708 | 0.5206 | 0.6997 | 0.116 |
| sharpe | 0.5651 | 0.0252 | 0.5303 | 0.6075 | 0.044 |
| sortino | 0.6898 | 0.0314 | 0.6427 | 0.7410 | 0.045 |
| calmar | 0.1814 | 0.0114 | 0.1629 | 0.1971 | 0.063 |
| ulcer | 8.5504 | 0.5775 | 8.0260 | 9.4065 | 0.068 |
| max_drawdown | -0.2861 | 0.0020 | -0.2901 | -0.2852 | 0.007 |

Reading the verdict: 'stable' (headline CV < 0.02) means a single seed is representative; 'moderate' (< 0.10) warrants reporting a seed band; 'unstable' (>= 0.10) calls for a seed-averaged ensemble (DL) or best-of-k by log-likelihood (HMM) as the headline result.
