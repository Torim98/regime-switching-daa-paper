# Seed Sensitivity of the Production Config (5 runs)

Each model is re-run on the production hyperparameters and the full walk-forward fold set, varying only its random source: the EM initialization (random_state) for HMM and HMM_Uni, the seed base of the 5-member production ensemble for LSTM and Transformer (run k trains a complete ensemble from the disjoint seed block k*5..k*5+4, so the spread is that of the estimator behind the headline results, not of a single network the pipeline never uses). MSM is deterministic and shown as the zero-variance control. Metrics are the pooled OOS values (CV = std / |mean|).

## Summary (headline metrics)

| Model | CAGR mean | CAGR std | CAGR CV | Martin mean | Martin std | Martin CV | Verdict |
|---|---|---|---|---|---|---|---|
| MSM | 0.0460 | 0.0000 | 0.000 | 1.2204 | 0.0000 | 0.000 | stable |
| HMM | 0.0344 | 0.0005 | 0.014 | 0.7259 | 0.0123 | 0.017 | stable |
| HMM_Uni | 0.0452 | 0.0001 | 0.003 | 1.1905 | 0.0031 | 0.003 | stable |
| LSTM | 0.0637 | 0.0024 | 0.037 | 1.1746 | 0.1996 | 0.170 | unstable |
| Transformer | 0.0441 | 0.0020 | 0.045 | 0.6371 | 0.0806 | 0.127 | moderate |

## Per-model detail

### MSM (1 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0460 | 0.0000 | 0.0460 | 0.0460 | 0.000 |
| martin | 1.2204 | 0.0000 | 1.2204 | 1.2204 | 0.000 |
| sharpe | 0.7757 | 0.0000 | 0.7757 | 0.7757 | 0.000 |
| sortino | 0.8721 | 0.0000 | 0.8721 | 0.8721 | 0.000 |
| calmar | 0.4215 | 0.0000 | 0.4215 | 0.4215 | 0.000 |
| ulcer | 3.7678 | 0.0000 | 3.7678 | 3.7678 | 0.000 |
| max_drawdown | -0.1091 | 0.0000 | -0.1091 | -0.1091 | 0.000 |

### HMM (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0344 | 0.0005 | 0.0338 | 0.0348 | 0.014 |
| martin | 0.7259 | 0.0123 | 0.7102 | 0.7422 | 0.017 |
| sharpe | 0.6362 | 0.0082 | 0.6259 | 0.6438 | 0.013 |
| sortino | 0.6686 | 0.0083 | 0.6577 | 0.6755 | 0.013 |
| calmar | 0.2392 | 0.0015 | 0.2371 | 0.2404 | 0.006 |
| ulcer | 4.7349 | 0.0273 | 4.6808 | 4.7549 | 0.006 |
| max_drawdown | -0.1437 | 0.0010 | -0.1445 | -0.1424 | 0.007 |

### HMM_Uni (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0452 | 0.0001 | 0.0450 | 0.0453 | 0.003 |
| martin | 1.1905 | 0.0031 | 1.1843 | 1.1921 | 0.003 |
| sharpe | 0.7637 | 0.0018 | 0.7601 | 0.7646 | 0.002 |
| sortino | 0.8571 | 0.0021 | 0.8529 | 0.8581 | 0.002 |
| calmar | 0.4104 | 0.0011 | 0.4083 | 0.4109 | 0.003 |
| ulcer | 3.7960 | 0.0002 | 3.7959 | 3.7965 | 0.000 |
| max_drawdown | -0.1101 | 0.0000 | -0.1101 | -0.1101 | 0.000 |

### LSTM (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0637 | 0.0024 | 0.0597 | 0.0660 | 0.037 |
| martin | 1.1746 | 0.1996 | 0.9255 | 1.5077 | 0.170 |
| sharpe | 0.7050 | 0.0225 | 0.6687 | 0.7360 | 0.032 |
| sortino | 0.8391 | 0.0298 | 0.7902 | 0.8743 | 0.035 |
| calmar | 0.2692 | 0.0316 | 0.2325 | 0.3174 | 0.117 |
| ulcer | 5.5499 | 0.7789 | 4.3754 | 6.4493 | 0.140 |
| max_drawdown | -0.2392 | 0.0247 | -0.2771 | -0.2078 | 0.103 |

### Transformer (5 run(s))

| Metric | mean | std | min | max | CV |
|---|---|---|---|---|---|
| cagr | 0.0441 | 0.0020 | 0.0415 | 0.0468 | 0.045 |
| martin | 0.6371 | 0.0806 | 0.5057 | 0.7484 | 0.127 |
| sharpe | 0.6305 | 0.0224 | 0.6057 | 0.6620 | 0.036 |
| sortino | 0.6471 | 0.0241 | 0.6208 | 0.6811 | 0.037 |
| calmar | 0.1609 | 0.0172 | 0.1314 | 0.1840 | 0.107 |
| ulcer | 7.0084 | 0.6822 | 6.1265 | 8.1964 | 0.097 |
| max_drawdown | -0.2765 | 0.0235 | -0.3156 | -0.2492 | 0.085 |

Reading the verdict: 'stable' (headline CV < 0.02) means a single seed is representative; 'moderate' (< 0.10) warrants reporting a seed band; 'unstable' (>= 0.10) means a single run must NOT be reported as a point estimate. Because the DL rows already measure the seed-averaged production ensemble, an 'unstable' verdict there cannot be fixed by averaging more of the same: report the band, raise walk_forward.dl_ensemble_size (variance shrinks ~1/sqrt(N)), or treat the model as not reliably estimable on this sample. For HMM, best-of-k by train log-likelihood (n_init) is the corresponding remedy.
