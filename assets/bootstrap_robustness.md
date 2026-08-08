# Bootstrap Robustness: Block vs. Stationary

Monte-Carlo depletion analysis on 10,000 paths per cell, seed 42, mean block length 20 trading days. Both runs share the same seed and paths; only the resampling scheme differs (fixed-length block bootstrap vs. the stationary bootstrap of Politis & Romano 1994).

|                                | Depletion (Block)   | 95% CI (Block)   | Depletion (Stationary)   | 95% CI (Stationary)   | Delta Depletion (pp)   | Median Final (Block)   | Median Final (Stationary)   | Delta Median (EUR)   |
|:-------------------------------|:--------------------|:-----------------|:-------------------------|:----------------------|:-----------------------|:-----------------------|:----------------------------|:---------------------|
| ('Standard', 'LSTM')           | 0.72%               | [0.57%, 0.91%]   | 0.60%                    | [0.47%, 0.77%]        | -0.12                  | 1,577,740 EUR          | 1,533,695 EUR               | -44,045              |
| ('Aggressive', 'Buy Hold')     | 32.86%              | [31.95%, 33.79%] | 33.04%                   | [32.12%, 33.97%]      | +0.18                  | 290,538 EUR            | 270,638 EUR                 | -19,899              |
| ('Aggressive', 'Transformer')  | 52.37%              | [51.39%, 53.35%] | 54.89%                   | [53.91%, 55.86%]      | +2.52                  | 0 EUR                  | 0 EUR                       | +0                   |
| ('Standard', 'HMM Uni')        | 0.43%               | [0.32%, 0.58%]   | 0.60%                    | [0.47%, 0.77%]        | +0.17                  | 670,234 EUR            | 634,032 EUR                 | -36,202              |
| ('Standard', 'Transformer')    | 2.57%               | [2.28%, 2.90%]   | 2.97%                    | [2.65%, 3.32%]        | +0.40                  | 598,135 EUR            | 566,457 EUR                 | -31,678              |
| ('Low_Capital', 'HMM')         | 50.40%              | [49.42%, 51.38%] | 51.99%                   | [51.01%, 52.97%]      | +1.59                  | 0 EUR                  | 0 EUR                       | +0                   |
| ('Low_Capital', 'Buy Hold')    | 14.76%              | [14.08%, 15.47%] | 14.07%                   | [13.40%, 14.77%]      | -0.69                  | 407,997 EUR            | 390,858 EUR                 | -17,138              |
| ('Standard', 'MSM')            | 0.42%               | [0.31%, 0.57%]   | 0.57%                    | [0.44%, 0.74%]        | +0.15                  | 681,333 EUR            | 643,115 EUR                 | -38,218              |
| ('Standard', 'HMM')            | 7.93%               | [7.42%, 8.48%]   | 8.88%                    | [8.34%, 9.45%]        | +0.95                  | 237,723 EUR            | 230,545 EUR                 | -7,178               |
| ('Aggressive', 'HMM Uni')      | 46.19%              | [45.21%, 47.17%] | 49.73%                   | [48.75%, 50.71%]      | +3.54                  | 27,113 EUR             | 1,657 EUR                   | -25,456              |
| ('Standard', 'Buy Hold')       | 3.93%               | [3.57%, 4.33%]   | 3.30%                    | [2.97%, 3.67%]        | -0.63                  | 1,069,451 EUR          | 1,040,838 EUR               | -28,613              |
| ('Aggressive', 'LSTM')         | 17.80%              | [17.06%, 18.56%] | 17.01%                   | [16.29%, 17.76%]      | -0.79                  | 660,320 EUR            | 615,996 EUR                 | -44,324              |
| ('Low_Capital', 'MSM')         | 9.53%               | [8.97%, 10.12%]  | 11.82%                   | [11.20%, 12.47%]      | +2.29                  | 213,874 EUR            | 193,529 EUR                 | -20,345              |
| ('Aggressive', 'HMM')          | 89.05%              | [88.42%, 89.65%] | 89.54%                   | [88.92%, 90.12%]      | +0.49                  | 0 EUR                  | 0 EUR                       | +0                   |
| ('Aggressive', 'MSM')          | 45.09%              | [44.12%, 46.07%] | 48.97%                   | [47.99%, 49.95%]      | +3.88                  | 34,547 EUR             | 9,310 EUR                   | -25,237              |
| ('Low_Capital', 'HMM Uni')     | 10.09%              | [9.51%, 10.70%]  | 12.46%                   | [11.83%, 13.12%]      | +2.37                  | 208,751 EUR            | 188,836 EUR                 | -19,915              |
| ('Low_Capital', 'LSTM')        | 5.21%               | [4.79%, 5.66%]   | 4.43%                    | [4.04%, 4.85%]        | -0.78                  | 670,680 EUR            | 643,501 EUR                 | -27,179              |
| ('Low_Capital', 'Transformer') | 18.89%              | [18.13%, 19.67%] | 20.18%                   | [19.40%, 20.98%]      | +1.29                  | 172,000 EUR            | 157,519 EUR                 | -14,482              |

**Robustness summary.** Across 18 (scenario, strategy) cells, the largest depletion-rate difference between the block and stationary bootstrap is 3.88 pp. The strategy ranking by depletion rate is identical under both methods in 2/3 scenarios.

The sign of every regime model's depletion advantage over Buy Hold is preserved under both methods in 15/15 model-scenario comparisons, so the tail-protection findings do not hinge on the resampling scheme.
