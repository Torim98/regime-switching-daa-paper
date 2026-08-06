# Bootstrap Robustness: Block vs. Stationary

Monte-Carlo depletion analysis on 10,000 paths per cell, seed 42, mean block length 20 trading days. Both runs share the same seed and paths; only the resampling scheme differs (fixed-length block bootstrap vs. the stationary bootstrap of Politis & Romano 1994).

|                                | Depletion (Block)   | 95% CI (Block)   | Depletion (Stationary)   | 95% CI (Stationary)   | Delta Depletion (pp)   | Median Final (Block)   | Median Final (Stationary)   | Delta Median (EUR)   |
|:-------------------------------|:--------------------|:-----------------|:-------------------------|:----------------------|:-----------------------|:-----------------------|:----------------------------|:---------------------|
| ('Standard', 'Transformer')    | 2.09%               | [1.83%, 2.39%]   | 1.55%                    | [1.33%, 1.81%]        | -0.54                  | 1,068,219 EUR          | 1,067,052 EUR               | -1,168               |
| ('Standard', 'HMM Uni')        | 0.43%               | [0.32%, 0.58%]   | 0.60%                    | [0.47%, 0.77%]        | +0.17                  | 670,235 EUR            | 634,035 EUR                 | -36,200              |
| ('Standard', 'MSM')            | 0.42%               | [0.31%, 0.57%]   | 0.57%                    | [0.44%, 0.74%]        | +0.15                  | 681,336 EUR            | 643,117 EUR                 | -38,219              |
| ('Standard', 'Buy Hold')       | 3.93%               | [3.57%, 4.33%]   | 3.30%                    | [2.97%, 3.67%]        | -0.63                  | 1,069,437 EUR          | 1,040,835 EUR               | -28,603              |
| ('Low_Capital', 'Buy Hold')    | 14.76%              | [14.08%, 15.47%] | 14.06%                   | [13.39%, 14.76%]      | -0.70                  | 407,995 EUR            | 390,860 EUR                 | -17,135              |
| ('Standard', 'HMM')            | 7.93%               | [7.42%, 8.48%]   | 8.88%                    | [8.34%, 9.45%]        | +0.95                  | 237,727 EUR            | 230,547 EUR                 | -7,180               |
| ('Standard', 'LSTM')           | 1.10%               | [0.91%, 1.32%]   | 0.70%                    | [0.55%, 0.88%]        | -0.40                  | 886,515 EUR            | 856,293 EUR                 | -30,221              |
| ('Low_Capital', 'MSM')         | 9.53%               | [8.97%, 10.12%]  | 11.82%                   | [11.20%, 12.47%]      | +2.29                  | 213,873 EUR            | 193,527 EUR                 | -20,347              |
| ('Aggressive', 'Buy Hold')     | 32.86%              | [31.95%, 33.79%] | 33.04%                   | [32.12%, 33.97%]      | +0.18                  | 290,541 EUR            | 270,636 EUR                 | -19,905              |
| ('Low_Capital', 'HMM')         | 50.39%              | [49.41%, 51.37%] | 51.99%                   | [51.01%, 52.97%]      | +1.60                  | 0 EUR                  | 0 EUR                       | +0                   |
| ('Aggressive', 'HMM Uni')      | 46.19%              | [45.21%, 47.17%] | 49.73%                   | [48.75%, 50.71%]      | +3.54                  | 27,113 EUR             | 1,657 EUR                   | -25,456              |
| ('Aggressive', 'Transformer')  | 30.43%              | [29.54%, 31.34%] | 29.75%                   | [28.86%, 30.65%]      | -0.68                  | 290,860 EUR            | 290,526 EUR                 | -334                 |
| ('Aggressive', 'MSM')          | 45.09%              | [44.12%, 46.07%] | 48.97%                   | [47.99%, 49.95%]      | +3.88                  | 34,545 EUR             | 9,308 EUR                   | -25,236              |
| ('Aggressive', 'LSTM')         | 33.51%              | [32.59%, 34.44%] | 34.75%                   | [33.82%, 35.69%]      | +1.24                  | 170,842 EUR            | 149,329 EUR                 | -21,513              |
| ('Aggressive', 'HMM')          | 89.05%              | [88.42%, 89.65%] | 89.54%                   | [88.92%, 90.12%]      | +0.49                  | 0 EUR                  | 0 EUR                       | +0                   |
| ('Low_Capital', 'Transformer') | 10.44%              | [9.86%, 11.05%]  | 9.48%                    | [8.92%, 10.07%]       | -0.96                  | 404,091 EUR            | 403,422 EUR                 | -669                 |
| ('Low_Capital', 'HMM Uni')     | 10.09%              | [9.51%, 10.70%]  | 12.46%                   | [11.83%, 13.12%]      | +2.37                  | 208,749 EUR            | 188,833 EUR                 | -19,916              |
| ('Low_Capital', 'LSTM')        | 8.60%               | [8.07%, 9.17%]   | 8.36%                    | [7.83%, 8.92%]        | -0.24                  | 315,468 EUR            | 299,774 EUR                 | -15,694              |

**Robustness summary.** Across 18 (scenario, strategy) cells, the largest depletion-rate difference between the block and stationary bootstrap is 3.88 pp. The strategy ranking by depletion rate is identical under both methods in 2/3 scenarios.

The sign of every regime model's depletion advantage over Buy Hold is preserved under both methods in 15/15 model-scenario comparisons, so the tail-protection findings do not hinge on the resampling scheme.
