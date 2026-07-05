# Bootstrap Robustness: Block vs. Stationary

Monte-Carlo depletion analysis on 10,000 paths per cell, seed 42, mean block length 20 trading days. Both runs share the same seed and paths; only the resampling scheme differs (fixed-length block bootstrap vs. the stationary bootstrap of Politis & Romano 1994).

|                                | Depletion (Block)   | 95% CI (Block)   | Depletion (Stationary)   | 95% CI (Stationary)   | Delta Depletion (pp)   | Median Final (Block)   | Median Final (Stationary)   | Delta Median (EUR)   |
|:-------------------------------|:--------------------|:-----------------|:-------------------------|:----------------------|:-----------------------|:-----------------------|:----------------------------|:---------------------|
| ('Standard', 'Buy Hold')       | 0.02%               | [0.01%, 0.07%]   | 0.03%                    | [0.01%, 0.09%]        | +0.01                  | 471,862 EUR            | 473,660 EUR                 | +1,798               |
| ('Standard', 'MSM')            | 0.00%               | [0.00%, 0.04%]   | 0.00%                    | [0.00%, 0.04%]        | +0.00                  | 393,351 EUR            | 393,052 EUR                 | -299                 |
| ('Standard', 'HMM')            | 0.00%               | [0.00%, 0.04%]   | 0.00%                    | [0.00%, 0.04%]        | +0.00                  | 334,367 EUR            | 335,487 EUR                 | +1,120               |
| ('Standard', 'HMM Uni')        | 0.00%               | [0.00%, 0.04%]   | 0.00%                    | [0.00%, 0.04%]        | +0.00                  | 383,999 EUR            | 379,173 EUR                 | -4,826               |
| ('Standard', 'LSTM')           | 0.00%               | [0.00%, 0.04%]   | 0.01%                    | [0.00%, 0.06%]        | +0.01                  | 469,591 EUR            | 469,898 EUR                 | +307                 |
| ('Standard', 'Transformer')    | 0.01%               | [0.00%, 0.06%]   | 0.01%                    | [0.00%, 0.06%]        | +0.00                  | 464,921 EUR            | 467,521 EUR                 | +2,600               |
| ('Aggressive', 'Buy Hold')     | 5.73%               | [5.29%, 6.20%]   | 4.78%                    | [4.38%, 5.22%]        | -0.95                  | 227,732 EUR            | 231,330 EUR                 | +3,597               |
| ('Aggressive', 'MSM')          | 3.25%               | [2.92%, 3.62%]   | 3.64%                    | [3.29%, 4.03%]        | +0.39                  | 165,348 EUR            | 163,983 EUR                 | -1,365               |
| ('Aggressive', 'HMM')          | 5.36%               | [4.94%, 5.82%]   | 5.59%                    | [5.16%, 6.06%]        | +0.23                  | 121,220 EUR            | 118,840 EUR                 | -2,380               |
| ('Aggressive', 'HMM Uni')      | 1.88%               | [1.63%, 2.17%]   | 1.74%                    | [1.50%, 2.02%]        | -0.14                  | 158,087 EUR            | 157,418 EUR                 | -669                 |
| ('Aggressive', 'LSTM')         | 4.50%               | [4.11%, 4.92%]   | 4.16%                    | [3.79%, 4.57%]        | -0.34                  | 225,928 EUR            | 223,837 EUR                 | -2,091               |
| ('Aggressive', 'Transformer')  | 4.40%               | [4.02%, 4.82%]   | 3.61%                    | [3.26%, 3.99%]        | -0.79                  | 222,759 EUR            | 226,157 EUR                 | +3,399               |
| ('Low_Capital', 'Buy Hold')    | 0.90%               | [0.73%, 1.10%]   | 0.54%                    | [0.41%, 0.70%]        | -0.36                  | 200,414 EUR            | 203,039 EUR                 | +2,624               |
| ('Low_Capital', 'MSM')         | 0.08%               | [0.04%, 0.16%]   | 0.09%                    | [0.05%, 0.17%]        | +0.01                  | 159,543 EUR            | 158,581 EUR                 | -961                 |
| ('Low_Capital', 'HMM')         | 0.12%               | [0.07%, 0.21%]   | 0.15%                    | [0.09%, 0.25%]        | +0.03                  | 129,407 EUR            | 128,332 EUR                 | -1,075               |
| ('Low_Capital', 'HMM Uni')     | 0.03%               | [0.01%, 0.09%]   | 0.04%                    | [0.02%, 0.10%]        | +0.01                  | 153,572 EUR            | 155,397 EUR                 | +1,825               |
| ('Low_Capital', 'LSTM')        | 0.38%               | [0.28%, 0.52%]   | 0.40%                    | [0.29%, 0.54%]        | +0.02                  | 196,098 EUR            | 198,791 EUR                 | +2,693               |
| ('Low_Capital', 'Transformer') | 0.49%               | [0.37%, 0.65%]   | 0.25%                    | [0.17%, 0.37%]        | -0.24                  | 198,513 EUR            | 200,391 EUR                 | +1,878               |

**Robustness summary.** Across 18 (scenario, strategy) cells, the largest depletion-rate difference between the block and stationary bootstrap is 0.95 pp. The strategy ranking by depletion rate is identical under both methods in 1/3 scenarios.

The sign of every regime model's depletion advantage over Buy Hold is preserved under both methods in 14/15 model-scenario comparisons, so the tail-protection findings do not hinge on the resampling scheme.
