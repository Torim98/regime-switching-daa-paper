| Model       | Median MaxDD (Model)   | Median MaxDD (B&H)   | Δ Median   |   Wilcoxon p | H1 (α=0.05)   |
|:------------|:-----------------------|:---------------------|:-----------|-------------:|:--------------|
| MSM         | -65.12%                | -59.34%              | -5.78 pp   |     1        | rejected      |
| HMM         | -74.21%                | -59.34%              | -14.87 pp  |     1        | rejected      |
| HMM_Uni     | -65.46%                | -59.34%              | -6.13 pp   |     1        | rejected      |
| LSTM        | -55.04%                | -59.34%              | +4.29 pp   |     5.62e-28 | confirmed     |
| Transformer | -52.93%                | -59.34%              | +6.41 pp   |     1.04e-62 | confirmed     |