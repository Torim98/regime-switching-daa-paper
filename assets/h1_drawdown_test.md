|                                | Headline   | Median MaxDD (Model)   | Median MaxDD (B&H)   | Δ Median   |   Wilcoxon p | H1 (α=0.05)   |
|:-------------------------------|:-----------|:-----------------------|:---------------------|:-----------|-------------:|:--------------|
| ('Standard', 'MSM')            |            | -24.89%                | -37.53%              | +12.63 pp  |     0        | confirmed     |
| ('Standard', 'HMM')            |            | -58.44%                | -37.53%              | -20.91 pp  |     1        | rejected      |
| ('Standard', 'HMM_Uni')        |            | -25.30%                | -37.53%              | +12.23 pp  |     0        | confirmed     |
| ('Standard', 'LSTM')           |            | -28.24%                | -37.53%              | +9.29 pp   |     0        | confirmed     |
| ('Standard', 'Transformer')    |            | -33.29%                | -37.53%              | +4.23 pp   |     2.22e-81 | confirmed     |
| ('Aggressive', 'MSM')          | yes        | -98.27%                | -62.00%              | -36.27 pp  |     1        | rejected      |
| ('Aggressive', 'HMM')          | yes        | -100.00%               | -62.00%              | -38.00 pp  |     1        | rejected      |
| ('Aggressive', 'HMM_Uni')      | yes        | -99.70%                | -62.00%              | -37.70 pp  |     1        | rejected      |
| ('Aggressive', 'LSTM')         | yes        | -38.94%                | -62.00%              | +23.07 pp  |     0        | confirmed     |
| ('Aggressive', 'Transformer')  | yes        | -100.00%               | -62.00%              | -38.00 pp  |     1        | rejected      |
| ('Low_Capital', 'MSM')         |            | -45.92%                | -43.41%              | -2.51 pp   |     8.82e-07 | confirmed     |
| ('Low_Capital', 'HMM')         |            | -100.00%               | -43.41%              | -56.59 pp  |     1        | rejected      |
| ('Low_Capital', 'HMM_Uni')     |            | -47.25%                | -43.41%              | -3.84 pp   |     0.0228   | confirmed     |
| ('Low_Capital', 'LSTM')        |            | -31.67%                | -43.41%              | +11.75 pp  |     0        | confirmed     |
| ('Low_Capital', 'Transformer') |            | -56.58%                | -43.41%              | -13.17 pp  |     1        | rejected      |