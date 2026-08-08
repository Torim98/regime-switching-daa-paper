# Data Quality Report

- **Status:** Coverage ≥ 96.2 % · max. gap 1 days
- **Period (raw):** 1990-01-02 to 2026-07-31
- **End date mode:** Freeze (fixed cutoff)
- **Resolved end date:** `2026-07-31`
- **yfinance version:** `1.5.1`
- **Tickers:** ^GSPC, VUSTX, ^VIX, ^IRX, ^TNX
- **Generated at:** 2026-08-06 08:30

## 1. Coverage (Observed vs. Expected Trading Days)
| Ticker   | From       | To         |   Obs. Days |   Exp. Bd (Mon-Fri) |   Coverage % |
|:---------|:-----------|:-----------|------------:|--------------------:|-------------:|
| ^GSPC    | 1990-01-02 | 2026-07-31 |        9212 |                9544 |        96.52 |
| VUSTX    | 1990-01-02 | 2026-07-31 |        9212 |                9544 |        96.52 |
| ^VIX     | 1990-01-02 | 2026-07-31 |        9213 |                9544 |        96.53 |
| ^IRX     | 1990-01-02 | 2026-07-31 |        9181 |                9544 |        96.2  |
| ^TNX     | 1990-01-02 | 2026-07-31 |        9181 |                9544 |        96.2  |

_Note: expected trading days from `bdate_range` (Mon-Fri incl. holidays). ~96-97% is the holiday-induced lower bound, not data loss._

## 2. Missing Values (Raw Frame, Before ffill/dropna)
| Ticker   |   NaN (raw) |   NaN % |   Longest Gap (Days) | First Value   | Last Value   |
|:---------|------------:|--------:|---------------------:|:--------------|:-------------|
| ^GSPC    |           1 |   0.011 |                    1 | 1990-01-02    | 2026-07-31   |
| VUSTX    |           1 |   0.011 |                    1 | 1990-01-02    | 2026-07-31   |
| ^VIX     |           0 |   0     |                    0 | 1990-01-02    | 2026-07-31   |
| ^IRX     |          32 |   0.347 |                    1 | 1990-01-02    | 2026-07-31   |
| ^TNX     |          32 |   0.347 |                    1 | 1990-01-02    | 2026-07-31   |

## 3. Adjustment Plausibility (Daily Jumps of the Price Series)
| Ticker   |   Max. Abs. Daily Return |   Outlier Days (z>8) | Largest Jump (Date)   |
|:---------|-------------------------:|---------------------:|:----------------------|
| ^GSPC    |                   0.1277 |                   27 | 2020-03-16            |
| VUSTX    |                   0.0654 |                    6 | 1992-12-31            |

## 4. Largest Daily Moves (Crisis Plausibility)
| Ticker   |   Rank | Date       |   Log Return |
|:---------|-------:|:-----------|-------------:|
| ^GSPC    |      1 | 2020-03-16 |      -0.1277 |
| ^GSPC    |      2 | 2008-10-13 |       0.1096 |
| ^GSPC    |      3 | 2008-10-28 |       0.1025 |
| ^GSPC    |      4 | 2020-03-12 |      -0.0999 |
| ^GSPC    |      5 | 2008-10-15 |      -0.0947 |
| VUSTX    |      1 | 1992-12-31 |       0.0654 |
| VUSTX    |      2 | 2020-03-20 |       0.0632 |
| VUSTX    |      3 | 2020-03-17 |      -0.0605 |
| VUSTX    |      4 | 1992-12-11 |      -0.0586 |
| VUSTX    |      5 | 2020-03-10 |      -0.0499 |

## 5. Effect of Cleaning (Bronze → Silver)
| Metric                          |    Value |
|:--------------------------------|---------:|
| Rows raw (Bronze)               | 9213     |
| Rows cleaned (Silver)           | 9211     |
| Removed (dropna + return shift) |    2     |
| Removed %                       |    0.022 |
