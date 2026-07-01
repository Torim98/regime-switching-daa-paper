# Data Quality Report

- **Zeitraum (roh):** 1990-01-02 – 2026-07-01
- **End-Datum-Modus:** Rolling (dynamisch = letzter Handelstag)
- **Aufgelöstes Enddatum:** `2026-07-01`
- **yfinance-Version:** `1.5.1`
- **Ticker:** ^GSPC, VUSTX, ^VIX, ^IRX, ^TNX
- **Erzeugt am:** 2026-07-01 17:10

## 1. Coverage (beobachtete vs. erwartete Handelstage)
| Ticker   | Von        | Bis        |   Beob. Tage |   Erw. Bd (Mo–Fr) |   Coverage % |
|:---------|:-----------|:-----------|-------------:|------------------:|-------------:|
| ^GSPC    | 1990-01-02 | 2026-07-01 |         9191 |              9522 |        96.52 |
| VUSTX    | 1990-01-02 | 2026-06-30 |         9190 |              9521 |        96.52 |
| ^VIX     | 1990-01-02 | 2026-07-01 |         9192 |              9522 |        96.53 |
| ^IRX     | 1990-01-02 | 2026-07-01 |         9158 |              9522 |        96.18 |
| ^TNX     | 1990-01-02 | 2026-07-01 |         9158 |              9522 |        96.18 |

## 2. Fehlende Werte (Roh-Frame, vor ffill/dropna)
| Ticker   |   NaN (roh) |   NaN % |   Längste Lücke (Tage) | Erster Wert   | Letzter Wert   |
|:---------|------------:|--------:|-----------------------:|:--------------|:---------------|
| ^GSPC    |           1 |   0.011 |                      1 | 1990-01-02    | 2026-07-01     |
| VUSTX    |           2 |   0.022 |                      1 | 1990-01-02    | 2026-06-30     |
| ^VIX     |           0 |   0     |                      0 | 1990-01-02    | 2026-07-01     |
| ^IRX     |          34 |   0.37  |                      2 | 1990-01-02    | 2026-07-01     |
| ^TNX     |          34 |   0.37  |                      2 | 1990-01-02    | 2026-07-01     |

## 3. Adjustment-Plausibilität (Tagessprünge Preis-Serien)
| Ticker   |   Max |Tagesrendite| |   Tage |z|>8 | Größter Sprung (Datum)   |
|:---------|---------------------:|-------------:|:-------------------------|
| ^GSPC    |               0.1277 |           27 | 2020-03-16               |
| VUSTX    |               0.0654 |            6 | 1992-12-31               |

## 4. Größte Tagesbewegungen (Krisen-Plausibilität)
|              | Datum      |   Log-Rendite |
|:-------------|:-----------|--------------:|
| ('^GSPC', 1) | 2020-03-16 |       -0.1277 |
| ('^GSPC', 2) | 2008-10-13 |        0.1096 |
| ('^GSPC', 3) | 2008-10-28 |        0.1025 |
| ('^GSPC', 4) | 2020-03-12 |       -0.0999 |
| ('^GSPC', 5) | 2008-10-15 |       -0.0947 |
| ('VUSTX', 1) | 1992-12-31 |        0.0654 |
| ('VUSTX', 2) | 2020-03-20 |        0.0632 |
| ('VUSTX', 3) | 2020-03-17 |       -0.0605 |
| ('VUSTX', 4) | 1992-12-11 |       -0.0586 |
| ('VUSTX', 5) | 2020-03-10 |       -0.0499 |

## 5. Auswirkung der Bereinigung (Bronze → Silver)
| Kennzahl                         |     Wert |
|:---------------------------------|---------:|
| Zeilen roh (Bronze)              | 9192     |
| Zeilen bereinigt (Silver)        | 9156     |
| Entfernt (dropna + Return-Shift) |   36     |
| Entfernt %                       |    0.392 |
