
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem **eingefrorenen Datensatz** vom **1990-01-02** bis **2026-04-17** (Thesis-Freeze).

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   | Final Wealth   | Total Return   | Max Drawdown   |
|:------------|:---------------|:---------------|:---------------|
| Buy_Hold    | 2,414,155 €    | +382.83%       | -34.77%        |
| MSM         | 1,478,296 €    | +195.66%       | -24.92%        |
| HMM         | 1,923,021 €    | +284.60%       | -8.50%         |
| LSTM        | 2,021,356 €    | +304.27%       | -32.92%        |
| Transformer | 2,073,893 €    | +314.78%       | -29.66%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |     Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|--------:|-----------------:|-----------:|
| Returns_GSPC  |             0.000328 |           0.011389 | -0.1277 |  0.1096 |          -0.3606 |    10.8109 |
| Returns_VUSTX |             0.000275 |           0.007485 | -0.0605 |  0.1296 |           0.6392 |    14.374  |
| Returns       |             0.000307 |           0.006935 | -0.0662 |  0.0584 |          -0.2267 |     7.7444 |
| VIX           |            19.4665   |           7.76644  |  9.14   | 82.69   |           2.2009 |     8.6722 |
| TNX_10Y       |             4.23749  |           1.9312   |  0.499  |  9.09   |           0.3306 |    -0.6363 |
| IRX_3M        |             2.7037   |           2.20243  | -0.105  |  7.99   |           0.2016 |    -1.2556 |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.4874 | 4.4557e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -18.1912 | 2.4191e-30 |           -2.8619 | Ja           |
| Returns       |        -17.5173 | 4.3038e-30 |           -2.8619 | Ja           |
| VIX           |         -7.2788 | 1.5194e-10 |           -2.8619 | Ja           |
| TNX_10Y       |         -2.3513 | 0.15596    |           -2.8619 | Nein         |
| IRX_3M        |         -2.341  | 0.15908    |           -2.8619 | Nein         |

**Volatilitätscluster und Autokorrelation (Heteroskedastizität):**
![Volatility Clusters](../assets/eda_volatility_clusters.png)

### Feature-Korrelation
Pearson-Korrelationsmatrix der sechs Modell-Features zur Prüfung auf Multikollinearität.

![Feature Correlation Matrix](../assets/feature_correlation_matrix.png)

### SORR Kontext: Historische Drawdowns
Darstellung der extremsten Verlustphasen des 60/40 Portfolios als Motivation für den aktiven Kapitalschutz.
![Historical Drawdowns](../assets/eda_historical_drawdowns.png)

### 60/40 Portfolio Kapitalkurve
Die Abbildung zeigt die kumulierte Wertentwicklung des statischen Referenzportfolios (60% Aktien / 40% Anleihen).

![Capital Curve](../assets/capital_curve.png)

*   **Datenquelle:** S&P 500 (`^GSPC`) und Vanguard Long-Term Treasury (`VUSTX`).
*   **Reproduzierbarkeit:** Der bereinigte Datensatz inkl. aller Features ist hinterlegt unter: `data/02_feature_engineered_data.parquet`.

---

## 3. Regime-Erkennung der Einzelmodelle
Hier werden die Identifikations-Ergebnisse der Modell-Kategorien (Statistik, Clustering, Deep Learning) visualisiert.

### A. Markov-Switching-Modelle (Ökonometrie)
Identifikation von Bull- und Bear-Regimes mittels eines univariaten Zwei-Regime-Markov-Switching-Modells auf Basis der S&P 500-Renditen.
![Markov Models](../assets/msm_regimes.png)

### B. Hidden Markov Model (Unsupervised Clustering)
![HMM Regimes](../assets/hmm_regimes.png)

### C. LSTM-Netzwerk (Deep Learning)
Vorhersage der Marktphasen durch das neuronale Netzwerk (trainiert auf Pagan-Sossounov-Labels).
![LSTM Model](../assets/lstm_model.png)

### D. Transformer-Netzwerk (Attention-basierte Regime-Erkennung)
Klassifikation von Marktregimes mittels eines Transformer-Encoders mit Multi-Head Self-Attention und Positional Encoding. Im Gegensatz zu rekurrenten Architekturen (LSTM) verarbeitet der Transformer alle Zeitschritte einer Sequenz parallel und lernt über den Attention-Mechanismus, welche historischen Datenpunkte die höchste Relevanz für die aktuelle Regime-Klassifikation besitzen. Trainiert im Supervised-Setting auf Pagan-Sossounov-Labels.
![Transformer Model](../assets/transformer_model.png)

### E. Globaler Regime-Vergleich
Detaillierte Gegenüberstellung der Wahrscheinlichkeiten und harten Signale aller Modelle.
![Regime Comparison](../assets/regime_comparison.png)

### F. Hyperparameter-Optimierung (Optuna)
Bayessche Suche über den Hyperparameter-Raum aller vier Modelle mittels Walk-Forward-Validierung als innere CV. Optimierungsziel ist der mediane OOS-Sharpe-Ratio über die subgesampelten Folds; geprunete Trials nutzen den Median-Pruner. Die hier ausgewiesenen Werte wurden 1:1 in die `config.yaml` übernommen und für den finalen Walk-Forward-Lauf verwendet.

# Optuna — Beste Hyperparameter

_Generiert am 2026-04-19 19:52:21_  
Optimierungs-Metrik: **Sharpe (Median OOS)**

## Übersicht

| Modell | Best Score | ✓ Complete | ✂ Pruned | Total |
|:---|---:|---:|---:|---:|
| **MSM** | 1.3918 | 28 | 34 | 83 |
| **HMM** | 2.0117 | 35 | 15 | 59 |
| **LSTM** | 1.3393 | 30 | 20 | 53 |
| **Transformer** | 1.3537 | 22 | 28 | 54 |

### MSM — Best Score `1.3918`

| Parameter | Wert |
|:---|---:|
| `k_regimes` | `2` |
| `threshold` | `0.35` |

### HMM — Best Score `2.0117`

| Parameter | Wert |
|:---|---:|
| `n_components` | `2` |
| `covariance_type` | `full` |
| `threshold` | `0.5` |

### LSTM — Best Score `1.3393`

| Parameter | Wert |
|:---|---:|
| `window_size` | `120` |
| `units_l1` | `16` |
| `units_l2` | `128` |
| `learning_rate` | `1.254e-04` |
| `dropout` | `0.1` |
| `epochs` | `40` |
| `threshold` | `0.3` |

### Transformer — Best Score `1.3537`

| Parameter | Wert |
|:---|---:|
| `d_model` | `128` |
| `n_heads` | `8` |
| `n_layers` | `3` |
| `dim_feedforward` | `64` |
| `learning_rate` | `3.148e-05` |
| `dropout` | `0.25` |
| `epochs` | `20` |
| `window_size` | `120` |
| `threshold` | `0.45` |


**Diagnose-Plots pro Modell** (Optimization History · Param-Importance · Slice · Contour):

| Modell | History | Importance | Slice | Contour |
|:---|:---|:---|:---|:---|
| MSM         | ![](../assets/optuna_MSM_history.png)         | ![](../assets/optuna_MSM_importance.png)         | ![](../assets/optuna_MSM_slice.png)         | — ¹                                            |
| HMM         | ![](../assets/optuna_HMM_history.png)         | ![](../assets/optuna_HMM_importance.png)         | ![](../assets/optuna_HMM_slice.png)         | ![](../assets/optuna_HMM_contour.png)         |
| LSTM        | ![](../assets/optuna_LSTM_history.png)        | ![](../assets/optuna_LSTM_importance.png)        | ![](../assets/optuna_LSTM_slice.png)        | ![](../assets/optuna_LSTM_contour.png)        |
| Transformer | ![](../assets/optuna_Transformer_history.png) | ![](../assets/optuna_Transformer_importance.png) | ![](../assets/optuna_Transformer_slice.png) | ![](../assets/optuna_Transformer_contour.png) |

¹ MSM hat nur einen Hyperparameter (`threshold`) im Search-Space — der Contour-Plot wäre degeneriert und entfällt.

### G. Label-Konkordanz (Auswahl der Trainings-Labels)
Vergleich der Regime-Labeler (MSM, HMM, Pagan-Sossounov, Peak-to-Trough, Lunde-Timmermann, NBER) zur Begründung der Label-Wahl für die Supervised-Modelle. Pagan-Sossounov wurde aufgrund seiner hohen Konkordanz mit NBER-Rezessionsperioden als Trainingsziel für LSTM und Transformer gewählt.

![Label Concordance](../assets/label_concordance_matrix.png)
![Label Cohen's κ](../assets/label_kappa_matrix.png)
![Label Timeline](../assets/label_timeline_comparison.png)

---

## 4. Backtesting & Strategie-Evaluation
Die ökonomische Anwendung der Regime-Signale durch dynamische Umschichtung in den Geldmarkt.

### Walk-Forward-Schema
Rollierende Train/Test-Fenster über den gesamten Untersuchungszeitraum. Jede Zeile entspricht einem Fold; der blaue Balken markiert das Trainingsfenster, der orange Balken das OOS-Testfenster. Die strikte chronologische Trennung verhindert Look-ahead Bias.

![Walk-Forward-Schema](../assets/walk_forward_schema.png)

### Equity Curves im Vergleich
![Equity Curves](../assets/equity_curves.png)

### Annualisierte Performance-Metriken
Normalisierte Kennzahlen (CAGR, Sharpe, Sortino, Calmar) für den Vergleich über unterschiedlich lange Evaluationszeiträume.

| Strategie   | CAGR   | Ann. Volatilität   |   Sharpe Ratio |   Sortino Ratio | Max Drawdown   |   Calmar Ratio |   OOS-Tage |   OOS-Jahre |
|:------------|:-------|:-------------------|---------------:|----------------:|:---------------|---------------:|-----------:|------------:|
| Buy_Hold    | +6.41% | 11.24%             |          0.57  |           0.737 | -34.77%        |          0.184 |       6404 |        25.4 |
| MSM         | +4.37% | 6.61%              |          0.661 |           0.771 | -24.92%        |          0.175 |       6404 |        25.4 |
| HMM         | +5.46% | 5.23%              |          1.044 |           1.09  | -8.50%         |          0.642 |       6404 |        25.4 |
| LSTM        | +5.67% | 10.40%             |          0.545 |           0.67  | -32.92%        |          0.172 |       6404 |        25.4 |
| Transformer | +5.77% | 10.49%             |          0.55  |           0.695 | -29.66%        |          0.195 |       6404 |        25.4 |

### Klassifikationsmetriken (vs. NBER-Rezessionen als Ground Truth)
Vergleich der Modelle als binäre Rezessionsklassifikatoren (Precision, Recall, F1).

| Modell      |   Precision |   Recall |    F1 |   TN |   FP |   FN |   TP |
|:------------|------------:|---------:|------:|-----:|-----:|-----:|-----:|
| MSM         |       0.264 |    0.877 | 0.406 | 4389 | 1430 |   72 |  514 |
| HMM         |       0.182 |    0.94  | 0.305 | 3347 | 2472 |   35 |  551 |
| LSTM        |       0.236 |    0.285 | 0.258 | 5277 |  542 |  419 |  167 |
| Transformer |       0.357 |    0.345 | 0.351 | 5455 |  364 |  384 |  202 |

![Confusion Matrices](../assets/confusion_matrices.png)

**ROC- und Precision-Recall-Kurven** (schwellenunabhängiger Vergleich über `*_Prob`):

![ROC-Kurven](../assets/roc_curves.png)
![PR-Kurven](../assets/pr_curves.png)

### Signal-Churning & Whipsaw-Analyse
Quantifizierung der Wechselhäufigkeit und Anteil sehr kurzer Regime-Phasen („Whipsaws").

| Modell      |   Signalwechsel |   Whipsaws (<5T) | Whipsaw-Anteil   |   Ø Phase (Tage) |   Median Phase (Tage) | Kumul. Kosten   |
|:------------|----------------:|-----------------:|:-----------------|-----------------:|----------------------:|:----------------|
| MSM         |             316 |              163 | 51.4%            |             20.2 |                     4 | 31.60%          |
| HMM         |              92 |               15 | 16.1%            |             68.9 |                    25 | 9.20%           |
| LSTM        |              20 |                1 | 4.8%             |            305   |                    69 | 2.00%           |
| Transformer |              98 |               64 | 64.6%            |             64.7 |                     2 | 9.80%           |

### Regime-Wahrscheinlichkeits-Heatmap
Zeitverlauf der Bear-Wahrscheinlichkeiten aller Modelle.

![Regime Probability Heatmap](../assets/regime_probability_heatmap.png)

### Threshold-Sensitivität
Variation der Entscheidungs-Schwelle pro Modell. Zeigt, wie robust Final Wealth, Max Drawdown und Anzahl der Regime-Wechsel gegenüber einer veränderten Bull/Bear-Klassifikations-Grenze sind (Kap. 4.1 — Glättung).

**MSM**

|   Threshold | Final Wealth   | Max Drawdown   |   Wechsel |
|------------:|:---------------|:---------------|----------:|
|        0.25 | 1,571,369 €    | -18.19%        |       310 |
|        0.3  | 1,452,916 €    | -22.02%        |       310 |
|        0.35 | 1,478,296 €    | -24.92%        |       316 |
|        0.4  | 1,562,183 €    | -23.76%        |       314 |
|        0.5  | 1,577,677 €    | -23.20%        |       333 |

**HMM**

|   Threshold | Final Wealth   | Max Drawdown   |   Wechsel |
|------------:|:---------------|:---------------|----------:|
|        0.4  | 1,837,359 €    | -8.50%         |        94 |
|        0.45 | 1,899,381 €    | -8.50%         |        92 |
|        0.5  | 1,923,021 €    | -8.50%         |        92 |
|        0.55 | 1,933,632 €    | -8.50%         |        92 |
|        0.6  | 1,942,131 €    | -8.50%         |        92 |

**LSTM**

|   Threshold | Final Wealth   | Max Drawdown   |   Wechsel |
|------------:|:---------------|:---------------|----------:|
|         0.2 | 2,068,373 €    | -32.90%        |        20 |
|         0.3 | 2,021,356 €    | -32.92%        |        20 |
|         0.4 | 2,001,631 €    | -32.87%        |        22 |
|         0.5 | 2,048,033 €    | -32.87%        |        24 |

**Transformer**

|   Threshold | Final Wealth   | Max Drawdown   |   Wechsel |
|------------:|:---------------|:---------------|----------:|
|        0.3  | 1,886,854 €    | -30.26%        |       158 |
|        0.4  | 2,042,974 €    | -28.79%        |       126 |
|        0.45 | 2,073,893 €    | -29.66%        |        98 |
|        0.5  | 2,195,745 €    | -27.71%        |        90 |
|        0.6  | 2,307,030 €    | -27.71%        |        58 |

### Time-to-Recovery
Alle Drawdown-Phasen jenseits der Mindesttiefe (gemäß `extended.ttr_min_dd`) mit Peak-, Trough- und Recovery-Datum sowie Dauer in Handelstagen. Eine offene (noch nicht erholte) Phase wird im Recovery-Feld mit „—" markiert.

**Buy_Hold**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown-Dauer (T) |   Recovery-Dauer (T) |   Gesamt (T) |
|:-----------|:-----------|:-----------|:---------|---------------------:|---------------------:|-------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                   49 |                   43 |           92 |
| 2001-02-02 | 2002-07-23 | 2004-02-11 | -24.02%  |                  536 |                  568 |         1104 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                   63 |                  178 |          241 |
| 2007-11-01 | 2009-03-09 | 2011-02-11 | -34.66%  |                  494 |                  704 |         1198 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                   14 |                   67 |           81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                   33 |                  120 |          153 |
| 2015-04-16 | 2015-08-25 | 2016-03-29 | -8.25%   |                  131 |                  217 |          348 |
| 2016-08-01 | 2016-11-14 | 2017-02-21 | -5.64%   |                  105 |                   99 |          204 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                   10 |                  197 |          207 |
| 2018-08-30 | 2018-12-24 | 2019-03-21 | -11.45%  |                  116 |                   87 |          203 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                   26 |                   82 |          108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                   57 |                   39 |           96 |
| 2021-12-28 | 2022-10-14 | 2024-11-29 | -27.55%  |                  290 |                  777 |         1067 |
| 2024-12-09 | 2025-04-08 | 2025-07-03 | -12.22%  |                  120 |                   86 |          206 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                   29 |                   21 |           50 |

**MSM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown-Dauer (T) |   Recovery-Dauer (T) |   Gesamt (T) |
|:-----------|:-----------|:-----------|:---------|---------------------:|---------------------:|-------------:|
| 2001-02-02 | 2003-04-11 | 2006-09-25 | -23.93%  |                  798 |                 1263 |         2061 |
| 2007-06-05 | 2009-08-06 | 2010-09-13 | -12.74%  |                  793 |                  403 |         1196 |
| 2011-07-25 | 2011-10-03 | 2012-01-03 | -7.53%   |                   70 |                   92 |          162 |
| 2013-05-22 | 2013-08-21 | 2013-11-29 | -5.66%   |                   91 |                  100 |          191 |
| 2015-02-26 | 2015-09-28 | 2016-03-18 | -6.78%   |                  214 |                  172 |          386 |
| 2016-08-01 | 2016-11-14 | 2017-02-23 | -5.90%   |                  105 |                  101 |          206 |
| 2018-01-29 | 2019-01-14 | 2019-06-06 | -8.77%   |                  350 |                  143 |          493 |
| 2021-01-26 | 2021-03-25 | 2021-07-07 | -6.62%   |                   58 |                  104 |          162 |
| 2021-11-10 | 2022-05-19 | 2023-03-30 | -7.65%   |                  190 |                  315 |          505 |
| 2023-07-03 | 2023-09-22 | 2023-12-27 | -6.18%   |                   81 |                   96 |          177 |
| 2024-07-17 | 2025-05-06 | —          | -10.64%  |                  293 |                  nan |          nan |

**HMM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown-Dauer (T) |   Recovery-Dauer (T) |   Gesamt (T) |
|:-----------|:-----------|:-----------|:---------|---------------------:|---------------------:|-------------:|
| 2002-01-11 | 2002-06-04 | 2004-12-06 | -6.20%   |                  144 |                  916 |         1060 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                   33 |                  120 |          153 |
| 2016-08-01 | 2016-11-14 | 2017-02-21 | -5.64%   |                  105 |                   99 |          204 |
| 2018-01-29 | 2018-04-25 | 2018-08-20 | -5.66%   |                   86 |                  117 |          203 |
| 2018-08-30 | 2019-01-14 | 2019-03-28 | -6.68%   |                  137 |                   73 |          210 |
| 2023-07-20 | 2023-11-09 | 2023-12-14 | -7.94%   |                  112 |                   35 |          147 |
| 2024-12-09 | 2025-01-10 | 2025-07-03 | -5.72%   |                   32 |                  174 |          206 |

**LSTM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown-Dauer (T) |   Recovery-Dauer (T) |   Gesamt (T) |
|:-----------|:-----------|:-----------|:---------|---------------------:|---------------------:|-------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                   49 |                   43 |           92 |
| 2001-02-02 | 2002-07-23 | 2004-01-07 | -22.15%  |                  536 |                  533 |         1069 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                   63 |                  178 |          241 |
| 2007-11-01 | 2009-03-09 | 2011-05-31 | -32.81%  |                  494 |                  813 |         1307 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                   14 |                   67 |           81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                   33 |                  120 |          153 |
| 2015-04-16 | 2016-01-20 | 2016-07-01 | -10.87%  |                  279 |                  163 |          442 |
| 2016-08-01 | 2016-11-14 | 2017-02-21 | -5.64%   |                  105 |                   99 |          204 |
| 2018-01-29 | 2018-12-24 | 2019-04-05 | -12.08%  |                  329 |                  102 |          431 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                   26 |                   82 |          108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                   57 |                   39 |           96 |
| 2021-12-28 | 2022-10-14 | 2024-11-29 | -27.55%  |                  290 |                  777 |         1067 |
| 2024-12-09 | 2025-04-08 | 2025-07-03 | -12.22%  |                  120 |                   86 |          206 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                   29 |                   21 |           50 |

**Transformer**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown-Dauer (T) |   Recovery-Dauer (T) |   Gesamt (T) |
|:-----------|:-----------|:-----------|:---------|---------------------:|---------------------:|-------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                   49 |                   43 |           92 |
| 2001-02-02 | 2003-03-11 | 2005-02-04 | -24.39%  |                  767 |                  696 |         1463 |
| 2006-01-17 | 2008-11-20 | 2008-12-16 | -13.74%  |                 1038 |                   26 |         1064 |
| 2009-01-05 | 2009-03-09 | 2009-07-30 | -18.97%  |                   63 |                  143 |          206 |
| 2010-05-04 | 2010-07-02 | 2010-09-13 | -5.36%   |                   59 |                   73 |          132 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                   14 |                   67 |           81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                   33 |                  120 |          153 |
| 2015-04-16 | 2016-11-14 | 2017-06-02 | -9.33%   |                  578 |                  200 |          778 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                   10 |                  197 |          207 |
| 2018-08-30 | 2018-12-24 | 2019-03-21 | -11.45%  |                  116 |                   87 |          203 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                   26 |                   82 |          108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                   57 |                   39 |           96 |
| 2021-12-28 | 2023-10-27 | 2025-09-30 | -29.51%  |                  668 |                  704 |         1372 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                   29 |                   21 |           50 |

### Krisen-Performance
Return und Max Drawdown während historischer Krisenperioden — der zentrale Nachweis für den Tail-Risk-Schutz der Regime-Switching-Modelle.

| Krise                                | ('Return', 'Buy_Hold')   | ('Return', 'HMM')   | ('Return', 'LSTM')   | ('Return', 'MSM')   | ('Return', 'Transformer')   | ('Max Drawdown', 'Buy_Hold')   | ('Max Drawdown', 'HMM')   | ('Max Drawdown', 'LSTM')   | ('Max Drawdown', 'MSM')   | ('Max Drawdown', 'Transformer')   |
|:-------------------------------------|:-------------------------|:--------------------|:---------------------|:--------------------|:----------------------------|:-------------------------------|:--------------------------|:---------------------------|:--------------------------|:----------------------------------|
| COVID Crash (2020-02 – 2020-03)      | -8.24%                   | +2.21%              | -8.24%               | +0.73%              | -8.24%                      | -18.53%                        | -0.36%                    | -18.53%                    | -1.81%                    | -18.53%                           |
| Dot-Com (2000-03 – 2002-10)          | -14.76%                  | +0.31%              | -12.66%              | -18.18%             | -15.45%                     | -24.79%                        | -6.36%                    | -22.94%                    | -21.03%                   | -24.79%                           |
| EU-Schuldenkrise (2011-07 – 2011-11) | +4.10%                   | +0.01%              | +4.10%               | -1.37%              | +4.10%                      | -7.24%                         | 0.00%                     | -7.24%                     | -8.17%                    | -7.24%                            |
| GFC (2007-10 – 2009-03)              | -25.67%                  | +2.77%              | -23.37%              | -8.75%              | -5.43%                      | -34.77%                        | -2.56%                    | -32.92%                    | -9.57%                    | -19.46%                           |
| Zinsanstieg (2022-01 – 2022-10)      | -24.20%                  | -1.71%              | -24.20%              | -4.21%              | -24.64%                     | -26.98%                        | -1.82%                    | -26.98%                    | -6.55%                    | -26.98%                           |

### Switch-Timing relativ zum Drawdown-Peak
Zeitlicher Abstand zwischen dem ersten Bear-Signal des Modells und dem Drawdown-Trough des Buy & Hold-Portfolios je Krise. Positiv = Modell reagierte frühzeitig, negativ = zu spät.

| Krise   | Modell      | DD-Trough   | 1. Bear-Signal   |   Lead (Tage) |
|:--------|:------------|:------------|:-----------------|--------------:|
| GFC     | MSM         | 2009-03-09  | 2007-10-01       |           525 |
| COVID   | MSM         | 2020-03-18  | 2020-02-24       |            23 |
| 2022    | MSM         | 2022-10-14  | 2022-01-05       |           282 |
| GFC     | HMM         | 2009-03-09  | 2007-10-01       |           525 |
| COVID   | HMM         | 2020-03-18  | 2020-02-21       |            26 |
| 2022    | HMM         | 2022-10-14  | 2022-01-05       |           282 |
| GFC     | LSTM        | 2009-03-09  | 2007-10-01       |           525 |
| COVID   | LSTM        | 2020-03-18  |                  |           nan |
| 2022    | LSTM        | 2022-10-14  |                  |           nan |
| GFC     | Transformer | 2009-03-09  | 2007-11-20       |           475 |
| COVID   | Transformer | 2020-03-18  |                  |           nan |
| 2022    | Transformer | 2022-10-14  | 2022-10-27       |           -13 |

### Drawdown-Verlauf
![Drawdown](../assets/drawdown.png)

### Rollierender Sharpe Ratio
Zeitvariierender, risikoadjustierter Rendite-Vergleich über ein rollendes 252-Tage-Fenster.

![Rolling Sharpe](../assets/rolling_sharpe.png)

### Umfassende Kennzahlen-Matrix
Detaillierte statistische Analyse inklusive risikoadjustierter Kennzahlen (Sharpe, Sortino, Calmar).

| Strategie   | Total Return   | CAGR (p.a.)   | Volatilität   | Max Drawdown   |   Sharpe Ratio |   Sortino Ratio |   Calmar Ratio |   Regime-Wechsel | Gesamtkosten (Gebühren)   |   Ulcer Index |
|:------------|:---------------|:--------------|:--------------|:---------------|---------------:|----------------:|---------------:|-----------------:|:--------------------------|--------------:|
| Buy Hold    | 384.52%        | 6.39%         | 11.23%        | -34.77%        |           0.61 |            0.8  |           0.18 |                0 | 0.00%                     |          8.88 |
| MSM         | 196.69%        | 4.36%         | 6.61%         | -24.92%        |           0.68 |            0.8  |           0.17 |              316 | 31.60%                    |          7.89 |
| HMM         | 285.95%        | 5.44%         | 5.23%         | -8.50%         |           1.04 |            1.1  |           0.64 |               92 | 9.30%                     |          2.49 |
| LSTM        | 305.69%        | 5.65%         | 10.39%        | -32.92%        |           0.58 |            0.73 |           0.17 |               20 | 2.00%                     |          8.76 |
| Transformer | 316.23%        | 5.75%         | 10.49%        | -29.66%        |           0.59 |            0.75 |           0.19 |               98 | 9.80%                     |          9.15 |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status           |
|:-------------------------------|:-------------|:-----------------|
| ('Standard', 'Buy Hold')       | 141,069.86 € | Kapitalerhalt    |
| ('Standard', 'MSM')            | 0.00 €       | Erschöpft (2021) |
| ('Standard', 'HMM')            | 75,647.87 €  | Kapitalerhalt    |
| ('Standard', 'LSTM')           | 0.00 €       | Erschöpft (2025) |
| ('Standard', 'Transformer')    | 15,792.04 €  | Kapitalerhalt    |
| ('Aggressive', 'Buy Hold')     | 0.00 €       | Erschöpft (2012) |
| ('Aggressive', 'MSM')          | 0.00 €       | Erschöpft (2010) |
| ('Aggressive', 'HMM')          | 0.00 €       | Erschöpft (2012) |
| ('Aggressive', 'LSTM')         | 0.00 €       | Erschöpft (2011) |
| ('Aggressive', 'Transformer')  | 0.00 €       | Erschöpft (2011) |
| ('Low_Capital', 'Buy Hold')    | 0.00 €       | Erschöpft (2016) |
| ('Low_Capital', 'MSM')         | 0.00 €       | Erschöpft (2013) |
| ('Low_Capital', 'HMM')         | 0.00 €       | Erschöpft (2016) |
| ('Low_Capital', 'LSTM')        | 0.00 €       | Erschöpft (2015) |
| ('Low_Capital', 'Transformer') | 0.00 €       | Erschöpft (2014) |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Low_Capital', 'HMM')         | 0.00%                     | 192,037.68 €        |
| ('Aggressive', 'MSM')          | 2.06%                     | 162,881.85 €        |
| ('Standard', 'Buy Hold')       | 0.00%                     | 496,792.52 €        |
| ('Standard', 'LSTM')           | 0.03%                     | 449,256.04 €        |
| ('Low_Capital', 'Transformer') | 0.56%                     | 191,144.41 €        |
| ('Aggressive', 'LSTM')         | 5.68%                     | 205,805.04 €        |
| ('Aggressive', 'HMM')          | 0.01%                     | 214,814.48 €        |
| ('Low_Capital', 'LSTM')        | 0.56%                     | 190,872.18 €        |
| ('Aggressive', 'Transformer')  | 5.06%                     | 215,817.72 €        |
| ('Standard', 'Transformer')    | 0.02%                     | 455,487.75 €        |
| ('Standard', 'MSM')            | 0.00%                     | 388,380.39 €        |
| ('Standard', 'HMM')            | 0.00%                     | 453,782.62 €        |
| ('Aggressive', 'Buy Hold')     | 4.58%                     | 245,081.83 €        |
| ('Low_Capital', 'MSM')         | 0.01%                     | 156,876.25 €        |
| ('Low_Capital', 'Buy Hold')    | 0.62%                     | 213,677.49 €        |

Verteilung der Endkapitalwerte:

![MCS Boxplots Standard](../assets/mcs_boxplot_standard.png)
![MCS Boxplots Aggressive](../assets/mcs_boxplot_aggressive.png)
![MCS Boxplots Low Capital](../assets/mcs_boxplot_low_capital.png)

Wahrscheinlichkeitskorridore:

Die schattierten Bereiche zeigen das 5% bis 95% Konfidenzintervall der Kapitalentwicklung.
![MCS Quantiles](../assets/mcs_quantiles.png)

### Depletion Rate mit 95%-Konfidenzintervall
Wilson-CI für die Ruin-Wahrscheinlichkeit (P[Endkapital ≤ 0]) je Szenario × Strategie.

|                                | Depletion Rate   | 95%-CI unten   | 95%-CI oben   | n_ruin / n_paths   |
|:-------------------------------|:-----------------|:---------------|:--------------|:-------------------|
| ('Standard', 'Buy_Hold')       | 0.00%            | 0.00%          | 0.04%         | 0/10000            |
| ('Standard', 'MSM')            | 0.00%            | 0.00%          | 0.04%         | 0/10000            |
| ('Standard', 'HMM')            | 0.00%            | 0.00%          | 0.04%         | 0/10000            |
| ('Standard', 'LSTM')           | 0.03%            | 0.01%          | 0.09%         | 3/10000            |
| ('Standard', 'Transformer')    | 0.02%            | 0.01%          | 0.07%         | 2/10000            |
| ('Aggressive', 'Buy_Hold')     | 4.58%            | 4.19%          | 5.01%         | 458/10000          |
| ('Aggressive', 'MSM')          | 2.06%            | 1.80%          | 2.36%         | 206/10000          |
| ('Aggressive', 'HMM')          | 0.01%            | 0.00%          | 0.06%         | 1/10000            |
| ('Aggressive', 'LSTM')         | 5.68%            | 5.24%          | 6.15%         | 568/10000          |
| ('Aggressive', 'Transformer')  | 5.06%            | 4.65%          | 5.51%         | 506/10000          |
| ('Low_Capital', 'Buy_Hold')    | 0.62%            | 0.48%          | 0.79%         | 62/10000           |
| ('Low_Capital', 'MSM')         | 0.01%            | 0.00%          | 0.06%         | 1/10000            |
| ('Low_Capital', 'HMM')         | 0.00%            | 0.00%          | 0.04%         | 0/10000            |
| ('Low_Capital', 'LSTM')        | 0.56%            | 0.43%          | 0.73%         | 56/10000           |
| ('Low_Capital', 'Transformer') | 0.56%            | 0.43%          | 0.73%         | 56/10000           |

### Hypothesentests (gepaarter Wilcoxon, α = 0.05)
**H1 — Regime-Switching reduziert MaxDD vs. Buy & Hold:**

| Modell      | Median MaxDD (Modell)   | Median MaxDD (B&H)   | Δ Median   |   Wilcoxon p | H1 (α=0.05)   |
|:------------|:------------------------|:---------------------|:-----------|-------------:|:--------------|
| MSM         | -68.80%                 | -57.03%              | -11.77 pp  |        1     | abgelehnt     |
| HMM         | -58.35%                 | -57.03%              | -1.32 pp   |        0.569 | abgelehnt     |
| LSTM        | -62.71%                 | -57.03%              | -5.68 pp   |        1     | abgelehnt     |
| Transformer | -61.07%                 | -57.03%              | -4.04 pp   |        1     | abgelehnt     |

**H2 — Transformer dominiert Ökonometrie und LSTM im Endvermögen:**

| Vergleich            | Median Transformer   | Median MSM   | Δ Median   |   Wilcoxon p | H2 (α=0.05)   | Median HMM   | Median LSTM   |
|:---------------------|:---------------------|:-------------|:-----------|-------------:|:--------------|:-------------|:--------------|
| Transformer vs. MSM  | 215,818 €            | 162,882 €    | +52,936 €  |    5.35e-206 | bestätigt     | nan          | nan           |
| Transformer vs. HMM  | 215,818 €            | nan          | +1,003 €   |    1.38e-08  | bestätigt     | 214,814 €    | nan           |
| Transformer vs. LSTM | 215,818 €            | nan          | +10,013 €  |    1.09e-05  | bestätigt     | nan          | 205,805 €     |

### Break-Even-Transaktionskosten
Ab welcher Kostenquote (in Basispunkten pro Umschichtung) verliert das aktive Switching seinen Renditevorteil gegenüber Buy & Hold?

| Modell      |   Final @10bps |   B&H Final |   Break-Even (bps) |
|:------------|---------------:|------------:|-------------------:|
| MSM         |          2.957 |       4.828 |                  0 |
| HMM         |          3.846 |       4.828 |                  0 |
| LSTM        |          4.043 |       4.828 |                  0 |
| Transformer |          4.148 |       4.828 |                  0 |

![Break-Even-Analyse](../assets/break_even_costs.png)

### Entnahmeraten-Sensitivität (3.5 % / 4 % / 5 %)
Robustheit der SORR-Ergebnisse bei variierenden jährlichen Entnahmen.

| Strategie   | ('Endkapital', '3.5%')   | ('Endkapital', '4.0%')   | ('Endkapital', '5.0%')   | ('Status', '3.5%')   | ('Status', '4.0%')   | ('Status', '5.0%')   |
|:------------|:-------------------------|:-------------------------|:-------------------------|:---------------------|:---------------------|:---------------------|
| Buy_Hold    | 1,091,710 €              | 901,582 €                | 521,326 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| HMM         | 848,192 €                | 693,683 €                | 384,665 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| LSTM        | 836,008 €                | 665,661 €                | 324,968 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| MSM         | 530,740 €                | 394,636 €                | 122,427 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| Transformer | 876,359 €                | 704,246 €                | 360,019 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |

---

## Forschungsnotizen & Methodik
- **Cash-Komponente:** Bei einem "Bear"-Signal schichtet die Strategie in den aktuellen Geldmarktzins (**^IRX**) um.
- **Vermeidung von Look-ahead Bias:** Alle Signale werden für das Backtesting um einen Tag zeitversetzt (`shift(1)`), um reale Handelsbedingungen zu simulieren.
- **Feature-Set:** Die Modelle nutzen Renditen, Volatilität (20d), SMA-Abstand, Momentum, VIX und Yield Spread.
- **Kostensimulation:** Es wird eine pauschale Gebühr von 10 Basispunkten (0,1%) pro Umschichtung berechnet.
- **SORR-Spezifika:** Bei Entnahmen in "Bull"-Phasen wird eine zusätzliche Liquiditätsgebühr von 0,1% auf den Entnahmebetrag erhoben (Asset-Verkäufe). In "Bear"-Phasen (Cash) entfällt diese.

---

## Pipeline-Laufzeiten

Ausführungszeiten der einzelnen Pipeline-Notebooks (monolithischer Notebook-Ansatz).

| Notebook | Start | Ende | Dauer (s) |
|----------|-------|------|-----------|
| 00_dependencies | 17:20:32 | 17:20:35 | 3.0 |
| 01_data_preprocessing | 17:20:35 | 17:20:42 | 6.9 |
| 02_feature_engineering | 17:20:42 | 17:20:47 | 4.8 |
| 03_regime_switching_models | 17:20:47 | 17:20:54 | 7.2 |
| 04_backtesting | 17:20:54 | 17:21:00 | 6.3 |
| 05_evaluation | 17:21:00 | 17:23:42 | 161.7 |
| **Gesamt** | | | **189.9** (3m 9.9s) |

---

## Modell-Persistierung

Status der Modell-Persistierung für diesen Pipeline-Durchlauf:

- **Persistierung:** AKTIV
- **Modell-Verzeichnis:** `../models`

| Modell | Datei | Status |
|:---|:---|:---|
| MSM | `msm_regime_model.pkl` | Neu trainiert |
| HMM | `hmm_regime_model.pkl` | Neu trainiert |
| LSTM | `lstm_regime_model.keras` | Neu trainiert |
| TRANSFORMER | `transformer_regime_model.pt` | Neu trainiert |

> **Hinweis:** Bei aktivierter Persistierung werden vortrainierte Modelle aus `../models` geladen, sofern die Dateien existieren. Andernfalls wird normal trainiert und das Ergebnis für zukünftige Läufe gespeichert. Bei Änderungen an Hyperparametern müssen die entsprechenden Modelldateien gelöscht werden.

---

**Zuletzt aktualisiert:** 19.04.2026 20:03<br>
**End date:** `2026-04-17`<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Walk-Forward-Validierung:** AKTIV (Modus: rolling, Train: 10J, Test: 12M, Step: 12M)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
