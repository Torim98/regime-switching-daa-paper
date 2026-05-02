
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem **eingefrorenen Datensatz** vom **1990-01-02** bis **2026-04-17** (Thesis-Freeze).

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   | Final Wealth   | Total Return   | Max Drawdown   |
|:------------|:---------------|:---------------|:---------------|
| Buy_Hold    | 2,414,155 €    | +384.52%       | -34.77%        |
| MSM         | 1,742,347 €    | +249.69%       | -28.68%        |
| HMM         | 1,766,700 €    | +254.58%       | -15.00%        |
| LSTM        | 2,484,328 €    | +398.61%       | -27.71%        |
| Transformer | 2,145,223 €    | +330.55%       | -27.71%        |

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

_Generiert am 2026-04-21 22:01:55_  
Optimierungs-Metrik: **Sharpe (Median OOS)**

## Übersicht

| Modell | Best Score | ✓ Complete | ✗ Pruned | Total |
|:---|---:|---:|---:|---:|
| **MSM** | 0.9308 | 23 | 27 | 50 |
| **HMM** | 1.2843 | 50 | 0 | 50 |
| **LSTM** | 1.4595 | 16 | 14 | 30 |
| **Transformer** | 1.0530 | 19 | 11 | 30 |

### MSM — Best Score `0.9308`

| Parameter | Wert |
|:---|---:|
| `threshold` | `0.7` |

### HMM — Best Score `1.2843`

| Parameter | Wert |
|:---|---:|
| `covariance_type` | `tied` |
| `threshold` | `0.35` |

### LSTM — Best Score `1.4595`

| Parameter | Wert |
|:---|---:|
| `window_size` | `120` |
| `units_l1` | `32` |
| `units_l2` | `64` |
| `learning_rate` | `1.053e-04` |
| `dropout` | `0.4` |
| `epochs` | `40` |
| `threshold` | `0.3` |

### Transformer — Best Score `1.0530`

| Parameter | Wert |
|:---|---:|
| `d_model` | `32` |
| `n_heads` | `4` |
| `n_layers` | `3` |
| `dim_feedforward` | `128` |
| `learning_rate` | `3.282e-05` |
| `dropout` | `0.1` |
| `epochs` | `40` |
| `window_size` | `40` |
| `threshold` | `0.55` |


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
| Buy_Hold    | +6.39% | 11.23%             |          0.609 |           0.798 | -34.77%        |          0.184 |       6404 |        25.4 |
| MSM         | +5.03% | 7.47%              |          0.697 |           0.87  | -28.68%        |          0.176 |       6404 |        25.4 |
| HMM         | +5.11% | 6.09%              |          0.848 |           0.9   | -15.00%        |          0.341 |       6404 |        25.4 |
| LSTM        | +6.53% | 10.86%             |          0.636 |           0.806 | -27.71%        |          0.236 |       6404 |        25.4 |
| Transformer | +5.91% | 10.17%             |          0.616 |           0.788 | -27.71%        |          0.213 |       6404 |        25.4 |

### Klassifikationsmetriken (vs. NBER-Rezessionen als Ground Truth)
Vergleich der Modelle als binäre Rezessionsklassifikatoren (Precision, Recall, F1).

| Modell      |   Precision |   Recall |    F1 |   TN |   FP |   FN |   TP |
|:------------|------------:|---------:|------:|-----:|-----:|-----:|-----:|
| MSM         |       0.308 |    0.742 | 0.436 | 4843 |  976 |  151 |  435 |
| HMM         |       0.173 |    0.845 | 0.287 | 3445 | 2374 |   91 |  495 |
| LSTM        |       0.351 |    0.278 | 0.31  | 5517 |  302 |  423 |  163 |
| Transformer |       0.476 |    0.408 | 0.439 | 5556 |  263 |  347 |  239 |

![Confusion Matrices](../assets/confusion_matrices.png)

**ROC- und Precision-Recall-Kurven** (schwellenunabhängiger Vergleich über `*_Prob`):

![ROC-Kurven](../assets/roc_curves.png)
![PR-Kurven](../assets/pr_curves.png)

### Signal-Churning & Whipsaw-Analyse
Quantifizierung der Wechselhäufigkeit und Anteil sehr kurzer Regime-Phasen („Whipsaws").

| Modell      |   Signalwechsel |   Whipsaws (<5T) | Whipsaw-Anteil   |   Ø Phase (Tage) |   Median Phase (Tage) | Kumul. Kosten   |
|:------------|----------------:|-----------------:|:-----------------|-----------------:|----------------------:|:----------------|
| MSM         |             371 |              214 | 57.5%            |             17.2 |                     3 | 37.20%          |
| HMM         |              71 |               16 | 22.2%            |             89   |                    32 | 7.10%           |
| LSTM        |              16 |                0 | 0.0%             |            376.8 |                    37 | 1.60%           |
| Transformer |             132 |               93 | 69.9%            |             48.2 |                     2 | 13.20%          |

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
|        0.4  | 1,799,202 €    | -15.26%        |        67 |
|        0.45 | 1,818,133 €    | -15.49%        |        61 |
|        0.5  | 1,788,340 €    | -17.05%        |        63 |
|        0.55 | 1,667,366 €    | -19.97%        |        81 |
|        0.6  | 1,521,907 €    | -25.69%        |       123 |

**LSTM**

|   Threshold | Final Wealth   | Max Drawdown   |   Wechsel |
|------------:|:---------------|:---------------|----------:|
|         0.2 | 2,483,648 €    | -27.71%        |        16 |
|         0.3 | 2,484,328 €    | -27.71%        |        16 |
|         0.4 | 2,507,880 €    | -27.71%        |        16 |
|         0.5 | 2,381,953 €    | -29.09%        |        16 |

**Transformer**

|   Threshold | Final Wealth   | Max Drawdown   |   Wechsel |
|------------:|:---------------|:---------------|----------:|
|        0.3  | 1,515,588 €    | -27.71%        |       114 |
|        0.4  | 1,592,761 €    | -29.36%        |       156 |
|        0.45 | 1,569,852 €    | -30.93%        |       150 |
|        0.5  | 1,786,668 €    | -27.71%        |       150 |
|        0.6  | 2,229,777 €    | -27.71%        |        90 |

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
| 2001-02-02 | 2003-03-11 | 2006-10-25 | -27.94%  |                  767 |                 1324 |         2091 |
| 2007-06-05 | 2009-06-23 | 2010-04-14 | -13.98%  |                  749 |                  295 |         1044 |
| 2010-05-04 | 2010-07-02 | 2010-09-14 | -5.86%   |                   59 |                   74 |          133 |
| 2011-07-25 | 2011-10-03 | 2011-12-29 | -7.98%   |                   70 |                   87 |          157 |
| 2015-02-26 | 2015-09-28 | 2015-10-26 | -5.45%   |                  214 |                   28 |          242 |
| 2015-12-02 | 2016-01-20 | 2016-03-17 | -5.96%   |                   49 |                   57 |          106 |
| 2016-08-01 | 2016-11-14 | 2017-02-21 | -5.84%   |                  105 |                   99 |          204 |
| 2018-01-29 | 2018-04-25 | 2018-08-24 | -6.26%   |                   86 |                  121 |          207 |
| 2018-08-30 | 2019-01-14 | 2019-06-07 | -9.57%   |                  137 |                  144 |          281 |
| 2020-09-03 | 2021-03-24 | 2021-07-23 | -8.65%   |                  202 |                  121 |          323 |
| 2021-12-28 | 2022-04-06 | 2023-07-18 | -12.42%  |                   99 |                  468 |          567 |
| 2023-07-20 | 2023-10-19 | 2024-06-05 | -10.89%  |                   91 |                  230 |          321 |
| 2024-12-09 | 2025-01-13 | 2025-09-04 | -6.20%   |                   35 |                  234 |          269 |
| 2025-10-29 | 2026-03-19 | —          | -6.25%   |                  141 |                  nan |          nan |

**HMM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown-Dauer (T) |   Recovery-Dauer (T) |   Gesamt (T) |
|:-----------|:-----------|:-----------|:---------|---------------------:|---------------------:|-------------:|
| 2002-03-07 | 2002-06-12 | 2003-12-18 | -6.68%   |                   97 |                  554 |          651 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                   63 |                  178 |          241 |
| 2008-05-20 | 2008-07-28 | 2009-08-20 | -7.18%   |                   69 |                  388 |          457 |
| 2013-05-22 | 2013-06-24 | 2014-10-21 | -5.37%   |                   33 |                  484 |          517 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                   10 |                  197 |          207 |
| 2018-08-30 | 2018-10-11 | 2019-11-26 | -5.40%   |                   42 |                  411 |          453 |
| 2020-02-21 | 2020-07-24 | 2021-11-05 | -5.89%   |                  154 |                  469 |          623 |
| 2021-11-10 | 2023-10-27 | 2024-02-01 | -14.75%  |                  716 |                   97 |          813 |
| 2024-12-09 | 2025-01-10 | 2025-06-12 | -5.33%   |                   32 |                  153 |          185 |
| 2025-10-29 | 2026-03-20 | 2026-04-14 | -5.17%   |                  142 |                   25 |          167 |

**LSTM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown-Dauer (T) |   Recovery-Dauer (T) |   Gesamt (T) |
|:-----------|:-----------|:-----------|:---------|---------------------:|---------------------:|-------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                   49 |                   43 |           92 |
| 2001-02-02 | 2002-07-23 | 2004-02-11 | -24.02%  |                  536 |                  568 |         1104 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                   63 |                  178 |          241 |
| 2007-07-20 | 2009-03-09 | 2009-11-16 | -25.99%  |                  598 |                  252 |          850 |
| 2010-05-04 | 2010-07-02 | 2010-09-13 | -5.36%   |                   59 |                   73 |          132 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                   14 |                   67 |           81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                   33 |                  120 |          153 |
| 2015-04-16 | 2015-08-25 | 2016-03-03 | -8.25%   |                  131 |                  191 |          322 |
| 2016-08-01 | 2016-11-14 | 2017-02-21 | -5.64%   |                  105 |                   99 |          204 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                   10 |                  197 |          207 |
| 2018-08-30 | 2018-12-24 | 2019-03-21 | -11.45%  |                  116 |                   87 |          203 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                   26 |                   82 |          108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                   57 |                   39 |           96 |
| 2021-12-28 | 2022-10-14 | 2024-11-29 | -27.55%  |                  290 |                  777 |         1067 |
| 2024-12-09 | 2025-04-08 | 2025-07-03 | -12.22%  |                  120 |                   86 |          206 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                   29 |                   21 |           50 |

**Transformer**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown-Dauer (T) |   Recovery-Dauer (T) |   Gesamt (T) |
|:-----------|:-----------|:-----------|:---------|---------------------:|---------------------:|-------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                   49 |                   43 |           92 |
| 2001-02-02 | 2002-07-23 | 2005-09-09 | -24.02%  |                  536 |                 1144 |         1680 |
| 2007-07-20 | 2008-10-27 | 2008-11-04 | -7.41%   |                  465 |                    8 |          473 |
| 2008-11-05 | 2008-11-20 | 2008-12-16 | -10.78%  |                   15 |                   26 |           41 |
| 2009-01-05 | 2009-02-02 | 2009-07-20 | -8.65%   |                   28 |                  168 |          196 |
| 2010-05-04 | 2010-08-30 | 2011-01-03 | -6.21%   |                  118 |                  126 |          244 |
| 2011-07-25 | 2011-08-08 | 2011-09-07 | -6.59%   |                   14 |                   30 |           44 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                   33 |                  120 |          153 |
| 2015-04-16 | 2016-03-10 | 2017-06-26 | -13.08%  |                  329 |                  473 |          802 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                   10 |                  197 |          207 |
| 2018-08-30 | 2018-12-24 | 2019-03-21 | -11.45%  |                  116 |                   87 |          203 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                   26 |                   82 |          108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                   57 |                   39 |           96 |
| 2021-12-28 | 2022-10-14 | 2024-11-29 | -27.55%  |                  290 |                  777 |         1067 |
| 2024-12-09 | 2025-04-09 | 2025-09-15 | -12.29%  |                  121 |                  159 |          280 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                   29 |                   21 |           50 |

### Krisen-Performance
Return und Max Drawdown während historischer Krisenperioden — der zentrale Nachweis für den Tail-Risk-Schutz der Regime-Switching-Modelle.

| Krise                                | ('Return', 'Buy_Hold')   | ('Return', 'HMM')   | ('Return', 'LSTM')   | ('Return', 'MSM')   | ('Return', 'Transformer')   | ('Max Drawdown', 'Buy_Hold')   | ('Max Drawdown', 'HMM')   | ('Max Drawdown', 'LSTM')   | ('Max Drawdown', 'MSM')   | ('Max Drawdown', 'Transformer')   |
|:-------------------------------------|:-------------------------|:--------------------|:---------------------|:--------------------|:----------------------------|:-------------------------------|:--------------------------|:---------------------------|:--------------------------|:----------------------------------|
| COVID Crash (2020-02 – 2020-03)      | -8.24%                   | -3.60%              | -8.24%               | +5.87%              | -8.24%                      | -18.53%                        | -6.01%                    | -18.53%                    | -1.81%                    | -18.53%                           |
| Dot-Com (2000-03 – 2002-10)          | -14.76%                  | -2.42%              | -14.76%              | -24.71%             | -15.57%                     | -24.79%                        | -7.19%                    | -24.79%                    | -25.86%                   | -24.79%                           |
| EU-Schuldenkrise (2011-07 – 2011-11) | +4.10%                   | -2.40%              | +4.10%               | -1.84%              | +4.52%                      | -7.24%                         | -3.54%                    | -7.24%                     | -8.61%                    | -7.24%                            |
| GFC (2007-10 – 2009-03)              | -25.67%                  | -4.49%              | -14.52%              | -11.15%             | -0.76%                      | -34.77%                        | -7.50%                    | -25.39%                    | -12.74%                   | -13.26%                           |
| Zinsanstieg (2022-01 – 2022-10)      | -24.20%                  | -8.68%              | -24.20%              | -7.74%              | -24.20%                     | -26.98%                        | -8.78%                    | -26.98%                    | -11.72%                   | -26.98%                           |

### Switch-Timing relativ zum Drawdown-Peak
Zeitlicher Abstand zwischen dem ersten Bear-Signal des Modells und dem Drawdown-Trough des Buy & Hold-Portfolios je Krise. Positiv = Modell reagierte frühzeitig, negativ = zu spät.

| Krise   | Modell      | DD-Trough   | 1. Bear-Signal   |   Lead (Tage) |
|:--------|:------------|:------------|:-----------------|--------------:|
| GFC     | MSM         | 2009-03-09  | 2007-10-01       |           525 |
| COVID   | MSM         | 2020-03-18  | 2020-02-24       |            23 |
| 2022    | MSM         | 2022-10-14  | 2022-01-05       |           282 |
| GFC     | HMM         | 2009-03-09  | 2007-10-19       |           507 |
| COVID   | HMM         | 2020-03-18  | 2020-02-27       |            20 |
| 2022    | HMM         | 2022-10-14  | 2022-01-19       |           268 |
| GFC     | LSTM        | 2009-03-09  | 2007-10-01       |           525 |
| COVID   | LSTM        | 2020-03-18  |                  |           nan |
| 2022    | LSTM        | 2022-10-14  |                  |           nan |
| GFC     | Transformer | 2009-03-09  | 2007-12-28       |           437 |
| COVID   | Transformer | 2020-03-18  |                  |           nan |
| 2022    | Transformer | 2022-10-14  |                  |           nan |

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
| MSM         | 249.69%        | 5.03%         | 7.47%         | -28.68%        |           0.7  |            0.87 |           0.18 |              371 | 37.20%                    |          8.91 |
| HMM         | 254.58%        | 5.09%         | 6.09%         | -15.00%        |           0.85 |            0.9  |           0.34 |               71 | 7.20%                     |          3.65 |
| LSTM        | 398.61%        | 6.50%         | 10.86%        | -27.71%        |           0.64 |            0.81 |           0.23 |               16 | 1.60%                     |          7.74 |
| Transformer | 330.55%        | 5.89%         | 10.17%        | -27.71%        |           0.62 |            0.79 |           0.21 |              132 | 13.20%                    |          8.53 |

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
| ('Standard', 'MSM')            | 0.00 €       | Erschöpft (2020) |
| ('Standard', 'HMM')            | 162,926.37 € | Kapitalerhalt    |
| ('Standard', 'LSTM')           | 154,993.93 € | Kapitalerhalt    |
| ('Standard', 'Transformer')    | 57,499.58 €  | Kapitalerhalt    |
| ('Aggressive', 'Buy Hold')     | 0.00 €       | Erschöpft (2012) |
| ('Aggressive', 'MSM')          | 0.00 €       | Erschöpft (2010) |
| ('Aggressive', 'HMM')          | 0.00 €       | Erschöpft (2014) |
| ('Aggressive', 'LSTM')         | 0.00 €       | Erschöpft (2012) |
| ('Aggressive', 'Transformer')  | 0.00 €       | Erschöpft (2011) |
| ('Low_Capital', 'Buy Hold')    | 0.00 €       | Erschöpft (2016) |
| ('Low_Capital', 'MSM')         | 0.00 €       | Erschöpft (2012) |
| ('Low_Capital', 'HMM')         | 0.00 €       | Erschöpft (2019) |
| ('Low_Capital', 'LSTM')        | 0.00 €       | Erschöpft (2016) |
| ('Low_Capital', 'Transformer') | 0.00 €       | Erschöpft (2015) |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 10.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Low_Capital', 'LSTM')        | 0.27%                     | 218,280.88 €        |
| ('Standard', 'Transformer')    | 0.01%                     | 464,701.22 €        |
| ('Aggressive', 'MSM')          | 2.25%                     | 190,664.14 €        |
| ('Standard', 'HMM')            | 0.00%                     | 423,754.26 €        |
| ('Low_Capital', 'Buy Hold')    | 0.62%                     | 213,677.49 €        |
| ('Standard', 'LSTM')           | 0.02%                     | 500,679.90 €        |
| ('Aggressive', 'HMM')          | 0.53%                     | 192,216.25 €        |
| ('Aggressive', 'LSTM')         | 3.73%                     | 248,069.01 €        |
| ('Low_Capital', 'Transformer') | 0.38%                     | 196,158.02 €        |
| ('Standard', 'MSM')            | 0.00%                     | 423,636.54 €        |
| ('Aggressive', 'Transformer')  | 4.06%                     | 223,872.59 €        |
| ('Standard', 'Buy Hold')       | 0.00%                     | 496,792.52 €        |
| ('Aggressive', 'Buy Hold')     | 4.58%                     | 245,081.83 €        |
| ('Low_Capital', 'MSM')         | 0.05%                     | 175,778.42 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 176,790.63 €        |

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
| ('Standard', 'LSTM')           | 0.02%            | 0.01%          | 0.07%         | 2/10000            |
| ('Standard', 'Transformer')    | 0.01%            | 0.00%          | 0.06%         | 1/10000            |
| ('Aggressive', 'Buy_Hold')     | 4.58%            | 4.19%          | 5.01%         | 458/10000          |
| ('Aggressive', 'MSM')          | 2.25%            | 1.98%          | 2.56%         | 225/10000          |
| ('Aggressive', 'HMM')          | 0.53%            | 0.41%          | 0.69%         | 53/10000           |
| ('Aggressive', 'LSTM')         | 3.73%            | 3.38%          | 4.12%         | 373/10000          |
| ('Aggressive', 'Transformer')  | 4.06%            | 3.69%          | 4.46%         | 406/10000          |
| ('Low_Capital', 'Buy_Hold')    | 0.62%            | 0.48%          | 0.79%         | 62/10000           |
| ('Low_Capital', 'MSM')         | 0.05%            | 0.02%          | 0.12%         | 5/10000            |
| ('Low_Capital', 'HMM')         | 0.00%            | 0.00%          | 0.04%         | 0/10000            |
| ('Low_Capital', 'LSTM')        | 0.27%            | 0.19%          | 0.39%         | 27/10000           |
| ('Low_Capital', 'Transformer') | 0.38%            | 0.28%          | 0.52%         | 38/10000           |

### Hypothesentests (gepaarter Wilcoxon, α = 0.05)
**H1 — Regime-Switching reduziert MaxDD vs. Buy & Hold:**

| Modell      | Median MaxDD (Modell)   | Median MaxDD (B&H)   | Δ Median   |   Wilcoxon p | H1 (α=0.05)   |
|:------------|:------------------------|:---------------------|:-----------|-------------:|:--------------|
| MSM         | -63.96%                 | -57.03%              | -6.93 pp   |       1      | abgelehnt     |
| HMM         | -63.02%                 | -57.03%              | -5.99 pp   |       1      | abgelehnt     |
| LSTM        | -56.22%                 | -57.03%              | +0.81 pp   |       0.0635 | abgelehnt     |
| Transformer | -59.54%                 | -57.03%              | -2.51 pp   |       1      | abgelehnt     |

**H2 — Transformer dominiert Ökonometrie und LSTM im Endvermögen:**

| Vergleich            | Median Transformer   | Median MSM   | Δ Median   |   Wilcoxon p | H2 (α=0.05)   | Median HMM   | Median LSTM   |
|:---------------------|:---------------------|:-------------|:-----------|-------------:|:--------------|:-------------|:--------------|
| Transformer vs. MSM  | 223,873 €            | 190,664 €    | +33,208 €  |     2.47e-81 | bestätigt     | nan          | nan           |
| Transformer vs. HMM  | 223,873 €            | nan          | +31,656 €  |     1.95e-97 | bestätigt     | 192,216 €    | nan           |
| Transformer vs. LSTM | 223,873 €            | nan          | -24,196 €  |     1        | abgelehnt     | nan          | 248,069 €     |

### Break-Even-Transaktionskosten
Ab welcher Kostenquote (in Basispunkten pro Umschichtung) verliert das aktive Switching seinen Renditevorteil gegenüber Buy & Hold?

| Modell      |   Final @10bps |   B&H Final |   Break-Even (bps) |
|:------------|---------------:|------------:|-------------------:|
| MSM         |          3.485 |       4.828 |                  5 |
| HMM         |          3.533 |       4.828 |                  0 |
| LSTM        |          4.969 |       4.828 |                 30 |
| Transformer |          4.29  |       4.828 |                  5 |

![Break-Even-Analyse](../assets/break_even_costs.png)

### Entnahmeraten-Sensitivität (3.5 % / 4 % / 5 %)
Robustheit der SORR-Ergebnisse bei variierenden jährlichen Entnahmen.

| Strategie   | ('Endkapital', '3.5%')   | ('Endkapital', '4.0%')   | ('Endkapital', '5.0%')   | ('Status', '3.5%')   | ('Status', '4.0%')   | ('Status', '5.0%')   |
|:------------|:-------------------------|:-------------------------|:-------------------------|:---------------------|:---------------------|:---------------------|
| Buy_Hold    | 1,091,710 €              | 901,582 €                | 521,326 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| HMM         | 833,742 €                | 699,579 €                | 431,253 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| LSTM        | 1,129,174 €              | 934,338 €                | 544,666 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| MSM         | 606,962 €                | 443,893 €                | 117,755 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| Transformer | 930,513 €                | 755,911 €                | 406,705 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |

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

**Zuletzt aktualisiert:** 22.04.2026 06:34<br>
**End date:** `2026-04-17`<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Walk-Forward-Validierung:** AKTIV (Modus: rolling, Train: 10J, Test: 12M, Step: 12M)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
