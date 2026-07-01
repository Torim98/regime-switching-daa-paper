
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag (2026-07-01) und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   | Final Wealth   | Total Return   | Max Drawdown   |
|:------------|:---------------|:---------------|:---------------|
| Buy_Hold    | 2,108,475 €    | +321.69%       | -35.08%        |
| MSM         | 1,593,965 €    | +218.79%       | -28.88%        |
| HMM         | 2,262,437 €    | +352.49%       | -15.00%        |
| LSTM        | 2,149,975 €    | +330.00%       | -27.71%        |
| Transformer | 1,904,269 €    | +280.85%       | -29.25%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |     Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|--------:|-----------------:|-----------:|
| Returns_GSPC  |             0.00033  |           0.011375 | -0.1277 |  0.1096 |          -0.3616 |    10.8099 |
| Returns_VUSTX |             0.000216 |           0.007267 | -0.0605 |  0.0654 |          -0.0328 |     4.4871 |
| Returns       |             0.000284 |           0.006896 | -0.0662 |  0.0584 |          -0.2777 |     7.6158 |
| VIX           |            19.4579   |           7.74766  |  9.14   | 82.69   |           2.2085 |     8.7338 |
| TNX_10Y       |             4.2386   |           1.9262   |  0.499  |  9.09   |           0.3297 |    -0.6249 |
| IRX_3M        |             2.70845  |           2.19762  | -0.105  |  7.99   |           0.1958 |    -1.2511 |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.5514 | 4.1409e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -17.9428 | 2.8566e-30 |           -2.8619 | Ja           |
| Returns       |        -17.5538 | 4.1298e-30 |           -2.8619 | Ja           |
| VIX           |         -7.2922 | 1.4075e-10 |           -2.8619 | Ja           |
| TNX_10Y       |         -2.3569 | 0.15426    |           -2.8619 | Nein         |
| IRX_3M        |         -2.3378 | 0.16006    |           -2.8619 | Nein         |

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
| Buy_Hold    | +5.80% | 11.18%             |          0.56  |           0.731 | -35.08%        |          0.165 |       6452 |        25.6 |
| MSM         | +4.65% | 7.33%              |          0.657 |           0.817 | -28.88%        |          0.161 |       6452 |        25.6 |
| HMM         | +6.09% | 6.69%              |          0.917 |           1.049 | -15.00%        |          0.406 |       6452 |        25.6 |
| LSTM        | +5.88% | 10.80%             |          0.583 |           0.731 | -27.71%        |          0.212 |       6452 |        25.6 |
| Transformer | +5.38% | 10.83%             |          0.538 |           0.675 | -29.25%        |          0.184 |       6452 |        25.6 |

### Klassifikationsmetriken (vs. NBER-Rezessionen als Ground Truth)
Vergleich der Modelle als binäre Rezessionsklassifikatoren (Precision, Recall, F1).

| Modell      |   Precision |   Recall |    F1 |   TN |   FP |   FN |   TP |
|:------------|------------:|---------:|------:|-----:|-----:|-----:|-----:|
| MSM         |       0.302 |    0.761 | 0.433 | 4837 | 1030 |  140 |  446 |
| HMM         |       0.233 |    0.845 | 0.366 | 4240 | 1627 |   91 |  495 |
| LSTM        |       0.166 |    0.172 | 0.169 | 5358 |  509 |  485 |  101 |
| Transformer |       0.431 |    0.311 | 0.361 | 5627 |  240 |  404 |  182 |

![Confusion Matrices](../assets/confusion_matrices.png)

**ROC- und Precision-Recall-Kurven** (schwellenunabhängiger Vergleich über `*_Prob`):

![ROC-Kurven](../assets/roc_curves.png)
![PR-Kurven](../assets/pr_curves.png)

### Signal-Churning & Whipsaw-Analyse
Quantifizierung der Wechselhäufigkeit und Anteil sehr kurzer Regime-Phasen („Whipsaws").

| Modell      |   Signalwechsel |   Whipsaws (<5T) | Whipsaw-Anteil   |   Ø Phase (Tage) |   Median Phase (Tage) | Kumul. Kosten   |
|:------------|----------------:|-----------------:|:-----------------|-----------------:|----------------------:|:----------------|
| MSM         |             371 |              210 | 56.5%            |             17.3 |                     3 | 37.10%          |
| HMM         |              91 |               29 | 31.5%            |             70.1 |                    19 | 9.10%           |
| LSTM        |              10 |                0 | 0.0%             |            586.6 |                    96 | 1.00%           |
| Transformer |              38 |               19 | 48.7%            |            165.5 |                     6 | 3.80%           |

### Regime-Wahrscheinlichkeits-Heatmap
Zeitverlauf der Bear-Wahrscheinlichkeiten aller Modelle.

![Regime Probability Heatmap](../assets/regime_probability_heatmap.png)

### Threshold-Sensitivität
Variation der Entscheidungs-Schwelle pro Modell. Zeigt, wie robust Final Wealth, Max Drawdown und Anzahl der Regime-Wechsel gegenüber einer veränderten Bull/Bear-Klassifikations-Grenze sind (Kap. 4.1 — Glättung).

**MSM**

|   Threshold | Final Wealth   | Max Drawdown   |   Wechsel |
|------------:|:---------------|:---------------|----------:|
|        0.25 | 1,525,528 €    | -14.84%        |       303 |
|        0.3  | 1,425,461 €    | -16.45%        |       315 |
|        0.35 | 1,410,094 €    | -19.99%        |       303 |
|        0.4  | 1,406,310 €    | -22.85%        |       311 |
|        0.5  | 1,496,576 €    | -23.39%        |       311 |

**HMM**

|   Threshold | Final Wealth   | Max Drawdown   |   Wechsel |
|------------:|:---------------|:---------------|----------:|
|        0.4  | 2,296,481 €    | -15.26%        |        79 |
|        0.45 | 2,291,461 €    | -17.07%        |        65 |
|        0.5  | 2,374,286 €    | -17.05%        |        61 |
|        0.55 | 2,381,224 €    | -17.77%        |        65 |
|        0.6  | 2,313,842 €    | -19.04%        |        75 |

**LSTM**

|   Threshold | Final Wealth   | Max Drawdown   |   Wechsel |
|------------:|:---------------|:---------------|----------:|
|         0.2 | 2,140,900 €    | -27.71%        |        10 |
|         0.3 | 2,149,975 €    | -27.71%        |        10 |
|         0.4 | 2,165,785 €    | -27.71%        |        10 |
|         0.5 | 2,143,347 €    | -27.71%        |        12 |

**Transformer**

|   Threshold | Final Wealth   | Max Drawdown   |   Wechsel |
|------------:|:---------------|:---------------|----------:|
|        0.3  | 1,573,580 €    | -28.46%        |        88 |
|        0.4  | 1,687,279 €    | -27.71%        |        76 |
|        0.45 | 1,800,753 €    | -27.71%        |        66 |
|        0.5  | 1,832,741 €    | -29.81%        |        62 |
|        0.6  | 1,970,019 €    | -29.36%        |        40 |

### Time-to-Recovery
Alle Drawdown-Phasen jenseits der Mindesttiefe (gemäß `extended.ttr_min_dd`) mit Peak-, Trough- und Recovery-Datum sowie Dauer in Handelstagen. Eine offene (noch nicht erholte) Phase wird im Recovery-Feld mit „—" markiert.

**Buy_Hold**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown-Dauer (T) |   Recovery-Dauer (T) |   Gesamt (T) |
|:-----------|:-----------|:-----------|:---------|---------------------:|---------------------:|-------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                   49 |                   43 |           92 |
| 2001-02-02 | 2002-07-23 | 2004-03-05 | -24.04%  |                  536 |                  591 |         1127 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                   63 |                  178 |          241 |
| 2007-11-01 | 2009-03-09 | 2011-04-28 | -34.97%  |                  494 |                  780 |         1274 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                   14 |                   67 |           81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                   33 |                  120 |          153 |
| 2015-03-23 | 2015-08-25 | 2016-04-13 | -8.39%   |                  155 |                  232 |          387 |
| 2016-08-01 | 2016-11-14 | 2017-04-17 | -5.64%   |                  105 |                  154 |          259 |
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
| 2001-02-02 | 2003-04-11 | 2007-02-15 | -28.15%  |                  798 |                 1406 |         2204 |
| 2007-06-05 | 2009-06-23 | 2010-04-21 | -14.10%  |                  749 |                  302 |         1051 |
| 2010-05-04 | 2010-07-02 | 2010-09-14 | -5.86%   |                   59 |                   74 |          133 |
| 2011-07-25 | 2011-10-03 | 2012-01-11 | -7.98%   |                   70 |                  100 |          170 |
| 2015-02-26 | 2016-01-20 | 2016-03-18 | -6.77%   |                  328 |                   58 |          386 |
| 2016-08-01 | 2016-11-14 | 2017-04-18 | -5.84%   |                  105 |                  155 |          260 |
| 2018-01-29 | 2018-04-25 | 2018-08-20 | -5.80%   |                   86 |                  117 |          203 |
| 2018-08-30 | 2019-01-14 | 2019-04-30 | -8.53%   |                  137 |                  106 |          243 |
| 2020-09-03 | 2021-03-24 | 2021-08-23 | -9.84%   |                  202 |                  152 |          354 |
| 2021-12-28 | 2022-04-06 | 2023-04-06 | -10.67%  |                   99 |                  365 |          464 |
| 2023-07-20 | 2023-10-19 | 2024-06-05 | -10.89%  |                   91 |                  230 |          321 |
| 2024-12-09 | 2025-01-13 | 2025-09-05 | -6.46%   |                   35 |                  235 |          270 |
| 2025-10-29 | 2026-03-19 | 2026-05-29 | -6.25%   |                  141 |                   71 |          212 |

**HMM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown-Dauer (T) |   Recovery-Dauer (T) |   Gesamt (T) |
|:-----------|:-----------|:-----------|:---------|---------------------:|---------------------:|-------------:|
| 2002-03-07 | 2002-06-12 | 2003-12-17 | -6.70%   |                   97 |                  553 |          650 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                   63 |                  178 |          241 |
| 2008-05-20 | 2008-07-28 | 2009-08-20 | -7.18%   |                   69 |                  388 |          457 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                   33 |                  120 |          153 |
| 2015-12-02 | 2016-01-20 | 2016-03-17 | -5.41%   |                   49 |                   57 |          106 |
| 2016-08-01 | 2016-11-14 | 2017-04-17 | -5.64%   |                  105 |                  154 |          259 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                   10 |                  197 |          207 |
| 2018-08-30 | 2018-10-11 | 2019-11-26 | -5.40%   |                   42 |                  411 |          453 |
| 2020-02-21 | 2020-07-24 | 2021-02-08 | -5.89%   |                  154 |                  199 |          353 |
| 2021-02-11 | 2021-03-18 | 2021-05-06 | -5.63%   |                   35 |                   49 |           84 |
| 2021-11-10 | 2023-10-27 | 2024-02-01 | -14.75%  |                  716 |                   97 |          813 |
| 2024-12-09 | 2025-04-08 | 2025-07-03 | -11.80%  |                  120 |                   86 |          206 |
| 2025-10-29 | 2026-03-20 | 2026-04-14 | -5.17%   |                  142 |                   25 |          167 |

**LSTM**

| Peak       | Trough     | Recovery   | Max DD   |   Drawdown-Dauer (T) |   Recovery-Dauer (T) |   Gesamt (T) |
|:-----------|:-----------|:-----------|:---------|---------------------:|---------------------:|-------------:|
| 2000-11-01 | 2000-12-20 | 2001-02-01 | -5.10%   |                   49 |                   43 |           92 |
| 2001-02-02 | 2002-07-23 | 2004-03-05 | -24.04%  |                  536 |                  591 |         1127 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                   63 |                  178 |          241 |
| 2007-11-01 | 2009-03-09 | 2010-01-14 | -25.77%  |                  494 |                  311 |          805 |
| 2010-05-04 | 2010-07-02 | 2010-09-13 | -5.36%   |                   59 |                   73 |          132 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                   14 |                   67 |           81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                   33 |                  120 |          153 |
| 2015-03-23 | 2016-01-20 | 2016-06-30 | -10.14%  |                  303 |                  162 |          465 |
| 2016-08-01 | 2016-11-14 | 2017-04-17 | -5.64%   |                  105 |                  154 |          259 |
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
| 2001-02-02 | 2002-07-23 | 2004-03-05 | -24.04%  |                  536 |                  591 |         1127 |
| 2004-03-08 | 2004-05-10 | 2004-11-04 | -6.37%   |                   63 |                  178 |          241 |
| 2007-07-20 | 2009-03-09 | 2010-09-14 | -29.10%  |                  598 |                  554 |         1152 |
| 2011-07-25 | 2011-08-08 | 2011-10-14 | -6.59%   |                   14 |                   67 |           81 |
| 2013-05-22 | 2013-06-24 | 2013-10-22 | -5.37%   |                   33 |                  120 |          153 |
| 2015-03-23 | 2016-01-05 | 2017-06-14 | -11.19%  |                  288 |                  526 |          814 |
| 2018-01-29 | 2018-02-08 | 2018-08-24 | -6.93%   |                   10 |                  197 |          207 |
| 2018-08-30 | 2018-12-24 | 2019-03-21 | -11.45%  |                  116 |                   87 |          203 |
| 2020-02-21 | 2020-03-18 | 2020-06-08 | -18.31%  |                   26 |                   82 |          108 |
| 2020-09-03 | 2020-10-30 | 2020-12-08 | -5.20%   |                   57 |                   39 |           96 |
| 2021-12-28 | 2022-10-14 | 2024-11-29 | -27.55%  |                  290 |                  777 |         1067 |
| 2024-12-09 | 2025-04-08 | 2025-08-08 | -12.22%  |                  120 |                  122 |          242 |
| 2026-02-26 | 2026-03-27 | 2026-04-17 | -6.69%   |                   29 |                   21 |           50 |

### Krisen-Performance
Return und Max Drawdown während historischer Krisenperioden — der zentrale Nachweis für den Tail-Risk-Schutz der Regime-Switching-Modelle.

| Krise                                | ('Return', 'Buy_Hold')   | ('Return', 'HMM')   | ('Return', 'LSTM')   | ('Return', 'MSM')   | ('Return', 'Transformer')   | ('Max Drawdown', 'Buy_Hold')   | ('Max Drawdown', 'HMM')   | ('Max Drawdown', 'LSTM')   | ('Max Drawdown', 'MSM')   | ('Max Drawdown', 'Transformer')   |
|:-------------------------------------|:-------------------------|:--------------------|:---------------------|:--------------------|:----------------------------|:-------------------------------|:--------------------------|:---------------------------|:--------------------------|:----------------------------------|
| COVID Crash (2020-02 – 2020-03)      | -8.55%                   | -3.60%              | -8.55%               | +5.87%              | -8.55%                      | -18.53%                        | -6.01%                    | -18.53%                    | -1.81%                    | -18.53%                           |
| Dot-Com (2000-03 – 2002-10)          | -14.78%                  | -2.45%              | -14.78%              | -24.50%             | -14.78%                     | -24.81%                        | -7.22%                    | -24.81%                    | -27.01%                   | -24.81%                           |
| EU-Schuldenkrise (2011-07 – 2011-11) | +4.10%                   | -2.40%              | +4.10%               | -1.84%              | +4.10%                      | -7.24%                         | -3.54%                    | -7.24%                     | -8.61%                    | -7.24%                            |
| GFC (2007-10 – 2009-03)              | -26.99%                  | -3.39%              | -16.44%              | -11.28%             | -19.17%                     | -35.08%                        | -7.50%                    | -25.90%                    | -12.86%                   | -28.13%                           |
| Zinsanstieg (2022-01 – 2022-10)      | -24.20%                  | -8.68%              | -24.20%              | -4.59%              | -24.20%                     | -26.98%                        | -8.78%                    | -26.98%                    | -9.96%                    | -26.98%                           |

### Switch-Timing relativ zum Drawdown-Peak
Zeitlicher Abstand zwischen dem ersten Bear-Signal des Modells und dem Drawdown-Trough des Buy & Hold-Portfolios je Krise. Positiv = Modell reagierte frühzeitig, negativ = zu spät.

| Krise   | Modell      | DD-Trough   | 1. Bear-Signal   |   Lead (Tage) |
|:--------|:------------|:------------|:-----------------|--------------:|
| GFC     | MSM         | 2009-03-09  | 2007-10-01       |           525 |
| COVID   | MSM         | 2020-03-18  | 2020-02-24       |            23 |
| 2022    | MSM         | 2022-10-14  | 2022-01-05       |           282 |
| GFC     | HMM         | 2009-03-09  | 2007-10-18       |           508 |
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
| Buy Hold    | 323.17%        | 5.78%         | 11.18%        | -35.08%        |           0.56 |            0.73 |           0.16 |                0 | 0.00%                     |          9.07 |
| MSM         | 219.91%        | 4.63%         | 7.33%         | -28.88%        |           0.66 |            0.82 |           0.16 |              371 | 37.20%                    |          9.39 |
| HMM         | 354.07%        | 6.07%         | 6.69%         | -15.00%        |           0.92 |            1.05 |           0.4  |               91 | 9.20%                     |          3.63 |
| LSTM        | 331.50%        | 5.86%         | 10.80%        | -27.71%        |           0.58 |            0.73 |           0.21 |               10 | 1.00%                     |          7.78 |
| Transformer | 282.19%        | 5.36%         | 10.83%        | -29.25%        |           0.54 |            0.68 |           0.18 |               38 | 3.80%                     |          8.28 |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status           |
|:-------------------------------|:-------------|:-----------------|
| ('Standard', 'Buy Hold')       | 13,947.46 €  | Kapitalerhalt    |
| ('Standard', 'MSM')            | 0.00 €       | Erschöpft (2018) |
| ('Standard', 'HMM')            | 302,869.78 € | Kapitalerhalt    |
| ('Standard', 'LSTM')           | 43,654.51 €  | Kapitalerhalt    |
| ('Standard', 'Transformer')    | 0.00 €       | Erschöpft (2025) |
| ('Aggressive', 'Buy Hold')     | 0.00 €       | Erschöpft (2011) |
| ('Aggressive', 'MSM')          | 0.00 €       | Erschöpft (2009) |
| ('Aggressive', 'HMM')          | 0.00 €       | Erschöpft (2014) |
| ('Aggressive', 'LSTM')         | 0.00 €       | Erschöpft (2011) |
| ('Aggressive', 'Transformer')  | 0.00 €       | Erschöpft (2011) |
| ('Low_Capital', 'Buy Hold')    | 0.00 €       | Erschöpft (2015) |
| ('Low_Capital', 'MSM')         | 0.00 €       | Erschöpft (2012) |
| ('Low_Capital', 'HMM')         | 0.00 €       | Erschöpft (2019) |
| ('Low_Capital', 'LSTM')        | 0.00 €       | Erschöpft (2015) |
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
| ('Aggressive', 'Buy Hold')     | 5.83%                     | 227,719.08 €        |
| ('Aggressive', 'Transformer')  | 6.14%                     | 208,699.58 €        |
| ('Low_Capital', 'Buy Hold')    | 0.70%                     | 198,223.30 €        |
| ('Standard', 'MSM')            | 0.00%                     | 413,304.46 €        |
| ('Low_Capital', 'Transformer') | 0.61%                     | 188,018.76 €        |
| ('Standard', 'Buy Hold')       | 0.00%                     | 472,038.98 €        |
| ('Aggressive', 'LSTM')         | 4.23%                     | 232,362.74 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 477,018.60 €        |
| ('Low_Capital', 'MSM')         | 0.05%                     | 170,351.55 €        |
| ('Aggressive', 'HMM')          | 0.19%                     | 247,824.76 €        |
| ('Standard', 'HMM')            | 0.00%                     | 493,848.91 €        |
| ('Low_Capital', 'LSTM')        | 0.50%                     | 201,765.54 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 212,787.47 €        |
| ('Aggressive', 'MSM')          | 2.29%                     | 180,225.34 €        |
| ('Standard', 'Transformer')    | 0.01%                     | 446,879.76 €        |

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
| ('Standard', 'LSTM')           | 0.00%            | 0.00%          | 0.04%         | 0/10000            |
| ('Standard', 'Transformer')    | 0.01%            | 0.00%          | 0.06%         | 1/10000            |
| ('Aggressive', 'Buy_Hold')     | 5.83%            | 5.39%          | 6.31%         | 583/10000          |
| ('Aggressive', 'MSM')          | 2.29%            | 2.01%          | 2.60%         | 229/10000          |
| ('Aggressive', 'HMM')          | 0.19%            | 0.12%          | 0.30%         | 19/10000           |
| ('Aggressive', 'LSTM')         | 4.23%            | 3.85%          | 4.64%         | 423/10000          |
| ('Aggressive', 'Transformer')  | 6.14%            | 5.69%          | 6.63%         | 614/10000          |
| ('Low_Capital', 'Buy_Hold')    | 0.70%            | 0.55%          | 0.88%         | 70/10000           |
| ('Low_Capital', 'MSM')         | 0.05%            | 0.02%          | 0.12%         | 5/10000            |
| ('Low_Capital', 'HMM')         | 0.00%            | 0.00%          | 0.04%         | 0/10000            |
| ('Low_Capital', 'LSTM')        | 0.50%            | 0.38%          | 0.66%         | 50/10000           |
| ('Low_Capital', 'Transformer') | 0.61%            | 0.48%          | 0.78%         | 61/10000           |

### Hypothesentests (gepaarter Wilcoxon, α = 0.05)
**H1 — Regime-Switching reduziert MaxDD vs. Buy & Hold:**

| Modell      | Median MaxDD (Modell)   | Median MaxDD (B&H)   | Δ Median   |   Wilcoxon p | H1 (α=0.05)   |
|:------------|:------------------------|:---------------------|:-----------|-------------:|:--------------|
| MSM         | -65.68%                 | -59.54%              | -6.14 pp   |    1         | abgelehnt     |
| HMM         | -52.99%                 | -59.54%              | +6.55 pp   |    6.62e-115 | bestätigt     |
| LSTM        | -58.69%                 | -59.54%              | +0.85 pp   |    0.00181   | bestätigt     |
| Transformer | -62.27%                 | -59.54%              | -2.73 pp   |    1         | abgelehnt     |

**H2 — Transformer dominiert Ökonometrie und LSTM im Endvermögen:**

| Vergleich            | Median Transformer   | Median MSM   | Δ Median   |   Wilcoxon p | H2 (α=0.05)   | Median HMM   | Median LSTM   |
|:---------------------|:---------------------|:-------------|:-----------|-------------:|:--------------|:-------------|:--------------|
| Transformer vs. MSM  | 208,700 €            | 180,225 €    | +28,474 €  |     1.18e-59 | bestätigt     | nan          | nan           |
| Transformer vs. HMM  | 208,700 €            | nan          | -39,125 €  |     1        | abgelehnt     | 247,825 €    | nan           |
| Transformer vs. LSTM | 208,700 €            | nan          | -23,663 €  |     1        | abgelehnt     | nan          | 232,363 €     |

### Break-Even-Transaktionskosten
Ab welcher Kostenquote (in Basispunkten pro Umschichtung) verliert das aktive Switching seinen Renditevorteil gegenüber Buy & Hold?

| Modell      |   Final @10bps |   B&H Final |   Break-Even (bps) |
|:------------|---------------:|------------:|-------------------:|
| MSM         |          3.188 |       4.217 |                  5 |
| HMM         |          4.525 |       4.217 |                 20 |
| LSTM        |          4.3   |       4.217 |                 30 |
| Transformer |          3.809 |       4.217 |                  0 |

![Break-Even-Analyse](../assets/break_even_costs.png)

### Entnahmeraten-Sensitivität (3.5 % / 4 % / 5 %)
Robustheit der SORR-Ergebnisse bei variierenden jährlichen Entnahmen.

| Strategie   | ('Endkapital', '3.5%')   | ('Endkapital', '4.0%')   | ('Endkapital', '5.0%')   | ('Status', '3.5%')   | ('Status', '4.0%')   | ('Status', '5.0%')   |
|:------------|:-------------------------|:-------------------------|:-------------------------|:---------------------|:---------------------|:---------------------|
| Buy_Hold    | 889,743 €                | 714,584 €                | 364,266 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| HMM         | 1,122,656 €              | 958,699 €                | 630,784 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| LSTM        | 924,424 €                | 748,270 €                | 395,962 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| MSM         | 498,691 €                | 341,426 €                | 26,896 €                 | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| Transformer | 774,350 €                | 611,981 €                | 287,242 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |

---

## Forschungsnotizen & Methodik
- **Cash-Komponente:** Bei einem "Bear"-Signal schichtet die Strategie in den aktuellen Geldmarktzins (**^IRX**) um.
- **Vermeidung von Look-ahead Bias:** Alle Signale werden für das Backtesting um einen Tag zeitversetzt (`shift(1)`), um reale Handelsbedingungen zu simulieren.
- **Feature-Set:** Die Modelle nutzen Renditen, Volatilität (20d), SMA-Abstand, Momentum, VIX und Yield Spread.
- **Kostensimulation:** Es wird eine pauschale Gebühr von 10 Basispunkten (0,1%) pro Umschichtung berechnet.
- **SORR-Spezifika:** Bei Entnahmen in "Bull"-Phasen wird eine zusätzliche Liquiditätsgebühr von 0,1% auf den Entnahmebetrag erhoben (Asset-Verkäufe). In "Bear"-Phasen (Cash) entfällt diese.

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

**Zuletzt aktualisiert:** 01.07.2026 07:10<br>
**End date:** `2026-07-01`<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Walk-Forward-Validierung:** AKTIV (Modus: rolling, Train: 10J, Test: 12M, Step: 12M)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch den Backtest-Service (Reporting).*
