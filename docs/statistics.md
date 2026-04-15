
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         4.7659 | +376.59%       | -34.77%        |
| MSM         |         2.9286 | +192.86%       | -24.92%        |
| HMM         |         2.9975 | +199.75%       | -23.55%        |
| LSTM        |         4.3734 | +337.34%       | -27.71%        |
| Transformer |         4.5012 | +350.12%       | -27.71%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |    Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|-------:|-----------------:|-----------:|
| Returns_GSPC  |             0.000326 |           0.01139  | -0.1277 | 0.1096 |          -0.3602 |    10.8106 |
| Returns_VUSTX |             0.000275 |           0.007485 | -0.0605 | 0.1296 |           0.6392 |    14.3747 |
| Returns       |             0.000305 |           0.006935 | -0.0662 | 0.0584 |          -0.2266 |     7.7455 |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.4874 | 4.4557e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -18.1867 | 2.4252e-30 |           -2.8619 | Ja           |
| Returns       |        -17.5106 | 4.3373e-30 |           -2.8619 | Ja           |

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

### F. Label-Konkordanz (Auswahl der Trainings-Labels)
Vergleich der Regime-Labeler (MSM, HMM, Pagan-Sossounov, Peak-to-Trough, Lunde-Timmermann, NBER) zur Begründung der Label-Wahl für die Supervised-Modelle. Pagan-Sossounov wurde aufgrund seiner hohen Konkordanz mit NBER-Rezessionsperioden als Trainingsziel für LSTM und Transformer gewählt.

![Label Concordance](../assets/label_concordance_matrix.png)
![Label Cohen's κ](../assets/label_kappa_matrix.png)
![Label Timeline](../assets/label_timeline_comparison.png)

---

## 4. Backtesting & Strategie-Evaluation
Die ökonomische Anwendung der Regime-Signale durch dynamische Umschichtung in den Geldmarkt.

### Equity Curves im Vergleich
![Equity Curves](../assets/equity_curves.png)

### Annualisierte Performance-Metriken
Normalisierte Kennzahlen (CAGR, Sharpe, Sortino, Calmar) für den Vergleich über unterschiedlich lange Evaluationszeiträume.

| Strategie   | CAGR   | Ann. Volatilität   |   Sharpe Ratio |   Sortino Ratio | Max Drawdown   |   Calmar Ratio |   OOS-Tage |   OOS-Jahre |
|:------------|:-------|:-------------------|---------------:|----------------:|:---------------|---------------:|-----------:|------------:|
| Buy_Hold    | +6.35% | 11.24%             |          0.566 |           0.731 | -34.77%        |          0.183 |       6401 |        25.4 |
| MSM         | +4.34% | 6.61%              |          0.656 |           0.764 | -24.92%        |          0.174 |       6401 |        25.4 |
| HMM         | +4.43% | 5.96%              |          0.744 |           0.668 | -23.55%        |          0.188 |       6401 |        25.4 |
| LSTM        | +6.00% | 10.72%             |          0.56  |           0.679 | -27.71%        |          0.216 |       6401 |        25.4 |
| Transformer | +6.12% | 10.37%             |          0.59  |           0.72  | -27.71%        |          0.221 |       6401 |        25.4 |

### Klassifikationsmetriken (vs. NBER-Rezessionen als Ground Truth)
Vergleich der Modelle als binäre Rezessionsklassifikatoren (Precision, Recall, F1).

| Modell      |   Precision |   Recall |    F1 |   TN |   FP |   FN |   TP |
|:------------|------------:|---------:|------:|-----:|-----:|-----:|-----:|
| MSM         |       0.265 |    0.877 | 0.407 | 4388 | 1428 |   72 |  514 |
| HMM         |       0.13  |    0.693 | 0.219 | 3096 | 2720 |  180 |  406 |
| LSTM        |       0.14  |    0.198 | 0.164 | 5106 |  710 |  470 |  116 |
| Transformer |       0.237 |    0.346 | 0.282 | 5163 |  653 |  383 |  203 |

![Confusion Matrices](../assets/confusion_matrices.png)

**ROC- und Precision-Recall-Kurven** (schwellenunabhängiger Vergleich über `*_Prob`):

![ROC-Kurven](../assets/roc_curves.png)
![PR-Kurven](../assets/pr_curves.png)

### Signal-Churning & Whipsaw-Analyse
Quantifizierung der Wechselhäufigkeit und Anteil sehr kurzer Regime-Phasen („Whipsaws").

| Modell      |   Signalwechsel |   Whipsaws (<5T) | Whipsaw-Anteil   |   Ø Phase (Tage) |   Median Phase (Tage) | Kumul. Kosten   |
|:------------|----------------:|-----------------:|:-----------------|-----------------:|----------------------:|:----------------|
| MSM         |             314 |              161 | 51.1%            |             20.3 |                     4 | 31.40%          |
| HMM         |              94 |               16 | 16.8%            |             67.4 |                    25 | 9.40%           |
| LSTM        |              29 |                2 | 6.7%             |            213.4 |                    18 | 2.90%           |
| Transformer |              34 |               20 | 57.1%            |            182.9 |                     2 | 3.40%           |

### Regime-Wahrscheinlichkeits-Heatmap
Zeitverlauf der Bear-Wahrscheinlichkeiten aller Modelle.

![Regime Probability Heatmap](../assets/regime_probability_heatmap.png)

### Krisen-Performance
Return und Max Drawdown während historischer Krisenperioden — der zentrale Nachweis für den Tail-Risk-Schutz der Regime-Switching-Modelle.

| Krise                                | ('Return', 'Buy_Hold')   | ('Return', 'HMM')   | ('Return', 'LSTM')   | ('Return', 'MSM')   | ('Return', 'Transformer')   | ('Max Drawdown', 'Buy_Hold')   | ('Max Drawdown', 'HMM')   | ('Max Drawdown', 'LSTM')   | ('Max Drawdown', 'MSM')   | ('Max Drawdown', 'Transformer')   |
|:-------------------------------------|:-------------------------|:--------------------|:---------------------|:--------------------|:----------------------------|:-------------------------------|:--------------------------|:---------------------------|:--------------------------|:----------------------------------|
| COVID Crash (2020-02 – 2020-03)      | -8.24%                   | +2.21%              | -8.24%               | +0.73%              | -8.24%                      | -18.53%                        | -0.36%                    | -18.53%                    | -1.81%                    | -18.53%                           |
| Dot-Com (2000-03 – 2002-10)          | -14.76%                  | +0.31%              | -14.76%              | -18.18%             | -14.76%                     | -24.79%                        | -6.36%                    | -24.79%                    | -21.03%                   | -24.79%                           |
| EU-Schuldenkrise (2011-07 – 2011-11) | +4.10%                   | +0.01%              | +4.10%               | -1.37%              | +4.10%                      | -7.24%                         | 0.00%                     | -7.24%                     | -8.17%                    | -7.24%                            |
| GFC (2007-10 – 2009-03)              | -25.67%                  | -16.95%             | -11.79%              | -8.75%              | -11.73%                     | -34.77%                        | -23.55%                   | -23.89%                    | -9.57%                    | -22.43%                           |
| Zinsanstieg (2022-01 – 2022-10)      | -24.20%                  | -1.71%              | -24.20%              | -4.21%              | -24.20%                     | -26.98%                        | -1.82%                    | -26.98%                    | -6.55%                    | -26.98%                           |

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
| GFC     | LSTM        | 2009-03-09  | 2007-10-02       |           524 |
| COVID   | LSTM        | 2020-03-18  |                  |           nan |
| 2022    | LSTM        | 2022-10-14  |                  |           nan |
| GFC     | Transformer | 2009-03-09  | 2007-10-16       |           510 |
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
| Buy Hold    | 378.26%        | 6.33%         | 11.23%        | -34.77%        |           0.6  |            0.79 |           0.18 |                0 | 0.00%                     |          8.88 |
| MSM         | 193.89%        | 4.32%         | 6.61%         | -24.92%        |           0.68 |            0.79 |           0.17 |              314 | 31.50%                    |          7.89 |
| HMM         | 200.80%        | 4.42%         | 5.95%         | -23.55%        |           0.76 |            0.69 |           0.19 |               94 | 9.50%                     |          8.58 |
| LSTM        | 338.87%        | 5.97%         | 10.71%        | -27.71%        |           0.6  |            0.73 |           0.22 |               29 | 2.90%                     |          7.73 |
| Transformer | 351.69%        | 6.09%         | 10.37%        | -27.71%        |           0.62 |            0.77 |           0.22 |               34 | 3.40%                     |          7.86 |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status           |
|:-------------------------------|:-------------|:-----------------|
| ('Standard', 'Buy Hold')       | 139,246.20 € | Kapitalerhalt    |
| ('Standard', 'MSM')            | 0.00 €       | Erschöpft (2021) |
| ('Standard', 'HMM')            | 0.00 €       | Erschöpft (2021) |
| ('Standard', 'LSTM')           | 113,328.35 € | Kapitalerhalt    |
| ('Standard', 'Transformer')    | 139,508.77 € | Kapitalerhalt    |
| ('Aggressive', 'Buy Hold')     | 0.00 €       | Erschöpft (2012) |
| ('Aggressive', 'MSM')          | 0.00 €       | Erschöpft (2010) |
| ('Aggressive', 'HMM')          | 0.00 €       | Erschöpft (2011) |
| ('Aggressive', 'LSTM')         | 0.00 €       | Erschöpft (2011) |
| ('Aggressive', 'Transformer')  | 0.00 €       | Erschöpft (2012) |
| ('Low_Capital', 'Buy Hold')    | 0.00 €       | Erschöpft (2016) |
| ('Low_Capital', 'MSM')         | 0.00 €       | Erschöpft (2013) |
| ('Low_Capital', 'HMM')         | 0.00 €       | Erschöpft (2014) |
| ('Low_Capital', 'LSTM')        | 0.00 €       | Erschöpft (2016) |
| ('Low_Capital', 'Transformer') | 0.00 €       | Erschöpft (2016) |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 497,471.48 €        |
| ('Standard', 'MSM')            | 0.00%                     | 387,448.46 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 480,481.09 €        |
| ('Standard', 'Transformer')    | 0.01%                     | 478,518.03 €        |
| ('Aggressive', 'Buy Hold')     | 4.41%                     | 244,180.29 €        |
| ('Aggressive', 'MSM')          | 1.98%                     | 160,416.93 €        |
| ('Aggressive', 'LSTM')         | 4.02%                     | 232,260.71 €        |
| ('Standard', 'HMM')            | 0.00%                     | 398,099.22 €        |
| ('Low_Capital', 'Buy Hold')    | 0.50%                     | 213,041.96 €        |
| ('Low_Capital', 'LSTM')        | 0.34%                     | 209,392.88 €        |
| ('Aggressive', 'Transformer')  | 4.18%                     | 233,138.58 €        |
| ('Low_Capital', 'HMM')         | 0.05%                     | 162,688.82 €        |
| ('Aggressive', 'HMM')          | 1.32%                     | 171,925.79 €        |
| ('Low_Capital', 'MSM')         | 0.03%                     | 155,761.25 €        |
| ('Low_Capital', 'Transformer') | 0.39%                     | 204,613.53 €        |

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
| ('Aggressive', 'Buy_Hold')     | 4.41%            | 4.02%          | 4.83%         | 441/10000          |
| ('Aggressive', 'MSM')          | 1.98%            | 1.72%          | 2.27%         | 198/10000          |
| ('Aggressive', 'HMM')          | 1.32%            | 1.11%          | 1.56%         | 132/10000          |
| ('Aggressive', 'LSTM')         | 4.02%            | 3.65%          | 4.42%         | 402/10000          |
| ('Aggressive', 'Transformer')  | 4.18%            | 3.81%          | 4.59%         | 418/10000          |
| ('Low_Capital', 'Buy_Hold')    | 0.50%            | 0.38%          | 0.66%         | 50/10000           |
| ('Low_Capital', 'MSM')         | 0.03%            | 0.01%          | 0.09%         | 3/10000            |
| ('Low_Capital', 'HMM')         | 0.05%            | 0.02%          | 0.12%         | 5/10000            |
| ('Low_Capital', 'LSTM')        | 0.34%            | 0.24%          | 0.47%         | 34/10000           |
| ('Low_Capital', 'Transformer') | 0.39%            | 0.29%          | 0.53%         | 39/10000           |

### Hypothesentests (gepaarter Wilcoxon, α = 0.05)
**H1 — Regime-Switching reduziert MaxDD vs. Buy & Hold:**

| Modell      | Median MaxDD (Modell)   | Median MaxDD (B&H)   | Δ Median   |   Wilcoxon p | H1 (α=0.05)   |
|:------------|:------------------------|:---------------------|:-----------|-------------:|:--------------|
| MSM         | -69.14%                 | -56.83%              | -12.31 pp  |        1     | abgelehnt     |
| HMM         | -66.50%                 | -56.83%              | -9.67 pp   |        1     | abgelehnt     |
| LSTM        | -58.41%                 | -56.83%              | -1.58 pp   |        0.998 | abgelehnt     |
| Transformer | -58.11%                 | -56.83%              | -1.28 pp   |        0.974 | abgelehnt     |

**H2 — Transformer dominiert Ökonometrie und LSTM im Endvermögen:**

| Vergleich            | Median Transformer   | Median MSM   | Δ Median   |   Wilcoxon p | H2 (α=0.05)   | Median HMM   | Median LSTM   |
|:---------------------|:---------------------|:-------------|:-----------|-------------:|:--------------|:-------------|:--------------|
| Transformer vs. MSM  | 233,139 €            | 160,417 €    | +72,722 €  |    0         | bestätigt     | nan          | nan           |
| Transformer vs. HMM  | 233,139 €            | nan          | +61,213 €  |    2.34e-288 | bestätigt     | 171,926 €    | nan           |
| Transformer vs. LSTM | 233,139 €            | nan          | +878 €     |    0.428     | abgelehnt     | nan          | 232,261 €     |

### Break-Even-Transaktionskosten
Ab welcher Kostenquote (in Basispunkten pro Umschichtung) verliert das aktive Switching seinen Renditevorteil gegenüber Buy & Hold?

| Modell      |   Final @10bps |   B&H Final |   Break-Even (bps) |
|:------------|---------------:|------------:|-------------------:|
| MSM         |          2.929 |       4.766 |                  0 |
| HMM         |          2.997 |       4.766 |                  0 |
| LSTM        |          4.373 |       4.766 |                  0 |
| Transformer |          4.501 |       4.766 |                  0 |

![Break-Even-Analyse](../assets/break_even_costs.png)

### Entnahmeraten-Sensitivität (3.5 % / 4 % / 5 %)
Robustheit der SORR-Ergebnisse bei variierenden jährlichen Entnahmen.

| Strategie   | ('Endkapital', '3.5%')   | ('Endkapital', '4.0%')   | ('Endkapital', '5.0%')   | ('Status', '3.5%')   | ('Status', '4.0%')   | ('Status', '5.0%')   |
|:------------|:-------------------------|:-------------------------|:-------------------------|:---------------------|:---------------------|:---------------------|
| Buy_Hold    | 1,077,600 €              | 889,929 €                | 514,588 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| HMM         | 524,550 €                | 384,632 €                | 104,797 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| LSTM        | 980,414 €                | 806,997 €                | 460,163 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| MSM         | 525,719 €                | 390,902 €                | 121,268 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |
| Transformer | 1,022,410 €              | 845,829 €                | 492,669 €                | Kapitalerhalt        | Kapitalerhalt        | Kapitalerhalt        |

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
| 00_dependencies | 16:55:50 | 16:55:53 | 2.9 |
| 01_data_preprocessing | 16:55:53 | 16:55:59 | 6.9 |
| 02_feature_engineering | 16:55:59 | 16:56:04 | 4.8 |
| 03_regime_switching_models | 16:56:04 | 16:56:11 | 7.1 |
| 04_backtesting | 16:56:11 | 16:56:17 | 6.1 |
| 05_evaluation | 16:56:17 | 16:59:10 | 172.4 |
| **Gesamt** | | | **200.2** (3m 20.2s) |

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
| TRANSFORMER | `transformer_regime_model.pt` | Geladen (persistiert) |

> **Hinweis:** Bei aktivierter Persistierung werden vortrainierte Modelle aus `../models` geladen, sofern die Dateien existieren. Andernfalls wird normal trainiert und das Ergebnis für zukünftige Läufe gespeichert. Bei Änderungen an Hyperparametern müssen die entsprechenden Modelldateien gelöscht werden.

---

**Zuletzt aktualisiert:** 15.04.2026 16:59<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Walk-Forward-Validierung:** AKTIV (Modus: rolling, Train: 10J, Test: 12M, Step: 12M)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
