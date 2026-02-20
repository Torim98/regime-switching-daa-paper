
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie         |   Final Wealth | Total Return   | Max Drawdown   |
|:------------------|---------------:|:---------------|:---------------|
| Buy_Hold          |         1.8951 | +89.51%        | -27.10%        |
| HMM               |         1.7155 | +71.55%        | -6.79%         |
| MS_Univariate     |         2.5036 | +150.36%       | -6.25%         |
| MS_Exo            |         2.4807 | +148.07%       | -5.44%         |
| LSTM              |         1.5812 | +58.12%        | -19.21%        |
| LSTM_Unsupervised |         1.5425 | +54.25%        | -13.68%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### 60/40 Portfolio Kapitalkurve
Die Abbildung zeigt die kumulierte Wertentwicklung des statischen Referenzportfolios (60% Aktien / 40% Anleihen).

![Capital Curve](./assets/capital_curve.png)

*   **Datenquelle:** S&P 500 (`^GSPC`) und Vanguard Long-Term Treasury (`VUSTX`).
*   **Reproduzierbarkeit:** Der bereinigte Datensatz inkl. aller Features ist hinterlegt unter: `data/02_feature_engineered_data.parquet`.

---

## 3. Regime-Erkennung der Einzelmodelle
Hier werden die Identifikations-Ergebnisse der drei Modell-Kategorien (Statistik, Clustering, Deep Learning) visualisiert.

### A. Hidden Markov Model (Unsupervised Clustering)
![HMM Regimes](./assets/hmm_regimes.png)

### B. Markov-Switching-Modelle (Ökonometrie)
Vergleich zwischen univariatem Ansatz und exogenem Ansatz (unter Berücksichtigung von VIX & Yield Spread).
![Markov Models](./assets/markov-models.png)

### C. LSTM-Netzwerk (Deep Learning)
Vorhersage der Marktphasen durch das neuronale Netzwerk (trainiert auf Markov-Labels).
![LSTM Model](./assets/lstm_model.png)

### D. Unsupervised LSTM-Netzwerk (Deep Learning)
Identifikation von Marktregimes mittels eines LSTM-Autoencoders in Kombination mit Gaussian Mixture Modeling (GMM). Im Gegensatz zum Supervised-Ansatz lernt dieses Modell ohne vordefinierte Labels (wie HMM oder Markov) und identifiziert Regime-Strukturen rein datengetrieben durch die Kompression und Rekonstruktion zeitlicher Sequenzen.
![Unsupervised LSTM Model](./assets/lstm_unsupervised_model.png)

### E. Globaler Regime-Vergleich
Detaillierte Gegenüberstellung der Wahrscheinlichkeiten und harten Signale aller Modelle.
![Regime Comparison](./assets/regime_comparison.png)

---

## 4. Backtesting & Strategie-Evaluation
Die ökonomische Anwendung der Regime-Signale durch dynamische Umschichtung in den Geldmarkt.

### Equity Curves im Vergleich
![Equity Curves](./assets/equity_curves.png)

### Umfassende Kennzahlen-Matrix
Detaillierte statistische Analyse inklusive risikoadjustierter Kennzahlen (Sharpe, Sortino, Calmar).

| Strategie         | Total Return   | CAGR (p.a.)   | Volatilität   | Max Drawdown   |   Sharpe Ratio |   Sortino Ratio |   Calmar Ratio |   Regime-Wechsel | Gesamtkosten (Gebühren)   |
|:------------------|:---------------|:--------------|:--------------|:---------------|---------------:|----------------:|---------------:|-----------------:|:--------------------------|
| Buy Hold          | 89.77%         | 9.54%         | 12.61%        | -27.10%        |           0.79 |            1.02 |           0.35 |                0 | 0.00%                     |
| HMM               | 71.78%         | 8.00%         | 4.96%         | -6.79%         |           1.58 |            1.47 |           1.18 |               29 | 2.90%                     |
| MS Univariate     | 150.69%        | 13.96%        | 6.33%         | -6.25%         |           2.1  |            2.73 |           2.23 |               42 | 4.20%                     |
| MS Exo            | 148.40%        | 13.81%        | 6.42%         | -5.44%         |           2.05 |            2.65 |           2.54 |               38 | 3.80%                     |
| LSTM              | 58.33%         | 6.75%         | 8.04%         | -19.21%        |           0.86 |            1.02 |           0.35 |               56 | 5.60%                     |
| LSTM Unsupervised | 54.46%         | 6.38%         | 8.16%         | -13.68%        |           0.8  |            0.83 |           0.47 |               25 | 2.60%                     |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](./assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                      | Endkapital   | Status        |
|:-------------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')             | 663,428.34 € | Kapitalerhalt |
| ('Standard', 'HMM')                  | 581,243.50 € | Kapitalerhalt |
| ('Standard', 'MS Univariate')        | 914,803.87 € | Kapitalerhalt |
| ('Standard', 'MS Exo')               | 898,189.16 € | Kapitalerhalt |
| ('Standard', 'LSTM')                 | 530,301.89 € | Kapitalerhalt |
| ('Standard', 'LSTM Unsupervised')    | 509,444.79 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')           | 492,189.10 € | Kapitalerhalt |
| ('Aggressive', 'HMM')                | 414,649.08 € | Kapitalerhalt |
| ('Aggressive', 'MS Univariate')      | 711,606.30 € | Kapitalerhalt |
| ('Aggressive', 'MS Exo')             | 691,899.65 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')               | 373,491.71 € | Kapitalerhalt |
| ('Aggressive', 'LSTM Unsupervised')  | 351,736.95 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')          | 340,749.17 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')               | 293,118.99 € | Kapitalerhalt |
| ('Low_Capital', 'MS Univariate')     | 480,973.83 € | Kapitalerhalt |
| ('Low_Capital', 'MS Exo')            | 469,967.28 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')              | 265,758.73 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM Unsupervised') | 252,949.79 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](./assets/sorr_sim_standard.png)
![SORR Aggressive](./assets/sorr_sim_aggressive.png)
![SORR Low Capital](./assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](./assets/mcs_paths.png)
|                                      | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')             | 0.00%                     | 746,397.52 €        |
| ('Standard', 'HMM')                  | 0.00%                     | 621,160.56 €        |
| ('Standard', 'MS Univariate')        | 0.00%                     | 1,232,297.50 €      |
| ('Standard', 'MS Exo')               | 0.00%                     | 1,191,794.59 €      |
| ('Standard', 'LSTM')                 | 0.00%                     | 524,424.66 €        |
| ('Standard', 'LSTM Unsupervised')    | 0.00%                     | 499,224.41 €        |
| ('Aggressive', 'Buy Hold')           | 1.60%                     | 453,140.46 €        |
| ('Aggressive', 'HMM')                | 0.00%                     | 355,301.78 €        |
| ('Aggressive', 'MS Univariate')      | 0.00%                     | 859,548.31 €        |
| ('Aggressive', 'MS Exo')             | 0.00%                     | 829,707.45 €        |
| ('Aggressive', 'LSTM')               | 0.90%                     | 271,166.30 €        |
| ('Aggressive', 'LSTM Unsupervised')  | 0.70%                     | 263,999.00 €        |
| ('Low_Capital', 'Buy Hold')          | 0.20%                     | 336,927.62 €        |
| ('Low_Capital', 'HMM')               | 0.00%                     | 282,740.23 €        |
| ('Low_Capital', 'MS Univariate')     | 0.00%                     | 620,279.47 €        |
| ('Low_Capital', 'MS Exo')            | 0.00%                     | 605,094.65 €        |
| ('Low_Capital', 'LSTM')              | 0.00%                     | 233,726.90 €        |
| ('Low_Capital', 'LSTM Unsupervised') | 0.00%                     | 217,507.84 €        |

Verteilung der Endkapitalwerte:

![MCS Boxplots Standard](./assets/mcs_boxplot_standard.png)
![MCS Boxplots Aggressive](./assets/mcs_boxplot_aggressive.png)
![MCS Boxplots Low Capital](./assets/mcs_boxplot_low_capital.png)

Wahrscheinlichkeitskorridore:

Die schattierten Bereiche zeigen das 5% bis 95% Konfidenzintervall der Kapitalentwicklung.
![MCS Quantiles](./assets/mcs_quantiles.png)

---

## Forschungsnotizen & Methodik
- **Cash-Komponente:** Bei einem "Bear"-Signal schichtet die Strategie in den aktuellen Geldmarktzins (**^IRX**) um.
- **Vermeidung von Look-ahead Bias:** Alle Signale werden für das Backtesting um einen Tag zeitversetzt (`shift(1)`), um reale Handelsbedingungen zu simulieren.
- **Feature-Set:** Die Modelle nutzen Renditen, Volatilität (20d), SMA-Abstand, Momentum, VIX und Yield Spread.
- **Kostensimulation:** Es wird eine pauschale Gebühr von 10 Basispunkten (0,1%) pro Umschichtung berechnet.
- **SORR-Spezifika:** Bei Entnahmen in "Bull"-Phasen wird eine zusätzliche Liquiditätsgebühr von 0,1% auf den Entnahmebetrag erhoben (Asset-Verkäufe). In "Bear"-Phasen (Cash) entfällt diese.

---
**Zuletzt aktualisiert:** 20.02.2026 13:19  
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
