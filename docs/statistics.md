
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie         |   Final Wealth | Total Return   | Max Drawdown   |
|:------------------|---------------:|:---------------|:---------------|
| Buy_Hold          |         1.9076 | +90.76%        | -27.10%        |
| HMM               |         1.7239 | +72.39%        | -6.79%         |
| MS_Univariate     |         2.5335 | +153.35%       | -6.25%         |
| MS_Exo            |         2.5    | +150.00%       | -5.44%         |
| LSTM              |         1.6084 | +60.84%        | -17.82%        |
| LSTM_Unsupervised |         1.7384 | +73.84%        | -13.71%        |

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
| Buy Hold          | 90.36%         | 9.58%         | 12.61%        | -27.10%        |           0.79 |            1.02 |           0.35 |                0 | 0.00%                     |
| HMM               | 72.03%         | 8.02%         | 4.95%         | -6.79%         |           1.59 |            1.47 |           1.18 |               29 | 2.90%                     |
| MS Univariate     | 152.82%        | 14.10%        | 6.33%         | -6.25%         |           2.12 |            2.75 |           2.26 |               42 | 4.20%                     |
| MS Exo            | 149.47%        | 13.88%        | 6.42%         | -5.44%         |           2.06 |            2.66 |           2.55 |               38 | 3.80%                     |
| LSTM              | 60.50%         | 6.96%         | 8.14%         | -17.82%        |           0.87 |            1.01 |           0.39 |               54 | 5.40%                     |
| LSTM Unsupervised | 73.48%         | 8.15%         | 8.79%         | -13.71%        |           0.94 |            0.99 |           0.59 |               13 | 1.40%                     |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](./assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                      | Endkapital   | Status        |
|:-------------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')             | 665,832.70 € | Kapitalerhalt |
| ('Standard', 'HMM')                  | 582,397.98 € | Kapitalerhalt |
| ('Standard', 'MS Univariate')        | 923,824.61 € | Kapitalerhalt |
| ('Standard', 'MS Exo')               | 902,487.80 € | Kapitalerhalt |
| ('Standard', 'LSTM')                 | 546,461.01 € | Kapitalerhalt |
| ('Standard', 'LSTM Unsupervised')    | 598,882.71 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')           | 494,251.71 € | Kapitalerhalt |
| ('Aggressive', 'HMM')                | 415,752.97 € | Kapitalerhalt |
| ('Aggressive', 'MS Univariate')      | 719,653.17 € | Kapitalerhalt |
| ('Aggressive', 'MS Exo')             | 695,558.95 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')               | 392,839.36 € | Kapitalerhalt |
| ('Aggressive', 'LSTM Unsupervised')  | 437,774.65 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')          | 342,305.96 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')               | 293,890.45 € | Kapitalerhalt |
| ('Low_Capital', 'MS Univariate')     | 486,237.62 € | Kapitalerhalt |
| ('Low_Capital', 'MS Exo')            | 472,516.39 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')              | 276,669.39 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM Unsupervised') | 305,626.94 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](./assets/sorr_sim_standard.png)
![SORR Aggressive](./assets/sorr_sim_aggressive.png)
![SORR Low Capital](./assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](./assets/mcs_paths.png)
|                                      | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')             | 0.00%                     | 736,964.39 €        |
| ('Standard', 'HMM')                  | 0.00%                     | 628,913.74 €        |
| ('Standard', 'MS Univariate')        | 0.00%                     | 1,228,023.05 €      |
| ('Standard', 'MS Exo')               | 0.00%                     | 1,199,628.64 €      |
| ('Standard', 'LSTM')                 | 0.00%                     | 517,328.45 €        |
| ('Standard', 'LSTM Unsupervised')    | 0.00%                     | 636,232.89 €        |
| ('Aggressive', 'Buy Hold')           | 2.10%                     | 454,490.37 €        |
| ('Aggressive', 'HMM')                | 0.00%                     | 353,134.09 €        |
| ('Aggressive', 'MS Univariate')      | 0.00%                     | 856,889.58 €        |
| ('Aggressive', 'MS Exo')             | 0.00%                     | 817,463.55 €        |
| ('Aggressive', 'LSTM')               | 1.40%                     | 286,665.23 €        |
| ('Aggressive', 'LSTM Unsupervised')  | 0.70%                     | 363,958.07 €        |
| ('Low_Capital', 'Buy Hold')          | 0.50%                     | 337,454.81 €        |
| ('Low_Capital', 'HMM')               | 0.00%                     | 283,150.62 €        |
| ('Low_Capital', 'MS Univariate')     | 0.00%                     | 614,535.11 €        |
| ('Low_Capital', 'MS Exo')            | 0.00%                     | 591,881.38 €        |
| ('Low_Capital', 'LSTM')              | 0.00%                     | 237,086.66 €        |
| ('Low_Capital', 'LSTM Unsupervised') | 0.00%                     | 291,787.25 €        |

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
**Zuletzt aktualisiert:** 21.02.2026 14:57  
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
