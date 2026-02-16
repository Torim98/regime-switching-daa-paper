
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie         |   Final Wealth | Total Return   | Max Drawdown   |
|:------------------|---------------:|:---------------|:---------------|
| Buy_Hold          |         1.938  | +93.80%        | -27.10%        |
| HMM               |         1.744  | +74.40%        | -6.79%         |
| MS_Univariate     |         2.574  | +157.40%       | -6.25%         |
| MS_Exo            |         2.5369 | +153.69%       | -5.44%         |
| LSTM              |         1.4968 | +49.68%        | -16.77%        |
| LSTM_Unsupervised |         1.7748 | +77.48%        | -7.85%         |

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
| Buy Hold          | 93.18%         | 9.83%         | 12.62%        | -27.10%        |           0.81 |            1.04 |           0.36 |                0 | 0.00%                     |
| HMM               | 73.84%         | 8.19%         | 4.96%         | -6.79%         |           1.62 |            1.5  |           1.21 |               29 | 3.00%                     |
| MS Univariate     | 156.57%        | 14.35%        | 6.33%         | -6.25%         |           2.16 |            2.79 |           2.3  |               42 | 4.20%                     |
| MS Exo            | 152.87%        | 14.12%        | 6.42%         | -5.44%         |           2.09 |            2.7  |           2.59 |               38 | 3.80%                     |
| LSTM              | 49.20%         | 5.86%         | 7.68%         | -16.77%        |           0.78 |            0.87 |           0.35 |               56 | 5.60%                     |
| LSTM Unsupervised | 76.91%         | 8.46%         | 7.66%         | -7.85%         |           1.1  |            1.28 |           1.08 |               21 | 2.20%                     |

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](./assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### Vergleich der Entnahmeszenarien
In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                      | Endkapital   | Status        |
|:-------------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')             | 676,642.08 € | Kapitalerhalt |
| ('Standard', 'HMM')                  | 585,004.69 € | Kapitalerhalt |
| ('Standard', 'MS Univariate')        | 937,966.15 € | Kapitalerhalt |
| ('Standard', 'MS Exo')               | 915,383.59 € | Kapitalerhalt |
| ('Standard', 'LSTM')                 | 500,764.28 € | Kapitalerhalt |
| ('Standard', 'LSTM Unsupervised')    | 609,661.30 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')           | 503,078.66 € | Kapitalerhalt |
| ('Aggressive', 'HMM')                | 414,483.10 € | Kapitalerhalt |
| ('Aggressive', 'MS Univariate')      | 731,033.21 € | Kapitalerhalt |
| ('Aggressive', 'MS Exo')             | 705,989.87 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')               | 353,627.28 € | Kapitalerhalt |
| ('Aggressive', 'LSTM Unsupervised')  | 444,716.86 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')          | 347,899.59 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')               | 294,065.88 € | Kapitalerhalt |
| ('Low_Capital', 'MS Univariate')     | 493,620.95 € | Kapitalerhalt |
| ('Low_Capital', 'MS Exo')            | 479,244.75 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')              | 251,267.07 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM Unsupervised') | 310,662.00 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](./assets/sorr_sim_standard.png)
![SORR Aggresive](./assets/sorr_sim_aggressiv.png)
![SORR Low Capital](./assets/sorr_sim_low_capital.png)

---

## 📝 Forschungsnotizen & Methodik
- **Cash-Komponente:** Bei einem "Bear"-Signal schichtet die Strategie in den aktuellen Geldmarktzins (**^IRX**) um.
- **Vermeidung von Look-ahead Bias:** Alle Signale werden für das Backtesting um einen Tag zeitversetzt (`shift(1)`), um reale Handelsbedingungen zu simulieren.
- **Feature-Set:** Die Modelle nutzen Renditen, Volatilität (20d), SMA-Abstand, Momentum, VIX und Yield Spread.
- **Kostensimulation:** Es wird eine pauschale Gebühr von 10 Basispunkten (0,1%) pro Umschichtung berechnet.
- **SORR-Spezifika:** Bei Entnahmen in "Bull"-Phasen wird eine zusätzliche Liquiditätsgebühr von 0,1% auf den Entnahmebetrag erhoben (Asset-Verkäufe). In "Bear"-Phasen (Cash) entfällt diese.

---
**Zuletzt aktualisiert:** 03.02.2026 13:57  
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
