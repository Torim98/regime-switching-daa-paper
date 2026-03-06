
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         1.8578 | +85.78%        | -27.10%        |
| MSM         |         2.4543 | +145.43%       | -6.25%         |
| HMM         |         1.7078 | +70.78%        | -5.64%         |
| LSTM        |         1.3964 | +39.64%        | -17.70%        |
| Transformer |         1.5829 | +58.29%        | -9.45%         |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

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
Vorhersage der Marktphasen durch das neuronale Netzwerk (trainiert auf Markov-Labels).
![LSTM Model](../assets/lstm_model.png)

### D. Transformer-Netzwerk (Attention-basierte Regime-Erkennung)
"Klassifikation von Marktregimes mittels eines Transformer-Encoders mit Multi-Head Self-Attention und Positional Encoding. Im Gegensatz zu rekurrenten Architekturen (LSTM) verarbeitet der Transformer alle Zeitschritte einer Sequenz parallel und lernt über den Attention-Mechanismus, welche historischen Datenpunkte die höchste Relevanz für die aktuelle Regime-Klassifikation besitzen. Trainiert im Supervised-Setting auf Markov-Labels.
![Transformer Model](../assets/transformer_model.png)

### E. Globaler Regime-Vergleich
Detaillierte Gegenüberstellung der Wahrscheinlichkeiten und harten Signale aller Modelle.
![Regime Comparison](../assets/regime_comparison.png)

---

## 4. Backtesting & Strategie-Evaluation
Die ökonomische Anwendung der Regime-Signale durch dynamische Umschichtung in den Geldmarkt.

### Equity Curves im Vergleich
![Equity Curves](../assets/equity_curves.png)

### Umfassende Kennzahlen-Matrix
Detaillierte statistische Analyse inklusive risikoadjustierter Kennzahlen (Sharpe, Sortino, Calmar).

| Strategie   | Total Return   | CAGR (p.a.)   | Volatilität   | Max Drawdown   |   Sharpe Ratio |   Sortino Ratio |   Calmar Ratio |   Regime-Wechsel | Gesamtkosten (Gebühren)   |
|:------------|:---------------|:--------------|:--------------|:---------------|---------------:|----------------:|---------------:|-----------------:|:--------------------------|
| Buy Hold    | 85.77%         | 9.20%         | 12.61%        | -27.10%        |           0.76 |            0.99 |           0.34 |                0 | 0.00%                     |
| MSM         | 145.42%        | 13.61%        | 6.34%         | -6.25%         |           2.05 |            2.67 |           2.18 |               42 | 4.20%                     |
| HMM         | 70.77%         | 7.90%         | 4.83%         | -5.64%         |           1.6  |            1.5  |           1.4  |               31 | 3.10%                     |
| LSTM        | 39.63%         | 4.86%         | 9.11%         | -17.70%        |           0.57 |            0.36 |           0.27 |               34 | 3.50%                     |
| Transformer | 58.28%         | 6.74%         | 7.02%         | -9.45%         |           0.97 |            1.11 |           0.71 |               64 | 6.40%                     |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status        |
|:-------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')       | 643,017.99 € | Kapitalerhalt |
| ('Standard', 'MSM')            | 888,374.26 € | Kapitalerhalt |
| ('Standard', 'HMM')            | 572,806.67 € | Kapitalerhalt |
| ('Standard', 'LSTM')           | 430,596.31 € | Kapitalerhalt |
| ('Standard', 'Transformer')    | 528,456.61 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')     | 471,507.89 € | Kapitalerhalt |
| ('Aggressive', 'MSM')          | 685,139.13 € | Kapitalerhalt |
| ('Aggressive', 'HMM')          | 404,166.71 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')         | 270,059.62 € | Kapitalerhalt |
| ('Aggressive', 'Transformer')  | 370,695.90 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')    | 328,640.76 € | Kapitalerhalt |
| ('Low_Capital', 'MSM')         | 465,279.51 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')         | 287,470.68 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')        | 204,845.55 € | Kapitalerhalt |
| ('Low_Capital', 'Transformer') | 264,487.06 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 763,736.03 €        |
| ('Standard', 'MSM')            | 0.00%                     | 1,212,155.35 €      |
| ('Standard', 'HMM')            | 0.00%                     | 633,079.65 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 409,785.93 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 514,581.69 €        |
| ('Aggressive', 'Buy Hold')     | 0.00%                     | 424,381.80 €        |
| ('Aggressive', 'MSM')          | 0.00%                     | 898,458.69 €        |
| ('Aggressive', 'HMM')          | 0.00%                     | 361,889.77 €        |
| ('Aggressive', 'LSTM')         | 4.00%                     | 195,924.44 €        |
| ('Aggressive', 'Transformer')  | 0.00%                     | 297,180.96 €        |
| ('Low_Capital', 'Buy Hold')    | 0.00%                     | 361,306.67 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 598,199.48 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 282,416.82 €        |
| ('Low_Capital', 'LSTM')        | 0.00%                     | 182,913.06 €        |
| ('Low_Capital', 'Transformer') | 0.00%                     | 225,464.37 €        |

Verteilung der Endkapitalwerte:

![MCS Boxplots Standard](../assets/mcs_boxplot_standard.png)
![MCS Boxplots Aggressive](../assets/mcs_boxplot_aggressive.png)
![MCS Boxplots Low Capital](../assets/mcs_boxplot_low_capital.png)

Wahrscheinlichkeitskorridore:

Die schattierten Bereiche zeigen das 5% bis 95% Konfidenzintervall der Kapitalentwicklung.
![MCS Quantiles](../assets/mcs_quantiles.png)

---

## Forschungsnotizen & Methodik
- **Cash-Komponente:** Bei einem "Bear"-Signal schichtet die Strategie in den aktuellen Geldmarktzins (**^IRX**) um.
- **Vermeidung von Look-ahead Bias:** Alle Signale werden für das Backtesting um einen Tag zeitversetzt (`shift(1)`), um reale Handelsbedingungen zu simulieren.
- **Feature-Set:** Die Modelle nutzen Renditen, Volatilität (20d), SMA-Abstand, Momentum, VIX und Yield Spread.
- **Kostensimulation:** Es wird eine pauschale Gebühr von 10 Basispunkten (0,1%) pro Umschichtung berechnet.
- **SORR-Spezifika:** Bei Entnahmen in "Bull"-Phasen wird eine zusätzliche Liquiditätsgebühr von 0,1% auf den Entnahmebetrag erhoben (Asset-Verkäufe). In "Bear"-Phasen (Cash) entfällt diese.

---
**Zuletzt aktualisiert:** 06.03.2026 15:26<br>
**Fast Mode Status zur Laufzeit:** TRUE (Development Mode)<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
