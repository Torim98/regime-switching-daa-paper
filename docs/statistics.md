
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie         |   Final Wealth | Total Return   | Max Drawdown   |
|:------------------|---------------:|:---------------|:---------------|
| Buy_Hold          |         1.8901 | +89.01%        | -27.10%        |
| HMM               |         1.7254 | +72.54%        | -5.64%         |
| MS_Univariate     |         2.5103 | +151.03%       | -6.25%         |
| MS_Exo            |         2.4457 | +144.57%       | -5.44%         |
| LSTM              |         1.6345 | +63.45%        | -12.50%        |
| LSTM_Unsupervised |         1.7827 | +78.27%        | -17.44%        |
| Transformer       |         1.4445 | +44.45%        | -8.03%         |

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

### A. Hidden Markov Model (Unsupervised Clustering)
![HMM Regimes](../assets/hmm_regimes.png)

### B. Markov-Switching-Modelle (Ökonometrie)
Vergleich zwischen univariatem Ansatz und exogenem Ansatz (unter Berücksichtigung von VIX & Yield Spread).
![Markov Models](../assets/markov-models.png)

### C. LSTM-Netzwerk (Deep Learning)
Vorhersage der Marktphasen durch das neuronale Netzwerk (trainiert auf Markov-Labels).
![LSTM Model](../assets/lstm_model.png)

### D. Unsupervised LSTM-Netzwerk (Deep Learning)
Identifikation von Marktregimes mittels eines LSTM-Autoencoders in Kombination mit Gaussian Mixture Modeling (GMM). Im Gegensatz zum Supervised-Ansatz lernt dieses Modell ohne vordefinierte Labels (wie HMM oder Markov) und identifiziert Regime-Strukturen rein datengetrieben durch die Kompression und Rekonstruktion zeitlicher Sequenzen.
![Unsupervised LSTM Model](../assets/lstm_unsupervised_model.png)

### E. Transformer-Netzwerk (Attention-basierte Regime-Erkennung)
"Klassifikation von Marktregimes mittels eines Transformer-Encoders mit Multi-Head Self-Attention und Positional Encoding. Im Gegensatz zu rekurrenten Architekturen (LSTM) verarbeitet der Transformer alle Zeitschritte einer Sequenz parallel und lernt über den Attention-Mechanismus, welche historischen Datenpunkte die höchste Relevanz für die aktuelle Regime-Klassifikation besitzen. Trainiert im Supervised-Setting auf Markov-Labels.
![Transformer Model](../assets/transformer_model.png)

### F. Globaler Regime-Vergleich
Detaillierte Gegenüberstellung der Wahrscheinlichkeiten und harten Signale aller Modelle.
![Regime Comparison](../assets/regime_comparison.png)

---

## 4. Backtesting & Strategie-Evaluation
Die ökonomische Anwendung der Regime-Signale durch dynamische Umschichtung in den Geldmarkt.

### Equity Curves im Vergleich
![Equity Curves](../assets/equity_curves.png)

### Umfassende Kennzahlen-Matrix
Detaillierte statistische Analyse inklusive risikoadjustierter Kennzahlen (Sharpe, Sortino, Calmar).

| Strategie         | Total Return   | CAGR (p.a.)   | Volatilität   | Max Drawdown   |   Sharpe Ratio |   Sortino Ratio |   Calmar Ratio |   Regime-Wechsel | Gesamtkosten (Gebühren)   |
|:------------------|:---------------|:--------------|:--------------|:---------------|---------------:|----------------:|---------------:|-----------------:|:--------------------------|
| Buy Hold          | 88.92%         | 9.45%         | 12.61%        | -27.10%        |           0.78 |            1.01 |           0.35 |                0 | 0.00%                     |
| HMM               | 72.46%         | 8.04%         | 4.84%         | -5.64%         |           1.63 |            1.53 |           1.43 |               31 | 3.10%                     |
| MS Univariate     | 150.91%        | 13.95%        | 6.33%         | -6.25%         |           2.1  |            2.73 |           2.23 |               42 | 4.20%                     |
| MS Exo            | 144.46%        | 13.53%        | 6.41%         | -5.44%         |           2.02 |            2.61 |           2.49 |               38 | 3.80%                     |
| LSTM              | 63.37%         | 7.22%         | 7.93%         | -12.50%        |           0.92 |            1.06 |           0.58 |               40 | 4.00%                     |
| LSTM Unsupervised | 78.20%         | 8.55%         | 9.58%         | -17.44%        |           0.91 |            0.94 |           0.49 |               25 | 2.60%                     |
| Transformer       | 44.39%         | 5.35%         | 6.10%         | -8.03%         |           0.89 |            0.93 |           0.67 |               58 | 5.80%                     |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                      | Endkapital   | Status        |
|:-------------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')             | 656,434.52 € | Kapitalerhalt |
| ('Standard', 'HMM')                  | 580,929.07 € | Kapitalerhalt |
| ('Standard', 'MS Univariate')        | 912,128.07 € | Kapitalerhalt |
| ('Standard', 'MS Exo')               | 878,483.74 € | Kapitalerhalt |
| ('Standard', 'LSTM')                 | 552,928.46 € | Kapitalerhalt |
| ('Standard', 'LSTM Unsupervised')    | 614,326.06 € | Kapitalerhalt |
| ('Standard', 'Transformer')          | 476,382.26 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')           | 483,522.72 € | Kapitalerhalt |
| ('Aggressive', 'HMM')                | 412,096.61 € | Kapitalerhalt |
| ('Aggressive', 'MS Univariate')      | 706,660.94 € | Kapitalerhalt |
| ('Aggressive', 'MS Exo')             | 672,181.35 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')               | 394,562.74 € | Kapitalerhalt |
| ('Aggressive', 'LSTM Unsupervised')  | 448,334.79 € | Kapitalerhalt |
| ('Aggressive', 'Transformer')        | 329,054.14 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')          | 336,223.45 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')               | 292,279.96 € | Kapitalerhalt |
| ('Low_Capital', 'MS Univariate')     | 478,787.80 € | Kapitalerhalt |
| ('Low_Capital', 'MS Exo')            | 458,322.78 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')              | 278,968.50 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM Unsupervised') | 313,265.21 € | Kapitalerhalt |
| ('Low_Capital', 'Transformer')       | 236,719.98 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                      | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')             | 0.00%                     | 785,197.19 €        |
| ('Standard', 'HMM')                  | 0.00%                     | 617,166.38 €        |
| ('Standard', 'MS Univariate')        | 0.00%                     | 1,270,890.43 €      |
| ('Standard', 'MS Exo')               | 0.00%                     | 1,235,744.00 €      |
| ('Standard', 'LSTM')                 | 0.00%                     | 542,486.69 €        |
| ('Standard', 'LSTM Unsupervised')    | 0.00%                     | 688,640.10 €        |
| ('Standard', 'Transformer')          | 0.00%                     | 425,701.54 €        |
| ('Aggressive', 'Buy Hold')           | 1.00%                     | 463,722.00 €        |
| ('Aggressive', 'HMM')                | 0.00%                     | 381,834.11 €        |
| ('Aggressive', 'MS Univariate')      | 0.00%                     | 875,246.73 €        |
| ('Aggressive', 'MS Exo')             | 0.00%                     | 849,639.80 €        |
| ('Aggressive', 'LSTM')               | 0.00%                     | 317,240.65 €        |
| ('Aggressive', 'LSTM Unsupervised')  | 1.00%                     | 394,444.63 €        |
| ('Aggressive', 'Transformer')        | 0.00%                     | 189,067.84 €        |
| ('Low_Capital', 'Buy Hold')          | 1.00%                     | 349,751.16 €        |
| ('Low_Capital', 'HMM')               | 0.00%                     | 293,172.90 €        |
| ('Low_Capital', 'MS Univariate')     | 0.00%                     | 597,320.83 €        |
| ('Low_Capital', 'MS Exo')            | 0.00%                     | 572,427.30 €        |
| ('Low_Capital', 'LSTM')              | 0.00%                     | 241,217.15 €        |
| ('Low_Capital', 'LSTM Unsupervised') | 0.00%                     | 289,464.42 €        |
| ('Low_Capital', 'Transformer')       | 0.00%                     | 163,202.75 €        |

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
**Zuletzt aktualisiert:** 03.03.2026 18:10<br>
**Fast Mode Status zur Laufzeit:** TRUE (Development Mode)<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
