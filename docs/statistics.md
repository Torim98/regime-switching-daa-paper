
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         1.8402 | +84.02%        | -27.10%        |
| MSM         |         2.431  | +143.10%       | -6.25%         |
| HMM         |         1.708  | +70.80%        | -5.64%         |
| LSTM        |         1.2366 | +23.66%        | -19.77%        |
| Transformer |         1.4426 | +44.26%        | -9.65%         |

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
| Buy Hold    | 85.03%         | 9.14%         | 12.61%        | -27.10%        |           0.76 |            0.98 |           0.34 |                0 | 0.00%                     |
| MSM         | 144.44%        | 13.54%        | 6.34%         | -6.25%         |           2.04 |            2.65 |           2.17 |               42 | 4.20%                     |
| HMM         | 71.74%         | 7.99%         | 4.83%         | -5.64%         |           1.62 |            1.52 |           1.42 |               31 | 3.10%                     |
| LSTM        | 24.34%         | 3.14%         | 9.14%         | -19.77%        |           0.39 |            0.25 |           0.16 |               48 | 4.90%                     |
| Transformer | 45.06%         | 5.43%         | 6.26%         | -9.65%         |           0.88 |            0.95 |           0.56 |               54 | 5.40%                     |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status        |
|:-------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')       | 641,991.68 € | Kapitalerhalt |
| ('Standard', 'MSM')            | 886,651.63 € | Kapitalerhalt |
| ('Standard', 'HMM')            | 577,574.47 € | Kapitalerhalt |
| ('Standard', 'LSTM')           | 367,262.00 € | Kapitalerhalt |
| ('Standard', 'Transformer')    | 481,742.29 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')     | 472,087.01 € | Kapitalerhalt |
| ('Aggressive', 'MSM')          | 685,316.72 € | Kapitalerhalt |
| ('Aggressive', 'HMM')          | 408,896.48 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')         | 214,590.65 € | Kapitalerhalt |
| ('Aggressive', 'Transformer')  | 335,617.69 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')    | 328,560.12 € | Kapitalerhalt |
| ('Low_Capital', 'MSM')         | 464,879.34 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')         | 290,318.68 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')        | 169,466.75 € | Kapitalerhalt |
| ('Low_Capital', 'Transformer') | 240,337.17 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 780,121.94 €        |
| ('Standard', 'MSM')            | 0.00%                     | 1,193,149.78 €      |
| ('Standard', 'HMM')            | 0.00%                     | 633,457.01 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 317,816.01 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 451,244.64 €        |
| ('Aggressive', 'Buy Hold')     | 0.00%                     | 457,463.39 €        |
| ('Aggressive', 'MSM')          | 0.00%                     | 882,459.44 €        |
| ('Aggressive', 'HMM')          | 0.00%                     | 349,665.47 €        |
| ('Aggressive', 'LSTM')         | 18.00%                    | 131,935.79 €        |
| ('Aggressive', 'Transformer')  | 0.00%                     | 243,425.68 €        |
| ('Low_Capital', 'Buy Hold')    | 0.00%                     | 372,501.22 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 595,451.16 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 282,791.44 €        |
| ('Low_Capital', 'LSTM')        | 0.00%                     | 131,711.86 €        |
| ('Low_Capital', 'Transformer') | 0.00%                     | 194,498.13 €        |

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

## Pipeline-Laufzeiten

Ausführungszeiten der einzelnen Pipeline-Notebooks (monolithischer Notebook-Ansatz).

| Notebook | Start | Ende | Dauer (s) |
|----------|-------|------|-----------|
| 00_dependencies | 12:17:53 | 12:17:58 | 5.2 |
| 01_data_preprocessing | 12:17:58 | 12:18:08 | 9.6 |
| 02_feature_engineering | 12:18:08 | 12:18:17 | 9.6 |
| 03_regime_switching_models | 12:18:17 | 12:23:50 | 332.2 |
| 04_backtesting | 12:23:50 | 12:23:57 | 7.0 |
| 05_evaluation | 12:23:57 | 12:24:22 | 25.3 |
| **Gesamt** | | | **388.9** (6m 28.9s) |

---

---

**Zuletzt aktualisiert:** 08.03.2026 12:24<br>
**Fast Mode Status zur Laufzeit:** TRUE (Development Mode)<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
