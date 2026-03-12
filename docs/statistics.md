
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         1.838  | +83.80%        | -27.10%        |
| MSM         |         2.4281 | +142.81%       | -6.25%         |
| HMM         |         1.7106 | +71.06%        | -5.64%         |
| LSTM        |         1.5429 | +54.29%        | -18.79%        |
| Transformer |         1.4745 | +47.45%        | -8.94%         |

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
| Buy Hold    | 83.85%         | 9.04%         | 12.61%        | -27.10%        |           0.75 |            0.97 |           0.33 |                0 | 0.00%                     |
| MSM         | 142.87%        | 13.44%        | 6.35%         | -6.25%         |           2.02 |            2.64 |           2.15 |               42 | 4.20%                     |
| HMM         | 71.11%         | 7.93%         | 4.82%         | -5.64%         |           1.61 |            1.51 |           1.41 |               31 | 3.10%                     |
| LSTM        | 54.33%         | 6.36%         | 7.88%         | -18.79%        |           0.82 |            0.93 |           0.34 |               57 | 5.70%                     |
| Transformer | 47.49%         | 5.68%         | 6.67%         | -8.94%         |           0.86 |            0.94 |           0.63 |               77 | 7.70%                     |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status        |
|:-------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')       | 636,330.23 € | Kapitalerhalt |
| ('Standard', 'MSM')            | 879,137.20 € | Kapitalerhalt |
| ('Standard', 'HMM')            | 573,899.30 € | Kapitalerhalt |
| ('Standard', 'LSTM')           | 520,574.75 € | Kapitalerhalt |
| ('Standard', 'Transformer')    | 487,323.40 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')     | 466,587.92 € | Kapitalerhalt |
| ('Aggressive', 'MSM')          | 677,996.09 € | Kapitalerhalt |
| ('Aggressive', 'HMM')          | 404,921.29 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')         | 369,915.95 € | Kapitalerhalt |
| ('Aggressive', 'Transformer')  | 337,244.27 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')    | 325,217.37 € | Kapitalerhalt |
| ('Low_Capital', 'MSM')         | 460,435.28 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')         | 288,013.58 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')        | 262,125.25 € | Kapitalerhalt |
| ('Low_Capital', 'Transformer') | 242,367.66 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 727,287.74 €        |
| ('Standard', 'MSM')            | 0.00%                     | 1,217,158.94 €      |
| ('Standard', 'HMM')            | 0.00%                     | 617,382.91 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 521,137.82 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 477,771.25 €        |
| ('Aggressive', 'Buy Hold')     | 1.80%                     | 462,533.00 €        |
| ('Aggressive', 'MSM')          | 0.00%                     | 847,416.47 €        |
| ('Aggressive', 'HMM')          | 0.00%                     | 349,355.72 €        |
| ('Aggressive', 'LSTM')         | 0.50%                     | 262,243.36 €        |
| ('Aggressive', 'Transformer')  | 0.30%                     | 235,015.17 €        |
| ('Low_Capital', 'Buy Hold')    | 0.40%                     | 345,286.76 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 608,670.10 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 290,776.29 €        |
| ('Low_Capital', 'LSTM')        | 0.00%                     | 231,230.59 €        |
| ('Low_Capital', 'Transformer') | 0.10%                     | 207,104.49 €        |

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
| 00_dependencies | 07:15:16 | 07:15:20 | 3.7 |
| 01_data_preprocessing | 07:15:20 | 07:15:25 | 5.3 |
| 02_feature_engineering | 07:15:25 | 07:15:30 | 4.9 |
| 03_regime_switching_models | 07:15:30 | 07:36:53 | 1282.8 |
| 04_backtesting | 07:36:53 | 07:37:00 | 7.1 |
| 05_evaluation | 07:37:00 | 07:39:42 | 162.8 |
| **Gesamt** | | | **1466.6** (24m 26.6s) |

---

## Modell-Persistierung

Status der Modell-Persistierung für diesen Pipeline-Durchlauf:

- **Persistierung:** AKTIV
- **Modell-Verzeichnis:** `../models`

| Modell | Datei | Status |
|:---|:---|:---|
| MSM | `msm_regime_model.pkl` | Geladen (persistiert) |
| HMM | `hmm_regime_model.pkl` | Geladen (persistiert) |
| LSTM | `lstm_regime_model.keras` | Geladen (persistiert) |
| TRANSFORMER | `transformer_regime_model.pt` | Geladen (persistiert) |

> **Hinweis:** Bei aktivierter Persistierung werden vortrainierte Modelle aus `../models` geladen, sofern die Dateien existieren. Andernfalls wird normal trainiert und das Ergebnis für zukünftige Läufe gespeichert. Bei Änderungen an Hyperparametern müssen die entsprechenden Modelldateien gelöscht werden.

---

**Zuletzt aktualisiert:** 12.03.2026 07:39<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
