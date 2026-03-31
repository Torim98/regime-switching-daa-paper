
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         1.7427 | +74.27%        | -27.10%        |
| MSM         |         2.4325 | +143.25%       | -6.25%         |
| HMM         |         1.707  | +70.70%        | -5.64%         |
| LSTM        |         1.7884 | +78.84%        | -10.53%        |
| Transformer |         1.4718 | +47.18%        | -13.64%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |    Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|-------:|-----------------:|-----------:|
| Returns_GSPC  |             0.00038  |           0.011373 | -0.1198 | 0.1158 |          -0.1452 |    10.6555 |
| Returns_VUSTX |             0.000302 |           0.007511 | -0.0587 | 0.1384 |           0.8301 |    17.2616 |
| Returns       |             0.000349 |           0.006928 | -0.0633 | 0.0622 |          -0.0422 |     7.7657 |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.6758 | 3.6274e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -18.1504 | 2.4769e-30 |           -2.8619 | Ja           |
| Returns       |        -17.5568 | 4.1158e-30 |           -2.8619 | Ja           |

**Volatilitätscluster und Autokorrelation (Heteroskedastizität):**
![Volatility Clusters](../assets/eda_volatility_clusters.png)

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
| Buy Hold    | 74.40%         | 8.23%         | 12.68%        | -27.10%        |           0.69 |            0.89 |           0.3  |                0 | 0.00%                     |
| MSM         | 143.44%        | 13.49%        | 6.31%         | -6.25%         |           2.04 |            2.63 |           2.16 |               43 | 4.30%                     |
| HMM         | 70.82%         | 7.91%         | 4.79%         | -5.64%         |           1.62 |            1.49 |           1.4  |               31 | 3.10%                     |
| LSTM        | 78.97%         | 8.63%         | 8.49%         | -10.53%        |           1.02 |            1.24 |           0.82 |               69 | 6.90%                     |
| Transformer | 47.29%         | 5.66%         | 7.43%         | -13.64%        |           0.78 |            0.91 |           0.42 |               93 | 9.30%                     |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status        |
|:-------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')       | 605,451.56 € | Kapitalerhalt |
| ('Standard', 'MSM')            | 884,445.69 € | Kapitalerhalt |
| ('Standard', 'HMM')            | 576,134.46 € | Kapitalerhalt |
| ('Standard', 'LSTM')           | 615,833.78 € | Kapitalerhalt |
| ('Standard', 'Transformer')    | 493,430.21 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')     | 445,532.08 € | Kapitalerhalt |
| ('Aggressive', 'MSM')          | 684,804.37 € | Kapitalerhalt |
| ('Aggressive', 'HMM')          | 409,343.02 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')         | 448,419.57 € | Kapitalerhalt |
| ('Aggressive', 'Transformer')  | 347,616.97 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')    | 309,964.44 € | Kapitalerhalt |
| ('Low_Capital', 'MSM')         | 464,120.31 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')         | 290,083.53 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')        | 313,695.53 € | Kapitalerhalt |
| ('Low_Capital', 'Transformer') | 247,453.71 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 669,702.30 €        |
| ('Standard', 'MSM')            | 0.00%                     | 1,143,087.22 €      |
| ('Standard', 'HMM')            | 0.00%                     | 609,677.85 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 660,409.39 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 459,852.46 €        |
| ('Aggressive', 'Buy Hold')     | 3.30%                     | 387,469.17 €        |
| ('Aggressive', 'MSM')          | 0.00%                     | 791,155.95 €        |
| ('Aggressive', 'HMM')          | 0.00%                     | 336,593.76 €        |
| ('Aggressive', 'LSTM')         | 0.20%                     | 375,270.79 €        |
| ('Aggressive', 'Transformer')  | 1.70%                     | 210,805.93 €        |
| ('Low_Capital', 'Buy Hold')    | 0.60%                     | 308,924.62 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 570,051.07 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 279,771.55 €        |
| ('Low_Capital', 'LSTM')        | 0.10%                     | 300,312.15 €        |
| ('Low_Capital', 'Transformer') | 0.20%                     | 197,101.87 €        |

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
| 00_dependencies | 11:16:58 | 11:17:02 | 3.8 |
| 01_data_preprocessing | 11:17:02 | 11:17:10 | 8.1 |
| 02_feature_engineering | 11:17:10 | 11:17:15 | 5.0 |
| 03_regime_switching_models | 11:17:15 | 11:19:27 | 132.5 |
| 04_backtesting | 11:19:27 | 11:19:34 | 6.8 |
| 05_evaluation | 11:19:34 | 11:22:14 | 160.2 |
| **Gesamt** | | | **316.4** (5m 16.4s) |

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

**Zuletzt aktualisiert:** 31.03.2026 11:22<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
