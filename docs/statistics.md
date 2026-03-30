
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         1.7378 | +73.78%        | -27.10%        |
| MSM         |         2.4434 | +144.34%       | -6.25%         |
| HMM         |         1.7067 | +70.67%        | -5.64%         |
| LSTM        |         1.4105 | +41.05%        | -15.72%        |
| Transformer |         1.4716 | +47.16%        | -13.64%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |    Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|-------:|-----------------:|-----------:|
| Returns_GSPC  |             0.000381 |           0.011374 | -0.1198 | 0.1158 |          -0.1453 |    10.6544 |
| Returns_VUSTX |             0.000301 |           0.00751  | -0.0587 | 0.1384 |           0.8305 |    17.2712 |
| Returns       |             0.000349 |           0.006929 | -0.0633 | 0.0622 |          -0.0421 |     7.7648 |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.6808 | 3.6088e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -18.1253 | 2.5148e-30 |           -2.8619 | Ja           |
| Returns       |        -17.5082 | 4.3491e-30 |           -2.8619 | Ja           |

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
| Buy Hold    | 73.91%         | 8.20%         | 12.68%        | -27.10%        |           0.69 |            0.89 |           0.3  |                0 | 0.00%                     |
| MSM         | 144.52%        | 13.58%        | 6.32%         | -6.25%         |           2.05 |            2.65 |           2.17 |               43 | 4.30%                     |
| HMM         | 70.80%         | 7.92%         | 4.80%         | -5.64%         |           1.62 |            1.49 |           1.4  |               31 | 3.10%                     |
| LSTM        | 41.16%         | 5.03%         | 9.83%         | -15.72%        |           0.55 |            0.43 |           0.32 |               75 | 7.60%                     |
| Transformer | 47.27%         | 5.67%         | 7.44%         | -13.64%        |           0.78 |            0.91 |           0.42 |               93 | 9.30%                     |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status        |
|:-------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')       | 603,776.68 € | Kapitalerhalt |
| ('Standard', 'MSM')            | 889,344.41 € | Kapitalerhalt |
| ('Standard', 'HMM')            | 576,051.92 € | Kapitalerhalt |
| ('Standard', 'LSTM')           | 438,259.77 € | Kapitalerhalt |
| ('Standard', 'Transformer')    | 493,359.77 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')     | 444,299.59 € | Kapitalerhalt |
| ('Aggressive', 'MSM')          | 689,386.88 € | Kapitalerhalt |
| ('Aggressive', 'HMM')          | 409,284.29 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')         | 277,747.25 € | Kapitalerhalt |
| ('Aggressive', 'Transformer')  | 347,567.36 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')    | 309,106.98 € | Kapitalerhalt |
| ('Low_Capital', 'MSM')         | 466,954.13 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')         | 290,041.94 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')        | 209,451.69 € | Kapitalerhalt |
| ('Low_Capital', 'Transformer') | 247,418.39 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 659,028.77 €        |
| ('Standard', 'MSM')            | 0.00%                     | 1,159,629.47 €      |
| ('Standard', 'HMM')            | 0.00%                     | 609,673.49 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 479,354.68 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 466,537.59 €        |
| ('Aggressive', 'Buy Hold')     | 2.70%                     | 390,580.52 €        |
| ('Aggressive', 'MSM')          | 0.00%                     | 802,614.35 €        |
| ('Aggressive', 'HMM')          | 0.00%                     | 333,388.47 €        |
| ('Aggressive', 'LSTM')         | 2.90%                     | 224,586.78 €        |
| ('Aggressive', 'Transformer')  | 2.10%                     | 209,058.11 €        |
| ('Low_Capital', 'Buy Hold')    | 0.30%                     | 316,361.11 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 584,344.76 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 281,113.37 €        |
| ('Low_Capital', 'LSTM')        | 0.10%                     | 197,229.21 €        |
| ('Low_Capital', 'Transformer') | 0.10%                     | 195,544.68 €        |

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
| 00_dependencies | 10:04:25 | 10:04:30 | 4.4 |
| 01_data_preprocessing | 10:04:30 | 10:04:54 | 24.2 |
| 02_feature_engineering | 10:04:54 | 10:04:59 | 5.6 |
| 03_regime_switching_models | 10:04:59 | 10:52:41 | 2861.7 |
| 04_backtesting | 10:52:41 | 10:52:48 | 7.0 |
| 05_evaluation | 10:52:48 | 10:55:33 | 165.0 |
| **Gesamt** | | | **3067.9** (51m 7.9s) |

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

**Zuletzt aktualisiert:** 30.03.2026 10:55<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
