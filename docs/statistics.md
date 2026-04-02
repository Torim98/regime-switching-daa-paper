
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         1.6723 | +67.23%        | -27.71%        |
| MSM         |         2.3799 | +137.99%       | -5.81%         |
| HMM         |         1.6896 | +68.96%        | -5.66%         |
| LSTM        |         1.3462 | +34.62%        | -24.39%        |
| Transformer |         1.6434 | +64.34%        | -9.35%         |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |    Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|-------:|-----------------:|-----------:|
| Returns_GSPC  |             0.000319 |           0.01139  | -0.1277 | 0.1096 |          -0.3601 |    10.8157 |
| Returns_VUSTX |             0.000274 |           0.007488 | -0.0605 | 0.1296 |           0.6395 |    14.3661 |
| Returns       |             0.000301 |           0.006935 | -0.0662 | 0.0584 |          -0.2272 |     7.7526 |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.4501 | 4.6576e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -18.1528 | 2.4733e-30 |           -2.8619 | Ja           |
| Returns       |        -17.4499 | 4.659e-30  |           -2.8619 | Ja           |

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
| Buy Hold    | 66.04%         | 7.48%         | 12.71%        | -27.71%        |           0.63 |            0.81 |           0.27 |                0 | 0.00%                     |
| MSM         | 136.30%        | 13.01%        | 6.31%         | -5.81%         |           1.98 |            2.54 |           2.24 |               43 | 4.30%                     |
| HMM         | 67.76%         | 7.64%         | 4.79%         | -5.66%         |           1.56 |            1.44 |           1.35 |               31 | 3.10%                     |
| LSTM        | 33.67%         | 4.21%         | 10.44%        | -24.39%        |           0.45 |            0.37 |           0.17 |               95 | 9.60%                     |
| Transformer | 63.17%         | 7.21%         | 7.21%         | -9.35%         |           1    |            1.11 |           0.77 |              119 | 11.90%                    |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status        |
|:-------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')       | 562,173.72 € | Kapitalerhalt |
| ('Standard', 'MSM')            | 850,009.98 € | Kapitalerhalt |
| ('Standard', 'HMM')            | 559,422.91 € | Kapitalerhalt |
| ('Standard', 'LSTM')           | 424,704.67 € | Kapitalerhalt |
| ('Standard', 'Transformer')    | 550,543.13 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')     | 401,351.78 € | Kapitalerhalt |
| ('Aggressive', 'MSM')          | 651,125.35 € | Kapitalerhalt |
| ('Aggressive', 'HMM')          | 391,806.11 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')         | 278,530.01 € | Kapitalerhalt |
| ('Aggressive', 'Transformer')  | 391,354.51 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')    | 283,696.92 € | Kapitalerhalt |
| ('Low_Capital', 'MSM')         | 443,711.11 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')         | 279,781.48 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')        | 206,097.91 € | Kapitalerhalt |
| ('Low_Capital', 'Transformer') | 277,263.00 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 593,127.51 €        |
| ('Standard', 'MSM')            | 0.00%                     | 1,100,961.33 €      |
| ('Standard', 'HMM')            | 0.00%                     | 598,768.52 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 389,762.95 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 564,418.53 €        |
| ('Aggressive', 'Buy Hold')     | 4.00%                     | 327,009.79 €        |
| ('Aggressive', 'MSM')          | 0.00%                     | 762,598.11 €        |
| ('Aggressive', 'HMM')          | 0.00%                     | 330,459.10 €        |
| ('Aggressive', 'LSTM')         | 8.60%                     | 156,615.80 €        |
| ('Aggressive', 'Transformer')  | 0.00%                     | 297,457.14 €        |
| ('Low_Capital', 'Buy Hold')    | 0.90%                     | 269,015.53 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 552,119.23 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 271,612.21 €        |
| ('Low_Capital', 'LSTM')        | 0.80%                     | 149,482.15 €        |
| ('Low_Capital', 'Transformer') | 0.00%                     | 262,254.02 €        |

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
| 00_dependencies | 14:23:25 | 14:23:29 | 3.8 |
| 01_data_preprocessing | 14:23:29 | 14:23:38 | 8.4 |
| 02_feature_engineering | 14:23:38 | 14:23:42 | 4.9 |
| 03_regime_switching_models | 14:23:42 | 15:04:34 | 2451.3 |
| 04_backtesting | 15:04:34 | 15:04:41 | 7.2 |
| 05_evaluation | 15:04:41 | 15:07:21 | 159.9 |
| **Gesamt** | | | **2635.5** (43m 55.5s) |

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

**Zuletzt aktualisiert:** 02.04.2026 15:07<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
