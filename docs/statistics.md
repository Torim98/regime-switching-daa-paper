
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         1.64   | +64.00%        | -27.71%        |
| MSM         |         2.3642 | +136.42%       | -5.81%         |
| HMM         |         1.6917 | +69.17%        | -5.66%         |
| LSTM        |         1.5736 | +57.36%        | -10.23%        |
| Transformer |         1.3157 | +31.57%        | -18.13%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |    Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|-------:|-----------------:|-----------:|
| Returns_GSPC  |             0.000316 |           0.011387 | -0.1277 | 0.1096 |          -0.3612 |    10.8279 |
| Returns_VUSTX |             0.000274 |           0.007489 | -0.0605 | 0.1296 |           0.6394 |    14.3625 |
| Returns       |             0.000299 |           0.006933 | -0.0662 | 0.0584 |          -0.228  |     7.7607 |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.3937 | 4.9924e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -18.1513 | 2.4755e-30 |           -2.8619 | Ja           |
| Returns       |        -17.3921 | 5.0023e-30 |           -2.8619 | Ja           |

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
| Buy Hold    | 64.13%         | 7.30%         | 12.69%        | -27.71%        |           0.62 |            0.8  |           0.26 |                0 | 0.00%                     |
| MSM         | 136.60%        | 13.03%        | 6.32%         | -5.81%         |           1.98 |            2.54 |           2.24 |               43 | 4.30%                     |
| HMM         | 69.29%         | 7.78%         | 4.80%         | -5.66%         |           1.59 |            1.46 |           1.37 |               31 | 3.10%                     |
| LSTM        | 57.48%         | 6.67%         | 7.72%         | -10.23%        |           0.88 |            1.02 |           0.65 |               79 | 7.90%                     |
| Transformer | 31.67%         | 3.99%         | 7.54%         | -18.13%        |           0.56 |            0.63 |           0.22 |               79 | 7.90%                     |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status        |
|:-------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')       | 560,587.89 € | Kapitalerhalt |
| ('Standard', 'MSM')            | 854,746.92 € | Kapitalerhalt |
| ('Standard', 'HMM')            | 569,652.61 € | Kapitalerhalt |
| ('Standard', 'LSTM')           | 534,769.83 € | Kapitalerhalt |
| ('Standard', 'Transformer')    | 432,569.06 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')     | 404,556.25 € | Kapitalerhalt |
| ('Aggressive', 'MSM')          | 657,790.30 € | Kapitalerhalt |
| ('Aggressive', 'HMM')          | 403,559.52 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')         | 383,206.28 € | Kapitalerhalt |
| ('Aggressive', 'Transformer')  | 297,094.22 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')    | 284,342.19 € | Kapitalerhalt |
| ('Low_Capital', 'MSM')         | 447,195.94 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')         | 286,427.20 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')        | 270,340.71 € | Kapitalerhalt |
| ('Low_Capital', 'Transformer') | 214,383.16 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.10%                     | 596,647.41 €        |
| ('Standard', 'MSM')            | 0.00%                     | 1,092,078.29 €      |
| ('Standard', 'HMM')            | 0.00%                     | 599,912.09 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 519,539.57 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 362,669.56 €        |
| ('Aggressive', 'Buy Hold')     | 4.50%                     | 329,272.98 €        |
| ('Aggressive', 'MSM')          | 0.00%                     | 742,830.97 €        |
| ('Aggressive', 'HMM')          | 0.00%                     | 328,694.09 €        |
| ('Aggressive', 'LSTM')         | 0.50%                     | 266,991.09 €        |
| ('Aggressive', 'Transformer')  | 7.40%                     | 133,452.56 €        |
| ('Low_Capital', 'Buy Hold')    | 0.90%                     | 268,817.95 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 539,891.71 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 274,297.92 €        |
| ('Low_Capital', 'LSTM')        | 0.10%                     | 230,682.86 €        |
| ('Low_Capital', 'Transformer') | 0.60%                     | 146,587.45 €        |

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
| 00_dependencies | 17:12:38 | 17:12:42 | 3.9 |
| 01_data_preprocessing | 17:12:42 | 17:12:51 | 8.6 |
| 02_feature_engineering | 17:12:51 | 17:12:56 | 5.0 |
| 03_regime_switching_models | 17:12:56 | 17:13:15 | 19.3 |
| 04_backtesting | 17:13:15 | 17:13:22 | 6.9 |
| 05_evaluation | 17:13:22 | 17:16:02 | 160.1 |
| **Gesamt** | | | **203.8** (3m 23.8s) |

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

**Zuletzt aktualisiert:** 31.03.2026 17:16<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
