
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
| LSTM        |         1.3134 | +31.34%        | -24.39%        |
| Transformer |         1.3651 | +36.51%        | -12.15%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |    Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|-------:|-----------------:|-----------:|
| Returns_GSPC  |             0.000316 |           0.011387 | -0.1277 | 0.1096 |          -0.3612 |    10.8279 |
| Returns_VUSTX |             0.000274 |           0.007489 | -0.0605 | 0.1296 |           0.6394 |    14.3626 |
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
| HMM         | 69.30%         | 7.78%         | 4.80%         | -5.66%         |           1.59 |            1.46 |           1.37 |               31 | 3.10%                     |
| LSTM        | 31.44%         | 3.97%         | 10.42%        | -24.39%        |           0.43 |            0.35 |           0.16 |               95 | 9.60%                     |
| Transformer | 36.61%         | 4.54%         | 7.88%         | -12.15%        |           0.6  |            0.69 |           0.37 |              121 | 12.10%                    |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status        |
|:-------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')       | 560,588.11 € | Kapitalerhalt |
| ('Standard', 'MSM')            | 854,746.70 € | Kapitalerhalt |
| ('Standard', 'HMM')            | 569,653.31 € | Kapitalerhalt |
| ('Standard', 'LSTM')           | 420,697.22 € | Kapitalerhalt |
| ('Standard', 'Transformer')    | 444,565.76 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')     | 404,556.47 € | Kapitalerhalt |
| ('Aggressive', 'MSM')          | 657,790.11 € | Kapitalerhalt |
| ('Aggressive', 'HMM')          | 403,560.17 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')         | 278,787.77 € | Kapitalerhalt |
| ('Aggressive', 'Transformer')  | 301,467.45 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')    | 284,342.32 € | Kapitalerhalt |
| ('Low_Capital', 'MSM')         | 447,195.82 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')         | 286,427.61 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')        | 205,115.18 € | Kapitalerhalt |
| ('Low_Capital', 'Transformer') | 219,040.02 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.10%                     | 596,646.83 €        |
| ('Standard', 'MSM')            | 0.00%                     | 1,092,078.33 €      |
| ('Standard', 'HMM')            | 0.00%                     | 599,912.89 €        |
| ('Standard', 'LSTM')           | 0.10%                     | 392,723.83 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 409,448.63 €        |
| ('Aggressive', 'Buy Hold')     | 4.50%                     | 329,272.21 €        |
| ('Aggressive', 'MSM')          | 0.00%                     | 742,829.63 €        |
| ('Aggressive', 'HMM')          | 0.00%                     | 328,694.72 €        |
| ('Aggressive', 'LSTM')         | 8.10%                     | 158,935.26 €        |
| ('Aggressive', 'Transformer')  | 4.30%                     | 164,918.18 €        |
| ('Low_Capital', 'Buy Hold')    | 0.90%                     | 268,819.04 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 539,891.71 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 274,298.21 €        |
| ('Low_Capital', 'LSTM')        | 1.10%                     | 154,884.64 €        |
| ('Low_Capital', 'Transformer') | 0.30%                     | 166,601.42 €        |

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
| 00_dependencies | 17:23:57 | 17:24:01 | 4.0 |
| 01_data_preprocessing | 17:24:01 | 17:24:09 | 8.1 |
| 02_feature_engineering | 17:24:09 | 17:24:14 | 5.0 |
| 03_regime_switching_models | 17:24:14 | 18:09:55 | 2741.4 |
| 04_backtesting | 18:09:55 | 18:10:04 | 8.5 |
| 05_evaluation | 18:10:04 | 18:12:43 | 159.3 |
| **Gesamt** | | | **2926.3** (48m 46.3s) |

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

**Zuletzt aktualisiert:** 31.03.2026 18:12<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
