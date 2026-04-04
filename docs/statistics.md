
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         1.6649 | +66.49%        | -27.71%        |
| MSM         |         2.3633 | +136.33%       | -5.81%         |
| HMM         |         1.6778 | +67.78%        | -5.66%         |
| LSTM        |         1.3101 | +31.01%        | -21.99%        |
| Transformer |         1.6586 | +65.86%        | -15.16%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |    Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|-------:|-----------------:|-----------:|
| Returns_GSPC  |             0.00032  |           0.01139  | -0.1277 | 0.1096 |          -0.3602 |    10.8172 |
| Returns_VUSTX |             0.000274 |           0.007488 | -0.0605 | 0.1296 |           0.6393 |    14.3662 |
| Returns       |             0.000301 |           0.006934 | -0.0662 | 0.0584 |          -0.2273 |     7.7535 |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.4523 | 4.6454e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -18.1632 | 2.4582e-30 |           -2.8619 | Ja           |
| Returns       |        -17.457  | 4.6195e-30 |           -2.8619 | Ja           |

**Volatilitätscluster und Autokorrelation (Heteroskedastizität):**
![Volatility Clusters](../assets/eda_volatility_clusters.png)

### Feature-Korrelation
Pearson-Korrelationsmatrix der sechs Modell-Features zur Prüfung auf Multikollinearität.

![Feature Correlation Matrix](../assets/feature_correlation_matrix.png)

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
| Buy Hold    | 67.41%         | 7.60%         | 12.71%        | -27.71%        |           0.64 |            0.83 |           0.27 |                0 | 0.00%                     |
| MSM         | 137.64%        | 13.10%        | 6.30%         | -5.81%         |           1.99 |            2.56 |           2.26 |               43 | 4.30%                     |
| HMM         | 68.71%         | 7.72%         | 4.78%         | -5.66%         |           1.58 |            1.45 |           1.36 |               31 | 3.10%                     |
| LSTM        | 31.74%         | 4.00%         | 10.18%        | -21.99%        |           0.44 |            0.35 |           0.18 |               81 | 8.20%                     |
| Transformer | 66.78%         | 7.55%         | 8.31%         | -15.16%        |           0.92 |            1.02 |           0.5  |              133 | 13.30%                    |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status        |
|:-------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')       | 568,281.60 € | Kapitalerhalt |
| ('Standard', 'MSM')            | 856,643.77 € | Kapitalerhalt |
| ('Standard', 'HMM')            | 564,126.10 € | Kapitalerhalt |
| ('Standard', 'LSTM')           | 398,430.95 € | Kapitalerhalt |
| ('Standard', 'Transformer')    | 570,086.41 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')     | 407,011.00 € | Kapitalerhalt |
| ('Aggressive', 'MSM')          | 657,711.02 € | Kapitalerhalt |
| ('Aggressive', 'HMM')          | 396,471.39 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')         | 242,283.52 € | Kapitalerhalt |
| ('Aggressive', 'Transformer')  | 411,791.50 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')    | 287,212.09 € | Kapitalerhalt |
| ('Low_Capital', 'MSM')         | 447,675.34 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')         | 282,590.76 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')        | 187,009.43 € | Kapitalerhalt |
| ('Low_Capital', 'Transformer') | 289,286.88 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 584,925.72 €        |
| ('Standard', 'MSM')            | 0.00%                     | 1,095,143.88 €      |
| ('Standard', 'HMM')            | 0.00%                     | 596,009.29 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 386,989.74 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 579,124.67 €        |
| ('Aggressive', 'Buy Hold')     | 4.20%                     | 330,000.03 €        |
| ('Aggressive', 'MSM')          | 0.00%                     | 758,497.87 €        |
| ('Aggressive', 'HMM')          | 0.00%                     | 329,994.70 €        |
| ('Aggressive', 'LSTM')         | 9.10%                     | 148,517.50 €        |
| ('Aggressive', 'Transformer')  | 0.60%                     | 317,500.84 €        |
| ('Low_Capital', 'Buy Hold')    | 0.80%                     | 271,587.46 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 549,684.16 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 272,284.37 €        |
| ('Low_Capital', 'LSTM')        | 1.10%                     | 143,164.95 €        |
| ('Low_Capital', 'Transformer') | 0.00%                     | 269,047.75 €        |

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
| 00_dependencies | 14:27:20 | 14:27:22 | 2.7 |
| 01_data_preprocessing | 14:27:22 | 14:27:29 | 6.7 |
| 02_feature_engineering | 14:27:29 | 14:27:34 | 4.7 |
| 03_regime_switching_models | 14:27:34 | 14:27:47 | 13.0 |
| 04_backtesting | 14:27:47 | 14:27:51 | 4.8 |
| 05_evaluation | 14:27:51 | 14:29:57 | 125.9 |
| **Gesamt** | | | **157.8** (2m 37.8s) |

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

**Zuletzt aktualisiert:** 04.04.2026 14:30<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
