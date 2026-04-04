
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
| LSTM        |         1.3602 | +36.02%        | -20.21%        |
| Transformer |         1.5022 | +50.22%        | -12.00%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |     Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|--------:|-----------------:|-----------:|
| Returns_GSPC  |             0.00032  |           0.01139  | -0.1277 |  0.1096 |          -0.3602 |    10.8172 |
| Returns_VUSTX |             0.000274 |           0.007488 | -0.0605 |  0.1296 |           0.6393 |    14.3663 |
| Returns       |             0.000301 |           0.006934 | -0.0662 |  0.0584 |          -0.2273 |     7.7535 |
| VIX           |            19.4658   |           7.77018  |  9.14   | 82.69   |           2.2003 |     8.6632 |
| TNX_10Y       |             4.23742  |           1.93226  |  0.499  |  9.09   |           0.3305 |    -0.6388 |
| IRX_3M        |             2.70271  |           2.20343  | -0.105  |  7.99   |           0.2028 |    -1.2566 |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.4523 | 4.6454e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -18.1632 | 2.4582e-30 |           -2.8619 | Ja           |
| Returns       |        -17.457  | 4.6195e-30 |           -2.8619 | Ja           |
| VIX           |         -7.2372 | 1.9255e-10 |           -2.8619 | Ja           |
| TNX_10Y       |         -2.3496 | 0.15648    |           -2.8619 | Nein         |
| IRX_3M        |         -2.3393 | 0.15961    |           -2.8619 | Nein         |

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
| LSTM        | 36.77%         | 4.55%         | 9.96%         | -20.21%        |           0.5  |            0.38 |           0.23 |               77 | 7.80%                     |
| Transformer | 51.05%         | 6.04%         | 7.39%         | -12.00%        |           0.83 |            0.92 |           0.5  |              137 | 13.70%                    |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status        |
|:-------------------------------|:-------------|:--------------|
| ('Standard', 'Buy Hold')       | 568,281.51 € | Kapitalerhalt |
| ('Standard', 'MSM')            | 856,644.01 € | Kapitalerhalt |
| ('Standard', 'HMM')            | 564,125.82 € | Kapitalerhalt |
| ('Standard', 'LSTM')           | 430,668.87 € | Kapitalerhalt |
| ('Standard', 'Transformer')    | 501,372.59 € | Kapitalerhalt |
| ('Aggressive', 'Buy Hold')     | 407,010.91 € | Kapitalerhalt |
| ('Aggressive', 'MSM')          | 657,711.21 € | Kapitalerhalt |
| ('Aggressive', 'HMM')          | 396,471.12 € | Kapitalerhalt |
| ('Aggressive', 'LSTM')         | 278,763.26 € | Kapitalerhalt |
| ('Aggressive', 'Transformer')  | 349,038.37 € | Kapitalerhalt |
| ('Low_Capital', 'Buy Hold')    | 287,212.04 € | Kapitalerhalt |
| ('Low_Capital', 'MSM')         | 447,675.47 € | Kapitalerhalt |
| ('Low_Capital', 'HMM')         | 282,590.59 € | Kapitalerhalt |
| ('Low_Capital', 'LSTM')        | 207,766.12 € | Kapitalerhalt |
| ('Low_Capital', 'Transformer') | 250,045.48 € | Kapitalerhalt |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 584,924.82 €        |
| ('Standard', 'MSM')            | 0.00%                     | 1,095,144.02 €      |
| ('Standard', 'HMM')            | 0.00%                     | 596,009.52 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 424,799.13 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 488,284.43 €        |
| ('Aggressive', 'Buy Hold')     | 4.20%                     | 330,001.18 €        |
| ('Aggressive', 'MSM')          | 0.00%                     | 758,499.04 €        |
| ('Aggressive', 'HMM')          | 0.00%                     | 329,994.21 €        |
| ('Aggressive', 'LSTM')         | 6.20%                     | 175,484.61 €        |
| ('Aggressive', 'Transformer')  | 0.40%                     | 231,992.01 €        |
| ('Low_Capital', 'Buy Hold')    | 0.80%                     | 271,586.50 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 549,684.73 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 272,284.52 €        |
| ('Low_Capital', 'LSTM')        | 0.40%                     | 163,993.76 €        |
| ('Low_Capital', 'Transformer') | 0.00%                     | 207,781.64 €        |

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
| 00_dependencies | 19:45:30 | 19:45:33 | 3.2 |
| 01_data_preprocessing | 19:45:33 | 19:45:40 | 6.8 |
| 02_feature_engineering | 19:45:40 | 19:45:45 | 4.7 |
| 03_regime_switching_models | 19:45:45 | 19:49:15 | 209.6 |
| 04_backtesting | 19:49:15 | 19:49:19 | 4.8 |
| 05_evaluation | 19:49:19 | 19:51:14 | 114.6 |
| **Gesamt** | | | **343.7** (5m 43.7s) |

---

## Modell-Persistierung

Status der Modell-Persistierung für diesen Pipeline-Durchlauf:

- **Persistierung:** AKTIV
- **Modell-Verzeichnis:** `../models`

| Modell | Datei | Status |
|:---|:---|:---|
| MSM | `msm_regime_model.pkl` | Neu trainiert |
| HMM | `hmm_regime_model.pkl` | Neu trainiert |
| LSTM | `lstm_regime_model.keras` | Neu trainiert |
| TRANSFORMER | `transformer_regime_model.pt` | Neu trainiert |

> **Hinweis:** Bei aktivierter Persistierung werden vortrainierte Modelle aus `../models` geladen, sofern die Dateien existieren. Andernfalls wird normal trainiert und das Ergebnis für zukünftige Läufe gespeichert. Bei Änderungen an Hyperparametern müssen die entsprechenden Modelldateien gelöscht werden.

---

**Zuletzt aktualisiert:** 04.04.2026 20:00<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
