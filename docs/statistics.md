
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         4.723  | +372.30%       | -34.77%        |
| MSM         |         2.9282 | +192.82%       | -24.92%        |
| HMM         |         3.6453 | +264.53%       | -8.50%         |
| LSTM        |         4.4544 | +345.44%       | -27.71%        |
| Transformer |         4.5206 | +352.06%       | -27.71%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |    Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|-------:|-----------------:|-----------:|
| Returns_GSPC  |             0.000324 |           0.01139  | -0.1277 | 0.1096 |          -0.36   |    10.8118 |
| Returns_VUSTX |             0.000274 |           0.007486 | -0.0605 | 0.1296 |           0.6394 |    14.3742 |
| Returns       |             0.000304 |           0.006935 | -0.0662 | 0.0584 |          -0.2264 |     7.7476 |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.4865 | 4.4603e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -18.1803 | 2.434e-30  |           -2.8619 | Ja           |
| Returns       |        -17.5053 | 4.3639e-30 |           -2.8619 | Ja           |

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
Vorhersage der Marktphasen durch das neuronale Netzwerk (trainiert auf Pagan-Sossounov-Labels).
![LSTM Model](../assets/lstm_model.png)

### D. Transformer-Netzwerk (Attention-basierte Regime-Erkennung)
Klassifikation von Marktregimes mittels eines Transformer-Encoders mit Multi-Head Self-Attention und Positional Encoding. Im Gegensatz zu rekurrenten Architekturen (LSTM) verarbeitet der Transformer alle Zeitschritte einer Sequenz parallel und lernt über den Attention-Mechanismus, welche historischen Datenpunkte die höchste Relevanz für die aktuelle Regime-Klassifikation besitzen. Trainiert im Supervised-Setting auf Pagan-Sossounov-Labels.
![Transformer Model](../assets/transformer_model.png)

### E. Globaler Regime-Vergleich
Detaillierte Gegenüberstellung der Wahrscheinlichkeiten und harten Signale aller Modelle.
![Regime Comparison](../assets/regime_comparison.png)

### F. Label-Konkordanz (Auswahl der Trainings-Labels)
Vergleich der Regime-Labeler (MSM, HMM, Pagan-Sossounov, Peak-to-Trough, Lunde-Timmermann, NBER) zur Begründung der Label-Wahl für die Supervised-Modelle. Pagan-Sossounov wurde aufgrund seiner hohen Konkordanz mit NBER-Rezessionsperioden als Trainingsziel für LSTM und Transformer gewählt.

![Label Concordance](../assets/label_concordance_matrix.png)
![Label Timeline](../assets/label_timeline_comparison.png)

---

## 4. Backtesting & Strategie-Evaluation
Die ökonomische Anwendung der Regime-Signale durch dynamische Umschichtung in den Geldmarkt.

### Equity Curves im Vergleich
![Equity Curves](../assets/equity_curves.png)

### Annualisierte Performance-Metriken
Normalisierte Kennzahlen (CAGR, Sharpe, Sortino, Calmar) für den Vergleich über unterschiedlich lange Evaluationszeiträume.

| Strategie   | CAGR   | Ann. Volatilität   |   Sharpe Ratio |   Sortino Ratio | Max Drawdown   |   Calmar Ratio |   OOS-Tage |   OOS-Jahre |
|:------------|:-------|:-------------------|---------------:|----------------:|:---------------|---------------:|-----------:|------------:|
| Buy_Hold    | +6.32% | 11.24%             |          0.562 |           0.726 | -34.77%        |          0.182 |       6400 |        25.4 |
| MSM         | +4.34% | 6.61%              |          0.656 |           0.764 | -24.92%        |          0.174 |       6400 |        25.4 |
| HMM         | +5.24% | 5.05%              |          1.037 |           1.028 | -8.50%         |          0.616 |       6400 |        25.4 |
| LSTM        | +6.07% | 10.84%             |          0.56  |           0.692 | -27.71%        |          0.219 |       6400 |        25.4 |
| Transformer | +6.13% | 9.71%              |          0.632 |           0.764 | -27.71%        |          0.221 |       6400 |        25.4 |

### Krisen-Performance
Return und Max Drawdown während historischer Krisenperioden — der zentrale Nachweis für den Tail-Risk-Schutz der Regime-Switching-Modelle.

| Krise                                | ('Return', 'Buy_Hold')   | ('Return', 'HMM')   | ('Return', 'LSTM')   | ('Return', 'MSM')   | ('Return', 'Transformer')   | ('Max Drawdown', 'Buy_Hold')   | ('Max Drawdown', 'HMM')   | ('Max Drawdown', 'LSTM')   | ('Max Drawdown', 'MSM')   | ('Max Drawdown', 'Transformer')   |
|:-------------------------------------|:-------------------------|:--------------------|:---------------------|:--------------------|:----------------------------|:-------------------------------|:--------------------------|:---------------------------|:--------------------------|:----------------------------------|
| COVID Crash (2020-02 – 2020-03)      | -8.24%                   | +2.21%              | -8.24%               | +0.73%              | -8.24%                      | -18.53%                        | -0.36%                    | -18.53%                    | -1.81%                    | -18.53%                           |
| Dot-Com (2000-03 – 2002-10)          | -14.76%                  | +0.31%              | -14.76%              | -18.18%             | -14.76%                     | -24.79%                        | -6.36%                    | -24.79%                    | -21.03%                   | -24.79%                           |
| EU-Schuldenkrise (2011-07 – 2011-11) | +4.10%                   | +0.01%              | +4.10%               | -1.37%              | +1.42%                      | -7.24%                         | 0.00%                     | -7.24%                     | -8.17%                    | -7.24%                            |
| GFC (2007-10 – 2009-03)              | -25.67%                  | +1.01%              | -17.40%              | -8.75%              | -4.25%                      | -34.77%                        | -2.90%                    | -27.70%                    | -9.57%                    | -9.23%                            |
| Zinsanstieg (2022-01 – 2022-10)      | -24.20%                  | -1.71%              | -24.20%              | -4.21%              | -24.20%                     | -26.98%                        | -1.82%                    | -26.98%                    | -6.55%                    | -26.98%                           |

### Drawdown-Verlauf
![Drawdown](../assets/drawdown.png)

### Rollierender Sharpe Ratio
Zeitvariierender, risikoadjustierter Rendite-Vergleich über ein rollendes 252-Tage-Fenster.

![Rolling Sharpe](../assets/rolling_sharpe.png)

### Umfassende Kennzahlen-Matrix
Detaillierte statistische Analyse inklusive risikoadjustierter Kennzahlen (Sharpe, Sortino, Calmar).

| Strategie   | Total Return   | CAGR (p.a.)   | Volatilität   | Max Drawdown   |   Sharpe Ratio |   Sortino Ratio |   Calmar Ratio |   Regime-Wechsel | Gesamtkosten (Gebühren)   |
|:------------|:---------------|:--------------|:--------------|:---------------|---------------:|----------------:|---------------:|-----------------:|:--------------------------|
| Buy Hold    | 373.95%        | 6.30%         | 11.23%        | -34.77%        |           0.6  |            0.79 |           0.18 |                0 | 0.00%                     |
| MSM         | 193.85%        | 4.32%         | 6.61%         | -24.92%        |           0.68 |            0.79 |           0.17 |              314 | 31.50%                    |
| HMM         | 265.80%        | 5.22%         | 5.05%         | -8.50%         |           1.04 |            1.03 |           0.61 |               96 | 9.70%                     |
| LSTM        | 347.00%        | 6.05%         | 10.84%        | -27.71%        |           0.6  |            0.75 |           0.22 |               18 | 1.80%                     |
| Transformer | 353.64%        | 6.11%         | 9.70%         | -27.71%        |           0.66 |            0.81 |           0.22 |               38 | 3.80%                     |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital   | Status           |
|:-------------------------------|:-------------|:-----------------|
| ('Standard', 'Buy Hold')       | 137,992.74 € | Kapitalerhalt    |
| ('Standard', 'MSM')            | 0.00 €       | Erschöpft (2021) |
| ('Standard', 'HMM')            | 4,773.76 €   | Kapitalerhalt    |
| ('Standard', 'LSTM')           | 89,697.84 €  | Kapitalerhalt    |
| ('Standard', 'Transformer')    | 109,341.85 € | Kapitalerhalt    |
| ('Aggressive', 'Buy Hold')     | 0.00 €       | Erschöpft (2012) |
| ('Aggressive', 'MSM')          | 0.00 €       | Erschöpft (2010) |
| ('Aggressive', 'HMM')          | 0.00 €       | Erschöpft (2012) |
| ('Aggressive', 'LSTM')         | 0.00 €       | Erschöpft (2012) |
| ('Aggressive', 'Transformer')  | 0.00 €       | Erschöpft (2012) |
| ('Low_Capital', 'Buy Hold')    | 0.00 €       | Erschöpft (2016) |
| ('Low_Capital', 'MSM')         | 0.00 €       | Erschöpft (2013) |
| ('Low_Capital', 'HMM')         | 0.00 €       | Erschöpft (2015) |
| ('Low_Capital', 'LSTM')        | 0.00 €       | Erschöpft (2015) |
| ('Low_Capital', 'Transformer') | 0.00 €       | Erschöpft (2016) |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'HMM')            | 0.00%                     | 439,739.20 €        |
| ('Standard', 'Buy Hold')       | 0.00%                     | 497,000.32 €        |
| ('Aggressive', 'Buy Hold')     | 4.59%                     | 245,599.29 €        |
| ('Low_Capital', 'Buy Hold')    | 0.53%                     | 212,454.13 €        |
| ('Standard', 'MSM')            | 0.00%                     | 388,020.49 €        |
| ('Low_Capital', 'MSM')         | 0.03%                     | 155,326.34 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 478,670.28 €        |
| ('Aggressive', 'HMM')          | 0.02%                     | 204,691.21 €        |
| ('Aggressive', 'MSM')          | 2.11%                     | 161,941.41 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 480,114.47 €        |
| ('Aggressive', 'LSTM')         | 4.86%                     | 228,169.28 €        |
| ('Low_Capital', 'LSTM')        | 0.47%                     | 206,503.79 €        |
| ('Low_Capital', 'Transformer') | 0.28%                     | 207,305.81 €        |
| ('Aggressive', 'Transformer')  | 3.11%                     | 240,154.44 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 185,934.81 €        |

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
| 00_dependencies | 19:17:06 | 19:17:08 | 2.7 |
| 01_data_preprocessing | 19:17:08 | 19:17:15 | 6.6 |
| 02_feature_engineering | 19:17:15 | 19:17:20 | 4.6 |
| 03_regime_switching_models | 19:17:20 | 20:22:38 | 3918.8 |
| 04_backtesting | 20:22:38 | 20:22:45 | 6.5 |
| 05_evaluation | 20:22:45 | 20:25:20 | 155.1 |
| **Gesamt** | | | **4094.3** (68m 14.3s) |

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
| TRANSFORMER | `transformer_regime_model.pt` | Geladen (persistiert) |

> **Hinweis:** Bei aktivierter Persistierung werden vortrainierte Modelle aus `../models` geladen, sofern die Dateien existieren. Andernfalls wird normal trainiert und das Ergebnis für zukünftige Läufe gespeichert. Bei Änderungen an Hyperparametern müssen die entsprechenden Modelldateien gelöscht werden.

---

**Zuletzt aktualisiert:** 15.04.2026 09:42<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Walk-Forward-Validierung:** AKTIV (Modus: rolling, Train: 10J, Test: 12M, Step: 12M)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
