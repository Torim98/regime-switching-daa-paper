
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         9.78   | +878.00%       | -34.77%        |
| MSM         |         5.4783 | +447.83%       | -22.27%        |
| HMM         |         5.515  | +451.50%       | -11.43%        |
| LSTM        |         3.7143 | +271.43%       | -15.82%        |
| Transformer |         3.1624 | +216.24%       | -13.23%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |    Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|-------:|-----------------:|-----------:|
| Returns_GSPC  |             0.000323 |           0.01139  | -0.1277 | 0.1096 |          -0.3598 |    10.8123 |
| Returns_VUSTX |             0.000274 |           0.007486 | -0.0605 | 0.1296 |           0.6396 |    14.3741 |
| Returns       |             0.000303 |           0.006935 | -0.0662 | 0.0584 |          -0.2266 |     7.7495 |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.4826 | 4.4809e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -18.1717 | 2.446e-30  |           -2.8619 | Ja           |
| Returns       |        -17.4968 | 4.4069e-30 |           -2.8619 | Ja           |

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

### Annualisierte Performance-Metriken
Normalisierte Kennzahlen (CAGR, Sharpe, Sortino, Calmar) für den Vergleich über unterschiedlich lange Evaluationszeiträume.

| Strategie   | CAGR   | Ann. Volatilität   |   Sharpe Ratio |   Sortino Ratio | Max Drawdown   |   Calmar Ratio |   OOS-Tage |   OOS-Jahre |
|:------------|:-------|:-------------------|---------------:|----------------:|:---------------|---------------:|-----------:|------------:|
| Buy_Hold    | +8.08% | 11.22%             |          0.72  |           0.94  | -34.77%        |          0.232 |       7400 |        29.4 |
| MSM         | +5.96% | 7.43%              |          0.802 |           0.954 | -22.27%        |          0.268 |       7400 |        29.4 |
| HMM         | +5.99% | 5.43%              |          1.103 |           1.101 | -11.43%        |          0.524 |       7400 |        29.4 |
| LSTM        | +4.57% | 5.96%              |          0.767 |           0.773 | -15.82%        |          0.289 |       7400 |        29.4 |
| Transformer | +4.00% | 5.52%              |          0.725 |           0.735 | -13.23%        |          0.302 |       7400 |        29.4 |

### Krisen-Performance
Return und Max Drawdown während historischer Krisenperioden — der zentrale Nachweis für den Tail-Risk-Schutz der Regime-Switching-Modelle.

| Krise                                | ('Return', 'Buy_Hold')   | ('Return', 'HMM')   | ('Return', 'LSTM')   | ('Return', 'MSM')   | ('Return', 'Transformer')   | ('Max Drawdown', 'Buy_Hold')   | ('Max Drawdown', 'HMM')   | ('Max Drawdown', 'LSTM')   | ('Max Drawdown', 'MSM')   | ('Max Drawdown', 'Transformer')   |
|:-------------------------------------|:-------------------------|:--------------------|:---------------------|:--------------------|:----------------------------|:-------------------------------|:--------------------------|:---------------------------|:--------------------------|:----------------------------------|
| COVID Crash (2020-02 – 2020-03)      | -8.24%                   | +0.13%              | -2.69%               | +0.73%              | -0.47%                      | -18.53%                        | -0.00%                    | -3.36%                     | -1.81%                    | -1.81%                            |
| Dot-Com (2000-03 – 2002-10)          | -5.04%                   | +1.14%              | +1.62%               | -13.11%             | -8.12%                      | -20.82%                        | -8.45%                    | -8.36%                     | -16.33%                   | -12.21%                           |
| EU-Schuldenkrise (2011-07 – 2011-11) | +4.10%                   | -3.34%              | -0.16%               | -1.37%              | -1.05%                      | -7.24%                         | -4.46%                    | -1.29%                     | -8.17%                    | -1.90%                            |
| GFC (2007-10 – 2009-03)              | -25.67%                  | +1.78%              | -6.68%               | -4.19%              | -4.32%                      | -34.77%                        | -1.17%                    | -10.85%                    | -5.71%                    | -7.29%                            |
| Zinsanstieg (2022-01 – 2022-10)      | -24.20%                  | +0.46%              | -10.64%              | -4.82%              | -5.10%                      | -26.98%                        | -1.28%                    | -15.38%                    | -7.15%                    | -5.69%                            |

### Drawdown-Verlauf
![Drawdown](../assets/drawdown.png)

### Rollierender Sharpe Ratio
Zeitvariierender, risikoadjustierter Rendite-Vergleich über ein rollendes 252-Tage-Fenster.

![Rolling Sharpe](../assets/rolling_sharpe.png)

### Umfassende Kennzahlen-Matrix
Detaillierte statistische Analyse inklusive risikoadjustierter Kennzahlen (Sharpe, Sortino, Calmar).

| Strategie   | Total Return   | CAGR (p.a.)   | Volatilität   | Max Drawdown   |   Sharpe Ratio |   Sortino Ratio |   Calmar Ratio |   Regime-Wechsel | Gesamtkosten (Gebühren)   |
|:------------|:---------------|:--------------|:--------------|:---------------|---------------:|----------------:|---------------:|-----------------:|:--------------------------|
| Buy Hold    | 878.08%        | 7.77%         | 11.21%        | -34.77%        |           0.75 |            0.99 |           0.22 |                0 | 0.00%                     |
| MSM         | 447.88%        | 5.74%         | 7.42%         | -22.27%        |           0.82 |            0.98 |           0.26 |              485 | 46.90%                    |
| HMM         | 451.54%        | 5.76%         | 5.43%         | -11.43%        |           1.1  |            1.1  |           0.5  |              136 | 13.10%                    |
| LSTM        | 271.46%        | 4.40%         | 5.96%         | -15.82%        |           0.78 |            0.79 |           0.28 |              181 | 18.10%                    |
| Transformer | 216.26%        | 3.85%         | 5.51%         | -13.23%        |           0.74 |            0.75 |           0.29 |              247 | 24.70%                    |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital     | Status           |
|:-------------------------------|:---------------|:-----------------|
| ('Standard', 'Buy Hold')       | 1,790,937.85 € | Kapitalerhalt    |
| ('Standard', 'MSM')            | 384,583.73 €   | Kapitalerhalt    |
| ('Standard', 'HMM')            | 373,534.67 €   | Kapitalerhalt    |
| ('Standard', 'LSTM')           | 300,123.85 €   | Kapitalerhalt    |
| ('Standard', 'Transformer')    | 0.00 €         | Erschöpft (2023) |
| ('Aggressive', 'Buy Hold')     | 0.00 €         | Erschöpft (2024) |
| ('Aggressive', 'MSM')          | 0.00 €         | Erschöpft (2012) |
| ('Aggressive', 'HMM')          | 0.00 €         | Erschöpft (2011) |
| ('Aggressive', 'LSTM')         | 0.00 €         | Erschöpft (2014) |
| ('Aggressive', 'Transformer')  | 0.00 €         | Erschöpft (2010) |
| ('Low_Capital', 'Buy Hold')    | 454,670.72 €   | Kapitalerhalt    |
| ('Low_Capital', 'MSM')         | 0.00 €         | Erschöpft (2018) |
| ('Low_Capital', 'HMM')         | 0.00 €         | Erschöpft (2017) |
| ('Low_Capital', 'LSTM')        | 0.00 €         | Erschöpft (2021) |
| ('Low_Capital', 'Transformer') | 0.00 €         | Erschöpft (2014) |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 632,119.68 €        |
| ('Standard', 'MSM')            | 0.00%                     | 489,343.21 €        |
| ('Standard', 'HMM')            | 0.00%                     | 478,311.61 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 392,734.33 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 374,371.43 €        |
| ('Aggressive', 'Buy Hold')     | 1.40%                     | 340,304.17 €        |
| ('Aggressive', 'MSM')          | 1.10%                     | 234,534.93 €        |
| ('Aggressive', 'HMM')          | 0.00%                     | 240,864.42 €        |
| ('Aggressive', 'LSTM')         | 1.30%                     | 171,491.63 €        |
| ('Aggressive', 'Transformer')  | 1.60%                     | 152,190.02 €        |
| ('Low_Capital', 'Buy Hold')    | 0.10%                     | 286,132.50 €        |
| ('Low_Capital', 'MSM')         | 0.10%                     | 207,072.50 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 209,463.03 €        |
| ('Low_Capital', 'LSTM')        | 0.00%                     | 163,528.76 €        |
| ('Low_Capital', 'Transformer') | 0.00%                     | 153,446.99 €        |

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
| 00_dependencies | 08:53:09 | 08:53:11 | 2.8 |
| 01_data_preprocessing | 08:53:11 | 08:53:18 | 6.7 |
| 02_feature_engineering | 08:53:18 | 08:53:23 | 4.7 |
| 03_regime_switching_models | 08:53:23 | 09:28:34 | 2111.4 |
| 04_backtesting | 09:28:34 | 09:28:40 | 6.1 |
| 05_evaluation | 09:28:40 | 09:30:44 | 124.0 |
| **Gesamt** | | | **2255.7** (37m 35.7s) |

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

**Zuletzt aktualisiert:** 12.04.2026 10:18<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Walk-Forward-Validierung:** AKTIV (Modus: rolling, Train: 5J, Test: 6M, Step: 6M)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
