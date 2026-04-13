
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         8.8991 | +789.91%       | -34.77%        |
| MSM         |         5.5548 | +455.48%       | -27.66%        |
| HMM         |         5.221  | +422.10%       | -16.30%        |
| LSTM        |         4.4167 | +341.67%       | -15.43%        |
| Transformer |         3.7003 | +270.03%       | -21.98%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |    Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|-------:|-----------------:|-----------:|
| Returns_GSPC  |             0.000323 |           0.01139  | -0.1277 | 0.1096 |          -0.3598 |    10.8123 |
| Returns_VUSTX |             0.000274 |           0.007486 | -0.0605 | 0.1296 |           0.6396 |    14.3739 |
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
Vorhersage der Marktphasen durch das neuronale Netzwerk (trainiert auf HMM-Labels).
![LSTM Model](../assets/lstm_model.png)

### D. Transformer-Netzwerk (Attention-basierte Regime-Erkennung)
"Klassifikation von Marktregimes mittels eines Transformer-Encoders mit Multi-Head Self-Attention und Positional Encoding. Im Gegensatz zu rekurrenten Architekturen (LSTM) verarbeitet der Transformer alle Zeitschritte einer Sequenz parallel und lernt über den Attention-Mechanismus, welche historischen Datenpunkte die höchste Relevanz für die aktuelle Regime-Klassifikation besitzen. Trainiert im Supervised-Setting auf HMM-Labels.
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
| Buy_Hold    | +7.59% | 11.35%             |          0.668 |           0.88  | -34.77%        |          0.218 |       7533 |        29.9 |
| MSM         | +5.90% | 7.12%              |          0.829 |           0.975 | -27.66%        |          0.213 |       7533 |        29.9 |
| HMM         | +5.68% | 5.46%              |          1.042 |           1.032 | -16.30%        |          0.349 |       7533 |        29.9 |
| LSTM        | +5.09% | 6.44%              |          0.791 |           0.819 | -15.43%        |          0.33  |       7533 |        29.9 |
| Transformer | +4.47% | 6.53%              |          0.686 |           0.735 | -21.98%        |          0.204 |       7533 |        29.9 |

### Krisen-Performance
Return und Max Drawdown während historischer Krisenperioden — der zentrale Nachweis für den Tail-Risk-Schutz der Regime-Switching-Modelle.

| Krise                                | ('Return', 'Buy_Hold')   | ('Return', 'HMM')   | ('Return', 'LSTM')   | ('Return', 'MSM')   | ('Return', 'Transformer')   | ('Max Drawdown', 'Buy_Hold')   | ('Max Drawdown', 'HMM')   | ('Max Drawdown', 'LSTM')   | ('Max Drawdown', 'MSM')   | ('Max Drawdown', 'Transformer')   |
|:-------------------------------------|:-------------------------|:--------------------|:---------------------|:--------------------|:----------------------------|:-------------------------------|:--------------------------|:---------------------------|:--------------------------|:----------------------------------|
| COVID Crash (2020-02 – 2020-03)      | -8.24%                   | +0.13%              | +0.13%               | +0.73%              | +0.20%                      | -18.53%                        | -0.00%                    | -0.00%                     | -1.81%                    | -0.09%                            |
| Dot-Com (2000-03 – 2002-10)          | -13.31%                  | -5.90%              | -7.96%               | -20.49%             | -8.88%                      | -27.27%                        | -16.30%                   | -15.43%                    | -24.98%                   | -21.98%                           |
| EU-Schuldenkrise (2011-07 – 2011-11) | +4.10%                   | -3.34%              | -0.76%               | -1.37%              | -2.85%                      | -7.24%                         | -4.46%                    | -2.28%                     | -8.17%                    | -8.33%                            |
| GFC (2007-10 – 2009-03)              | -25.67%                  | +1.78%              | +1.78%               | +0.83%              | +1.78%                      | -34.77%                        | -1.17%                    | -1.17%                     | -2.86%                    | -1.17%                            |
| Zinsanstieg (2022-01 – 2022-10)      | -24.20%                  | +0.46%              | -0.29%               | -3.41%              | +1.30%                      | -26.98%                        | -1.28%                    | -3.87%                     | -6.35%                    | 0.00%                             |

### Drawdown-Verlauf
![Drawdown](../assets/drawdown.png)

### Rollierender Sharpe Ratio
Zeitvariierender, risikoadjustierter Rendite-Vergleich über ein rollendes 252-Tage-Fenster.

![Rolling Sharpe](../assets/rolling_sharpe.png)

### Umfassende Kennzahlen-Matrix
Detaillierte statistische Analyse inklusive risikoadjustierter Kennzahlen (Sharpe, Sortino, Calmar).

| Strategie   | Total Return   | CAGR (p.a.)   | Volatilität   | Max Drawdown   |   Sharpe Ratio |   Sortino Ratio |   Calmar Ratio |   Regime-Wechsel | Gesamtkosten (Gebühren)   |
|:------------|:---------------|:--------------|:--------------|:---------------|---------------:|----------------:|---------------:|-----------------:|:--------------------------|
| Buy Hold    | 789.98%        | 7.56%         | 11.35%        | -34.77%        |           0.7  |            0.94 |           0.22 |                0 | 0.00%                     |
| MSM         | 455.53%        | 5.88%         | 7.11%         | -27.66%        |           0.84 |            1    |           0.21 |              465 | 46.00%                    |
| HMM         | 422.15%        | 5.66%         | 5.46%         | -16.30%        |           1.04 |            1.04 |           0.35 |              136 | 13.10%                    |
| LSTM        | 341.71%        | 5.08%         | 6.44%         | -15.43%        |           0.8  |            0.84 |           0.33 |               69 | 7.00%                     |
| Transformer | 270.06%        | 4.46%         | 6.52%         | -21.98%        |           0.7  |            0.76 |           0.2  |              137 | 13.80%                    |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital     | Status           |
|:-------------------------------|:---------------|:-----------------|
| ('Standard', 'Buy Hold')       | 1,324,083.30 € | Kapitalerhalt    |
| ('Standard', 'MSM')            | 204,766.14 €   | Kapitalerhalt    |
| ('Standard', 'HMM')            | 129,761.03 €   | Kapitalerhalt    |
| ('Standard', 'LSTM')           | 17,902.53 €    | Kapitalerhalt    |
| ('Standard', 'Transformer')    | 0.00 €         | Erschöpft (2023) |
| ('Aggressive', 'Buy Hold')     | 0.00 €         | Erschöpft (2017) |
| ('Aggressive', 'MSM')          | 0.00 €         | Erschöpft (2010) |
| ('Aggressive', 'HMM')          | 0.00 €         | Erschöpft (2008) |
| ('Aggressive', 'LSTM')         | 0.00 €         | Erschöpft (2008) |
| ('Aggressive', 'Transformer')  | 0.00 €         | Erschöpft (2008) |
| ('Low_Capital', 'Buy Hold')    | 169,281.99 €   | Kapitalerhalt    |
| ('Low_Capital', 'MSM')         | 0.00 €         | Erschöpft (2014) |
| ('Low_Capital', 'HMM')         | 0.00 €         | Erschöpft (2013) |
| ('Low_Capital', 'LSTM')        | 0.00 €         | Erschöpft (2012) |
| ('Low_Capital', 'Transformer') | 0.00 €         | Erschöpft (2011) |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 592,732.77 €        |
| ('Standard', 'MSM')            | 0.00%                     | 478,325.30 €        |
| ('Aggressive', 'HMM')          | 0.07%                     | 227,295.87 €        |
| ('Standard', 'HMM')            | 0.00%                     | 466,591.54 €        |
| ('Low_Capital', 'Buy Hold')    | 0.29%                     | 269,857.12 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 428,929.78 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 399,427.13 €        |
| ('Aggressive', 'LSTM')         | 0.47%                     | 198,353.85 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 205,175.28 €        |
| ('Aggressive', 'Transformer')  | 1.33%                     | 169,347.40 €        |
| ('Aggressive', 'Buy Hold')     | 2.20%                     | 329,487.69 €        |
| ('Aggressive', 'MSM')          | 0.51%                     | 234,855.11 €        |
| ('Low_Capital', 'Transformer') | 0.00%                     | 161,829.28 €        |
| ('Low_Capital', 'LSTM')        | 0.00%                     | 181,603.56 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 201,422.45 €        |

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
| 00_dependencies | 12:07:49 | 12:07:52 | 2.8 |
| 01_data_preprocessing | 12:07:52 | 12:07:59 | 6.8 |
| 02_feature_engineering | 12:07:59 | 12:08:03 | 4.8 |
| 03_regime_switching_models | 12:08:03 | 13:34:24 | 5180.8 |
| 04_backtesting | 13:34:24 | 13:34:30 | 5.9 |
| 05_evaluation | 13:34:30 | 13:36:28 | 118.2 |
| **Gesamt** | | | **5319.3** (88m 39.3s) |

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

**Zuletzt aktualisiert:** 13.04.2026 15:30<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Walk-Forward-Validierung:** AKTIV (Modus: rolling, Train: 5J, Test: 6M, Step: 6M)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
