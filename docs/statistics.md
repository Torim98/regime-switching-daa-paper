
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie   |   Final Wealth | Total Return   | Max Drawdown   |
|:------------|---------------:|:---------------|:---------------|
| Buy_Hold    |         8.8991 | +789.91%       | -34.77%        |
| MSM         |         5.5548 | +455.48%       | -27.66%        |
| HMM         |         5.2293 | +422.93%       | -16.30%        |
| LSTM        |         4.4714 | +347.14%       | -21.98%        |
| Transformer |         3.7792 | +277.92%       | -21.98%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
| Zeitreihe     |   Mittelwert (tägl.) |   Std.Abw. (tägl.) |     Min |     Max |   Schiefe (Skew) |   Kurtosis |
|:--------------|---------------------:|-------------------:|--------:|--------:|-----------------:|-----------:|
| Returns_GSPC  |             0.000324 |           0.01139  | -0.1277 |  0.1096 |          -0.36   |    10.8118 |
| Returns_VUSTX |             0.000274 |           0.007486 | -0.0605 |  0.1296 |           0.6393 |    14.3741 |
| Returns       |             0.000304 |           0.006935 | -0.0662 |  0.0584 |          -0.2264 |     7.7476 |
| VIX           |            19.4671   |           7.76808  |  9.14   | 82.69   |           2.2002 |     8.6667 |
| TNX_10Y       |             4.23747  |           1.93162  |  0.499  |  9.09   |           0.3306 |    -0.6373 |
| IRX_3M        |             2.7033   |           2.20283  | -0.105  |  7.99   |           0.2021 |    -1.256  |

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
| Zeitreihe     |   ADF-Statistik |     p-Wert |   Krit. Wert (5%) | Stationär?   |
|:--------------|----------------:|-----------:|------------------:|:-------------|
| Returns_GSPC  |        -17.4865 | 4.4603e-30 |           -2.8619 | Ja           |
| Returns_VUSTX |        -18.1803 | 2.434e-30  |           -2.8619 | Ja           |
| Returns       |        -17.5053 | 4.3639e-30 |           -2.8619 | Ja           |
| VIX           |         -7.2694 | 1.6034e-10 |           -2.8619 | Ja           |
| TNX_10Y       |         -2.3505 | 0.15621    |           -2.8619 | Nein         |
| IRX_3M        |         -2.3404 | 0.15927    |           -2.8619 | Nein         |

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
| HMM         | +5.69% | 5.46%              |          1.043 |           1.033 | -16.30%        |          0.349 |       7533 |        29.9 |
| LSTM        | +5.14% | 6.98%              |          0.737 |           0.809 | -21.98%        |          0.234 |       7533 |        29.9 |
| Transformer | +4.55% | 6.68%              |          0.681 |           0.73  | -21.98%        |          0.207 |       7533 |        29.9 |

### Krisen-Performance
Return und Max Drawdown während historischer Krisenperioden — der zentrale Nachweis für den Tail-Risk-Schutz der Regime-Switching-Modelle.

| Krise                                | ('Return', 'Buy_Hold')   | ('Return', 'HMM')   | ('Return', 'LSTM')   | ('Return', 'MSM')   | ('Return', 'Transformer')   | ('Max Drawdown', 'Buy_Hold')   | ('Max Drawdown', 'HMM')   | ('Max Drawdown', 'LSTM')   | ('Max Drawdown', 'MSM')   | ('Max Drawdown', 'Transformer')   |
|:-------------------------------------|:-------------------------|:--------------------|:---------------------|:--------------------|:----------------------------|:-------------------------------|:--------------------------|:---------------------------|:--------------------------|:----------------------------------|
| COVID Crash (2020-02 – 2020-03)      | -8.24%                   | +0.13%              | +0.13%               | +0.73%              | -0.37%                      | -18.53%                        | -0.00%                    | -0.00%                     | -1.81%                    | -1.81%                            |
| Dot-Com (2000-03 – 2002-10)          | -13.31%                  | -5.90%              | -2.70%               | -20.49%             | -9.24%                      | -27.27%                        | -16.30%                   | -21.98%                    | -24.98%                   | -21.98%                           |
| EU-Schuldenkrise (2011-07 – 2011-11) | +4.10%                   | -3.34%              | +1.13%               | -1.37%              | -5.44%                      | -7.24%                         | -4.46%                    | -7.24%                     | -8.17%                    | -11.07%                           |
| GFC (2007-10 – 2009-03)              | -25.67%                  | +1.78%              | -0.63%               | +0.83%              | +1.01%                      | -34.77%                        | -1.17%                    | -2.34%                     | -2.86%                    | -1.37%                            |
| Zinsanstieg (2022-01 – 2022-10)      | -24.20%                  | +0.46%              | -2.62%               | -3.41%              | +2.66%                      | -26.98%                        | -1.28%                    | -3.96%                     | -6.35%                    | -1.18%                            |

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
| HMM         | 422.97%        | 5.67%         | 5.46%         | -16.30%        |           1.04 |            1.04 |           0.35 |              136 | 13.10%                    |
| LSTM        | 347.18%        | 5.12%         | 6.98%         | -21.98%        |           0.75 |            0.83 |           0.23 |               90 | 9.00%                     |
| Transformer | 277.95%        | 4.53%         | 6.67%         | -21.98%        |           0.7  |            0.76 |           0.21 |              225 | 22.60%                    |

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/transaction_costs.png)

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

|                                | Endkapital     | Status           |
|:-------------------------------|:---------------|:-----------------|
| ('Standard', 'Buy Hold')       | 1,324,083.84 € | Kapitalerhalt    |
| ('Standard', 'MSM')            | 204,768.08 €   | Kapitalerhalt    |
| ('Standard', 'HMM')            | 131,806.28 €   | Kapitalerhalt    |
| ('Standard', 'LSTM')           | 60,120.44 €    | Kapitalerhalt    |
| ('Standard', 'Transformer')    | 0.00 €         | Erschöpft (2025) |
| ('Aggressive', 'Buy Hold')     | 0.00 €         | Erschöpft (2017) |
| ('Aggressive', 'MSM')          | 0.00 €         | Erschöpft (2010) |
| ('Aggressive', 'HMM')          | 0.00 €         | Erschöpft (2008) |
| ('Aggressive', 'LSTM')         | 0.00 €         | Erschöpft (2008) |
| ('Aggressive', 'Transformer')  | 0.00 €         | Erschöpft (2009) |
| ('Low_Capital', 'Buy Hold')    | 169,282.33 €   | Kapitalerhalt    |
| ('Low_Capital', 'MSM')         | 0.00 €         | Erschöpft (2014) |
| ('Low_Capital', 'HMM')         | 0.00 €         | Erschöpft (2013) |
| ('Low_Capital', 'LSTM')        | 0.00 €         | Erschöpft (2013) |
| ('Low_Capital', 'Transformer') | 0.00 €         | Erschöpft (2012) |

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/sorr_sim_standard.png)
![SORR Aggressive](../assets/sorr_sim_aggressive.png)
![SORR Low Capital](../assets/sorr_sim_low_capital.png)

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/mcs_paths.png)
|                                | Ruin-Wahrscheinlichkeit   | Median Endkapital   |
|:-------------------------------|:--------------------------|:--------------------|
| ('Standard', 'Buy Hold')       | 0.00%                     | 592,731.46 €        |
| ('Aggressive', 'MSM')          | 0.51%                     | 234,857.52 €        |
| ('Standard', 'HMM')            | 0.00%                     | 467,071.89 €        |
| ('Low_Capital', 'HMM')         | 0.00%                     | 201,551.71 €        |
| ('Standard', 'LSTM')           | 0.00%                     | 432,841.70 €        |
| ('Aggressive', 'HMM')          | 0.07%                     | 227,626.21 €        |
| ('Aggressive', 'Buy Hold')     | 2.20%                     | 329,486.86 €        |
| ('Standard', 'MSM')            | 0.00%                     | 478,323.95 €        |
| ('Low_Capital', 'LSTM')        | 0.00%                     | 183,100.97 €        |
| ('Low_Capital', 'Transformer') | 0.01%                     | 164,276.49 €        |
| ('Standard', 'Transformer')    | 0.00%                     | 402,287.44 €        |
| ('Low_Capital', 'Buy Hold')    | 0.29%                     | 269,857.62 €        |
| ('Aggressive', 'LSTM')         | 0.65%                     | 200,357.63 €        |
| ('Low_Capital', 'MSM')         | 0.00%                     | 205,174.61 €        |
| ('Aggressive', 'Transformer')  | 1.51%                     | 173,936.63 €        |

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
| TRANSFORMER | `transformer_regime_model.pt` | Neu trainiert |

> **Hinweis:** Bei aktivierter Persistierung werden vortrainierte Modelle aus `../models` geladen, sofern die Dateien existieren. Andernfalls wird normal trainiert und das Ergebnis für zukünftige Läufe gespeichert. Bei Änderungen an Hyperparametern müssen die entsprechenden Modelldateien gelöscht werden.

---

**Zuletzt aktualisiert:** 14.04.2026 12:08<br>
**Fast Mode Status zur Laufzeit:** FALSE (Full Run)<br>
**Walk-Forward-Validierung:** AKTIV (Modus: rolling, Train: 5J, Test: 6M, Step: 6M)<br>
**Modell-Persistierung:** AKTIV<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
