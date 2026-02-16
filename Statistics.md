
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

| Strategie     |   Final Wealth | Total Return   | Max Drawdown   |
|:--------------|---------------:|:---------------|:---------------|
| Buy_Hold      |         1.9352 | +93.52%        | -27.10%        |
| HMM           |         1.7415 | +74.15%        | -6.79%         |
| MS_Univariate |         2.5566 | +155.66%       | -6.25%         |
| MS_Exo        |         2.5332 | +153.32%       | -5.44%         |
| LSTM          |         1.5616 | +56.16%        | -19.75%        |

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### 60/40 Portfolio Kapitalkurve
Die Abbildung zeigt die kumulierte Wertentwicklung des statischen Referenzportfolios (60% Aktien / 40% Anleihen).

![Capital Curve](./assets/capital_curve.png)

*   **Datenquelle:** S&P 500 (`^GSPC`) und Vanguard Long-Term Treasury (`VUSTX`).
*   **Reproduzierbarkeit:** Der bereinigte Datensatz inkl. aller Features ist hinterlegt unter: `data/02_feature_engineered_data.parquet`.

---

## 3. Regime-Erkennung der Einzelmodelle
Hier werden die Identifikations-Ergebnisse der drei Modell-Kategorien (Statistik, Clustering, Deep Learning) visualisiert.

### A. Hidden Markov Model (Unsupervised Clustering)
![HMM Regimes](./assets/hmm_regimes.png)

### B. Markov-Switching-Modelle (Ökonometrie)
Vergleich zwischen univariatem Ansatz und exogenem Ansatz (unter Berücksichtigung von VIX & Yield Spread).
![Markov Models](./assets/markov-models.png)

### C. LSTM-Netzwerk (Deep Learning)
Vorhersage der Marktphasen durch das neuronale Netzwerk (trainiert auf Markov-Labels).
![LSTM Model](./assets/lstm_model.png)

### D. Globaler Regime-Vergleich
Detaillierte Gegenüberstellung der Wahrscheinlichkeiten und harten Signale aller Modelle.
![Regime Comparison](./assets/regime_comparison.png)

---

## 4. Backtesting & Strategie-Evaluation
Die ökonomische Anwendung der Regime-Signale durch dynamische Umschichtung in den Geldmarkt.

### Equity Curves im Vergleich
![Equity Curves](./assets/equity_curves.png)

### Umfassende Kennzahlen-Matrix
Detaillierte statistische Analyse inklusive risikoadjustierter Kennzahlen (Sharpe, Sortino, Calmar).

| Strategie     | Total Return   | CAGR (p.a.)   | Volatilität   | Max Drawdown   |   Sharpe Ratio |   Sortino Ratio |   Calmar Ratio |   Regime-Wechsel | Gesamtkosten (Gebühren)   |
|:--------------|:---------------|:--------------|:--------------|:---------------|---------------:|----------------:|---------------:|-----------------:|:--------------------------|
| Buy Hold      | 92.90%         | 9.80%         | 12.62%        | -27.10%        |           0.81 |            1.04 |           0.36 |                0 | 0.00%                     |
| HMM           | 73.59%         | 8.17%         | 4.96%         | -6.79%         |           1.61 |            1.49 |           1.2  |               29 | 3.00%                     |
| MS Univariate | 154.84%        | 14.24%        | 6.33%         | -6.25%         |           2.14 |            2.78 |           2.28 |               42 | 4.20%                     |
| MS Exo        | 152.51%        | 14.09%        | 6.43%         | -5.44%         |           2.09 |            2.7  |           2.59 |               38 | 3.80%                     |
| LSTM          | 55.66%         | 6.50%         | 8.14%         | -19.75%        |           0.82 |            0.94 |           0.33 |               68 | 6.80%                     |

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](./assets/transaction_costs.png)

---

## 📝 Forschungsnotizen & Methodik
- **Cash-Komponente:** Bei einem "Bear"-Signal schichtet die Strategie in den aktuellen Geldmarktzins (**^IRX**) um.
- **Vermeidung von Look-ahead Bias:** Alle Signale werden für das Backtesting um einen Tag zeitversetzt (`shift(1)`), um reale Handelsbedingungen zu simulieren.
- **Feature-Set:** Die Modelle nutzen Renditen, Volatilität (20d), SMA-Abstand, Momentum, VIX und Yield Spread.
- **Kostensimulation:** Es wird eine pauschale Gebühr von 10 Basispunkten (0,1%) pro Umschichtung berechnet.

---
**Zuletzt aktualisiert:** 03.02.2026 10:58  
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
