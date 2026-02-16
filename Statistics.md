
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen Tag und werden automatisiert aktualisiert.

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

[//]: # (Dynamischer Import von performance_summary.md)
| Strategie     |   Final Wealth | Total Return   | Max Drawdown   |
|:--------------|---------------:|:---------------|:---------------|
| Buy_Hold      |         1.9352 | +93.52%        | -27.10%        |
| MS_Univariate |         2.6662 | +166.62%       | -5.80%         |
| MS_Exogenous  |         2.6312 | +163.12%       | -5.44%         |
| LSTM_Regime   |         1.6955 | +69.55%        | -14.93%        |
| HMM_Based     |         1.7944 | +79.44%        | -6.53%         |

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

## 🤖 3. Regime-Erkennung der Einzelmodelle
Hier werden die Identifikations-Ergebnisse der drei Modell-Kategorien (Statistik, Clustering, Deep Learning) visualisiert.

### A. Hidden Markov Model (Unsupervised Clustering)
![HMM Regimes](./assets/hmm_regimes.png)

### B. Markov-Switching-Modelle (Ökonometrie)
Vergleich zwischen univariatem Ansatz und exogenem Ansatz (unter Berücksichtigung von VIX & Yield Spread).
![Markov Models](./assets/markov-models.png)

### C. LSTM-Netzwerk (Deep Learning)
Vorhersage der Marktphasen durch das neuronale Netzwerk (trainiert auf Markov-Labels).
![LSTM Model](./assets/lstm_model.png)

### 🔄 D. Globaler Regime-Vergleich
Detaillierte Gegenüberstellung der Wahrscheinlichkeiten und harten Signale aller Modelle.
![Regime Comparison](./assets/regime_comparison.png)

---

## 🏁 4. Backtesting & Strategie-Evaluation
Die ökonomische Anwendung der Regime-Signale durch dynamische Umschichtung in den Geldmarkt.

### Equity Curves im Vergleich
![Equity Curves](./assets/equity_curves.png)

### Umfassende Kennzahlen-Matrix
Detaillierte statistische Analyse inklusive risikoadjustierter Kennzahlen (Sharpe, Sortino, Calmar).

[//]: # (Dynamischer Import von evaluation_table.md)
| Strategie     | Total Return   | CAGR (p.a.)   | Volatilität   | Max Drawdown   |   Sharpe Ratio |   Sortino Ratio |   Calmar Ratio |   Regime-Wechsel |
|:--------------|:---------------|:--------------|:--------------|:---------------|---------------:|----------------:|---------------:|-----------------:|
| Buy_Hold      | 92.90%         | 9.80%         | 12.62%        | -27.10%        |           0.81 |            1.04 |           0.36 |                0 |
| MS_Univariate | 165.77%        | 14.93%        | 6.33%         | -5.80%         |           2.24 |            2.88 |           2.57 |               42 |
| MS_Exogenous  | 162.28%        | 14.71%        | 6.42%         | -5.44%         |           2.18 |            2.79 |           2.7  |               38 |
| LSTM_Regime   | 69.01%         | 7.76%         | 8.02%         | -14.93%        |           0.97 |            1.12 |           0.52 |               70 |
| HMM_Based     | 78.87%         | 8.63%         | 4.96%         | -6.53%         |           1.7  |            1.56 |           1.32 |               29 |

---

## 📝 Forschungsnotizen & Methodik
- **Cash-Komponente:** Bei einem "Bear"-Signal schichtet die Strategie in den aktuellen Geldmarktzins (**^IRX**) um.
- **Vermeidung von Look-ahead Bias:** Alle Signale werden für das Backtesting um einen Tag zeitversetzt (`shift(1)`), um reale Handelsbedingungen zu simulieren.
- **Feature-Set:** Die Modelle nutzen Renditen, Volatilität (20d), SMA-Abstand, Momentum, VIX und Yield Spread.

---
**Zuletzt aktualisiert:** 03.02.2026 09:43  
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
