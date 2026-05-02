# Limitations & Scope Boundaries

Dieses Dokument beschreibt bewusste Abgrenzungen und Design-Entscheidungen der Implementierung, die bei der Interpretation der Ergebnisse berücksichtigt werden sollten.

---

## 1. Steuerliche Effekte (Tax Modeling)

**Status:** Bewusst ausgegrenzt (Thesis Tabelle 1: Out of Scope)

Die Backtesting-Simulation berücksichtigt **Transaktionskosten** (0,1% pro Rebalancing-Event), modelliert jedoch **keine steuerlichen Effekte** auf realisierte Kapitalerträge.

### Begründung

1. **Relativer Vergleich bleibt valide:** Alle vier Modelle (MSM, HMM, LSTM, Transformer) sowie die Buy-&-Hold-Benchmark unterliegen identischen steuerlichen Rahmenbedingungen. Eine Kapitalertragsteuer würde die absoluten Renditen aller Strategien gleichermaßen reduzieren, ohne die relative Rangfolge der risikoadjustierten Metriken (Sharpe Ratio, Sortino Ratio, Calmar Ratio) zu verändern.

2. **Jurisdiktions-Neutralität:** Steuerliche Regelungen unterscheiden sich erheblich zwischen Jurisdiktionen (z. B. deutsche Abgeltungssteuer 26,375% inkl. Solidaritätszuschlag vs. US Capital Gains Tax mit Short-/Long-Term-Differenzierung). Eine Modellierung würde die Ergebnisse an ein spezifisches Steuersystem binden und die Generalisierbarkeit einschränken.

### Einschränkung: Differenzielle Steuerbelastung durch Signalfrequenz

Modelle mit **höherer Signalfrequenz** (mehr Regime-Switches) generieren mehr steuerpflichtige Rebalancing-Events als Modelle mit stabilen Signalen:

| Modell | Signalstabilität | Steuerliche Betroffenheit |
|--------|-----------------|--------------------------|
| HMM | Hohe Stabilität (wenige Switches) | Gering |
| MSM | Moderate Stabilität | Moderat |
| LSTM | Tendenziell höhere Frequenz | Erhöht |
| Transformer | Variabel | Variabel |

Würde eine Kapitalertragsteuer modelliert, würden Modelle mit vielen kurzen Regime-Wechseln relativ stärker belastet als stabile Modelle. Dies könnte die ohnehin in der Evaluation sichtbaren Transaktionskosten-Nachteile frequenter Switcher weiter verstärken.

### Mögliche Erweiterung

- Deutsche Abgeltungssteuer (26,375% inkl. Soli) auf realisierte Gewinne pro Rebalancing
- US Short-Term vs. Long-Term Capital Gains Differenzierung (Haltedauer < / > 1 Jahr)
- Steueroptimierte Rebalancing-Strategien (Tax-Loss Harvesting)

---

## 2. Zwei-Regime-Annahme

**Status:** Bewusste Design-Entscheidung

Alle Modelle operieren mit exakt **zwei Regimes** (Bull/Bear):
- MSM: `k_regimes: 2`
- HMM: `n_components: 2`
- LSTM / Transformer: Binäre Klassifikation (Sigmoid-Output)

### Begründung

Die Zwei-Regime-Annahme folgt der dominierenden Literatur zu Markov-Switching-Modellen und ermöglicht eine klare, interpretierbare Handlungsregel (Risk-On vs. Risk-Off). Sie bildet die Grundlage für die binäre Allokationsstrategie (100% Equity vs. 100% Safe Haven).

### Einschränkung

Finanzmärkte können mehr als zwei Zustände aufweisen (z. B. Bull, Bear, Seitwärts/Recovery). Eine höhere Regimeanzahl (k=3+) könnte granularere Allokationsstufen ermöglichen (z. B. 100% / 60% / 0% Equity), würde aber die Modellkomplexität und das Overfitting-Risiko erhöhen. Zudem wäre die direkte Vergleichbarkeit zwischen ökonometrischen und DL-Modellen erschwert, da die Label-Zuordnung bei k>2 nicht mehr trivial ist.

---

## 3. Datenquelle Yahoo Finance

**Status:** Bewusste Wahl mit bekannten Trade-offs

Alle Marktdaten werden über die `yfinance`-Bibliothek bezogen (Yahoo Finance API).

### Begründung

Yahoo Finance bietet kostenfreien Zugang zu adjustierten historischen Kursdaten mit ausreichender Abdeckung des Untersuchungszeitraums (ab 1990). Für eine akademische Arbeit mit Fokus auf Methodenvergleich (nicht auf Live-Trading) ist die Datenqualität ausreichend.

### Einschränkung

- **Keine garantierte API-Stabilität:** Yahoo Finance bietet keine offizielle API; `yfinance` nutzt inoffizielle Endpunkte, die sich ändern können.
- **Survivorship Bias:** Die verwendeten Indizes (S&P 500, VUSTX) unterliegen Survivorship Bias, da nur überlebende Unternehmen/Fonds enthalten sind.
- **Adjustierte Kurse:** Die Adjustierung von Splits und Dividenden durch Yahoo ist nicht immer transparent dokumentiert.
- **Keine Intraday-Daten:** Die Pipeline arbeitet mit Tagesschlusskursen. Intraday-Regime-Switches werden nicht erfasst.

Für produktive Anwendungen wären institutionelle Datenanbieter (Bloomberg, Refinitiv) vorzuziehen.

---

## 4. Asset-Universum

**Status:** Bewusst eingeschränkt

Das Portfolio besteht aus zwei US-Assetklassen:
- **Risk-Asset:** 60% S&P 500 (`^GSPC`) + 40% US Long-Term Bonds (`VUSTX`)
- **Safe Haven:** 3-Monats-Treasury Bill Rate (`^IRX`) als Cash-Proxy

### Begründung

Das klassische 60/40-Portfolio dient als etablierter Benchmark in der Finanzliteratur und ermöglicht eine fokussierte Untersuchung der Regime-Switching-Wirkung ohne Interferenz durch Multi-Asset-Allokationsentscheidungen.

### Einschränkung

- **Nur US-Markt:** Ergebnisse sind nicht direkt auf andere Märkte (Europa, Emerging Markets) übertragbar, da Regime-Dynamiken regional variieren können.
- **Nur 2 Assetklassen:** Diversifikationseffekte durch Rohstoffe, Immobilien, internationale Anleihen oder Kryptowährungen werden nicht abgebildet.
- **Korrelationsannahme:** Die 2022er Periode (gleichzeitiger Einbruch von Aktien und Anleihen) zeigt, dass die historisch negative Korrelation nicht dauerhaft gelten muss.

---

## 5. Walk-Forward-Konfiguration

**Status:** Bewusste Parameterwahl mit Trade-offs

Die Walk-Forward-Validierung nutzt folgende Konfiguration:
- **Modus:** Rolling Window (nicht expandierend)
- **Train-Fenster:** 10 Jahre (`train_window_years: 10`)
- **Test-Fenster:** 12 Monate (`test_window_months: 12`)
- **Schrittweite:** 12 Monate (`step_months: 12`, nicht-überlappend)

### Begründung

Die 10J/12M/12M-Konfiguration balanciert zwischen ausreichend Trainingsdaten für stabile Modellschätzungen und genügend OOS-Folds für eine belastbare Evaluation. Nicht-überlappende Folds vermeiden Autokorrelation zwischen Test-Perioden.

### Einschränkung

- **Rolling vs. Expanding:** Ein expandierendes Fenster würde insbesondere den späteren Folds mehr Trainingsdaten geben, könnte aber ältere, weniger relevante Marktregimes übergewichten.
- **DL-Modelle bei 10 Jahren Training:** LSTM und Transformer profitieren typischerweise von größeren Datensätzen. Längere Trainingsfenster könnten die DL-Performance verbessern, würden aber die Anzahl verfügbarer Folds reduzieren.
- **12-Monats-Folds:** Kürzere Folds (z. B. 6 Monate) würden mehr Datenpunkte für die Evaluation liefern, erhöhen aber die Rechenzeit und das Risiko instabiler Schätzungen bei ökonometrischen Modellen.

---

## 6. HMM Cold-Start bei kurzen Testfenstern

**Status:** Bekannte Limitation

Das Hidden Markov Model benötigt eine **Anlaufphase** (Cold Start), um die Zustandsverteilung aus den Emissionen zu schätzen. Bei kurzen Walk-Forward-Testfenstern (z.B. 6 Monate = ca. 126 Handelstage) können die ersten Vorhersagen eines neuen Folds instabil sein, da das Modell die Zustandssequenz aus den initialen Beobachtungen ableiten muss.

### Auswirkung

In der Pipeline wird dieses Problem durch die Verwendung der `predict_proba`-Methode auf dem gesamten Testfenster abgemildert (Forward-Algorithmus nutzt die gesamte Sequenz). Dennoch kann die Zustandszuordnung am Fold-Beginn weniger zuverlässig sein als in der Fold-Mitte.

Siehe Thesis Kapitel 3.5.5 für eine detaillierte Diskussion.