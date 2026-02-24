# Dynamische Asset-Allokation mittels Regime-Switching-Modellen

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dieses Repository enthält den Code und die Analysen meiner Master-Thesis zum Thema: 
**"Dynamische Asset-Allokation mittels Regime-Switching-Modellen: Ein Vergleich ökonometrischer Modelle und moderner Machine-Learning-Verfahren zur Reduktion von Maximum Drawdowns"**.

## Ziel der Arbeit

Das Kernziel dieser Arbeit ist der **systematische Vergleich** zwischen zwei Paradigmen der Finanzmarktanalyse zur Identifikation von Marktregimes: klassischen **ökonometrischen Modellen** und modernen **Machine-Learning-Verfahren**. 

Statische Anlagestrategien leiden in Zeiten zunehmender Marktvolatilität und komplexer Krisenzyklen (wie der Dotcom-Blase, der Finanzkrise 2008 oder der Zinswende 2022) besonders unter dem **Sequence of Returns Risk (SORR)**, das Risiko, dass Markteinbrüche zu Beginn einer Entnahmephase das Kapital irreversibel schädigen. Dieses Projekt untersucht, wie eine **Dynamische Asset-Allokation (DAA)**, gestützt auf eine **automatisierte Regime-Erkennung** (Bulle vs. Bär) genutzt werden kann, um durch rechtzeitige Umschichtung in den Geldmarkt (Cash) das Portfolio-Risiko zu glätten und die Kapitalsicherheit im Ruhestand massiv zu erhöhen.

### Die Untersuchungsschwerpunkte sind:

1.  **Modell-Vergleich:** Evaluierung der Vorhersagegüte von statistisch fundierten ökonometrischen Modellen (z. B. Markov-Switching) gegenüber hochflexiblen Machine-Learning-Architekturen (z. B. LSTM-Netzwerken, Transformer) unter Einbeziehung von Makro-Indikatoren (VIX, Yield Spreads,...).
2.  **Risikominimierung:** Quantifizierung des Mehrwerts dieser Modelle zur signifikanten **Reduktion von Maximum Drawdowns**. Es wird geprüft, ob die Modelle in der Lage sind, rechtzeitig Signale zur Umschichtung von Risiko-Assets (Aktien) in sichere Häfen (Geldmarkt/Bonds) zu generieren.
3.  **SORR-Prävention:** Die praktische Anwendung fokussiert sich auf die Minderung des **Sequence of Returns Risk (SORR)**. Hierbei soll bewiesen werden, dass eine regimebasierte Steuerung das Pfadrisiko des Vermögensaufbaus glättet und somit die Kapitalsicherheit, insbesondere in der kritischen Phase kurz vor oder zu Beginn der Entnahmephase (Renteneintritt), massiv erhöht.

Durch diesen Vergleich soll aufgezeigt werden, ob die höhere Komplexität moderner KI-Verfahren gegenüber bewährter Ökonometrie einen ökonomisch messbaren Vorteil in der risikoadjustierten Performance liefert.

---

## Methodik & Modelle

In diesem Projekt werden zwei verschiedene Ansätze zur Regime-Erkennung verglichen:

1. **Ökonometrische Modelle:** Diese Modelle basieren auf der Annahme, dass Finanzmärkte stochastischen Prozessen folgen und Regimes als verborgene Zustände (Latent States) mathematisch modelliert werden können.
*   **Markov-Switching-Modelle (MSM):** Ein klassisches Regressionsverfahren, bei dem Parameter (wie Mittelwert und Varianz der Rendite) zwischen Zuständen springen. Die Wechselwahrscheinlichkeiten werden über eine Übergangsmatrix berechnet. In diesem Projekt wird zudem ein **Exogenes MSM** eingesetzt, das Makro-Daten (VIX, Yield Spread) als erklärende Variablen nutzt.
*   **Hidden-Markov-Modelle (HMM):** Ein Unsupervised-Learning-Ansatz aus der Statistik. Das HMM identifiziert Cluster in den Datenverteilungen, um Phasen hoher und niedriger Volatilität voneinander zu trennen, ohne dass vorab gelabelte Daten nötig sind.
2. **Moderne Machine-Learning-Verfahren:** Dieser Ansatz nutzt die Fähigkeit von künstlichen neuronalen Netzen, hochkomplexe, nicht-lineare Zusammenhänge in großen Datenmengen zu identifizieren, ohne explizite statistische Verteilungsannahmen vorauszusetzen.
*   **LSTM-Netzwerke (Long Short-Term Memory):** Eine spezialisierte Form von Recurrent Neural Networks (RNN), die über ein "Gedächtnis" für zeitliche Abhängigkeiten verfügen. In dieser Arbeit wird das LSTM in einem **Supervised-Learning-Setting** eingesetzt: Es lernt, die durch die ökonometrischen Modelle identifizierten Regime-Wechsel unter Berücksichtigung von Zeitreihen-Fenstern (Sequenzen) vorherzusagen.
*   **Unsupervised LSTM (Deep Clustering):** Ein innovativer Ansatz mittels LSTM-Autoencoder. Hierbei lernt das Netzwerk, Markt-Sequenzen in einen hochdimensionalen latenten Raum zu komprimieren, um die „Essenz“ der Marktdynamik zu erfassen. Durch ein anschließendes Clustering (Gaussian Mixture Modeling (GMM)) werden Regimes rein datengetrieben identifiziert, ohne den Einfluss vordefinierter statistischer Labels. Dies dient als objektive Kontrollinstanz, um zu prüfen, ob das RNN eigenständige Risikomuster erkennt, die klassischen Modellen verborgen bleiben.
*   **Transformer-Netzwerk (Multi-Head Self-Attention):** Eine Attention-basierte Architektur, die im Gegensatz zu rekurrenten Netzwerken **alle Zeitschritte einer Sequenz parallel** verarbeiten kann. Durch den Multi-Head Self-Attention-Mechanismus lernt das Modell, welche historischen Zeitpunkte innerhalb eines Fensters die stärkste Relevanz für die aktuelle Regime-Klassifikation besitzen. Ein Positional Encoding bewahrt dabei die zeitliche Ordnung der Inputdaten. Der Transformer wird im **Supervised-Setting** (trainiert auf Markov-Labels) eingesetzt und dient dem Test der Hypothese H2: Ob Attention-basierte Architekturen eine höhere Vorhersagegüte als ökonometrische Modelle und rekurrente Netze erreichen.

---

## Technologie-Stack

Für die Umsetzung der Forschungsumgebung wurde ein moderner Data-Science-Stack gewählt, der Stabilität mit hoher Rechenleistung kombiniert:

*   **Programmiersprache:** Python 3.10+
*   **Datenquellen:** Yahoo Finance API (`yfinance`)
*   **Datenverarbeitung:** `Pandas`, `NumPy`, `PyArrow` (Parquet-Engine)
*   **Ökonometrie & Statistik:** `Statsmodels` (Markov-Regression), `hmmlearn` (Hidden Markov Models), `SciPy`
*   **Machine Learning:** `TensorFlow` / `Keras` (LSTM-Architekturen), `PyTorch` (Transformer), `Scikit-Learn`
*   **Reporting:** `Matplotlib` (Visualisierung), `Tabulate` (Markdown-Export)

---

## Architektur

Das Projekt folgt einem **modularen Pipeline-Design**. Anstatt eines monolithischen Skripts ist der Workflow in spezialisierte Teilschritte unterteilt, um die Reproduzierbarkeit und Skalierbarkeit zu gewährleisten. Ein zentrales Master-Notebook (`regime-switching-daa.ipynb`) orchestriert die Ausführung der einzelnen Module in der korrekten Reihenfolge.

---

## Engineering-Konzepte

Hinter der Pipeline stehen fortgeschrittene Konzepte der Software-Entwicklung und Finanzmathematik, um die Validität der Ergebnisse sicherzustellen:

### Daten-Persistierung & Entkopplung
Um Notebooks voneinander zu entkoppeln und den Arbeitsspeicher effizient zu nutzen, werden Zwischenergebnisse im **Apache Parquet-Format** gespeichert. Parquet bietet gegenüber CSV eine höhere Performance und erhält die Integrität der Datentypen (insb. Zeitstempel), was für die Zeitreihenanalyse essentiell ist.

### Vermeidung von Look-ahead Bias
Ein kritischer Aspekt im Backtesting ist die Vermeidung von Informationslecks aus der Zukunft. Alle generierten Handelssignale werden systematisch um einen Zeitschritt ($T+1$) verschoben. Entscheidungen werden somit ausschließlich auf Basis der zum Handelszeitpunkt verfügbaren historischen Informationen getroffen.

### Data-Driven Automation (Dynamic Matching)
Das Framework ist **vollständig dynamisch** aufgebaut. Ein spezialisierter Such-Algorithmus identifiziert neue Modell-Outputs automatisch anhand eines definierten Namensschemas (`Modell_Signal`). Dadurch können neue Modell-Architekturen integriert werden, ohne den Code für das Backtesting, die Evaluation oder das Reporting manuell anpassen zu müssen.

### Realitätsnahe Kostensimulation
Die Simulation berücksichtigt reale Marktreibungen:
*   **Transaktionskosten:** Jede Umschichtung zwischen Portfolio und Cash wird mit einer Gebühr (0,1%) belegt.
*   **Liquiditätsgebühren:** Bei Entnahmen in investierten Marktphasen werden zusätzliche Verkaufsgebühren simuliert, während Entnahmen aus Cash-Beständen spesenfrei erfolgen.

### Automated Reporting (Live-Docs)
Die Datei `statistics.md` wird am Ende jedes Pipeline-Durchlaufs neu generiert. Hierbei werden Markdown-Tabellen und PNG-Assets direkt in das Dokument eingebettet, was eine lückenlose und stets aktuelle Dokumentation der Forschungsergebnisse ermöglicht.

---

## Die Research-Pipeline (Modularer Aufbau)

Das Projekt ist als vollautomatisierte Pipeline konzipiert. Jedes Modul baut auf den persistierten Daten des Vorgängers auf:

1.  **`00_dependencies`**: Initialisierung der Forschungsumgebung.
2.  **`01_data_preprocessing`**: Download (YFinance) und Bereinigung von Multi-Asset-Daten (Aktien, Bonds, Cash).
3.  **`02_feature_engineering`**: Berechnung technischer und makroökonomischer Indikatoren.
4.  **`03_regime_switching_models`**: Training und Hyperparameteroptimierung der Regime-Switching-Modelle.
5.  **`04_backtesting`**: Simulation realer Investitionsszenarien inkl. variabler Entnahmen und Transaktionskosten.
6.  **`05_evaluation`**: Stress-Tests mittels Block-Bootstrap zur statistischen Validierung der Ergebnisse.
7.  **`99_generate_report`**: Automatisierte Zusammenführung aller Ergebnisse in die Dokumentation.

---

## Aktuelle Ergebnisse (Live-Update)

Die folgenden Grafiken werden automatisch generiert und repräsentieren den aktuellen Stand der Backtesting-Simulation auf dem S&P 500 / Long-Bond (60/40) Portfolio.

### 1. Performance-Vergleich (Equity Curves)
Vergleich der kumulierten Rendite zwischen der statischen "Buy & Hold"-Strategie und den aktiven Regime-Switching-Modellen.

![Equity Curves](./assets/equity_curves.png)

### 2. Regime-Erkennung im Detail
Visualisierung der berechneten Wahrscheinlichkeiten für ein Bärenmarkt-Regime über den Testzeitraum.

![Regime Comparison](./assets/regime_comparison.png)

### 3. Zusammenfassung der Kennzahlen
| Kategorie | Kennzahlen | Relevanz für die Thesis |
| :--- | :--- | :--- |
| **Risikoschutz (SORR)** | **Max Drawdown, Calmar Ratio** | Die primäre Zielgröße. Misst die Effektivität der Verlustvermeidung und das Verhältnis von Rendite zu maximalem Drawdown. |
| **Risikoadjustierung** | **Sortino Ratio, Sharpe Ratio** | Bewertet, ob die Modelle eine Überrendite pro Risikoeinheit liefern. Die Sortino Ratio ist hierbei zentral, da sie gezielt das Abwärtsrisiko (Downside) betrachtet. |
| **Wachstumsdynamik** | **Total Return, CAGR (p.a.)** | Zeigt, ob die Modelle trotz der defensiven Ausrichtung in Bärenmärkten langfristig in der Lage sind, den Markt (Buy & Hold) zu schlagen. |
| **Modellstabilität** | **Regime-Wechsel, Volatilität** | Evaluiert die praktische Umsetzbarkeit. Eine hohe Anzahl an Wechseln ("Churning") deutet auf Instabilität und hohe Transaktionskosten hin. |

### 4. Risikoprofil (SORR Stress-Test)
Simulation einer Entnahmephase: Wie lange reicht das Kapital unter Berücksichtigung von Marktschocks und monatlichen Rentenzahlungen?

![SORR Standard](./assets/sorr_sim_standard.png)

### 5. Statistische Siginifika
nz (Monte-Carlo-Simulation (MCS)
Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.

![MCS Boxplots Standard](./assets/mcs_boxplot_standard.png)

👉 **Detaillierte statistische Auswertungen, Tabellen und Einzelauswertungen findest du in der [statistics.md](./docs/statistics.md).**

---

## Projektstruktur
- `jupyter/` : Ablageort der Jupyter-Notebook-Files mit der gesamten Pipeline.
- `assets/` : Ordner für persistierte Grafiken und Statistiken.
- `config/` : Konfigurationsparameter der Research-Pipeline.
- `data/` : Lokale Cache-Daten der Yahoo Finance API.
- `docs/` : Begleitende Projektdokumentation.
- `README.md` : Projektübersicht.
- `Statistics.md` : Tiefergehende Analyse der Modellergebnisse.

---

## Installation & Start

1. **Repository klonen:**
   ```bash
   git clone https://github.com/Torim98/regime-switching-daa.git
2. **Pipeline ausführen:**
   Starte das Master-Notebook `regime-switching-daa.ipynb` im Verzeichnis `jupyter/`. Dies triggert alle Teilschritte und aktualisiert automatisch alle Grafiken und Statistiken.
   Ggf. muss in der `00_dependencies.ipynb` Code auskommentiert werden, um Abhängigkeiten automatisch zu installieren.
---

## Ausblick & Offene Punkte (Roadmap)

Um die Robustheit und Praxistauglichkeit der dynamischen Asset-Allokation weiter zu steigern, sind folgende Entwicklungsschritte geplant:

### 1. Modell-Erweiterungen
*   **Hyperparameter-Optimierung:** Implementierung einer systematischen (ggf. automatisierten) Suche nach optimalen Parametern (z. B. Sensitivitätsanalyse der `window_size`).

### 2. Analyse & Infrastruktur
*   **Workflow-Optimierung:** Kontinuierliche Verfeinerung der Repository-Struktur und der automatisierten Dokumentations-Pipelines für eine maximale Reproduzierbarkeit.
*   **Erweiterte Visualisierung:** Aufbau umfangreicherer Dashboards zur explorativen Datenanalyse und zur grafischen Aufarbeitung der Modell-Fehlentscheidungen.
*   **Systemarchitektur-Diagramm:** Erstellung eines detaillierten konzeptionellen Architekturdiagramms (Flow-Chart), um den modularen Datenfluss, die Interaktionen zwischen den Paradigmen (Ökonometrie & ML) sowie die Persistierungs-Logik innerhalb der Pipeline visuell darzustellen.

---

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Details findest du in der Datei [LICENSE](./LICENSE).

**Autor:** Tom Maurer B.Sc.<br>
**Betreuer:** Prof. Dr. Christian Müller-Kett<br>
**Akademischer Kontext:** Master-Thesis Projekt im Bereich Quantitative Finance / Data Science.
