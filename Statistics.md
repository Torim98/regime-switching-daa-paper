\# Detaillierte statistische Auswertung



Diese Seite bietet eine tiefgreifende Analyse der Modellergebnisse. Alle Daten beziehen sich auf den \*\*Out-of-Sample Testzeitraum\*\* (letzte 20% der Daten) und werden bei jedem Durchlauf der Pipeline automatisch aktualisiert.



---



\## 1. Performance-Matrix im Vergleich

Die folgende Tabelle vergleicht die statistische Güte aller untersuchten Strategien.



\[//]: # (Der folgende Inhalt wird durch die Datei assets/key\_metrics.md repräsentiert)

\[Hier klicken, um die aktuelle Tabelle direkt zu öffnen](./assets/key\_metrics.md)



| Strategie | CAGR | Max Drawdown | Sortino Ratio | Regime-Wechsel |

| :--- | :--- | :--- | :--- | :--- |

| \*\*Buy \& Hold\*\* | \*siehe Tabelle\* | \*siehe Tabelle\* | \*siehe Tabelle\* | 0 |

| \*\*MS Univariat\*\* | ... | ... | ... | ... |

| \*\*MS Exogen\*\* | ... | ... | ... | ... |

| \*\*HMM Based\*\* | ... | ... | ... | ... |

| \*\*LSTM Regime\*\* | ... | ... | ... | ... |



> \*\*Erkenntnis:\*\* Achte besonders auf die \*\*Sortino Ratio\*\* und den \*\*Max Drawdown\*\*. Eine überlegene Strategie zur Minderung des \*\*Sequence of Returns Risk (SORR)\*\* sollte hier signifikante Verbesserungen gegenüber der Buy \& Hold Benchmark zeigen.



---



\## 2. Analyse der Regime-Dynamik

Ein zentraler Aspekt der Arbeit ist die Untersuchung, wie "nervös" oder "stabil" die einzelnen Modelle schalten.



!\[Detaillierter Signalvergleich](./assets/regime\_comparison\_detail.png)



\### Beobachtungen zur Modell-Charakteristik:

\*   \*\*Markov-Modelle:\*\* Neigen oft zu einer höheren Handelsfrequenz ("Churning"), reagieren aber sehr schnell auf Änderungen der Volatilität.

\*   \*\*HMM (Clustering):\*\* Liefert oft die stabilsten Regime-Blöcke, da es auf der statistischen Verteilung der Daten basiert.

\*   \*\*LSTM:\*\* Versucht, die nicht-linearen Muster in den Sequenzen zu glätten. Es ist im Idealfall weniger anfällig für "Fehlsignale" (Rauschen) als das rein statistische Univariat-Modell.



---



\## 3. Risikoprofil \& Drawdown-Analyse

Der Schutz des Kapitals in Bärenmärkten ist die wichtigste Voraussetzung für eine sichere Entnahmephase im Alter.



!\[Equity Curves](./assets/equity\_curves.png)



\### Risikokennzahlen (SORR-Fokus):

\- \*\*Maximum Drawdown (MDD):\*\* Misst den maximalen kumulierten Verlust. Ein MDD-Wert unter -15% wird in dieser Studie als Erfolg gewertet, um das SORR-Problem zu entschärfen.

\- \*\*Calmar Ratio:\*\* Je höher dieser Wert, desto besser wurde die Rendite im Verhältnis zum eingegangenen Risiko (Drawdown) erzielt.



---



\## Aktualität

Die Daten werden mit jeder Pipeline-Ausführung aktualisiert.

