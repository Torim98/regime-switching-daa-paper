"""Generierung des statistics.md Master-Reports für das Git-Repository."""

import datetime
import os


def load_markdown_asset(filepath: str, fallback: str = "") -> str:
    """Markdown-Datei laden. Gibt Fallback-String zurück wenn nicht gefunden."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return fallback


def build_model_persistence_table(cfg) -> tuple[str, str]:
    """
    Pro Modell prüfen, ob ein persistiertes Modell geladen wurde.
    Gibt (persist_status_text, model_persistence_table_md) zurück.
    """
    persist_enabled = cfg.model_persistence.enabled
    persist_dir = cfg.model_persistence.models_dir
    persist_status_text = "AKTIV" if persist_enabled else "DEAKTIVIERT"

    model_status_rows = []
    for key in ["msm", "hmm", "lstm", "transformer"]:
        filename = getattr(cfg.model_persistence.files, key)
        filepath = os.path.join(persist_dir, filename)
        file_exists = os.path.exists(filepath)
        if persist_enabled and file_exists:
            status = "Geladen (persistiert)"
        else:
            status = "Neu trainiert"
        model_status_rows.append(f"| {key.upper()} | `{filename}` | {status} |")

    model_persistence_table = "\n".join(
        [
            "| Modell | Datei | Status |",
            "|:---|:---|:---|",
        ]
        + model_status_rows
    )

    return persist_status_text, model_persistence_table


def generate_statistics_report(cfg) -> str:
    """
    Liest alle generierten Markdown/PNG-Assets und baut daraus
    das Master-Dokument statistics.md.

    Lädt alle Teilberichte (EDA, Evaluation, SORR, MCS, Timing etc.)
    und fügt sie in ein einheitliches Markdown-Template ein.

    Gibt den fertigen Markdown-String zurück.
    """
    ASSETS_DIR = cfg.paths.assets_dir

    # Alle Markdown-Assets laden
    eda_desc_stats_md = load_markdown_asset(cfg.asset_path("eda_descriptive_stats"))
    eda_adf_tests_md = load_markdown_asset(cfg.asset_path("eda_adf_tests"))
    evaluation_table_md = load_markdown_asset(cfg.asset_path("evaluation_table"))
    performance_summary_md = load_markdown_asset(cfg.asset_path("performance_summary"))
    sorr_summary_md = load_markdown_asset(
        os.path.join(ASSETS_DIR, cfg.asset_path("sorr_summary"))
    )
    mcs_summary_md = load_markdown_asset(
        os.path.join(ASSETS_DIR, cfg.asset_path("mcs_summary"))
    )
    feature_corr_table_md = load_markdown_asset(cfg.asset_path("feature_correlation_table"))
    pipeline_timing_md = load_markdown_asset(
        os.path.join(ASSETS_DIR, cfg.asset_path("pipeline_timing")),
        fallback="*Keine Timing-Daten verfügbar.*",
    )
    annualized_metrics_md = load_markdown_asset(
        cfg.asset_path("annualized_metrics"),
        fallback="*Keine annualisierten Metriken verfügbar.*",
    )
    crisis_performance_md = load_markdown_asset(
        cfg.asset_path("crisis_performance"),
        fallback="*Keine Krisen-Performance verfügbar.*",
    )

    # --- Issue #13: Extended Evaluation ---
    classification_md = load_markdown_asset(
        cfg.asset_path("classification_metrics"),
        fallback="*Keine Klassifikationsmetriken verfügbar.*",
    )
    churning_md = load_markdown_asset(
        cfg.asset_path("churning_stats"),
        fallback="*Keine Churning-Statistik verfügbar.*",
    )
    depletion_ci_md = load_markdown_asset(
        cfg.asset_path("depletion_ci"),
        fallback="*Keine Depletion-CI verfügbar.*",
    )
    h1_md = load_markdown_asset(
        cfg.asset_path("h1_drawdown"),
        fallback="*H1-Test noch nicht durchgeführt.*",
    )
    h2_md = load_markdown_asset(
        cfg.asset_path("h2_transformer"),
        fallback="*H2-Test noch nicht durchgeführt.*",
    )
    break_even_md = load_markdown_asset(
        cfg.asset_path("break_even_table"),
        fallback="*Keine Break-Even-Analyse verfügbar.*",
    )
    withdrawal_sensitivity_md = load_markdown_asset(
        cfg.asset_path("withdrawal_sensitivity"),
        fallback="*Keine Entnahmeraten-Sensitivität verfügbar.*",
    )
    switch_timing_md = load_markdown_asset(
        cfg.asset_path("switch_timing"),
        fallback="*Keine Switch-Timing-Analyse verfügbar.*",
    )
    optuna_best_params_md = load_markdown_asset(
        cfg.asset_path("optuna_best_params"),
        fallback="*Keine Optuna-Optimierung durchgeführt.*",
    )

    # Threshold-Sensitivität pro Modell laden
    threshold_models = list(cfg.evaluation.extended.f1_models)
    threshold_sections = []
    for m in threshold_models:
        md = load_markdown_asset(
            cfg.asset_path("threshold_sensitivity").replace("{model}", m),
            fallback=f"*Keine Threshold-Sensitivity-Tabelle für {m} verfügbar.*",
        )
        threshold_sections.append(f"**{m}**\n\n{md}")
    threshold_sensitivity_md = "\n\n".join(threshold_sections)

    # Time-to-Recovery pro Modell laden (Buy_Hold + alle f1_models)
    ttr_models = ["Buy_Hold"] + threshold_models
    ttr_sections = []
    for m in ttr_models:
        md = load_markdown_asset(
            cfg.asset_path("ttr_table").replace("{model}", m),
            fallback=f"*Keine Time-to-Recovery-Tabelle für {m} verfügbar.*",
        )
        ttr_sections.append(f"**{m}**\n\n{md}")
    ttr_md = "\n\n".join(ttr_sections)

    # Fast Mode Status aus config auslesen
    fast_mode_enabled = cfg.fast_mode.enabled
    fast_mode_status = "TRUE (Development Mode)" if fast_mode_enabled else "FALSE (Full Run)"

    # Walk-Forward Status
    wf_enabled = getattr(cfg.walk_forward, "enabled", False)
    if wf_enabled:
        wf_status = (
            f"AKTIV (Modus: {cfg.walk_forward.mode}, "
            f"Train: {cfg.walk_forward.train_window_years}J, "
            f"Test: {cfg.walk_forward.test_window_months}M, "
            f"Step: {cfg.walk_forward.step_months}M)"
        )
    else:
        wf_status = "DEAKTIVIERT (Single 80/20 Split)"

    # Model Persistence Status
    persist_status_text, model_persistence_table = build_model_persistence_table(cfg)
    persist_dir = cfg.model_persistence.models_dir

    timestamp = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")

    if cfg.data.end_date_is_frozen:
        data_window_note = (
            f"Alle Auswertungen basieren auf dem **eingefrorenen Datensatz** "
            f"vom **{cfg.data.start_date}** bis **{cfg.data.end_date}** "
            f"(Thesis-Freeze)."
        )
    else:
        data_window_note = (
            f"Alle Auswertungen basieren auf dem Datensatz bis zum gestrigen "
            f"Tag ({cfg.data.end_date}) und werden automatisiert aktualisiert."
        )

    stats_md_content = f"""
# Detaillierte statistische Auswertung & Forschungsergebnisse

Diese Seite dokumentiert die numerischen und grafischen Ergebnisse der Forschungs-Pipeline. {data_window_note}

---

## 1. Executive Summary: Performance & Risiko
Ein direkter Vergleich der Kernkennzahlen über den gesamten **Out-of-Sample Testzeitraum**.

{performance_summary_md}

> **Kernaussage:** Vergleiche den **Max Drawdown** der aktiven Strategien mit der Buy & Hold Benchmark. Ziel der Arbeit ist eine signifikante Reduktion dieses Werts zur Minderung des SORR.

---

## 2. Datenbasis & Baseline Portfolio
Grundlage der Untersuchung ist ein globaler Multi-Asset-Ansatz.

### Explorative Datenanalyse (EDA)
**Deskriptive Statistik der Basiszeitreihen:**
{eda_desc_stats_md}

**Prüfung auf Stationarität (Augmented Dickey-Fuller Test):**
{eda_adf_tests_md}

**Volatilitätscluster und Autokorrelation (Heteroskedastizität):**
![Volatility Clusters](../assets/{cfg.paths.assets.eda_volatility_clusters})

### Feature-Korrelation
Pearson-Korrelationsmatrix der sechs Modell-Features zur Prüfung auf Multikollinearität.

![Feature Correlation Matrix](../assets/{cfg.paths.assets.feature_correlation_matrix})

### SORR Kontext: Historische Drawdowns
Darstellung der extremsten Verlustphasen des 60/40 Portfolios als Motivation für den aktiven Kapitalschutz.
![Historical Drawdowns](../assets/{cfg.paths.assets.eda_historical_drawdowns})

### 60/40 Portfolio Kapitalkurve
Die Abbildung zeigt die kumulierte Wertentwicklung des statischen Referenzportfolios (60% Aktien / 40% Anleihen).

![Capital Curve](../assets/{cfg.paths.assets.capital_curve})

*   **Datenquelle:** S&P 500 (`^GSPC`) und Vanguard Long-Term Treasury (`VUSTX`).
*   **Reproduzierbarkeit:** Der bereinigte Datensatz inkl. aller Features ist hinterlegt unter: `data/02_feature_engineered_data.parquet`.

---

## 3. Regime-Erkennung der Einzelmodelle
Hier werden die Identifikations-Ergebnisse der Modell-Kategorien (Statistik, Clustering, Deep Learning) visualisiert.

### A. Markov-Switching-Modelle (Ökonometrie)
Identifikation von Bull- und Bear-Regimes mittels eines univariaten Zwei-Regime-Markov-Switching-Modells auf Basis der S&P 500-Renditen.
![Markov Models](../assets/{cfg.paths.assets.markov_model})

### B. Hidden Markov Model (Unsupervised Clustering)
![HMM Regimes](../assets/{cfg.paths.assets.hmm_regimes})

### C. LSTM-Netzwerk (Deep Learning)
Vorhersage der Marktphasen durch das neuronale Netzwerk (trainiert auf Pagan-Sossounov-Labels).
![LSTM Model](../assets/{cfg.paths.assets.lstm_model})

### D. Transformer-Netzwerk (Attention-basierte Regime-Erkennung)
Klassifikation von Marktregimes mittels eines Transformer-Encoders mit Multi-Head Self-Attention und Positional Encoding. Im Gegensatz zu rekurrenten Architekturen (LSTM) verarbeitet der Transformer alle Zeitschritte einer Sequenz parallel und lernt über den Attention-Mechanismus, welche historischen Datenpunkte die höchste Relevanz für die aktuelle Regime-Klassifikation besitzen. Trainiert im Supervised-Setting auf Pagan-Sossounov-Labels.
![Transformer Model](../assets/{cfg.paths.assets.transformer_model})

### E. Globaler Regime-Vergleich
Detaillierte Gegenüberstellung der Wahrscheinlichkeiten und harten Signale aller Modelle.
![Regime Comparison](../assets/{cfg.paths.assets.regime_comparison})

### F. Hyperparameter-Optimierung (Optuna)
Bayessche Suche über den Hyperparameter-Raum aller vier Modelle mittels Walk-Forward-Validierung als innere CV. Optimierungsziel ist der mediane OOS-Sharpe-Ratio über die subgesampelten Folds; geprunete Trials nutzen den Median-Pruner. Die hier ausgewiesenen Werte wurden 1:1 in die `config.yaml` übernommen und für den finalen Walk-Forward-Lauf verwendet.

{optuna_best_params_md}

**Diagnose-Plots pro Modell** (Optimization History · Param-Importance · Slice · Contour):

| Modell | History | Importance | Slice | Contour |
|:---|:---|:---|:---|:---|
| MSM         | ![](../assets/optuna_MSM_history.png)         | ![](../assets/optuna_MSM_importance.png)         | ![](../assets/optuna_MSM_slice.png)         | ![](../assets/optuna_MSM_contour.png)         |
| HMM         | ![](../assets/optuna_HMM_history.png)         | ![](../assets/optuna_HMM_importance.png)         | ![](../assets/optuna_HMM_slice.png)         | ![](../assets/optuna_HMM_contour.png)         |
| LSTM        | ![](../assets/optuna_LSTM_history.png)        | ![](../assets/optuna_LSTM_importance.png)        | ![](../assets/optuna_LSTM_slice.png)        | ![](../assets/optuna_LSTM_contour.png)        |
| Transformer | ![](../assets/optuna_Transformer_history.png) | ![](../assets/optuna_Transformer_importance.png) | ![](../assets/optuna_Transformer_slice.png) | ![](../assets/optuna_Transformer_contour.png) |

### G. Label-Konkordanz (Auswahl der Trainings-Labels)
Vergleich der Regime-Labeler (MSM, HMM, Pagan-Sossounov, Peak-to-Trough, Lunde-Timmermann, NBER) zur Begründung der Label-Wahl für die Supervised-Modelle. Pagan-Sossounov wurde aufgrund seiner hohen Konkordanz mit NBER-Rezessionsperioden als Trainingsziel für LSTM und Transformer gewählt.

![Label Concordance](../assets/{cfg.paths.assets.label_concordance_matrix})
![Label Cohen's κ](../assets/{cfg.paths.assets.label_kappa_matrix})
![Label Timeline](../assets/{cfg.paths.assets.label_timeline_comparison})

---

## 4. Backtesting & Strategie-Evaluation
Die ökonomische Anwendung der Regime-Signale durch dynamische Umschichtung in den Geldmarkt.

### Walk-Forward-Schema
Rollierende Train/Test-Fenster über den gesamten Untersuchungszeitraum. Jede Zeile entspricht einem Fold; der blaue Balken markiert das Trainingsfenster, der orange Balken das OOS-Testfenster. Die strikte chronologische Trennung verhindert Look-ahead Bias.

![Walk-Forward-Schema](../assets/{cfg.paths.assets.walk_forward_schema})

### Equity Curves im Vergleich
![Equity Curves](../assets/{cfg.paths.assets.equity_curves})

### Annualisierte Performance-Metriken
Normalisierte Kennzahlen (CAGR, Sharpe, Sortino, Calmar) für den Vergleich über unterschiedlich lange Evaluationszeiträume.

{annualized_metrics_md}

### Klassifikationsmetriken (vs. NBER-Rezessionen als Ground Truth)
Vergleich der Modelle als binäre Rezessionsklassifikatoren (Precision, Recall, F1).

{classification_md}

![Confusion Matrices](../assets/{cfg.paths.assets.confusion_matrices})

**ROC- und Precision-Recall-Kurven** (schwellenunabhängiger Vergleich über `*_Prob`):

![ROC-Kurven](../assets/{cfg.paths.assets.roc_curves})
![PR-Kurven](../assets/{cfg.paths.assets.pr_curves})

### Signal-Churning & Whipsaw-Analyse
Quantifizierung der Wechselhäufigkeit und Anteil sehr kurzer Regime-Phasen („Whipsaws").

{churning_md}

### Regime-Wahrscheinlichkeits-Heatmap
Zeitverlauf der Bear-Wahrscheinlichkeiten aller Modelle.

![Regime Probability Heatmap](../assets/{cfg.paths.assets.regime_probability_heatmap})

### Threshold-Sensitivität
Variation der Entscheidungs-Schwelle pro Modell. Zeigt, wie robust Final Wealth, Max Drawdown und Anzahl der Regime-Wechsel gegenüber einer veränderten Bull/Bear-Klassifikations-Grenze sind (Kap. 4.1 — Glättung).

{threshold_sensitivity_md}

### Time-to-Recovery
Alle Drawdown-Phasen jenseits der Mindesttiefe (gemäß `extended.ttr_min_dd`) mit Peak-, Trough- und Recovery-Datum sowie Dauer in Handelstagen. Eine offene (noch nicht erholte) Phase wird im Recovery-Feld mit „—" markiert.

{ttr_md}

### Krisen-Performance
Return und Max Drawdown während historischer Krisenperioden — der zentrale Nachweis für den Tail-Risk-Schutz der Regime-Switching-Modelle.

{crisis_performance_md}

### Switch-Timing relativ zum Drawdown-Peak
Zeitlicher Abstand zwischen dem ersten Bear-Signal des Modells und dem Drawdown-Trough des Buy & Hold-Portfolios je Krise. Positiv = Modell reagierte frühzeitig, negativ = zu spät.

{switch_timing_md}

### Drawdown-Verlauf
![Drawdown](../assets/{cfg.paths.assets.drawdown})

### Rollierender Sharpe Ratio
Zeitvariierender, risikoadjustierter Rendite-Vergleich über ein rollendes 252-Tage-Fenster.

![Rolling Sharpe](../assets/{cfg.paths.assets.rolling_sharpe})

### Umfassende Kennzahlen-Matrix
Detaillierte statistische Analyse inklusive risikoadjustierter Kennzahlen (Sharpe, Sortino, Calmar).

{evaluation_table_md}

### Transaktionskosten

Diese Grafik zeigt die kumulierten Transaktionskosten im Zeitverlauf. Steile Anstiege deuten auf instabile Regime-Wechsel ("Churning") hin.

![Transaction Costs](../assets/{cfg.paths.assets.transaction_costs})

Stress-Test: Sequence of Returns Risk (SORR)
Außerdem wurde die Überlebensdauer des Kapitals in einer simulierten Entnahmephase (Ruhestandsszenario) durchgeführt.

### SORR-Simulation: Vergleich der Entnahmeszenarien

In dieser Tabelle werden verschiedene Stress-Szenarien (Standard, Aggressiv, Geringes Kapital) gegenübergestellt.

{sorr_summary_md}

Abbildung der Kapitalentwicklung der unterschiedlichen Szenarien:
![SORR Standard](../assets/{cfg.paths.assets.sorr_sim_standard})
![SORR Aggressive](../assets/{cfg.paths.assets.sorr_sim_aggressive})
![SORR Low Capital](../assets/{cfg.paths.assets.sorr_sim_low_capital})

### MCS: Block-Bootstrap Robustness-Check

Um die statistische Signifikanz zu prüfen, wurden 1.000 künstliche Marktpfade mittels Block-Bootstrap simuliert.
![MCS Paths](../assets/{cfg.paths.assets.mcs_paths})
{mcs_summary_md}

Verteilung der Endkapitalwerte:

![MCS Boxplots Standard](../assets/{cfg.paths.assets.mcs_boxplot_standard})
![MCS Boxplots Aggressive](../assets/{cfg.paths.assets.mcs_boxplot_aggressive})
![MCS Boxplots Low Capital](../assets/{cfg.paths.assets.mcs_boxplot_low_capital})

Wahrscheinlichkeitskorridore:

Die schattierten Bereiche zeigen das 5% bis 95% Konfidenzintervall der Kapitalentwicklung.
![MCS Quantiles](../assets/{cfg.paths.assets.mcs_quantiles})

### Depletion Rate mit 95%-Konfidenzintervall
Wilson-CI für die Ruin-Wahrscheinlichkeit (P[Endkapital ≤ 0]) je Szenario × Strategie.

{depletion_ci_md}

### Hypothesentests (gepaarter Wilcoxon, α = 0.05)
**H1 — Regime-Switching reduziert MaxDD vs. Buy & Hold:**

{h1_md}

**H2 — Transformer dominiert Ökonometrie und LSTM im Endvermögen:**

{h2_md}

### Break-Even-Transaktionskosten
Ab welcher Kostenquote (in Basispunkten pro Umschichtung) verliert das aktive Switching seinen Renditevorteil gegenüber Buy & Hold?

{break_even_md}

![Break-Even-Analyse](../assets/{cfg.paths.assets.break_even_plot})

### Entnahmeraten-Sensitivität (3.5 % / 4 % / 5 %)
Robustheit der SORR-Ergebnisse bei variierenden jährlichen Entnahmen.

{withdrawal_sensitivity_md}

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

{pipeline_timing_md}

---

## Modell-Persistierung

Status der Modell-Persistierung für diesen Pipeline-Durchlauf:

- **Persistierung:** {persist_status_text}
- **Modell-Verzeichnis:** `{persist_dir}`

{model_persistence_table}

> **Hinweis:** Bei aktivierter Persistierung werden vortrainierte Modelle aus `{persist_dir}` geladen, sofern die Dateien existieren. Andernfalls wird normal trainiert und das Ergebnis für zukünftige Läufe gespeichert. Bei Änderungen an Hyperparametern müssen die entsprechenden Modelldateien gelöscht werden.

---

**Zuletzt aktualisiert:** {timestamp}<br>
**End date:** `{cfg.data.end_date}`<br>
**Fast Mode Status zur Laufzeit:** {fast_mode_status}<br>
**Walk-Forward-Validierung:** {wf_status}<br>
**Modell-Persistierung:** {persist_status_text}<br>
*Generiert durch die automatisierte ETL-Pipeline (Notebook 99).*
"""

    return stats_md_content