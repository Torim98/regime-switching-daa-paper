# How to Add a New ML Model to the Research Pipeline

> **Ziel:** Schritt-für-Schritt-Anleitung zur Integration eines neuen Regime-Switching-Modells in die bestehende Research-Pipeline. Der Guide nutzt die **Signal-Schnittstelle**, den **Dynamic-Matching-Mechanismus** und das **zentrale Konfigurationsmanagement** des Frameworks, sodass ein neues Modell automatisch in Backtesting, Evaluation und Reporting erscheint, ohne dass downstream Code angepasst werden muss.

---

## Inhaltsverzeichnis

1. [Voraussetzungen](#1-voraussetzungen)
2. [Die Signal-Schnittstelle (Interface-Spezifikation)](#2-die-signal-schnittstelle-interface-spezifikation)
3. [Hyperparameter in der zentralen Config registrieren](#3-hyperparameter-in-der-zentralen-config-registrieren)
4. [Schritt-für-Schritt-Anleitung](#4-schritt-für-schritt-anleitung)
   - [Schritt 1: Config-Eintrag erstellen](#schritt-1-config-eintrag-erstellen)
   - [Schritt 2: Daten laden und Features vorbereiten](#schritt-2-daten-laden-und-features-vorbereiten)
   - [Schritt 3: Modell implementieren und trainieren (mit Config)](#schritt-3-modell-implementieren-und-trainieren-mit-config)
   - [Schritt 4: Wahrscheinlichkeiten und Signale erzeugen](#schritt-4-wahrscheinlichkeiten-und-signale-erzeugen)
   - [Schritt 5: Signale in den DataFrame schreiben](#schritt-5-signale-in-den-dataframe-schreiben)
   - [Schritt 6: DataFrame speichern](#schritt-6-dataframe-speichern)
5. [Warum das funktioniert: Dynamic Matching](#5-warum-das-funktioniert-dynamic-matching)
6. [Look-Ahead Bias Prevention (T+1 Shift)](#6-look-ahead-bias-prevention-t1-shift)
7. [Code-Template (Copy & Paste)](#7-code-template-copy--paste)
8. [Validierungs-Checkliste](#8-validierungs-checkliste)
9. [Referenz-Implementierungen](#9-referenz-implementierungen)
10. [FAQ & Troubleshooting](#10-faq--troubleshooting)

---

## 1. Voraussetzungen

Bevor du ein neues Modell integrierst, stelle sicher, dass:

- [ ] Die Pipeline-Module `00_dependencies`, `01_data_preprocessing` und `02_feature_engineering` erfolgreich durchgelaufen sind
- [ ] Die Datei `data/02_feature_engineered_data.parquet` existiert und aktuell ist
- [ ] Die benötigten Python-Pakete für dein Modell installiert sind (ggf. in `00_dependencies.ipynb` ergänzen)
- [ ] Du den Aufbau von `jupyter/03_regime_switching_models.ipynb` grundlegend verstanden hast

---

## 2. Die Signal-Schnittstelle (Interface-Spezifikation)

Das zentrale Designprinzip der Pipeline ist die **standardisierte Signal-Schnittstelle**. Jedes Modell muss exakt **zwei Spalten** im gemeinsamen `test_df` DataFrame erzeugen:

### Spalte 1: `<Modell>_Prob` (Regime-Wahrscheinlichkeit)

| Eigenschaft | Spezifikation |
|:---|:---|
| **Namenskonvention** | `<Modellname>_Prob` |
| **Datentyp** | `float64` |
| **Wertebereich** | `0.0` bis `1.0` |
| **Semantik** | Wahrscheinlichkeit, dass der aktuelle Tag ein **Bear-Regime** (Krise) ist |
| **Beispiel** | `HMM_Prob`, `LSTM_Prob`, `Transformer_Prob` |

### Spalte 2: `<Modell>_Signal` (Binäres Handelssignal)

| Eigenschaft | Spezifikation |
|:---|:---|
| **Namenskonvention** | `<Modellname>_Signal` |
| **Datentyp** | `int` (0 oder 1) |
| **Wertebereich** | `0` = **Bull** (investiert), `1` = **Bear** (Cash/Geldmarkt) |
| **Semantik** | Binäre Entscheidung, abgeleitet aus `_Prob` mittels Threshold (i.d.R. `>= 0.5`) |
| **Beispiel** | `HMM_Signal`, `LSTM_Signal`, `Transformer_Signal` |

### Namenskonvention

```
<Modellname>_Prob     →  z.B. MyModel_Prob
<Modellname>_Signal   →  z.B. MyModel_Signal
```

**Regeln für `<Modellname>`:**
- Verwende **PascalCase** oder **Snake_Case** mit Großbuchstaben (z.B. `Transformer`, `LSTM_Unsupervised`, `MS_Exo`)
- **Keine Leerzeichen**, verwende stattdessen Unterstriche
- Der Name muss **eindeutig** sein und darf nicht mit dem Namen eines bestehenden Modells kollidieren
- Der Suffix `_Signal` ist **reserviert** und wird vom Dynamic-Matching-Algorithmus als Erkennungsmerkmal verwendet

### Bestehende Modelle als Referenz

| Modellname        | `_Prob`-Spalte           | `_Signal`-Spalte           | Paradigma                  |
| :---------------- | :----------------------- | :------------------------- | :------------------------- |
| HMM               | `HMM_Prob`               | `HMM_Signal`               | Ökonometrie (Unsupervised) |
| MS_Univariate     | `MS_Univariate_Prob`     | `MS_Univariate_Signal`     | Ökonometrie (Regression)   |
| MS_Exo            | `MS_Exo_Prob`            | `MS_Exo_Signal`            | Ökonometrie (Exogen)       |
| LSTM              | `LSTM_Prob`              | `LSTM_Signal`              | ML (Supervised)            |
| LSTM_Unsupervised | `LSTM_Unsupervised_Prob` | `LSTM_Unsupervised_Signal` | ML (Unsupervised)          |

---

## 3. Hyperparameter in der zentralen Config registrieren

> **Wichtig:** Seit [Issue #3](https://github.com/Torim98/regime-switching-daa/issues/3) werden **alle Pipeline-Parameter zentral** in [`config/config.yaml`](../config/config.yaml) verwaltet. Hardcoded Hyperparameter in Notebooks sind nicht mehr erlaubt.

### Architektur-Übersicht

```
config/
├── config.yaml          # Single Source of Truth — alle Parameter
└── config_loader.py     # PipelineConfig-Klasse + Singleton `cfg`
```

Die `config.yaml` ist hierarchisch nach Pipeline-Sektionen gegliedert:

```yaml
models:
  hmm:
    n_components: 2
    covariance_type: "full"
    ...
  lstm:
    window_size: 30
    epochs: 30
    ...
  my_model:              # ← Dein neues Modell hier eintragen
    param_1: value_1
    ...
```

### So lädst du die Config in einem Notebook

```python
# Am Anfang jedes Notebooks — Config laden
import sys; sys.path.insert(0, "../config")
from config_loader import cfg

# Zugriff per Dot-Notation:
cfg.models.hmm.n_components          # → 2
cfg.models.lstm.epochs               # → 30
cfg.features.model_features          # → ['Returns', 'Vol_20', ...]
cfg.data_path("test_data")           # → "../data/03_test_df_data.parquet"
cfg.asset_path("equity_curves")      # → "../assets/equity_curves.png"
cfg.transaction_cost_rate             # → 0.001 (convenience property)
```

### So fügst du dein Modell zur Config hinzu

**Schritt A:** Öffne `config/config.yaml` und ergänze unter der `models:`-Sektion einen neuen Block:

```yaml
models:
  # ... bestehende Modelle (hmm, markov_switching, lstm, lstm_unsupervised) ...

  my_model:                    # ← Schlüsselname in snake_case
    window_size: 20            # Beispiel-Hyperparameter
    n_heads: 4                 # Beispiel für Transformer
    n_layers: 2
    d_model: 64
    dropout: 0.1
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    threshold: 0.5             # Signal-Threshold
```

**Schritt B:** Greife im Notebook auf die Parameter zu:

```python
# Statt hardcoded: window_size = 20
my_cfg = cfg.models.my_model
window_size = my_cfg.window_size      # → 20
n_heads     = my_cfg.n_heads          # → 4
epochs      = my_cfg.epochs           # → 50
threshold   = my_cfg.threshold        # → 0.5
```

### Vorteile der zentralen Konfiguration

| Aspekt                         | Vorher (Hardcoded)                            | Nachher (Config)                                                                                                              |
| :----------------------------- | :-------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| **Änderung eines Parameters**  | Notebook öffnen, Zelle suchen, manuell ändern | `config.yaml` editieren, propagiert automatisch                                                                               |
| **Reproduzierbarkeit**         | Parameter über Notebooks verstreut            | Alles an einer Stelle, versioniert in Git                                                                                     |
| **Hyperparameter-Optimierung** | Manuell im Code anpassen                      | Optuna kann `config.yaml` programmatisch überschreiben ([Issue #2](https://github.com/Torim98/regime-switching-daa/issues/2)) |
| **Fast Mode (Entwicklung)**    | Jedes Modell einzeln anpassen                 | `fast_mode.enabled: true` reduziert Epochs/MCS-Paths automatisch                                                              |

### Fast Mode für neue Modelle (optional)

Falls du einen Fast-Mode-Override für dein Modell hinzufügen möchtest:

```yaml
fast_mode:
  enabled: false
  overrides:
    lstm_epochs: 5
    lstm_unsupervised_epochs: 10
    mcs_n_paths: 100
    my_model_epochs: 5          # ← Neuer Override
```

Dann im `config_loader.py` (Klasse `PipelineConfig.__init__`) ergänzen:

```python
if self.fast_mode.enabled:
    # ... bestehende overrides ...
    self.models.my_model.epochs = self.fast_mode.overrides.my_model_epochs
```

---

## 4. Schritt-für-Schritt-Anleitung

Alle Änderungen erfolgen in **zwei Dateien**: `config/config.yaml` (Hyperparameter) und `jupyter/03_regime_switching_models.ipynb` (Modell-Code). Füge die neue Zelle **nach den bestehenden Modellen und vor der finalen Speicher-Zelle** ein.

### Schritt 1: Config-Eintrag erstellen

Ergänze `config/config.yaml` unter `models:` mit den Hyperparametern deines Modells:

```yaml
models:
  # ... bestehende Modelle ...

  my_model:
    window_size: 20
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    dropout: 0.1
    threshold: 0.5
    # ... weitere modellspezifische Parameter ...
```

### Schritt 2: Daten laden und Features vorbereiten

Der DataFrame `df` ist zum Zeitpunkt deiner Zelle bereits geladen. Die verfügbaren Features sind ebenfalls in der Config definiert:

```python
# Features aus der zentralen Config laden
feature_cols = cfg.features.model_features
# → ['Returns', 'Vol_20', 'Distance_SMA', 'Momentum', 'VIX', 'Yield_Spread']
```

> **Hinweis:** Du kannst eine Teilmenge dieser Features nutzen oder, falls erforderlich, im Feature-Engineering-Notebook (`02_feature_engineering.ipynb`) zusätzliche Features berechnen und die Feature-Liste in `config.yaml` erweitern.

### Schritt 3: Modell implementieren und trainieren (mit Config)

Implementiere dein Modell in der neuen Zelle. **Alle Hyperparameter kommen aus der Config**. Kein Hardcoding!

```python
from my_model_library import MyModel

# Hyperparameter aus der zentralen Config laden
my_cfg = cfg.models.my_model

# Feature-Matrix vorbereiten
feature_cols = cfg.features.model_features
X = df[feature_cols].dropna()

# Modell initialisieren — alle Parameter aus config.yaml
model = MyModel(
    window_size=my_cfg.window_size,
    epochs=my_cfg.epochs,
    batch_size=my_cfg.batch_size,
    learning_rate=my_cfg.learning_rate,
    dropout=my_cfg.dropout,
)
model.fit(X)

# Regime-Wahrscheinlichkeiten berechnen (Bear-Wahrscheinlichkeit)
bear_probabilities = model.predict_proba(X)  # Array mit Werten 0.0 - 1.0
```

### Schritt 4: Wahrscheinlichkeiten und Signale erzeugen

Leite aus den Bear-Wahrscheinlichkeiten das binäre Signal ab. Der Threshold kommt ebenfalls aus der Config:

```python
import numpy as np

# Modellname definieren (WICHTIG: Einheitlich für _Prob und _Signal verwenden!)
MODEL_NAME = "MyModel"

# Threshold aus Config (oder Default 0.5)
threshold = my_cfg.threshold

# Wahrscheinlichkeiten in den DataFrame schreiben
df[f'{MODEL_NAME}_Prob'] = bear_probabilities

# Binäres Signal ableiten
df[f'{MODEL_NAME}_Signal'] = (df[f'{MODEL_NAME}_Prob'] >= threshold).astype(int)
```

> ⚠️ **Wichtig:** Stelle sicher, dass `_Prob` die **Bear-Wahrscheinlichkeit** enthält (hohe Werte = Krise). Falls dein Modell die Bull-Wahrscheinlichkeit ausgibt, invertiere sie: `bear_prob = 1 - bull_prob`.

### Schritt 5: Signale in den DataFrame schreiben

Deine Spalten `MyModel_Prob` und `MyModel_Signal` sind jetzt Teil von `df`. Prüfe direkt nach der Zuweisung, ob die Signale plausibel sind:

```python
# Quick-Sanity-Check
print(f"\n{MODEL_NAME} — Statistik nach Regimes:")
print(df.groupby(f'{MODEL_NAME}_Signal')[['Returns', 'VIX', 'Yield_Spread', f'{MODEL_NAME}_Prob']].mean())
print(f"\nSignal-Verteilung:\n{df[f'{MODEL_NAME}_Signal'].value_counts()}")
```

**Erwartete Plausibilitätskriterien:**
- Regime `1` (Bear) sollte **niedrigere Returns** und **höheren VIX** aufweisen als Regime `0` (Bull)
- Falls dies umgekehrt ist, müssen die Labels getauscht werden (das Modell hat die Regimes vertauscht zugeordnet)

### Schritt 6: DataFrame speichern

Die **letzte Zelle** des Notebooks speichert den `test_df` DataFrame automatisch — unter Verwendung des in der Config definierten Pfads:

```python
# Diese Zelle existiert bereits am Ende von 03_regime_switching_models.ipynb
# Deine neuen Spalten werden automatisch mit persistiert!
test_df.to_parquet(cfg.data_path("test_data"))
# → speichert nach "../data/03_test_df_data.parquet"
```

> Du musst an der Speicher-Zelle **nichts ändern**, solange deine neuen Spalten korrekt im `df` / `test_df` DataFrame enthalten sind.

---

## 5. Warum das funktioniert: Dynamic Matching

Die nachfolgenden Pipeline-Module (`04_backtesting`, `05_evaluation`, `99_statistics_md`) nutzen einen **dynamischen Such-Algorithmus**, der alle Modell-Signale automatisch erkennt:

```python
# Aus 04_backtesting.ipynb — Kernlogik des Dynamic Matching:
signal_cols = [col for col in test_df.columns if col.endswith('_Signal')]

for sig_col in signal_cols:
    model_name = sig_col.rsplit('_', 1)[0]  # Extrahiert "HMM" aus "HMM_Signal"
    print(f"Berechne Backtest für {model_name}...")
    backtesting_results[model_name] = backtest(test_df, sig_col, fee=cfg.transaction_cost_rate)
```

**Das bedeutet:**
1. **Backtesting** erkennt jede Spalte, die auf `_Signal` endet, und führt einen vollständigen Backtest durch
2. **Evaluation** berechnet für jedes erkannte Modell automatisch alle Kennzahlen (Sharpe, Sortino, Calmar, Max Drawdown etc.)
3. **Reporting** generiert Equity Curves, Statistik-Tabellen und SORR-Simulationen für alle Modelle

➡️ **Du musst keinen Code in `04_backtesting.ipynb`, `05_evaluation.ipynb` oder `99_statistics_md.ipynb` anpassen.** Es genügt, die Signal-Schnittstelle korrekt zu implementieren und die Hyperparameter in der Config zu registrieren.

---

## 6. Look-Ahead Bias Prevention (T+1 Shift)

Ein kritisches Konzept, das das Backtesting-Modul automatisch handhabt:

```python
# Aus 04_backtesting.ipynb — Automatischer T+1 Shift:
trading_signal = df[signal_col].shift(cfg.backtesting.signal_shift).fillna(0)
# signal_shift ist in config.yaml definiert (Standard: 1)
```

**Was passiert hier?**
- Das Backtesting verschiebt jedes Signal um **einen Handelstag** in die Zukunft (`shift(1)`)
- Ein Signal, das am Tag `T` berechnet wurde, wird erst am Tag `T+1` als Handelsentscheidung angewandt
- Dies verhindert **Look-Ahead Bias**: Entscheidungen basieren ausschließlich auf historisch verfügbaren Informationen
- Der Shift-Wert ist über `backtesting.signal_shift` in der `config.yaml` konfigurierbar (Standard: `1`)

**Das bedeutet für dich:**
- Du musst den Shift **NICHT** selbst implementieren, das Backtesting erledigt das
- Du darfst den Shift **NICHT** doppelt anwenden (also nicht schon in `03_regime_switching_models.ipynb`)
- Die Spalten `_Prob` und `_Signal` enthalten die Werte **zum Zeitpunkt der Berechnung** (Tag `T`)

---

## 7. Code-Template (Copy & Paste)

### A. Config-Eintrag (`config/config.yaml`)

Füge unter `models:` hinzu:

```yaml
  my_model:                       # ← snake_case Schlüsselname
    # --- Architektur ---
    window_size: 20               # Input-Sequenzlänge
    units: 64                     # Modellgröße (z.B. Hidden Units, d_model)
    dropout: 0.1                  # Regularisierung
    # --- Training ---
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    validation_split: 0.1
    # --- Signal ---
    threshold: 0.5                # Bear-Signal wenn Prob >= threshold
```

### B. Notebook-Zelle (`jupyter/03_regime_switching_models.ipynb`)

Füge die folgende Vorlage als **neue Zelle** ein (vor der finalen Speicher-Zelle):

```python
# =============================================================================
# Neues Modell: <Modellname>
# Typ: <Supervised | Unsupervised | Regression | ...>
# Beschreibung: <Kurze Beschreibung des Ansatzes>
# Config-Key: models.my_model
# =============================================================================

MODEL_NAME = "MyModel"  # ← Hier anpassen (kein _Signal / _Prob Suffix!)

# --- 0. Hyperparameter aus zentraler Config laden ---
my_cfg = cfg.models.my_model      # Alle Parameter aus config/config.yaml

# --- 1. Features vorbereiten ---
feature_cols = cfg.features.model_features
X = df[feature_cols].dropna()

# --- 2. Modell trainieren (alle Hyperparameter aus Config!) ---
# from my_library import MyModel
# model = MyModel(
#     window_size=my_cfg.window_size,
#     units=my_cfg.units,
#     dropout=my_cfg.dropout,
#     epochs=my_cfg.epochs,
#     batch_size=my_cfg.batch_size,
#     learning_rate=my_cfg.learning_rate,
#     validation_split=my_cfg.validation_split,
# )
# model.fit(X)

# --- 3. Bear-Wahrscheinlichkeiten berechnen ---
# bear_probs = model.predict_proba(X)  # Werte zwischen 0.0 und 1.0
# HINWEIS: Falls dein Modell Bull-Wahrscheinlichkeiten liefert → bear_probs = 1 - bull_probs

# --- 4. Ergebnisse in den DataFrame schreiben ---
# df[f'{MODEL_NAME}_Prob'] = np.nan  # Initialisierung (wegen dropna oben)
# df.loc[X.index, f'{MODEL_NAME}_Prob'] = bear_probs
# df[f'{MODEL_NAME}_Signal'] = (df[f'{MODEL_NAME}_Prob'] >= my_cfg.threshold).astype(int)

# --- 5. Sanity Check ---
print(f"\n{'='*60}")
print(f"   {MODEL_NAME} — Regime-Statistik")
print(f"{'='*60}")
print(df.groupby(f'{MODEL_NAME}_Signal')[['Returns', 'VIX', 'Yield_Spread', f'{MODEL_NAME}_Prob']].mean())
print(f"\nSignal-Verteilung:\n{df[f'{MODEL_NAME}_Signal'].value_counts()}")

# --- 6. Plausibilitäts-Check: Bear-Regime sollte niedrigere Returns haben ---
mean_returns_by_regime = df.groupby(f'{MODEL_NAME}_Signal')['Returns'].mean()
if mean_returns_by_regime.get(1, 0) > mean_returns_by_regime.get(0, 0):
    print(f"\n WARNUNG: {MODEL_NAME} Bear-Regime (1) hat höhere Returns als Bull (0)!")
    print("    → Labels könnten vertauscht sein. Bitte prüfen und ggf. invertieren:")
    print(f"    → df['{MODEL_NAME}_Signal'] = 1 - df['{MODEL_NAME}_Signal']")
else:
    print(f"\n {MODEL_NAME} Plausibilitäts-Check bestanden.")
```

---

## 8. Validierungs-Checkliste

Führe nach der Integration folgende Prüfungen durch:

### Formale Prüfungen
- [ ] Die Spalte `<Modell>_Prob` existiert in `test_df` und enthält `float`-Werte zwischen `0.0` und `1.0`
- [ ] Die Spalte `<Modell>_Signal` existiert in `test_df` und enthält ausschließlich `0` oder `1`
- [ ] Keine `NaN`-Werte in `<Modell>_Signal` (im Testzeitraum)
- [ ] Der Modellname kollidiert nicht mit bestehenden Namen (`HMM`, `MS_Univariate`, `MS_Exo`, `LSTM`, `LSTM_Unsupervised`)

### Config-Prüfungen
- [ ] Neuer Eintrag unter `models:` in `config/config.yaml` angelegt
- [ ] **Keine** Hyperparameter hardcoded im Notebook, alles kommt aus `cfg.models.<name>`
- [ ] Config-Key stimmt mit dem Zugriff im Notebook überein (z.B. `cfg.models.my_model`)
- [ ] Falls Fast-Mode-Override gewünscht: Eintrag in `fast_mode.overrides` und `config_loader.py` ergänzt

### Inhaltliche Prüfungen
- [ ] Bear-Regime (Signal = 1) weist im Durchschnitt **niedrigere Returns** auf als Bull-Regime (Signal = 0)
- [ ] Bear-Regime weist im Durchschnitt **höheren VIX** auf als Bull-Regime
- [ ] Die Signalverteilung ist plausibel (nicht 99% ein Regime)

### Pipeline-Integration
- [ ] `jupyter/03_regime_switching_models.ipynb` läuft fehlerfrei durch
- [ ] Die Datei `data/03_test_df_data.parquet` wird erfolgreich aktualisiert
- [ ] `jupyter/04_backtesting.ipynb` erkennt das neue Modell automatisch und berechnet Equity Curves
- [ ] `jupyter/05_evaluation.ipynb` berechnet Kennzahlen (Sharpe, Sortino, Calmar, Max Drawdown) für das neue Modell
- [ ] `jupyter/99_statistics_md.ipynb` generiert die aktualisierte `statistics.md` mit dem neuen Modell
- [ ] Die Grafiken in `assets/` (Equity Curves, Regime Comparison, MCS Boxplots etc.) enthalten das neue Modell

### End-to-End Test (empfohlen)
```bash
# Vollständigen Pipeline-Durchlauf starten:
# Öffne jupyter/regime-switching-daa.ipynb und führe alle Zellen aus.
```

---

## 9. Referenz-Implementierungen

Die folgenden bestehenden Modelle in `jupyter/03_regime_switching_models.ipynb` dienen als Referenz. Alle laden ihre Hyperparameter aus der zentralen `config.yaml`:

### A. HMM (Hidden Markov Model) — Unsupervised, Ökonometrie
- **Bibliothek:** `hmmlearn`
- **Ansatz:** Identifiziert Cluster in den Datenverteilungen ohne gelabelte Daten
- **Output:** `HMM_Prob`, `HMM_Signal`
- **Config-Key:** `models.hmm` (n_components, covariance_type, n_iter, random_state)
- **Besonderheit:** Erfordert nach dem Training einen Check, ob Regime 0 oder 1 dem Bear-Regime entspricht (Label-Alignment)

### B. Markov-Switching-Modelle (MS_Univariate, MS_Exo) — Ökonometrie
- **Bibliothek:** `statsmodels` (MarkovRegression)
- **Ansatz:** Regressionsmodell mit zustandsabhängigen Parametern
- **Output:** `MS_Univariate_Prob` / `MS_Univariate_Signal` und `MS_Exo_Prob` / `MS_Exo_Signal`
- **Config-Key:** `models.markov_switching` (k_regimes, switching_variance)
- **Besonderheit:** `MS_Exo` nutzt zusätzlich exogene Variablen (VIX, Yield Spread) als erklärende Variablen

### C. LSTM (Supervised) — Machine Learning
- **Bibliothek:** `TensorFlow` / `Keras`
- **Ansatz:** Supervised Learning auf Markov-Labels; lernt Regime-Wechsel aus Zeitreihen-Sequenzen (Windows)
- **Output:** `LSTM_Prob`, `LSTM_Signal`
- **Config-Key:** `models.lstm` (window_size, units, epochs, batch_size, learning_rate, dropout, activation, optimizer, loss, metrics, validation_split, verbose, labels)
- **Besonderheit:** Nutzt ein rollierendes Fenster (`window_size`) als Input-Sequenz. Labels stammen aus dem MS_Univariate-Modell (konfigurierbar via `models.lstm.labels`)

### D. LSTM Unsupervised (Deep Clustering) — Machine Learning
- **Bibliothek:** `TensorFlow` / `Keras` + `Scikit-Learn` (GMM)
- **Ansatz:** LSTM-Autoencoder komprimiert Markt-Sequenzen in latenten Raum; Gaussian Mixture Model clustert die Embeddings
- **Output:** `LSTM_Unsupervised_Prob`, `LSTM_Unsupervised_Signal`
- **Config-Key:** `models.lstm_unsupervised` (window_size, train_test_split, encoder_units, decoder_units, activation, optimizer, loss, epochs, batch_size, validation_split, n_components, gmm_n_init, gmm_random_state)
- **Besonderheit:** Rein datengetrieben, keine vordefinierten Labels; dient als objektive Kontrollinstanz

### Config-Mapping Übersicht

| Modell | Config-Key | Wichtigste Parameter |
|:---|:---|:---|
| HMM | `cfg.models.hmm` | `n_components`, `covariance_type`, `n_iter`, `random_state` |
| MS_Univariate / MS_Exo | `cfg.models.markov_switching` | `k_regimes`, `switching_variance` |
| LSTM | `cfg.models.lstm` | `window_size`, `units`, `epochs`, `batch_size`, `learning_rate`, `dropout` |
| LSTM_Unsupervised | `cfg.models.lstm_unsupervised` | `window_size`, `encoder_units`, `decoder_units`, `epochs`, `n_components` |
| **Dein Modell** | `cfg.models.my_model` | *deine Parameter* |

---

## 10. FAQ & Troubleshooting

### Mein Modell taucht nicht im Backtesting auf
**Ursache:** Die Signal-Spalte endet nicht exakt auf `_Signal` oder enthält `NaN`-Werte.
**Lösung:** Prüfe den Spaltennamen und stelle sicher, dass keine fehlenden Werte vorhanden sind:
```python
assert df['MyModel_Signal'].isna().sum() == 0, "NaN-Werte im Signal gefunden!"
assert df['MyModel_Signal'].isin([0, 1]).all(), "Signal enthält Werte außerhalb von {0, 1}!"
```

### Die Equity Curve meines Modells ist identisch mit Buy & Hold
**Ursache:** Das Modell gibt fast ausschließlich Signal `0` (Bull) aus.
**Lösung:** Prüfe die Signalverteilung und den Threshold:
```python
print(df['MyModel_Signal'].value_counts(normalize=True))
# Ggf. Threshold in config.yaml anpassen: models.my_model.threshold
```

### Bear-Regime hat höhere Returns als Bull-Regime
**Ursache:** Die Labels sind vertauscht (häufig bei Unsupervised-Modellen).
**Lösung:** Invertiere Signal und Wahrscheinlichkeit:
```python
df['MyModel_Signal'] = 1 - df['MyModel_Signal']
df['MyModel_Prob'] = 1 - df['MyModel_Prob']
```

### Config-Fehler: `AttributeError: 'SimpleNamespace' object has no attribute 'my_model'`
**Ursache:** Der Config-Eintrag in `config.yaml` fehlt oder der Key-Name stimmt nicht überein.
**Lösung:** Prüfe, dass unter `models:` ein Eintrag `my_model:` existiert (snake_case, korrekte Einrückung mit 2 Spaces). Starte ggf. den Kernel neu, damit `cfg` neu geladen wird.

### Wie ändere ich einen Hyperparameter für einen erneuten Lauf?
**Lösung:** Editiere **nur** `config/config.yaml` — z.B. `models.my_model.epochs: 100`. Starte dann die Pipeline neu. Die Änderung propagiert automatisch über `cfg` in alle Notebooks.

### Mein Modell braucht zusätzliche Features, die noch nicht existieren
**Lösung:** Erweitere `jupyter/02_feature_engineering.ipynb` um die neuen Features und ergänze sie in `config.yaml` unter `features.model_features`. Stelle sicher, dass die Features im `df` DataFrame persistiert werden (via `02_feature_engineered_data.parquet`).

### Wie viele Modelle kann die Pipeline verarbeiten?
**Antwort:** Es gibt kein technisches Limit. Der Dynamic-Matching-Algorithmus erkennt beliebig viele `_Signal`-Spalten. Beachte jedoch, dass mehr Modelle die Laufzeit der Evaluation (insb. Monte-Carlo-Simulation mit `evaluation.mcs.n_paths` Pfaden) verlängern. Nutze `fast_mode.enabled: true` in der Config für schnellere Entwicklungszyklen.

### Kann ich den Fast Mode für Entwicklung nutzen?
**Antwort:** Ja! Setze in `config.yaml`:
```yaml
fast_mode:
  enabled: true
  overrides:
    lstm_epochs: 5
    lstm_unsupervised_epochs: 10
    mcs_n_paths: 100
```
Dies reduziert Training-Epochs und MCS-Pfade automatisch. Vergiss nicht, vor dem finalen Run `fast_mode.enabled: false` zu setzen.