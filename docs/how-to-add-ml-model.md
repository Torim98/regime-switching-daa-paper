# How to Add a New ML Model to the Research Pipeline

> **Ziel:** Schritt-für-Schritt-Anleitung zur Integration eines neuen Regime-Switching-Modells in die bestehende Microservice-Pipeline. Der Guide nutzt die **Signal-Schnittstelle**, den **Dynamic-Matching-Mechanismus** und das **zentrale Konfigurationsmanagement** des Frameworks, sodass ein neues Modell automatisch in Backtesting, Evaluation und Reporting erscheint, ohne dass downstream Code angepasst werden muss.

---

## Inhaltsverzeichnis

1. [Voraussetzungen](#1-voraussetzungen)
2. [Die Signal-Schnittstelle (Interface-Spezifikation)](#2-die-signal-schnittstelle-interface-spezifikation)
3. [Hyperparameter in der zentralen Config registrieren](#3-hyperparameter-in-der-zentralen-config-registrieren)
4. [Schritt-für-Schritt-Anleitung](#4-schritt-für-schritt-anleitung)
   - [Schritt 1: Config-Eintrag erstellen](#schritt-1-config-eintrag-erstellen)
   - [Schritt 2: Trainingslogik in src/models/ implementieren](#schritt-2-trainingslogik-in-srcmodels-implementieren)
   - [Schritt 3: Plot-Funktion in src/models/plots.py](#schritt-3-plot-funktion)
   - [Schritt 4: Route im Model Service registrieren](#schritt-4-route-im-model-service-registrieren)
   - [Schritt 5: In /train-all aufnehmen](#schritt-5-in-train-all-aufnehmen)
   - [Schritt 6: Modell-Persistierung (optional)](#schritt-6-modell-persistierung-optional)
   - [Schritt 7: Docker-Rebuild & Test](#schritt-7-docker-rebuild--test)
5. [Warum das funktioniert: Dynamic Matching](#5-warum-das-funktioniert-dynamic-matching)
6. [Look-Ahead Bias Prevention (T+1 Shift)](#6-look-ahead-bias-prevention-t1-shift)
7. [Dokumentation aktualisieren](#7-dokumentation-aktualisieren)
8. [Code-Template (Copy & Paste)](#8-code-template-copy--paste)
9. [Validierungs-Checkliste](#9-validierungs-checkliste)
10. [Referenz-Implementierungen](#10-referenz-implementierungen)
11. [FAQ & Troubleshooting](#11-faq--troubleshooting)

---

## 1. Voraussetzungen

Bevor du ein neues Modell integrierst, stelle sicher, dass:

- [ ] Der Data Service erfolgreich gelaufen ist (`POST /data/ingest`) und `data/silver/03_feature_engineered_data.parquet` existiert und aktuell ist
- [ ] Die benötigten Python-Pakete für dein Modell in `pyproject.toml` ergänzt sind (`ml`-Extra für schwere Frameworks)
- [ ] Du den Aufbau von `services/model_service/routes.py` und der Module unter `src/models/` grundlegend verstanden hast

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
- Verwende **PascalCase** oder **Snake_Case** mit Großbuchstaben (z.B. , `MSM`, `HMM`, `Transformer`, `LSTM`)
- **Keine Leerzeichen**, verwende stattdessen Unterstriche
- Der Name muss **eindeutig** sein und darf nicht mit dem Namen eines bestehenden Modells kollidieren
- Der Suffix `_Signal` ist **reserviert** und wird vom Dynamic-Matching-Algorithmus als Erkennungsmerkmal verwendet

### Bestehende Modelle als Referenz

| Modellname        | `_Prob`-Spalte           | `_Signal`-Spalte           | Paradigma                  |
| :---------------- | :----------------------- | :------------------------- | :------------------------- |
| MSM               | `MSM_Prob`               | `MSM_Signal`               | Ökonometrie (Regression)   |
| HMM               | `HMM_Prob`               | `HMM_Signal`               | Ökonometrie (Unsupervised) |
| LSTM              | `LSTM_Prob`              | `LSTM_Signal`              | ML (Supervised)            |
| Transformer       | `Transformer_Prob`       | `Transformer_Signal`       | ML (Attention-basiert)     |

---

## 3. Hyperparameter in der zentralen Config registrieren

> **Wichtig:** Seit [Issue #3](https://github.com/Torim98/regime-switching-daa/issues/3) werden **alle Pipeline-Parameter zentral** in [`config/config.yaml`](../config/config.yaml) verwaltet. Hardcoded Hyperparameter im Code sind nicht mehr erlaubt.

> **Hyperparameter-Optimierung:** Optuna kann `config.yaml`-Parameter programmatisch
> überschreiben ([Issue #2](https://github.com/Torim98/regime-switching-daa/issues/2)).
> Um ein neues Modell in die Optimierung aufzunehmen, muss in `src/backtest/optimize.py`
> eine `objective_<modell>`-Funktion mit den passenden `trial.suggest_*()`-Aufrufen
> definiert und in `_OBJECTIVE_MAP` registriert werden.

### Architektur-Übersicht

```
config/
├── config.yaml          # Single Source of Truth — alle Parameter
└── config_loader.py     # PipelineConfig-Klasse + Singleton `cfg`
                         # Methoden: data_path(), asset_path(), model_path()
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

### So lädst du die Config in Code

```python
# In src/ oder services/ — Config laden
from config.config_loader import cfg

# Zugriff per Dot-Notation:
cfg.models.hmm.n_components          # → 2
cfg.features.model_features          # → ['Returns', 'Vol_20', ...]
cfg.asset_path("equity_curves")      # → ".../assets/equity_curves.png"
cfg.model_path("lstm")               # → ".../models/lstm_regime_model.keras"
cfg.model_persistence.enabled        # → true/false
```

### So fügst du dein Modell zur Config hinzu

**Schritt A:** Öffne `config/config.yaml` und ergänze unter der `models:`-Sektion einen neuen Block:

```yaml
models:
  # ... bestehende Modelle (msm, hmm, lstm, transformer) ...

  my_model:                    # ← Schlüsselname in snake_case
    window_size: 20            # Beispiel-Hyperparameter
    n_heads: 4                 # Beispiel für Transformer
    n_layers: 2
    d_model: 64
    dropout: 0.1
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    threshold: *threshold       # Deckt sich mit den anderen Modellen; Bear-Signal wenn Prob >= threshold
```

### Farbe für Plots registrieren (optional)

Damit dein Modell in allen Plots eine konsistente Farbe erhält, ergänze unter `plotting.colors` einen Eintrag:

```yaml
plotting:
  colors:
    # ... bestehende Modelle ...
    MyModel: "tab:cyan"        # ← Deine Modellfarbe
```

> **Hinweis:** Dieser Schritt ist optional. Modelle ohne Eintrag erhalten 
> automatisch eine Farbe aus dem matplotlib Default-Cycle. Die Farbe wird 
> über `cfg.color_map` in allen Services konsistent verwendet.

Verfügbare Farbnamen: Alle [matplotlib Named Colors](https://matplotlib.org/stable/gallery/color/named_colors.html) 
und `tab:`-Palette (z.B. `tab:blue`, `tab:red`, `tab:cyan`).

**Schritt B:** Greife im Code auf die Parameter zu:

```python
# Statt hardcoded: window_size = 20
my_cfg = cfg.models.my_model
window_size = my_cfg.window_size      # → 20
n_heads     = my_cfg.n_heads          # → 4
epochs      = my_cfg.epochs           # → 50
threshold   = my_cfg.threshold        # → 0.5
```

### Vorteile der zentralen Konfiguration

| Aspekt                         | Vorher (Hardcoded)                | Nachher (Config)                                                                                                              |
| :----------------------------- | :-------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| **Änderung eines Parameters**  | Code suchen, manuell ändern       | `config.yaml` editieren, propagiert automatisch                                                                               |
| **Reproduzierbarkeit**         | Parameter über Module verstreut   | Alles an einer Stelle, versioniert in Git                                                                                     |
| **Hyperparameter-Optimierung** | Manuell im Code anpassen          | Optuna kann `config.yaml` programmatisch überschreiben ([Issue #2](https://github.com/Torim98/regime-switching-daa/issues/2)) |
| **Fast Mode (Entwicklung)**    | Jedes Modell einzeln anpassen     | `fast_mode.enabled: true` reduziert Epochs/MCS-Paths automatisch                                                              |

### Fast Mode für neue Modelle (optional)

Falls du einen Fast-Mode-Override für dein Modell hinzufügen möchtest:

```yaml
fast_mode:
  enabled: false
  overrides:
    lstm_epochs: 5
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

Alle Änderungen erfolgen in **vier Stellen**: `config/config.yaml` (Hyperparameter), `src/models/<name>.py` (Trainingslogik), `src/models/plots.py` (Regime-Plot) und `services/model_service/routes.py` (Service-Route).

> **Prinzip — Shared Business Logic:** Die Fachlogik lebt in `src/`, die Service-Route ruft sie nur auf. Dupliziere niemals Trainingslogik in der Route selbst.

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

### Schritt 2: Trainingslogik in src/models/ implementieren

Lege ein neues Modul `src/models/my_model.py` an. Die Funktion lädt alle Hyperparameter aus der Config (**kein Hardcoding!**), trainiert das Modell und schreibt die beiden Signal-Spalten in den DataFrame:

```python
# src/models/my_model.py
from my_model_library import MyModel

def train_my_model(df, feature_cols, cfg):
    """Trainiert MyModel und schreibt MyModel_Prob / MyModel_Signal in df."""
    my_cfg = cfg.models.my_model

    # Feature-Matrix vorbereiten
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

    # Bear-Wahrscheinlichkeit (hohe Werte = Krise) + binäres Signal ableiten
    df["MyModel_Prob"] = model.predict_proba(X)
    df["MyModel_Signal"] = (df["MyModel_Prob"] >= my_cfg.threshold).astype(int)
    return df, model
```

> **Wichtig:** Stelle sicher, dass `_Prob` die **Bear-Wahrscheinlichkeit** enthält (hohe Werte = Krise). Falls dein Modell die Bull-Wahrscheinlichkeit ausgibt, invertiere sie: `bear_prob = 1 - bull_prob`.

> **Features:** `feature_cols` stammt aus `cfg.features.model_features` (`['Returns', 'Vol_20', 'Distance_SMA', 'Momentum', 'VIX', 'Yield_Spread']`). Du kannst eine Teilmenge nutzen oder im Data Service zusätzliche Features berechnen und die Liste in `config.yaml` erweitern.

### Schritt 3: Plot-Funktion in src/models/plots.py

Ergänze eine `matplotlib.use("Agg")`-kompatible Plot-Funktion (kein `plt.show()`, stattdessen `plt.savefig()` + `plt.close(fig)`):

```python
# In src/models/plots.py ergänzen:

def plot_my_model(df, model_name, color, save_path):
    """Regime-Plot für MyModel."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    # ... Plotting-Logik ...
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

### Schritt 4: Route im Model Service registrieren

Öffne `services/model_service/routes.py`, importiere deine Funktionen und ergänze im `train_model()`-Endpoint (`POST /train/{model_name}`) einen `elif`-Block — analog zu den bestehenden Modellen (`if model_name == "msm": ... elif ...`):

```python
from src.models.my_model import train_my_model
from src.models.plots import plot_my_model
from src.models.common import validate_regime_signal

@router.post("/train/{model_name}")
def train_model(model_name: str):
    cfg = get_cfg()
    df = pd.read_parquet(cfg.data_path("feature_engineered"))
    feature_cols = cfg.features.model_features
    # ... if model_name == "msm": ... elif "hmm" / "lstm" / "transformer" ...

    elif model_name == "my_model":
        df, model = train_my_model(df, feature_cols, cfg)
        validate_regime_signal(df, "MyModel")
        df.to_parquet(cfg.data_path("feature_engineered"))
        plot_my_model(
            df, "MyModel",
            cfg.color_map.get("MyModel", "tab:cyan"),
            cfg.asset_path("my_model"),
        )
```

> **`validate_regime_signal`** (aus `src/models/common.py`) gibt Regime-Statistiken aus (Returns, VIX, Yield_Spread pro Signal), prüft Plausibilität (Bear-Regime `1` muss niedrigere Returns haben als Bull `0`) mit automatischer Label-Inversion (`auto_invert=True`) und führt formale Assertions durch (Spaltenexistenz, Wertebereich `[0,1]`, keine NaN).

> **Persistierung der Signale:** Die bestehenden Modelle schreiben ihre Signal-Spalten zurück in `feature_engineered` (`df.to_parquet(cfg.data_path("feature_engineered"))`). Der Backtest Service liest dieselbe Datei und erkennt deine `_Signal`-Spalte per Dynamic Matching automatisch — du musst nichts downstream anpassen.

### Schritt 5: In /train-all aufnehmen

Damit dein Modell bei `POST /models/train-all` mitläuft, nimm es in die Trainingsreihenfolge auf:

```python
@router.post("/train-all")
def train_all():
    results = []
    for model in ["msm", "hmm", "lstm", "transformer", "my_model"]:
        results.append(train_model(model))
    return results
```

> **Beachte die Reihenfolge:** Falls dein Modell auf vordefinierten Labels angewiesen ist (wie LSTM und Transformer auf Pagan-Sossounov), müssen diese Labels bereits im `feature_engineered_data` verfügbar sein (wird vom Data Service erzeugt).

### Schritt 6: Modell-Persistierung (optional)

Trainierte Modelle können im Ordner `models/` zwischengespeichert werden. Dies ist besonders nützlich, wenn das Training rechenintensiv ist (z.B. LSTM, Transformer).

#### Voraussetzung

In `config/config.yaml` existiert die Sektion `model_persistence`:

```yaml
model_persistence:
  enabled: true
  models_dir: "../models"
  files:
    msm: "msm_regime_model.pkl"
    hmm: "hmm_regime_model.pkl"
    scaler_hmm: "hmm_scaler.pkl"
    lstm: "lstm_regime_model.keras"
    scaler_lstm: "lstm_scaler.pkl"
    transformer: "transformer_regime_model.pt"
```

#### So fügst du dein Modell hinzu

**1.** Ergänze unter `model_persistence.files` einen Eintrag für dein Modell:

```yaml
model_persistence:
  files:
    # ... bestehende Einträge ...
    my_model: "my_model.pkl"           # ← Dateiname deines Modells
    scaler_my_model: "my_model_scaler.pkl"  # ← Falls ein Scaler nötig ist
```

**2.** Nutze in `src/models/my_model.py` `cfg.model_path("my_model")`, um den vollständigen Pfad zu erhalten.

**3.** Umschließe Training und Laden in deiner Trainingsfunktion mit einem `if/else`-Block:

```python
import os
from pathlib import Path

persist = cfg.model_persistence
model_file = cfg.model_path("my_model")

if persist.enabled and os.path.exists(model_file):
    # MODUS A: Gespeichertes Modell laden
    model = load_my_model(model_file)
else:
    # MODUS B: Normal trainieren + speichern
    model = train_my_model_impl(...)

    Path(persist.models_dir).mkdir(parents=True, exist_ok=True)
    save_my_model(model, model_file)
```

**4.** Falls dein Modell einen Scaler benötigt (z.B. `StandardScaler`, `MinMaxScaler`), persistiere diesen ebenfalls. Beim Laden **unbedingt** `transform()` statt `fit_transform()` verwenden!

#### Serialisierungs-Formate nach Bibliothek

| Bibliothek | Speichern | Laden | Dateiendung |
|:---|:---|:---|:---|
| `statsmodels` | `results.save(path)` | `sm.load(path)` | `.pkl` |
| `hmmlearn` / `sklearn` | `joblib.dump(model, path)` | `joblib.load(path)` | `.pkl` |
| `TensorFlow/Keras` | `model.save(path)` | `load_model(path)` | `.keras` |
| `PyTorch` | `torch.save(model.state_dict(), path)` | `model.load_state_dict(torch.load(path))` | `.pt` |

> **Hinweis:** Der Ordner `models/` ist in `.gitignore` eingetragen und wird beim ersten Speichern automatisch via `Path(...).mkdir(parents=True, exist_ok=True)` angelegt.

### Schritt 7: Docker-Rebuild & Test

```bash
docker-compose build model-service     # src/ wird via COPY src/ src/ ins Image übernommen
docker-compose up -d model-service

# Einzelnes Modell trainieren:
curl -X POST http://localhost:8002/models/train/my_model

# Oder alle Modelle:
curl -X POST http://localhost:8002/models/train-all
```

> Da die Trainingslogik in `src/` liegt und `src/` bereits im Dockerfile kopiert wird (`COPY src/ src/`), sind **keine Änderungen am Dockerfile nötig**.

---

## 5. Warum das funktioniert: Dynamic Matching

Die nachgelagerten Schritte (Backtest Service, Evaluation, Reporting) nutzen einen **dynamischen Such-Algorithmus**, der alle Modell-Signale automatisch erkennt:

```python
# Aus src/backtest/engine.py (run_all_backtests) — Kernlogik des Dynamic Matching:
signal_cols = [col for col in test_df.columns if col.endswith("_Signal")]

for sig_col in signal_cols:
    model_name = sig_col.rsplit("_", 1)[0]  # Extrahiert "HMM" aus "HMM_Signal"
    backtesting_results[model_name] = backtest(
        test_df, sig_col, signal_shift=cfg.backtesting.signal_shift,
        fee=cfg.transaction_cost_rate,
    )
```

**Das bedeutet:**
1. **Backtesting** erkennt jede Spalte, die auf `_Signal` endet, und führt einen vollständigen Backtest durch
2. **Evaluation** berechnet für jedes erkannte Modell automatisch alle Kennzahlen (Sharpe, Sortino, Calmar, Max Drawdown etc.)
3. **Reporting** generiert Equity Curves, Statistik-Tabellen und SORR-Simulationen für alle Modelle

**Du musst keinen Code im Backtest Service, in der Evaluation oder im Reporting (`src/backtest/`) anpassen.** Es genügt, die Signal-Schnittstelle korrekt zu implementieren und die Hyperparameter in der Config zu registrieren.

---

## 6. Look-Ahead Bias Prevention (T+1 Shift)

Ein kritisches Konzept, das das Backtesting-Modul automatisch handhabt:

```python
# Aus src/backtest/engine.py — Automatischer T+1 Shift:
trading_signal = df[signal_col].shift(signal_shift).fillna(0)
# signal_shift ist in config.yaml definiert (Standard: 1)
```

**Was passiert hier?**
- Das Backtesting verschiebt jedes Signal um **einen Handelstag** in die Zukunft (`shift(1)`)
- Ein Signal, das am Tag `T` berechnet wurde, wird erst am Tag `T+1` als Handelsentscheidung angewandt
- Dies verhindert **Look-Ahead Bias**: Entscheidungen basieren ausschließlich auf historisch verfügbaren Informationen
- Der Shift-Wert ist über `backtesting.signal_shift` in der `config.yaml` konfigurierbar (Standard: `1`)

**Das bedeutet für dich:**
- Du musst den Shift **NICHT** selbst implementieren, das Backtesting erledigt das
- Du darfst den Shift **NICHT** doppelt anwenden (also nicht schon in `src/models/<name>.py`)
- Die Spalten `_Prob` und `_Signal` enthalten die Werte **zum Zeitpunkt der Berechnung** (Tag `T`)

---

## 7. Dokumentation aktualisieren

Nach der erfolgreichen Integration des Modells in die Pipeline (Schritte 1–7) müssen **drei Dokumentationsebenen** aktualisiert werden, damit das neue Modell korrekt in der Projektdokumentation erscheint.

> **Hinweis:** Die quantitativen Tabellen (Performance Summary, Evaluation-Matrix, SORR-Summary, MCS-Summary) und die meisten Plots (Equity Curves, Regime Comparison, MCS-Boxplots) werden dank Dynamic Matching **automatisch** generiert. Die folgenden Schritte betreffen ausschließlich die **manuellen** Dokumentationsanpassungen.

### Schritt A: Asset-Pfad in `config.yaml` registrieren

Damit das Reporting den Modell-Plot referenzieren kann, muss unter `paths.assets` ein Eintrag hinzugefügt werden:

```yaml
paths:
  assets:
    # ... bestehende Einträge ...
    my_model: "my_model.png"           # ← Dateiname des Regime-Plots
```

> Der Plot selbst wird in `src/models/plots.py` erzeugt und unter `assets/` gespeichert. Die Config definiert lediglich den Dateinamen, über den `src/backtest/reporting.py` den Plot einbettet.

### Schritt B: Modell-Abschnitt im Reporting ergänzen (`statistics.md`)

Die Datei `docs/statistics.md` wird **vollständig automatisch** durch `src/backtest/reporting.py` generiert. Die Sektion **"Regime-Erkennung der Einzelmodelle"** enthält jedoch für jedes Modell einen manuell gepflegten Absatz mit Beschreibung und Bildverweis im f-String-Template.

Öffne `src/backtest/reporting.py` und füge innerhalb des f-Strings (`stats_md_content`) an der passenden Stelle einen neuen Unterabschnitt ein:

```python
### G. <Modellname> (<Paradigma>)
<Kurzbeschreibung des Modells: Ansatz, Besonderheit, Trainings-Setting.>
![<Modellname> Model](../assets/{cfg.paths.assets.my_model})
```

**Orientierung:** Die bestehenden Abschnitte folgen der Reihenfolge Ökonometrie → ML (Supervised) → Attention-basiert. Ordne dein Modell entsprechend ein.

> **Wichtig:** Bearbeite **niemals** `docs/statistics.md` direkt. Die Datei wird beim nächsten `POST /backtest/evaluate` überschrieben. Alle Änderungen müssen in `src/backtest/reporting.py` im f-String-Template erfolgen.

### Schritt C: README.md aktualisieren

Die `README.md` enthält im Abschnitt **"Methodik & Modelle"** eine Beschreibung aller Modellkategorien. Ergänze dort einen Eintrag für dein neues Modell:

- **Unter Punkt 1 (Ökonometrische Modelle)** — wenn es sich um ein statistisches Verfahren handelt
- **Unter Punkt 2 (Machine-Learning-Verfahren)** — wenn es sich um ein ML-/DL-Modell handelt

Verwende den Stil und Detailgrad der bestehenden Modellbeschreibungen als Vorlage. Ein Eintrag sollte mindestens enthalten:
1. **Modellname** und Architektur-Typ (fett)
2. Kurzbeschreibung des Ansatzes (1–2 Sätze)
3. Trainings-Setting (Supervised / Unsupervised) und ggf. Label-Quelle

### Schritt D (optional): Architektur-Dokumentation in `docs/`

Für komplexe Modelle (insb. neuronale Netze) empfiehlt es sich, eine dedizierte Architektur-Beschreibung unter `docs/` abzulegen.

Eine solche Datei sollte enthalten:
- Schematische Darstellung der Schichten / Module (als Text-Diagramm oder Bild)
- Input-/Output-Dimensionen
- Verweis auf den zugehörigen Config-Key (`cfg.models.<name>`)

### Schritt E (optional): Plot-Farbe in `config.yaml` registrieren

Ergänze unter `plotting.colors` eine Farbe für dein Modell, damit es in 
allen Vergleichsplots (Equity Curves, Regime Comparison, MCS) konsistent 
dargestellt wird.

---

## 8. Code-Template (Copy & Paste)

### A. Config-Eintrag (`config/config.yaml`)

```yaml
models:
  # ... bestehende Modelle ...
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

plotting:
  colors:
    # ... bestehende Modelle ...
    MyModel: "tab:cyan"           # ← Plot-Farbe (optional)
```

### B. Service-Modul (`src/models/my_model.py`)

Lege das Trainingsmodul an (Signatur analog zu den bestehenden Modellen):

```python
from my_model_library import MyModel

def train_my_model(df, feature_cols, cfg):
    """Trainiert MyModel und schreibt MyModel_Prob / MyModel_Signal in df."""
    my_cfg = cfg.models.my_model
    X = df[feature_cols].dropna()
    model = MyModel(
        window_size=my_cfg.window_size, epochs=my_cfg.epochs,
        batch_size=my_cfg.batch_size, learning_rate=my_cfg.learning_rate,
        dropout=my_cfg.dropout,
    )
    model.fit(X)
    df["MyModel_Prob"] = model.predict_proba(X)
    df["MyModel_Signal"] = (df["MyModel_Prob"] >= my_cfg.threshold).astype(int)
    return df, model
```

Die Registrierung im Model Service (`elif`-Block + `validate_regime_signal` + Plot) erfolgt wie in [Abschnitt 4, Schritt 4](#schritt-4-route-im-model-service-registrieren).

---

## 9. Validierungs-Checkliste

Führe nach der Integration folgende Prüfungen durch:

### Formale Prüfungen
- [ ] Die Spalte `<Modell>_Prob` existiert im DataFrame und enthält `float`-Werte zwischen `0.0` und `1.0`
- [ ] Die Spalte `<Modell>_Signal` existiert im DataFrame und enthält ausschließlich `0` oder `1`
- [ ] Keine `NaN`-Werte in `<Modell>_Signal` (im Testzeitraum)
- [ ] Der Modellname kollidiert nicht mit bestehenden Namen (`MSM`, `HMM`, `LSTM`, `Transformer`)

### Config-Prüfungen
- [ ] Neuer Eintrag unter `models:` in `config/config.yaml` angelegt
- [ ] **Keine** Hyperparameter hardcoded im Code, alles kommt aus `cfg.models.<name>`
- [ ] Config-Key stimmt mit dem Zugriff im Code überein (z.B. `cfg.models.my_model`)
- [ ] Falls Fast-Mode-Override gewünscht: Eintrag in `fast_mode.overrides` und `config_loader.py` ergänzt

### Persistierungs-Prüfungen (optional)
- [ ] Eintrag unter `model_persistence.files` in `config.yaml` angelegt (falls Persistierung gewünscht)
- [ ] `if/else`-Block für Load/Train in `src/models/<name>.py` implementiert
- [ ] Scaler wird bei Laden mit `transform()` statt `fit_transform()` verwendet
- [ ] `Path(persist.models_dir).mkdir(parents=True, exist_ok=True)` vor dem ersten Speichern aufgerufen

### Inhaltliche Prüfungen
- [ ] Bear-Regime (Signal = 1) weist im Durchschnitt **niedrigere Returns** auf als Bull-Regime (Signal = 0)
- [ ] Bear-Regime weist im Durchschnitt **höheren VIX** auf als Bull-Regime
- [ ] Die Signalverteilung ist plausibel (nicht 99% ein Regime)

### Dokumentations-Prüfungen
- [ ] Asset-Pfad für den Modell-Plot in `config.yaml` unter `paths.assets` registriert
- [ ] (Optional) Plot-Farbe unter `plotting.colors` in `config.yaml` registriert
- [ ] Neuer Abschnitt im Reporting (`src/backtest/reporting.py`, f-String-Template) eingefügt
- [ ] `README.md` — Modell in "Methodik & Modelle" beschrieben
- [ ] (Optional) Architektur-Dokumentation unter `docs/` angelegt
- [ ] `docs/statistics.md` enthält nach Pipeline-Durchlauf den neuen Modell-Abschnitt mit korrektem Bild

### Implementierungs-Prüfungen (Model Service)
- [ ] Trainingslogik in `src/models/` als wiederverwendbares Modul implementiert (nicht inline in der Service-Route)
- [ ] Plot-Funktion in `src/models/plots.py` mit `plt.close(fig)` (kein `plt.show()`)
- [ ] `elif model_name == "my_model":` Block in `services/model_service/routes.py` ergänzt
- [ ] Modell in der `train_all()`-Funktion des Model Service registriert
- [ ] Docker-Image neu gebaut (`docker-compose build model-service`)
- [ ] Endpunkt `POST /models/train/my_model` liefert HTTP 200

### Pipeline-Integration
- [ ] `POST /models/train/my_model` (bzw. `/train-all`) aktualisiert `data/silver/03_feature_engineered_data.parquet` mit `MyModel_Prob`/`MyModel_Signal`
- [ ] `POST /backtest/run` erkennt das neue Modell automatisch (Dynamic Matching) und berechnet Equity Curves
- [ ] `POST /backtest/evaluate` berechnet Kennzahlen (Sharpe, Sortino, Calmar, Max Drawdown) und generiert `statistics.md` mit dem neuen Modell
- [ ] Die Grafiken in `assets/` (Equity Curves, Regime Comparison, MCS Boxplots etc.) enthalten das neue Modell

### End-to-End Test (empfohlen)
```bash
# Vollständigen Pipeline-Durchlauf via Microservices:
docker-compose up --build -d
curl -X POST http://localhost:8001/data/ingest
curl -X POST http://localhost:8002/models/train-all
curl -X POST http://localhost:8003/backtest/run
curl -X POST http://localhost:8003/backtest/evaluate
```

---

## 10. Referenz-Implementierungen

Die folgenden bestehenden Modelle in `src/models/` (`msm.py`, `hmm.py`, `lstm.py`, `transformer.py`) dienen als Referenz. Alle laden ihre Hyperparameter aus der zentralen `config.yaml`:

### A. Markov-Switching (MSM) — Ökonometrie
- **Bibliothek:** `statsmodels` (MarkovRegression)
- **Ansatz:** Univariates Regressionsmodell mit zustandsabhängigen Parametern (switching variance)
- **Output:** `MSM_Prob`, `MSM_Signal`
- **Config-Key:** `models.msm` (k_regimes, switching_variance)

### B. HMM (Hidden Markov Model) — Unsupervised, Ökonometrie
- **Bibliothek:** `hmmlearn`
- **Ansatz:** Identifiziert Cluster in den Datenverteilungen ohne gelabelte Daten
- **Output:** `HMM_Prob`, `HMM_Signal`
- **Config-Key:** `models.hmm` (n_components, covariance_type, n_iter, random_state)
- **Besonderheit:** Erfordert nach dem Training einen Check, ob Regime 0 oder 1 dem Bear-Regime entspricht (Label-Alignment)

### C. LSTM (Supervised) — Machine Learning
- **Bibliothek:** `TensorFlow` / `Keras`
- **Ansatz:** Supervised Learning auf Pagan-Sossounov-Labels; lernt Regime-Wechsel aus Zeitreihen-Sequenzen (Windows)
- **Output:** `LSTM_Prob`, `LSTM_Signal`
- **Config-Key:** `models.lstm` (window_size, units_l1, units_l2, epochs, batch_size, learning_rate, dropout, activation, optimizer, loss, metrics, validation_split, verbose)
- **Besonderheit:** Nutzt ein rollierendes Fenster (`window_size`) als Input-Sequenz. Labels stammen aus dem Pagan-Sossounov-Algorithmus (konfigurierbar via `labels.supervised_label_source`; Vergleich der Label-Quellen via `POST /data/label-analysis`)

### D. Transformer (Supervised, Attention-basiert) — Machine Learning
- **Bibliothek:** `PyTorch` (`torch.nn.TransformerEncoder`)
- **Ansatz:** Transformer-Encoder mit Positional Encoding und Multi-Head Self-Attention für zeitreihenbasierte Regime-Klassifikation; Supervised auf Pagan-Sossounov-Labels
- **Output:** `Transformer_Prob`, `Transformer_Signal`
- **Config-Key:** `models.transformer` (window_size, d_model, n_heads, n_layers, dim_feedforward, dropout, epochs, batch_size, learning_rate, threshold, pos_weight_auto)
- **Besonderheit:** Nutzt BCEWithLogitsLoss mit automatischer Class-Balance-Gewichtung (sqrt pos_weight). Testet Hypothese H2 (Attention-Mechanismus vs. ökonometrische MSM). Dient als **Referenz-Implementierung** für die guide-konforme Signal-Schnittstelle (vollständiger Sanity Check, Assertions, Config-only Hyperparameter).

### Config-Mapping Übersicht

| Modell | Config-Key | Wichtigste Parameter |
|:---|:---|:---|
| MSM | `cfg.models.msm` | `k_regimes`, `switching_variance` |
| HMM | `cfg.models.hmm` | `n_components`, `covariance_type`, `n_iter`, `random_state` |
| LSTM | `cfg.models.lstm` | `window_size`, `units_l1`, `units_l2`, `epochs`, `batch_size`, `learning_rate`, `dropout` |
| Transformer | `cfg.models.transformer` | `window_size`, `d_model`, `n_heads`, `n_layers`, `epochs`, `threshold` |
| **Dein Modell** | `cfg.models.my_model` | *deine Parameter* |

---

## 11. FAQ & Troubleshooting

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
**Lösung:** `validate_regime_signal()` erkennt und korrigiert dies automatisch 
(`auto_invert=True`). Falls du die automatische Inversion deaktivieren möchtest:
```python
validate_regime_signal(df, MODEL_NAME, auto_invert=False)
```

### Config-Fehler: `AttributeError: 'SimpleNamespace' object has no attribute 'my_model'`
**Ursache:** Der Config-Eintrag in `config.yaml` fehlt oder der Key-Name stimmt nicht überein.
**Lösung:** Prüfe, dass unter `models:` ein Eintrag `my_model:` existiert (snake_case, korrekte Einrückung mit 2 Spaces). Starte ggf. den Service neu (`docker-compose restart model-service`), damit `cfg` neu geladen wird.

### Wie ändere ich einen Hyperparameter für einen erneuten Lauf?
**Lösung:** Editiere **nur** `config/config.yaml` — z.B. `models.my_model.epochs: 100`. Starte dann die Pipeline neu. Die Änderung propagiert automatisch über `cfg` in alle Services.

### Mein Modell braucht zusätzliche Features, die noch nicht existieren
**Lösung:** Erweitere das Feature-Engineering im Data Service (`src/data/feature_engineering.py`) um die neuen Features und ergänze sie in `config.yaml` unter `features.model_features`. Stelle sicher, dass die Features im `feature_engineered_data` persistiert werden.

### Wie viele Modelle kann die Pipeline verarbeiten?
**Antwort:** Es gibt kein technisches Limit. Der Dynamic-Matching-Algorithmus erkennt beliebig viele `_Signal`-Spalten. Beachte jedoch, dass mehr Modelle die Laufzeit der Evaluation (insb. Monte-Carlo-Simulation mit `evaluation.mcs.n_paths` Pfaden) verlängern. Nutze `fast_mode.enabled: true` in der Config für schnellere Entwicklungszyklen.

### Kann ich den Fast Mode für Entwicklung nutzen?
**Antwort:** Ja! Setze in `config.yaml`:
```yaml
fast_mode:
  enabled: true
  overrides:
    lstm_epochs: 5
    mcs_n_paths: 100
```
Dies reduziert Training-Epochs und MCS-Pfade automatisch. Vergiss nicht, vor dem finalen Run `fast_mode.enabled: false` zu setzen.

### Ich habe Hyperparameter geändert, aber die Ergebnisse sind identisch
**Ursache:** `model_persistence.enabled` ist `true` und ein altes Modell liegt noch unter `models/`.
**Lösung:** Lösche die betroffene Modelldatei aus `models/` (oder den gesamten Ordner), damit das Modell mit den neuen Parametern neu trainiert wird. Alternativ setze `model_persistence.enabled: false` in der `config.yaml`.

### Wie kann ich ein einzelnes Modell neu trainieren, ohne alle zu löschen?
**Lösung:** Lösche nur die spezifische Datei (z.B. `models/lstm_regime_model.keras`). Beim nächsten Pipeline-Run wird nur dieses Modell neu trainiert, alle anderen werden weiterhin aus dem Cache geladen.
