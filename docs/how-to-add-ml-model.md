# How to Add a New ML Model to the Research Pipeline

> **Goal:** Step-by-step guide for integrating a new regime-switching model into the existing microservice pipeline. The guide uses the framework's **signal interface**, the **dynamic matching mechanism**, and the **central configuration management**, so that a new model automatically appears in backtesting, evaluation, and reporting without requiring changes to downstream code.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [The Signal Interface (Interface Specification)](#2-the-signal-interface-interface-specification)
3. [Registering Hyperparameters in the Central Config](#3-registering-hyperparameters-in-the-central-config)
4. [Step-by-Step Guide](#4-step-by-step-guide)
   - [Step 1: Create a config entry](#step-1-create-a-config-entry)
   - [Step 2: Implement the training logic in src/models/](#step-2-implement-the-training-logic-in-srcmodels)
   - [Step 3: Plot function in src/models/plots.py](#step-3-plot-function)
   - [Step 4: Register the route in the Model Service](#step-4-register-the-route-in-the-model-service)
   - [Step 5: Include in /train-all](#step-5-include-in-train-all)
   - [Step 6: Model persistence (optional)](#step-6-model-persistence-optional)
   - [Step 7: Docker rebuild & test](#step-7-docker-rebuild--test)
5. [Why This Works: Dynamic Matching](#5-why-this-works-dynamic-matching)
6. [Look-Ahead Bias Prevention (T+1 Shift)](#6-look-ahead-bias-prevention-t1-shift)
7. [Updating the Documentation](#7-updating-the-documentation)
8. [Code Template (Copy & Paste)](#8-code-template-copy--paste)
9. [Validation Checklist](#9-validation-checklist)
10. [Reference Implementations](#10-reference-implementations)
11. [FAQ & Troubleshooting](#11-faq--troubleshooting)

---

## 1. Prerequisites

Before integrating a new model, make sure that:

- [ ] The Data Service has run successfully (`POST /data/ingest`) and `data/silver/03_feature_engineered_data.parquet` exists and is up to date
- [ ] The Python packages required for your model have been added to `pyproject.toml` (`ml` extra for heavy frameworks)
- [ ] You have a basic understanding of the structure of `services/model_service/routes.py` and the modules under `src/models/`

---

## 2. The Signal Interface (Interface Specification)

The central design principle of the pipeline is the **standardized signal interface**. Every model must produce exactly **two columns** in the shared `test_df` DataFrame:

### Column 1: `<Model>_Prob` (Regime Probability)

| Property | Specification |
|:---|:---|
| **Naming convention** | `<ModelName>_Prob` |
| **Data type** | `float64` |
| **Value range** | `0.0` to `1.0` |
| **Semantics** | Probability that the current day is a **bear regime** (crisis) |
| **Example** | `HMM_Prob`, `LSTM_Prob`, `Transformer_Prob` |

### Column 2: `<Model>_Signal` (Binary Trading Signal)

| Property | Specification |
|:---|:---|
| **Naming convention** | `<ModelName>_Signal` |
| **Data type** | `int` (0 or 1) |
| **Value range** | `0` = **bull** (invested), `1` = **bear** (cash/money market) |
| **Semantics** | Binary decision, derived from `_Prob` via threshold (usually `>= 0.5`) |
| **Example** | `HMM_Signal`, `LSTM_Signal`, `Transformer_Signal` |

### Naming Convention

```
<ModelName>_Prob     →  e.g. MyModel_Prob
<ModelName>_Signal   →  e.g. MyModel_Signal
```

**Rules for `<ModelName>`:**
- Use **PascalCase** or **Snake_Case** with capital letters (e.g. `MSM`, `HMM`, `Transformer`, `LSTM`)
- **No spaces**; use underscores instead
- The name must be **unique** and must not collide with the name of an existing model
- The suffix `_Signal` is **reserved** and used by the dynamic matching algorithm as the detection marker

### Existing Models as Reference

| Model name        | `_Prob` column           | `_Signal` column           | Paradigm                  			|
| :---------------- | :----------------------- | :------------------------- | :------------------------- 			|
| MSM               | `MSM_Prob`               | `MSM_Signal`               | Econometrics (regression)   			|
| HMM               | `HMM_Prob`               | `HMM_Signal`               | Econometrics (unsupervised) 			|
| HMM_Uni           | `HMM_Uni_Prob`           | `HMM_Uni_Signal`           | Econometrics (unsupervised, ablation)  |
| LSTM              | `LSTM_Prob`              | `LSTM_Signal`              | ML (supervised)            			|
| Transformer       | `Transformer_Prob`       | `Transformer_Signal`       | ML (attention-based)     			|

---

## 3. Registering Hyperparameters in the Central Config

> **Important:** Since [Issue #3](https://github.com/Torim98/regime-switching-daa/issues/3), **all pipeline parameters are managed centrally** in [`config/config.yaml`](../config/config.yaml). Hardcoded hyperparameters in code are no longer allowed.

> **Hyperparameter optimization:** Optuna can programmatically override
> `config.yaml` parameters ([Issue #2](https://github.com/Torim98/regime-switching-daa/issues/2)).
> To include a new model in the optimization, declare its search space under
> `optimization.search_spaces.<Model>` in `config.yaml`, define an
> `objective_<model>` function in `src/backtest/optimize.py` (sample via
> `_suggest_space(trial, cfg.optimization.search_spaces.<Model>)` and pool the
> per-fold OOS returns), and register it in `_OBJECTIVE_MAP`. Add the model to
> `optimization.grid_models` to search it exhaustively with a `GridSampler`
> instead of TPE. See [hyperparameter-optimization.md](./hyperparameter-optimization.md).

### Architecture Overview

```
config/
├── config.yaml          # Single source of truth: all parameters
└── config_loader.py     # PipelineConfig class + singleton `cfg`
                         # Methods: data_path(), asset_path(), model_path()
```

The `config.yaml` is structured hierarchically by pipeline section:

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
  my_model:              # ← Add your new model here
    param_1: value_1
    ...
```

### How to Load the Config in Code

```python
# In src/ or services/: load the config
from config.config_loader import cfg

# Access via dot notation:
cfg.models.hmm.n_components          # → 2
cfg.features.model_features          # → ['Returns', 'Vol_20', ...]
cfg.asset_path("equity_curves")      # → ".../assets/equity_curves.png"
cfg.model_path("lstm")               # → ".../models/lstm_regime_model.keras"
cfg.model_persistence.enabled        # → true/false
```

### How to Add Your Model to the Config

**Step A:** Open `config/config.yaml` and add a new block under the `models:` section:

```yaml
models:
  # ... existing models (msm, hmm, lstm, transformer) ...

  my_model:                    # ← key name in snake_case
    window_size: 20            # example hyperparameter
    n_heads: 4                 # example for Transformer
    n_layers: 2
    d_model: 64
    dropout: 0.1
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    threshold: *threshold       # matches the other models; bear signal if prob >= threshold
```

### Registering a Plot Color (Optional)

To give your model a consistent color in all plots, add an entry under `plotting.colors`:

```yaml
plotting:
  colors:
    # ... existing models ...
    MyModel: "tab:cyan"        # ← your model color
```

> **Note:** This step is optional. Models without an entry automatically
> receive a color from the matplotlib default cycle. The color is used
> consistently across all services via `cfg.color_map`.

Available color names: all [matplotlib named colors](https://matplotlib.org/stable/gallery/color/named_colors.html)
and the `tab:` palette (e.g. `tab:blue`, `tab:red`, `tab:cyan`).

**Step B:** Access the parameters in code:

```python
# Instead of hardcoded: window_size = 20
my_cfg = cfg.models.my_model
window_size = my_cfg.window_size      # → 20
n_heads     = my_cfg.n_heads          # → 4
epochs      = my_cfg.epochs           # → 50
threshold   = my_cfg.threshold        # → 0.5
```

### Advantages of the Central Configuration

| Aspect                         | Before (hardcoded)                | After (config)                                                                                                              |
| :----------------------------- | :-------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| **Changing a parameter**  | Search code, change manually       | Edit `config.yaml`, propagates automatically                                                                               |
| **Reproducibility**         | Parameters scattered across modules   | Everything in one place, versioned in Git                                                                                     |
| **Hyperparameter optimization** | Adjust manually in code          | Optuna can override `config.yaml` programmatically ([Issue #2](https://github.com/Torim98/regime-switching-daa/issues/2)) |
| **Fast mode (development)**    | Adjust each model individually     | `fast_mode.enabled: true` reduces epochs/MCS paths automatically                                                              |

### Fast Mode for New Models (Optional)

If you want to add a fast-mode override for your model:

```yaml
fast_mode:
  enabled: false
  overrides:
    lstm_epochs: 5
    mcs_n_paths: 100
    my_model_epochs: 5          # ← new override
```

Then add to `config_loader.py` (class `PipelineConfig.__init__`):

```python
if self.fast_mode.enabled:
    # ... existing overrides ...
    self.models.my_model.epochs = self.fast_mode.overrides.my_model_epochs
```

---

## 4. Step-by-Step Guide

All changes take place in **four locations**: `config/config.yaml` (hyperparameters), `src/models/<name>.py` (training logic), `src/models/plots.py` (regime plot), and `services/model_service/routes.py` (service route).

> **Principle: shared business logic.** The domain logic lives in `src/`; the service route only calls it. Never duplicate training logic in the route itself.

### Step 1: Create a Config Entry

Add your model's hyperparameters to `config/config.yaml` under `models:`:

```yaml
models:
  # ... existing models ...

  my_model:
    window_size: 20
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    dropout: 0.1
    threshold: 0.5
    # ... further model-specific parameters ...
```

### Step 2: Implement the Training Logic in src/models/

Create a new module `src/models/my_model.py`. The function loads all hyperparameters from the config (**no hardcoding!**), trains the model, and writes the two signal columns into the DataFrame:

```python
# src/models/my_model.py
from my_model_library import MyModel

def train_my_model(df, feature_cols, cfg):
    """Trains MyModel and writes MyModel_Prob / MyModel_Signal into df."""
    my_cfg = cfg.models.my_model

    # Prepare the feature matrix
    X = df[feature_cols].dropna()

    # Initialize the model: all parameters from config.yaml
    model = MyModel(
        window_size=my_cfg.window_size,
        epochs=my_cfg.epochs,
        batch_size=my_cfg.batch_size,
        learning_rate=my_cfg.learning_rate,
        dropout=my_cfg.dropout,
    )
    model.fit(X)

    # Bear probability (high values = crisis) + derive binary signal
    df["MyModel_Prob"] = model.predict_proba(X)
    df["MyModel_Signal"] = (df["MyModel_Prob"] >= my_cfg.threshold).astype(int)
    return df, model
```

> **Important:** Make sure that `_Prob` contains the **bear probability** (high values = crisis). If your model outputs the bull probability, invert it: `bear_prob = 1 - bull_prob`.

> **Features:** `feature_cols` comes from `cfg.features.model_features` (`['Returns', 'Vol_20', 'Distance_SMA', 'Momentum', 'VIX', 'Yield_Spread']`). You can use a subset, or compute additional features in the Data Service and extend the list in `config.yaml`.

### Step 3: Plot Function

Add a `matplotlib.use("Agg")`-compatible plot function in `src/models/plots.py` (no `plt.show()`; use `plt.savefig()` + `plt.close(fig)` instead):

```python
# Add to src/models/plots.py:

def plot_my_model(df, model_name, color, save_path):
    """Regime plot for MyModel."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    # ... plotting logic ...
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

### Step 4: Register the Route in the Model Service

Open `services/model_service/routes.py`, import your functions, and add an `elif` block in the `train_model()` endpoint (`POST /train/{model_name}`), analogous to the existing models (`if model_name == "msm": ... elif ...`):

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

> **`validate_regime_signal`** (from `src/models/common.py`) prints regime statistics (returns, VIX, yield spread per signal), checks plausibility (bear regime `1` must have lower returns than bull `0`) with automatic label inversion (`auto_invert=True`), and performs formal assertions (column existence, value range `[0,1]`, no NaN).

> **Signal persistence:** The existing models write their signal columns back to `feature_engineered` (`df.to_parquet(cfg.data_path("feature_engineered"))`). The Backtest Service reads the same file and detects your `_Signal` column automatically via dynamic matching; nothing needs to be adapted downstream.

### Step 5: Include in /train-all

To make your model run with `POST /models/train-all`, add it to the training order:

```python
@router.post("/train-all")
def train_all():
    results = []
    for model in ["msm", "hmm", "lstm", "transformer", "my_model"]:
        results.append(train_model(model))
    return results
```

> **Mind the order:** If your model depends on predefined labels (like LSTM and Transformer on Pagan-Sossounov), these labels must already be available in `feature_engineered_data` (created by the Data Service).

### Step 6: Model Persistence (Optional)

Trained models can be cached in the `models/` directory. This is particularly useful when training is computationally expensive (e.g. LSTM, Transformer).

#### Prerequisite

The `model_persistence` section exists in `config/config.yaml`:

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

#### How to Add Your Model

**1.** Add an entry for your model under `model_persistence.files`:

```yaml
model_persistence:
  files:
    # ... existing entries ...
    my_model: "my_model.pkl"           # ← file name of your model
    scaler_my_model: "my_model_scaler.pkl"  # ← if a scaler is required
```

**2.** In `src/models/my_model.py`, use `cfg.model_path("my_model")` to obtain the full path.

**3.** Wrap training and loading in your training function with an `if/else` block:

```python
import os
from pathlib import Path

persist = cfg.model_persistence
model_file = cfg.model_path("my_model")

if persist.enabled and os.path.exists(model_file):
    # MODE A: load the saved model
    model = load_my_model(model_file)
else:
    # MODE B: train normally + save
    model = train_my_model_impl(...)

    Path(persist.models_dir).mkdir(parents=True, exist_ok=True)
    save_my_model(model, model_file)
```

**4.** If your model requires a scaler (e.g. `StandardScaler`, `MinMaxScaler`), persist it as well. When loading, **always** use `transform()` instead of `fit_transform()`!

#### Serialization Formats by Library

| Library | Save | Load | File extension |
|:---|:---|:---|:---|
| `statsmodels` | `results.save(path)` | `sm.load(path)` | `.pkl` |
| `hmmlearn` / `sklearn` | `joblib.dump(model, path)` | `joblib.load(path)` | `.pkl` |
| `TensorFlow/Keras` | `model.save(path)` | `load_model(path)` | `.keras` |
| `PyTorch` | `torch.save(model.state_dict(), path)` | `model.load_state_dict(torch.load(path))` | `.pt` |

> **Note:** The `models/` directory is listed in `.gitignore` and is created automatically on first save via `Path(...).mkdir(parents=True, exist_ok=True)`.

### Step 7: Docker Rebuild & Test

```bash
docker-compose build model-service     # src/ is copied into the image via COPY src/ src/
docker-compose up -d model-service

# Train a single model:
curl -X POST http://localhost:8002/models/train/my_model

# Or all models:
curl -X POST http://localhost:8002/models/train-all
```

> Since the training logic lives in `src/` and `src/` is already copied in the Dockerfile (`COPY src/ src/`), **no Dockerfile changes are required**.

---

## 5. Why This Works: Dynamic Matching

The downstream steps (Backtest Service, evaluation, reporting) use a **dynamic matching algorithm** that detects all model signals automatically:

```python
# From src/backtest/engine.py (run_all_backtests): core logic of dynamic matching
signal_cols = [col for col in test_df.columns if col.endswith("_Signal")]

for sig_col in signal_cols:
    model_name = sig_col.rsplit("_", 1)[0]  # Extracts "HMM" from "HMM_Signal"
    backtesting_results[model_name] = backtest(
        test_df, sig_col, signal_shift=cfg.backtesting.signal_shift,
        fee=cfg.transaction_cost_rate,
    )
```

**This means:**
1. **Backtesting** detects every column ending in `_Signal` and runs a full backtest
2. **Evaluation** automatically computes all metrics (Sharpe, Sortino, Calmar, max drawdown, etc.) for every detected model
3. **Reporting** generates equity curves, statistics tables, and SORR simulations for all models

**You do not need to change any code in the Backtest Service, the evaluation, or the reporting (`src/backtest/`).** It is sufficient to implement the signal interface correctly and register the hyperparameters in the config.

---

## 6. Look-Ahead Bias Prevention (T+1 Shift)

A critical concept that the backtesting module handles automatically:

```python
# From src/backtest/engine.py: automatic T+1 shift
trading_signal = df[signal_col].shift(signal_shift).fillna(0)
# signal_shift is defined in config.yaml (default: 1)
```

**What happens here?**
- The backtesting shifts every signal by **one trading day** into the future (`shift(1)`)
- A signal computed on day `T` is applied as a trading decision only on day `T+1`
- This prevents **look-ahead bias**: decisions are based exclusively on historically available information
- The shift value is configurable via `backtesting.signal_shift` in `config.yaml` (default: `1`)

**What this means for you:**
- You do **NOT** need to implement the shift yourself; the backtesting takes care of it
- You must **NOT** apply the shift twice (i.e. not already in `src/models/<name>.py`)
- The columns `_Prob` and `_Signal` contain the values **at the time of computation** (day `T`)

---

## 7. Updating the Documentation

After successfully integrating the model into the pipeline (steps 1–7), **three documentation levels** must be updated so that the new model appears correctly in the project documentation.

> **Note:** The quantitative tables (performance summary, evaluation matrix, SORR summary, MCS summary) and most plots (equity curves, regime comparison, MCS boxplots) are generated **automatically** thanks to dynamic matching. The following steps concern only the **manual** documentation adjustments.

### Step A: Register the Asset Path in `config.yaml`

For the reporting to reference the model plot, an entry must be added under `paths.assets`:

```yaml
paths:
  assets:
    # ... existing entries ...
    my_model: "my_model.png"           # ← file name of the regime plot
```

> The plot itself is created in `src/models/plots.py` and stored under `assets/`. The config only defines the file name through which `src/backtest/reporting.py` embeds the plot.

### Step B: Add a Model Section to the Reporting (`statistics.md`)

The file `docs/statistics.md` is generated **fully automatically** by `src/backtest/reporting.py`. However, the section **"Regime detection of the individual models"** contains a manually maintained paragraph for each model with a description and image reference in the f-string template.

Open `src/backtest/reporting.py` and insert a new subsection at the appropriate position within the f-string (`stats_md_content`):

```python
### G. <ModelName> (<Paradigm>)
<Short description of the model: approach, distinctive feature, training setting.>
![<ModelName> Model](../assets/{cfg.paths.assets.my_model})
```

**Orientation:** The existing sections follow the order econometrics → ML (supervised) → attention-based. Place your model accordingly.

> **Important:** **Never** edit `docs/statistics.md` directly. The file is overwritten on the next `POST /backtest/evaluate`. All changes must be made in the f-string template in `src/backtest/reporting.py`.

### Step C: Update README.md

The `README.md` contains a description of all model categories in the section **"Methodology & Models"**. Add an entry for your new model there:

- **Under item 1 (econometric models)** if it is a statistical method
- **Under item 2 (machine-learning methods)** if it is an ML/DL model

Use the style and level of detail of the existing model descriptions as a template. An entry should contain at least:
1. **Model name** and architecture type (bold)
2. Short description of the approach (1–2 sentences)
3. Training setting (supervised / unsupervised) and, if applicable, label source

### Step D (Optional): Architecture Documentation in `docs/`

For complex models (especially neural networks), it is recommended to add a dedicated architecture description under `docs/`.

Such a file should contain:
- Schematic representation of the layers / modules (as text diagram or image)
- Input/output dimensions
- Reference to the corresponding config key (`cfg.models.<name>`)

### Step E (Optional): Register the Plot Color in `config.yaml`

Add a color for your model under `plotting.colors` so that it is displayed
consistently in all comparison plots (equity curves, regime comparison, MCS).

---

## 8. Code Template (Copy & Paste)

### A. Config Entry (`config/config.yaml`)

```yaml
models:
  # ... existing models ...
  my_model:                       # ← snake_case key name
    # --- Architecture ---
    window_size: 20               # input sequence length
    units: 64                     # model size (e.g. hidden units, d_model)
    dropout: 0.1                  # regularization
    # --- Training ---
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    validation_split: 0.1
    # --- Signal ---
    threshold: 0.5                # bear signal if prob >= threshold

plotting:
  colors:
    # ... existing models ...
    MyModel: "tab:cyan"           # ← plot color (optional)
```

### B. Service Module (`src/models/my_model.py`)

Create the training module (signature analogous to the existing models):

```python
from my_model_library import MyModel

def train_my_model(df, feature_cols, cfg):
    """Trains MyModel and writes MyModel_Prob / MyModel_Signal into df."""
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

Registration in the Model Service (`elif` block + `validate_regime_signal` + plot) follows [Section 4, Step 4](#step-4-register-the-route-in-the-model-service).

---

## 9. Validation Checklist

After integration, perform the following checks:

### Formal Checks
- [ ] The column `<Model>_Prob` exists in the DataFrame and contains `float` values between `0.0` and `1.0`
- [ ] The column `<Model>_Signal` exists in the DataFrame and contains only `0` or `1`
- [ ] No `NaN` values in `<Model>_Signal` (within the test period)
- [ ] The model name does not collide with existing names (`MSM`, `HMM`, `HMM_Uni`, `LSTM`, `Transformer`)

### Config Checks
- [ ] New entry created under `models:` in `config/config.yaml`
- [ ] **No** hyperparameters hardcoded in code; everything comes from `cfg.models.<name>`
- [ ] Config key matches the access in code (e.g. `cfg.models.my_model`)
- [ ] If a fast-mode override is desired: entry added in `fast_mode.overrides` and `config_loader.py`

### Persistence Checks (Optional)
- [ ] Entry created under `model_persistence.files` in `config.yaml` (if persistence is desired)
- [ ] `if/else` block for load/train implemented in `src/models/<name>.py`
- [ ] Scaler is used with `transform()` instead of `fit_transform()` when loading
- [ ] `Path(persist.models_dir).mkdir(parents=True, exist_ok=True)` called before the first save

### Content Checks
- [ ] Bear regime (signal = 1) shows **lower average returns** than bull regime (signal = 0)
- [ ] Bear regime shows a **higher average VIX** than bull regime
- [ ] The signal distribution is plausible (not 99% one regime)

### Documentation Checks
- [ ] Asset path for the model plot registered in `config.yaml` under `paths.assets`
- [ ] (Optional) Plot color registered under `plotting.colors` in `config.yaml`
- [ ] New section inserted in the reporting (`src/backtest/reporting.py`, f-string template)
- [ ] `README.md`: model described in "Methodology & Models"
- [ ] (Optional) Architecture documentation created under `docs/`
- [ ] `docs/statistics.md` contains the new model section with the correct image after a pipeline run

### Implementation Checks (Model Service)
- [ ] Training logic implemented in `src/models/` as a reusable module (not inline in the service route)
- [ ] Plot function in `src/models/plots.py` with `plt.close(fig)` (no `plt.show()`)
- [ ] `elif model_name == "my_model":` block added in `services/model_service/routes.py`
- [ ] Model registered in the `train_all()` function of the Model Service
- [ ] Docker image rebuilt (`docker-compose build model-service`)
- [ ] Endpoint `POST /models/train/my_model` returns HTTP 200

### Pipeline Integration
- [ ] `POST /models/train/my_model` (or `/train-all`) updates `data/silver/03_feature_engineered_data.parquet` with `MyModel_Prob`/`MyModel_Signal`
- [ ] `POST /backtest/run` detects the new model automatically (dynamic matching) and computes equity curves
- [ ] `POST /backtest/evaluate` computes metrics (Sharpe, Sortino, Calmar, max drawdown) and generates `statistics.md` with the new model
- [ ] The figures in `assets/` (equity curves, regime comparison, MCS boxplots, etc.) include the new model

### End-to-End Test (Recommended)
```bash
# Full pipeline run via microservices:
docker-compose up --build -d
curl -X POST http://localhost:8001/data/ingest
curl -X POST http://localhost:8002/models/train-all
curl -X POST http://localhost:8003/backtest/run
curl -X POST http://localhost:8003/backtest/evaluate
```

---

## 10. Reference Implementations

The following existing models in `src/models/` (`msm.py`, `hmm.py`, `lstm.py`, `transformer.py`) serve as references. All load their hyperparameters from the central `config.yaml`:

### A. Markov-Switching (MSM): Econometrics
- **Library:** `statsmodels` (MarkovRegression)
- **Approach:** Univariate regression model with state-dependent parameters (switching variance)
- **Output:** `MSM_Prob`, `MSM_Signal`
- **Config key:** `models.msm` (k_regimes, switching_variance)

### B. HMM (Hidden Markov Model): Unsupervised, Econometrics
- **Library:** `hmmlearn`
- **Approach:** Identifies clusters in the data distributions without labeled data
- **Output:** `HMM_Prob`, `HMM_Signal`
- **Config key:** `models.hmm` (n_components, covariance_type, n_iter, random_state)
- **Distinctive feature:** Requires a post-training check whether regime 0 or 1 corresponds to the bear regime (label alignment)

### B2. HMM_Uni (Univariate HMM): Ablation Variant, Econometrics
- **Library:** `hmmlearn` (identical code to HMM: `src/models/hmm.py`, `train_hmm_fold` is feature-agnostic)
- **Approach:** Like HMM, but with only `Returns` as input. Identical input space to the MSM. Separates the architectural effect from the feature contribution
- **Output:** `HMM_Uni_Prob`, `HMM_Uni_Signal`
- **Config key:** `models.hmm_uni` (features, n_components, covariance_type, n_iter, random_state, threshold)
- **Distinctive feature:** No dedicated module required; only a second config block that runs through the same fold function (`_run_hmm_fold` in `src/backtest/parallel.py`). Reference case showing that a new model can be integrated purely via configuration + orchestration.

### C. LSTM (Supervised): Machine Learning
- **Library:** `TensorFlow` / `Keras`
- **Approach:** Supervised learning on Pagan-Sossounov labels; learns regime switches from time-series sequences (windows)
- **Output:** `LSTM_Prob`, `LSTM_Signal`
- **Config key:** `models.lstm` (window_size, units_l1, units_l2, epochs, batch_size, learning_rate, dropout, activation, optimizer, loss, metrics, validation_split, verbose)
- **Distinctive feature:** Uses a rolling window (`window_size`) as input sequence. Labels come from the Pagan-Sossounov algorithm (configurable via `labels.supervised_label_source`; comparison of the label sources via `POST /data/label-analysis`)

### D. Transformer (Supervised, Attention-Based): Machine Learning
- **Library:** `PyTorch` (`torch.nn.TransformerEncoder`)
- **Approach:** Transformer encoder with positional encoding and multi-head self-attention for time-series-based regime classification; supervised on Pagan-Sossounov labels
- **Output:** `Transformer_Prob`, `Transformer_Signal`
- **Config key:** `models.transformer` (window_size, d_model, n_heads, n_layers, dim_feedforward, dropout, epochs, batch_size, learning_rate, threshold, pos_weight_auto)
- **Distinctive feature:** Uses BCEWithLogitsLoss with automatic class-balance weighting (sqrt pos_weight). Tests hypothesis H2 (attention mechanism vs. econometric MSM). Serves as the **reference implementation** for the guide-compliant signal interface (full sanity check, assertions, config-only hyperparameters).

### Config Mapping Overview

| Model | Config key | Most important parameters |
|:---|:---|:---|
| MSM | `cfg.models.msm` | `k_regimes`, `switching_variance` |
| HMM | `cfg.models.hmm` | `n_components`, `covariance_type`, `n_iter`, `random_state` |
| HMM_Uni | `cfg.models.hmm_uni` | `n_components`, `covariance_type`, `threshold` |
| LSTM | `cfg.models.lstm` | `window_size`, `units_l1`, `units_l2`, `epochs`, `batch_size`, `learning_rate`, `dropout` |
| Transformer | `cfg.models.transformer` | `window_size`, `d_model`, `n_heads`, `n_layers`, `epochs`, `threshold` |
| **Your model** | `cfg.models.my_model` | *your parameters* |

---

## 11. FAQ & Troubleshooting

### My model does not appear in the backtesting
**Cause:** The signal column does not end exactly in `_Signal` or contains `NaN` values.
**Solution:** Check the column name and make sure there are no missing values:
```python
assert df['MyModel_Signal'].isna().sum() == 0, "NaN values found in signal!"
assert df['MyModel_Signal'].isin([0, 1]).all(), "Signal contains values outside {0, 1}!"
```

### My model's equity curve is identical to buy & hold
**Cause:** The model outputs almost exclusively signal `0` (bull).
**Solution:** Check the signal distribution and the threshold:
```python
print(df['MyModel_Signal'].value_counts(normalize=True))
# If necessary, adjust the threshold in config.yaml: models.my_model.threshold
```

### The bear regime has higher returns than the bull regime
**Cause:** The labels are swapped (common with unsupervised models).
**Solution:** `validate_regime_signal()` detects and corrects this automatically
(`auto_invert=True`). If you want to disable the automatic inversion:
```python
validate_regime_signal(df, MODEL_NAME, auto_invert=False)
```

### Config error: `AttributeError: 'SimpleNamespace' object has no attribute 'my_model'`
**Cause:** The config entry in `config.yaml` is missing or the key name does not match.
**Solution:** Check that an entry `my_model:` exists under `models:` (snake_case, correct indentation with 2 spaces). If necessary, restart the service (`docker-compose restart model-service`) so that `cfg` is reloaded.

### How do I change a hyperparameter for a new run?
**Solution:** Edit **only** `config/config.yaml`, e.g. `models.my_model.epochs: 100`. Then restart the pipeline. The change propagates automatically via `cfg` into all services.

### My model needs additional features that do not exist yet
**Solution:** Extend the feature engineering in the Data Service (`src/data/feature_engineering.py`) with the new features and add them to `config.yaml` under `features.model_features`. Make sure the features are persisted in `feature_engineered_data`.

### How many models can the pipeline handle?
**Answer:** There is no technical limit. The dynamic matching algorithm detects any number of `_Signal` columns. Note, however, that more models increase the runtime of the evaluation (especially the Monte Carlo simulation with `evaluation.mcs.n_paths` paths). Use `fast_mode.enabled: true` in the config for faster development cycles.

### Can I use fast mode for development?
**Answer:** Yes. Set in `config.yaml`:
```yaml
fast_mode:
  enabled: true
  overrides:
    lstm_epochs: 5
    mcs_n_paths: 100
```
This automatically reduces training epochs and MCS paths. Do not forget to set `fast_mode.enabled: false` before the final run.

### I changed hyperparameters, but the results are identical
**Cause:** `model_persistence.enabled` is `true` and an old model still exists under `models/`.
**Solution:** Delete the affected model file from `models/` (or the entire directory) so that the model is retrained with the new parameters. Alternatively, set `model_persistence.enabled: false` in `config.yaml`.

### How can I retrain a single model without deleting all of them?
**Solution:** Delete only the specific file (e.g. `models/lstm_regime_model.keras`). On the next pipeline run, only this model is retrained; all others continue to be loaded from the cache.
