# Hyperparameter Optimization (Optuna)

> **Goal:** Describe how the paper-grade hyperparameter optimization (HPO) works: what it optimizes, over which data, with which samplers and budgets, and how selection is kept separate from evaluation. The design follows [Issue #5](https://github.com/Torim98/regime-switching-daa-paper/issues/5). Implementation: [src/backtest/optimize.py](../src/backtest/optimize.py) (search) and [src/backtest/hpo_analysis.py](../src/backtest/hpo_analysis.py) (post-HPO analysis). All settings live under `optimization:` in [config/config.yaml](../config/config.yaml).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Where the HPO Sits in the Pipeline](#2-where-the-hpo-sits-in-the-pipeline)
3. [The Objective: Pooled OOS Risk Metric](#3-the-objective-pooled-oos-risk-metric)
4. [Search Spaces](#4-search-spaces)
5. [Samplers and Trial Budgets](#5-samplers-and-trial-budgets)
6. [Training Protocol: Warm-Start, Early Stopping, Pruning](#6-training-protocol-warm-start-early-stopping-pruning)
7. [Selection vs. Evaluation (`tune_until`)](#7-selection-vs-evaluation-tune_until)
8. [Reproducibility and Study Versioning](#8-reproducibility-and-study-versioning)
9. [Post-HPO Analysis](#9-post-hpo-analysis)
10. [How to Run](#10-how-to-run)
11. [GPU Infrastructure](#11-gpu-infrastructure)
12. [Config Reference](#12-config-reference)

---

## 1. Overview

The HPO tunes the hyperparameters of all five regime models (MSM, HMM, HMM_Uni, LSTM, Transformer) using the walk-forward splitter as an inner cross-validation. Each trial trains a model per fold, concatenates the out-of-sample (OOS) net returns of all folds into one pooled series, and scores that series with a configurable risk metric that Optuna maximizes.

Key design choices, each of which is motivated below:

- **Objective** is a path-dependent risk metric (default: **Martin ratio**) on the **pooled** OOS series, not the fold-wise median Sharpe.
- **Econometric models** (MSM, HMM, HMM_Uni) are searched **exhaustively via a grid**; **deep-learning models** (LSTM, Transformer) via a **multivariate TPE sampler**.
- **Selection and evaluation are time-separated:** the HPO only sees the development folds (`tune_until`), while the final walk-forward run uses all folds, so the holdout stays selection-free.
- **Full reproducibility:** fixed seed, versioned study names, persistent SQLite storage.

---

## 2. Where the HPO Sits in the Pipeline

```
walk_forward_splits()          # identical splits for HPO and final run
        │
        ├── HPO (this document) ── inner CV on development folds only
        │        │
        │        └── best params ──> config.yaml (apply_best_params)
        │
        └── final walk-forward run ── all folds, frozen params ──> results
```

The HPO reuses the exact same fold generator as the final run
([walk_forward.py](../src/backtest/walk_forward.py)), so the inner CV and the
final evaluation are consistent by construction. The HPO never trains the
persisted production models; it only searches for the hyperparameters that are
then written into `config.yaml` and used by the final run.

There is **no look-ahead bias within a fold**: each fold fits its labels and
scaler on the train window only, and Optuna observes OOS metrics exclusively.

---

## 3. The Objective: Pooled OOS Risk Metric

### Pooled series instead of fold-median

The thesis (ch. 4.4) notes that a median over a sequence of mostly bullish
12-month test windows under-weights the rare crisis folds. Computing a
path-dependent tail metric on the **pooled** OOS return series instead lets
exactly those crisis segments drive the objective, which aligns the search with
the paper's sequence-of-returns-risk (SORR) / tail-risk goal.

Concretely, every objective function accumulates each fold's daily net returns
into a list and, at the end, calls `compute_oos_metrics(np.concatenate(pooled))`
([optimize.py](../src/backtest/optimize.py)).

### Available metrics

`cfg.optimization.metric` selects the objective. All metrics are normalized
internally to "higher = better", so the study direction is always `maximize`.

| Metric | Definition | Notes |
|:---|:---|:---|
| `martin` (default) | CAGR / Ulcer index | Integrates drawdown **depth and duration**; SORR-related; smoother than MaxDD |
| `sharpe` | annualized mean / std | Symmetric; reported as a robustness metric |
| `sortino` | annualized mean / downside deviation | Downside-focused, still moment-based |
| `calmar` | CAGR / \|MaxDD\| | MaxDD is dominated by a single event (noisy) |
| `ulcer` | RMS drawdown (%) | Loss metric, maximized as its negative |
| `max_drawdown` | worst peak-to-trough | Reported as its raw (negative) value |

The Ulcer index is computed via the canonical helper in
[evaluation.py](../src/backtest/evaluation.py) so the objective and the reported
Ulcer figure are identical. To keep the Martin ratio finite on a (near)
drawdown-free path, the Ulcer denominator is floored at 0.1.

### Full metric vector logged per trial

Every trial stores its **complete** metric vector (all of the above plus CAGR
and `n_oos_days`) as `user_attrs`. This is what makes the post-hoc
objective-sensitivity table possible (best config under Martin vs. Sharpe vs.
Sortino) without re-running the search.

### Failed / degenerate trials

A trial that fails on at least half of its folds, or produces fewer than 20
pooled OOS days, returns a large negative sentinel score so it is ranked last
without aborting the study.

---

## 4. Search Spaces

Search spaces are declared declaratively in `config.yaml` under
`optimization.search_spaces` and parsed into `trial.suggest_*` calls. The design
principle is: **un-bound every edge-of-range optimum from the thesis run,
eliminate dead dimensions, and encode constraints into the space instead of
pruning them away.**

### Econometric models (grid)

| Model | Parameter | Space |
|:---|:---|:---|
| MSM | `threshold` | 0.10 to 0.975, step 0.025 (36 values) |
| HMM | `covariance_type` | {full, diag, tied} |
| HMM | `threshold` | 0.10 to 0.975, step 0.025 |
| HMM_Uni | `threshold` | 0.10 to 0.975, step 0.025 |

`k_regimes` / `n_components` stays fixed at 2 (bull/bear premise, comparison axis
of Issue #3). `covariance_type` is **not** tuned for HMM_Uni: with univariate
input the 1x1 covariance makes full/diag/tied identical.

### Deep-learning models (TPE)

**LSTM** (7 dimensions):

| Parameter | Space |
|:---|:---|
| `window_size` | 20 to 250, step 10 |
| `units_l1` | {16, 32, 64, 128} |
| `units_l2` | {32, 64, 128, 256} |
| `batch_size` | {32, 64, 128} |
| `learning_rate` | 1e-5 to 1e-2, log scale |
| `dropout` | 0.0 to 0.6, step 0.05 |
| `threshold` | 0.10 to 0.90, step 0.05 |

**Transformer** (8 dimensions):

| Parameter | Space |
|:---|:---|
| `window_size` | 20 to 250, step 10 |
| `dmodel_nheads` | {16-2, 32-2, 32-4, 64-4, 64-8, 128-4, 128-8} |
| `n_layers` | 1 to 4 |
| `dim_feedforward` | {32, 64, 128, 256} |
| `batch_size` | {32, 64, 128} |
| `learning_rate` | 1e-5 to 1e-2, log scale |
| `dropout` | 0.0 to 0.5, step 0.05 |
| `threshold` | 0.10 to 0.90, step 0.05 |

The `(d_model, n_heads)` divisibility constraint (`d_model % n_heads == 0`) is
encoded as a **single categorical of valid pairs** (`"64-8"` and the like)
rather than being sampled as two dimensions and pruned on violation. This wastes
no trials on invalid combinations and keeps the pruning statistics
interpretable. `epochs` is not a search dimension (see
[Section 6](#6-training-protocol-warm-start-early-stopping-pruning)).

LSTM and Transformer share the same learning-rate range, window range and early
stopping, which makes the architecture comparison fair.

---

## 5. Samplers and Trial Budgets

The sampler is chosen per model from `optimization.grid_models`:

- **GridSampler** for the econometric models. The low-dimensional spaces are
  small enough to evaluate exhaustively, so the paper can state "global optimum
  within the grid" instead of "a sample of the space". The grid size (36 / 108 /
  36) overrides the `n_trials_per_model` value at runtime.
- **TPESampler** for the DL models, configured as
  `TPESampler(multivariate=True, group=True, seed=42, n_startup_trials=...)`.
  Multivariate sampling matters because `window_size x learning_rate x dropout`
  interact (per fANOVA).

| Model | Sampler | Budget |
|:---|:---|---:|
| MSM | Grid | 36 (exhaustive) |
| HMM | Grid | 108 (exhaustive) |
| HMM_Uni | Grid | 36 (exhaustive) |
| LSTM | TPE, 50 startup | 300 |
| Transformer | TPE, 60 startup | 400 |

The DL budgets are **minimum** budgets. The pre-registered stopping rule (Issue
#5) is: run in blocks of 100 trials and stop when the best value improves by
< 1 % over a block **and** the contour plots show no unexplored high-performance
region. The convergence review ([Section 9](#9-post-hpo-analysis)) supports this
decision after each block.

The current config also enqueues the existing config values as a warm-start
baseline trial for the TPE models (`enqueue_trial`), so the search starts from a
known-good point. Grid models do not enqueue a baseline (it would be an off-grid
point).

---

## 6. Training Protocol: Warm-Start, Early Stopping, Pruning

### Warm-start across folds

DL folds warm-start from the previous fold's weights inside the HPO trials,
exactly as the final walk-forward run does (`dl_warm_start: true`,
`dl_warm_start_epochs: 10`). This makes the HPO an image of the evaluated
procedure and cuts the per-trial cost: only the first (cold) fold trains up to
`dl_max_epochs`, later folds fine-tune for `dl_warm_start_epochs` epochs. On an
architecture mismatch a fold falls back to a cold start.

### Early stopping instead of an epochs dimension

`epochs` is not tuned. A large ceiling (`dl_max_epochs: 150`) combined with the
trainers' early stopping on validation loss governs the effective epoch count.
Both the LSTM ([lstm.py](../src/models/lstm.py)) and the Transformer
([transformer.py](../src/models/transformer.py)) restore the best weights.
This removes one redundant search dimension.

### Pruning

Pruning is **disabled for the DL headline run** (`pruning.enabled: false`). The
MedianPruner compares intermediate values over chronologically ordered folds, so
a config that protects capital in the rare crisis folds (exactly what the paper
seeks) looks mediocre in the early bullish folds and would be pruned. Keeping
100 % complete trials also yields unbiased fANOVA importances and contour plots.
Pruning should only be enabled under a compute-budget cap; then `n_warmup_steps`
must be large enough to cover the early crisis folds.

---

## 7. Selection vs. Evaluation (`tune_until`)

The most important validity safeguard: **the HPO is run on a development period
only, and the reported results come from an untouched holdout.**

- `optimization.tune_until: "2016-12-31"` restricts the HPO to folds whose OOS
  test window ends on or before that date. The development period contains the
  Dotcom crash (2000 to 2002) and the Global Financial Crisis (2007 to 2009), so
  the drawdown objective gets two full bear markets as optimization signal.
- The **final walk-forward run uses all folds** with frozen parameters. The
  holdout (COVID 2020, the 2022 rate shock) therefore tests generalization to
  *structurally different* crises rather than recognition of the tuning crises.

This is implemented by `_filter_splits_until` in
[optimize.py](../src/backtest/optimize.py); the final run
([walk_forward.py](../src/backtest/walk_forward.py)) is unchanged. Set
`tune_until` to `null` or `""` to tune on all folds.

The number of tested configurations is priced in explicitly via the Deflated
Sharpe Ratio and PBO ([Section 9](#9-post-hpo-analysis)).

---

## 8. Reproducibility and Study Versioning

- **Fixed seed** (`sampler.seed: 42`) for the TPE and grid samplers (Issue #5
  acceptance criterion).
- **Versioned study names** via `study_suffix` (e.g. `opt_LSTM_v2_martin`). A
  changed search space or objective must never resume into a study built under
  different distributions, which would contaminate the TPE posterior. The old
  study database is kept as an archive for the before/after comparison.
- **Persistent storage** in `models/optuna_studies.db` (SQLite). Because SQLite
  on Docker bind mounts is fragile under write pressure, the storage is opened
  with WAL journaling, `busy_timeout`, `synchronous=NORMAL` and pre-ping. Runs
  can be interrupted and resumed safely.
- Each study records its `metric` and `tune_until` as study-level `user_attrs`.

---

## 9. Post-HPO Analysis

[src/backtest/hpo_analysis.py](../src/backtest/hpo_analysis.py) operates on the
persisted studies and reuses the fold machinery from `optimize.py`, so its
numbers are identical to the search itself. It has two scopes:

- **`cheap`** (seconds, reads logged trial attrs only): convergence review +
  objective sensitivity.
- **`full`** (minutes to hours, re-trains DL models): additionally DSR, PBO and
  multi-seed re-evaluation.

| Building block | Purpose | Output asset |
|:---|:---|:---|
| `convergence_review` | Best value/trial, fANOVA importance, edge-of-range flags (optimum within one grid step of a bound) | `hpo_convergence.md` |
| `objective_sensitivity` | Best config under each candidate metric, valued across all metrics, to show the ranking is not objective-cherry-picked | `objective_sensitivity.md` |
| `deflated_sharpe_ratio` / `dsr_for_study` | P(true Sharpe > 0) after deflating for the number of tested configs (Bailey & Lopez de Prado) | `hpo_dsr.md` |
| `pbo_cscv` | Probability of Backtest Overfitting via Combinatorially Symmetric CV | `hpo_pbo.md` |
| `multiseed_reeval` | Top-N configs re-run over several global seeds; mean/std per metric, to guard DL rankings against seed sensitivity | `hpo_multiseed.md` |
| `apply_best_params` | Write each study's best params into `config.yaml` (comment- and anchor-preserving, validated via `yaml.safe_load`) | edits `config.yaml` |

All Markdown assets are embedded in [docs/statistics.md](./statistics.md)
(section G) and rendered on the dashboard Models page. The intended workflow is:
run the search, review convergence and edge-of-range flags, extend ranges if
needed, re-run, then `apply_best_params` and finally run the DSR/PBO/multi-seed
reports for the paper.

---

## 10. How to Run

The run is driven entirely from `config.yaml`; there are no API overrides, so a
run is reproducible from config alone.

### Via the Model Service API

| Endpoint | Effect |
|:---|:---|
| `POST /models/optimize/{model_name}` | Optimize a single model |
| `POST /models/optimize-all` | Optimize all five models sequentially |
| `POST /models/hpo-analysis?scope=cheap\|full` | Generate the post-HPO report assets |

Requires `walk_forward.enabled: true`. Results are persisted in
`models/optuna_studies.db`. The endpoints return `metric`, `best_score`,
`best_params` and `n_trials`. See
[docs/fastapi-endpoints.md](./fastapi-endpoints.md).

### Via the CLI (post-HPO analysis)

```bash
python -m src.backtest.hpo_analysis review
python -m src.backtest.hpo_analysis apply [--dry-run]
python -m src.backtest.hpo_analysis sensitivity [--save]
python -m src.backtest.hpo_analysis multiseed --model LSTM --top 5 --seeds 5
python -m src.backtest.hpo_analysis dsr --model LSTM
python -m src.backtest.hpo_analysis pbo --model LSTM --top 20
python -m src.backtest.hpo_analysis report --scope cheap|full
```

To switch the search on, set `optimization.enabled: true`. Once the best params
are in `config.yaml`, set it back to `false` so only the final walk-forward run
remains.

---

## 11. GPU Infrastructure

The DL search is GPU-bound. Native Windows TensorFlow has no GPU support since
2.10, so the HPO runs in the Linux Model Service container.

- The [Model Service Dockerfile](../services/model_service/Dockerfile) installs
  `tensorflow[and-cuda]` (which supplies the newer CUDA userspace libs) plus
  `torch` (cu124) with `--no-deps` so both frameworks share one ABI-compatible
  set of CUDA libraries. Only the driver comes from the host.
- [docker-compose.yml](../docker-compose.yml) reserves one NVIDIA GPU and sets
  `TF_FORCE_GPU_ALLOW_GROWTH=true` so TF does not grab all memory (LSTM/TF and
  Transformer/torch share the single GPU in one process).
- Studies run with `n_jobs = 1` on the GPU; parallelism is achieved by running
  the grid models on CPU cores alongside one DL study on the GPU.

---

## 12. Config Reference

All keys under `optimization:` in [config/config.yaml](../config/config.yaml):

| Key | Meaning |
|:---|:---|
| `enabled` | Run Optuna (`true`) or skip to the final run (`false`) |
| `metric` | Objective metric on the pooled OOS series (default `martin`) |
| `models` | Models to optimize |
| `grid_models` | Models searched exhaustively via GridSampler |
| `n_trials_per_model` | TPE budget per DL model (ignored for grid models) |
| `every_nth_fold_per_model` | Inner-CV fold subsampling (now `1` = all folds) |
| `sampler.seed` | Sampler seed (reproducibility) |
| `sampler.multivariate` / `group` | TPE multivariate sampling |
| `sampler.n_startup_trials` | Random exploration before TPE, per DL model |
| `pruning.enabled` | MedianPruner on/off (off for the headline run) |
| `pruning.n_startup_trials` / `n_warmup_steps` | Pruner warmup |
| `tune_until` | Selection cutoff date (development period) |
| `dl_max_epochs` | Epoch ceiling for DL training (with early stopping) |
| `study_suffix` | Versioned study-name suffix |
| `search_spaces` | Declarative per-model search spaces |
| `storage` | Optuna SQLite storage URL |

---

## Related Documents

- [docs/fastapi-endpoints.md](./fastapi-endpoints.md): API routes
- [docs/statistics.md](./statistics.md): auto-generated master report (section G embeds the HPO assets)
- [docs/how-to-add-ml-model.md](./how-to-add-ml-model.md): where hyperparameters are registered
- [Issue #5](https://github.com/Torim98/regime-switching-daa-paper/issues/5): design rationale and acceptance criteria
