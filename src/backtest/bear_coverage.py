"""
Bear-market coverage diagnostics for the walk-forward folds (Issue #8, part 1).

Motivation
----------
The fixed walk-forward configuration (10y train / 12m test / 12m step) raises the
question of whether the out-of-sample (OOS) test windows actually contain bear
markets. A 12-month fold can easily fall entirely inside a bull run, in which
case the fold contributes no crisis signal to the evaluation. This module
quantifies, per fold, how much bear-market exposure each OOS window (and its
training window) carries under the Pagan-Sossounov (2003) labeling scheme.

For every walk-forward fold it reports, for the OOS test window and the train
window separately:

- the share of days carrying a bear label,
- the number of bear phases that overlap the window,
- whether at least one COMPLETE bear phase (start and end) lies inside the window.

A "bear phase" is a maximal contiguous run of bear labels (label == 1) in the
Pagan-Sossounov series computed once over the full price history. A phase is
counted as "complete" for a window when both its first and its last bear day
fall inside that window (i.e. the peak-to-trough episode begins and ends within
the window). Note that a run touching the global data boundary cannot be
confirmed complete, which only affects the final fold; this is flagged in the
summary.

This is a descriptive diagnostic only. It does not change the pipeline; it just
documents the fold granularity discussed in `docs/limitations.md` (Section 5).

Reproduce with:
    python -m src.backtest.bear_coverage
"""

from __future__ import annotations

import pandas as pd

from src.backtest.walk_forward import walk_forward_splits
from src.data.labels.pagan_sossounov import label_pagan_sossounov


def _bear_phases(labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Extracts maximal contiguous bear phases (runs of label == 1).

    Returns
    -------
    list[tuple[pd.Timestamp, pd.Timestamp]]
        (start_ts, end_ts) for each bear run, in chronological order.
        start_ts is the first bear day of the run, end_ts the last.
    """
    is_bear = labels.astype(int).values == 1
    idx = labels.index
    phases: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    n = len(is_bear)
    i = 0
    while i < n:
        if is_bear[i]:
            j = i
            while j + 1 < n and is_bear[j + 1]:
                j += 1
            phases.append((idx[i], idx[j]))
            i = j + 1
        else:
            i += 1
    return phases


def _window_coverage(
    win_idx: pd.DatetimeIndex,
    labels: pd.Series,
    phases: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[float, int, bool]:
    """
    Computes the three coverage figures for a single window.

    Parameters
    ----------
    win_idx : pd.DatetimeIndex
        Trading days of the window (fold train or test index).
    labels : pd.Series
        Full bear/bull label series (0/1) indexed like the price history.
    phases : list[tuple]
        Bear phases from `_bear_phases` over the full series.

    Returns
    -------
    (bear_share, n_overlapping_phases, has_complete_phase)
    """
    win_start = win_idx.min()
    win_end = win_idx.max()

    bear_share = float(labels.loc[win_idx].mean())

    # A phase [s, e] overlaps [win_start, win_end] iff s <= win_end and e >= win_start.
    n_overlap = sum(1 for s, e in phases if s <= win_end and e >= win_start)

    # A phase is complete inside the window iff both endpoints fall inside it.
    has_complete = any(win_start <= s and e <= win_end for s, e in phases)

    return bear_share, n_overlap, has_complete


def compute_bear_coverage(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Builds the per-fold bear-coverage table.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered data with a monotone DatetimeIndex and a
        "Cumulative_Returns" column (used as the price series for labeling).
    cfg : PipelineConfig
        Provides `cfg.walk_forward.*` and `cfg.labels.pagan_sossounov`.

    Returns
    -------
    pd.DataFrame
        One row per fold, indexed by "Fold", with train/test coverage columns.
    """
    prices = df["Cumulative_Returns"].dropna()
    labels = label_pagan_sossounov(prices, **vars(cfg.labels.pagan_sossounov))
    phases = _bear_phases(labels)

    splits = walk_forward_splits(
        index=df.index,
        mode=cfg.walk_forward.mode,
        train_window_years=cfg.walk_forward.train_window_years,
        test_window_months=cfg.walk_forward.test_window_months,
        step_months=cfg.walk_forward.step_months,
        min_train_years=cfg.walk_forward.min_train_years,
    )

    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        tr_share, tr_overlap, tr_complete = _window_coverage(train_idx, labels, phases)
        te_share, te_overlap, te_complete = _window_coverage(test_idx, labels, phases)
        rows.append({
            "Fold": fold_id,
            "Test Start": test_idx.min().date().isoformat(),
            "Test End": test_idx.max().date().isoformat(),
            "Test Bear %": round(100.0 * te_share, 1),
            "Test Bear Phases": te_overlap,
            "Test Full Bear Phase": "Yes" if te_complete else "No",
            "Train Bear %": round(100.0 * tr_share, 1),
            "Train Bear Phases": tr_overlap,
            "Train Full Bear Phase": "Yes" if tr_complete else "No",
        })

    return pd.DataFrame(rows).set_index("Fold")


def _summary_lines(table: pd.DataFrame, min_phase_months: int = 4) -> list[str]:
    """Builds the 2 to 3 sentence summary shown below the table."""
    n_folds = len(table)
    n_test_complete = int((table["Test Full Bear Phase"] == "Yes").sum())
    n_test_any_bear = int((table["Test Bear Phases"] > 0).sum())
    n_test_zero_bear = int((table["Test Bear %"] == 0.0).sum())

    return [
        f"Across the {n_folds} walk-forward folds, {n_test_complete} OOS test windows "
        f"contain at least one complete Pagan-Sossounov bear phase (peak and trough "
        f"inside the 12-month window), while {n_test_any_bear} folds overlap at least "
        f"one bear phase and {n_test_zero_bear} folds carry no bear day at all.",
        f"Because the {min_phase_months}-month minimum-phase filter and the 12-month "
        "fold length rarely coincide, most crisis exposure enters the folds as partial "
        "(window-truncated) bear phases rather than as fully contained episodes, whereas "
        "every training window (10 years) spans several complete bear phases.",
        "A bear run that is still open at the global data boundary cannot be confirmed "
        "complete, so the classification is conservative for any fold whose window "
        "reaches the end of the sample.",
    ]


def render_bear_coverage_md(table: pd.DataFrame, min_phase_months: int = 4) -> str:
    """Renders the table plus summary as a Markdown document string."""
    md = table.to_markdown()
    summary = "\n\n".join(_summary_lines(table, min_phase_months))
    return f"{md}\n\n{summary}\n"


def generate_bear_coverage_report(cfg=None) -> str:
    """
    Loads the feature-engineered data, computes the coverage table, and writes
    it to `assets/bear_coverage.md`. Returns the output path.
    """
    if cfg is None:
        from config.config_loader import PipelineConfig
        cfg = PipelineConfig()

    df = pd.read_parquet(cfg.data_path("feature_engineered"))
    table = compute_bear_coverage(df, cfg)
    out_path = cfg.asset_path("bear_coverage")
    min_phase_months = cfg.labels.pagan_sossounov.min_phase_months
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_bear_coverage_md(table, min_phase_months))
    return out_path


if __name__ == "__main__":
    path = generate_bear_coverage_report()
    print(f"Bear-coverage report written: {path}")
