"""Data quality checks (Issue #2): coverage, missing values, adjustment plausibility.

Generates a Markdown report on the raw data (Bronze, BEFORE ffill/dropna) and
the effect of cleaning (Silver). Pure functions without config dependency;
the calling route (data_service) passes freeze metadata and writes the file.

Kept analogous to src/data/eda.py: DataFrame-based partial reports that are
assembled into the master report as Markdown tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError


# Price-based tickers (log-return plausibility) vs. level series (VIX/interest rates)
PRICE_TICKERS = ["^GSPC", "VUSTX"]


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _yfinance_version() -> str:
    """Installed yfinance version for the freeze documentation."""
    try:
        return version("yfinance")
    except PackageNotFoundError:
        return "unknown"


def _fmt_date(ts) -> str:
    """Timestamp -> 'YYYY-MM-DD' (robust against None/NaT)."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)) or pd.isna(ts):
        return "n/a"
    return pd.Timestamp(ts).date().isoformat()


def _longest_nan_run(series: pd.Series) -> int:
    """Longest contiguous run of missing values (in observations)."""
    isna = series.isna().to_numpy()
    if not isna.any():
        return 0
    max_run = run = 0
    for flag in isna:
        run = run + 1 if flag else 0
        if run > max_run:
            max_run = run
    return int(max_run)


def _robust_z(x: pd.Series) -> pd.Series:
    """MAD-based robust z-score (fat-tail-proof, consistent with RobustScaler)."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - med) / (1.4826 * mad)


# --------------------------------------------------------------------------- #
# Partial reports
# --------------------------------------------------------------------------- #
def coverage_report(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Coverage per ticker: period, observed vs. expected trading days.

    Expected trading days are approximated via `pd.bdate_range` (Mon-Fri).
    Exchange holidays are included, so the coverage is a conservative
    lower bound (below 100% even for gapless data due to holidays).
    """
    rows = []
    for col in raw_df.columns:
        valid = raw_df[col].dropna()
        if valid.empty:
            rows.append({
                "Ticker": col, "From": "n/a", "To": "n/a",
                "Obs. Days": 0, "Exp. Bd (Mon-Fri)": 0, "Coverage %": "n/a",
            })
            continue
        first, last = valid.index.min(), valid.index.max()
        expected = len(pd.bdate_range(first, last))
        observed = int(valid.shape[0])
        rows.append({
            "Ticker": col,
            "From": _fmt_date(first),
            "To": _fmt_date(last),
            "Obs. Days": observed,
            "Exp. Bd (Mon-Fri)": expected,
            "Coverage %": f"{100 * observed / expected:.2f}" if expected else "n/a",
        })
    return pd.DataFrame(rows).set_index("Ticker")


def missing_value_report(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Missing values per ticker in the RAW frame (before ffill/dropna).

    Makes the data loss that is silent in preprocessing.fill_missing_values() visible.
    """
    rows = []
    n = len(raw_df)
    for col in raw_df.columns:
        s = raw_df[col]
        n_missing = int(s.isna().sum())
        rows.append({
            "Ticker": col,
            "NaN (raw)": n_missing,
            "NaN %": f"{100 * n_missing / n:.3f}" if n else "n/a",
            "Longest Gap (Days)": _longest_nan_run(s),
            "First Value": _fmt_date(s.first_valid_index()),
            "Last Value": _fmt_date(s.last_valid_index()),
        })
    return pd.DataFrame(rows).set_index("Ticker")


def adjustment_jump_report(
    raw_df: pd.DataFrame,
    price_tickers: list[str] = PRICE_TICKERS,
    z_thresh: float = 8.0,
) -> pd.DataFrame:
    """Jump/outlier check on the log returns of the price series.

    Proxy for adjustment errors: a faulty split/dividend adjustment
    shows up as an implausible daily jump. Reported per ticker: the number of
    days with |robust z| > z_thresh and the largest daily move.
    """
    rows = []
    for col in price_tickers:
        if col not in raw_df.columns:
            continue
        s = raw_df[col].dropna()
        logret = np.log(s / s.shift(1)).dropna()
        if logret.empty:
            continue
        z = _robust_z(logret)
        rows.append({
            "Ticker": col,
            "Max. Abs. Daily Return": f"{logret.abs().max():.4f}",
            f"Outlier Days (z>{z_thresh:g})": int((z.abs() > z_thresh).sum()),
            "Largest Jump (Date)": _fmt_date(logret.abs().idxmax()),
        })
    return pd.DataFrame(rows).set_index("Ticker")


def worst_moves_report(
    raw_df: pd.DataFrame,
    price_tickers: list[str] = PRICE_TICKERS,
    top_n: int = 5,
) -> pd.DataFrame:
    """Top-N largest absolute daily moves per price series (plausibility view).

    Serves for cross-checking against known crisis days (e.g. 2008-10, 2020-03):
    jumps on known dates are plausible, isolated jumps are not.
    """
    rows = []
    for col in price_tickers:
        if col not in raw_df.columns:
            continue
        s = raw_df[col].dropna()
        logret = np.log(s / s.shift(1)).dropna()
        if logret.empty:
            continue
        top = logret.reindex(logret.abs().sort_values(ascending=False).index).head(top_n)
        for rank, (date, val) in enumerate(top.items(), start=1):
            rows.append({
                "Ticker": col,
                "Rank": rank,
                "Date": _fmt_date(date),
                "Log Return": f"{val:+.4f}",
            })
    return pd.DataFrame(rows).set_index("Ticker")


def cleaning_impact_report(
    raw_df: pd.DataFrame,
    preprocessed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Effect of cleaning: rows raw vs. preprocessed.

    The difference comprises start rows removed via dropna() (no ffill anchor)
    and the single row lost through the log-return construction (shift).
    """
    n_raw = len(raw_df)
    n_pre = len(preprocessed_df)
    dropped = n_raw - n_pre
    rows = [
        {"Metric": "Rows raw (Bronze)", "Value": n_raw},
        {"Metric": "Rows cleaned (Silver)", "Value": n_pre},
        {"Metric": "Removed (dropna + return shift)", "Value": dropped},
        {"Metric": "Removed %", "Value": f"{100 * dropped / n_raw:.3f}" if n_raw else "n/a"},
    ]
    return pd.DataFrame(rows).set_index("Metric")


# --------------------------------------------------------------------------- #
# Master report
# --------------------------------------------------------------------------- #
def build_data_quality_report(
    raw_df: pd.DataFrame,
    preprocessed_df: pd.DataFrame,
    *,
    freeze_date: str,
    is_frozen: bool,
    price_tickers: list[str] = PRICE_TICKERS,
) -> str:
    """Assemble the full data quality report as a Markdown string.

    Parameters
    ----------
    raw_df : Bronze frame (ticker columns, BEFORE ffill/dropna).
    preprocessed_df : Silver frame (after preprocess_pipeline).
    freeze_date : cfg.data.end_date (resolved end date).
    is_frozen : cfg.data.end_date_is_frozen (False = dynamic/rolling).
    """
    mode = "Freeze (fixed cutoff)" if is_frozen else "Rolling (dynamic = last trading day)"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    idx = raw_df.index
    span = f"{_fmt_date(idx.min())} to {_fmt_date(idx.max())}" if len(idx) else "n/a"

    # Compute partial reports once (reused for the verdict AND the sections)
    cov = coverage_report(raw_df)
    miss = missing_value_report(raw_df)
    cov_min = pd.to_numeric(cov["Coverage %"], errors="coerce").min()
    gap_max = int(miss["Longest Gap (Days)"].max()) if len(miss) else 0
    verdict = (
        f"Coverage ≥ {cov_min:.1f} % · max. gap {gap_max} days"
        if pd.notna(cov_min) else "n/a"
    )

    parts: list[str] = []
    parts.append("# Data Quality Report")
    parts.append("")
    parts.append(f"- **Status:** {verdict}")
    parts.append(f"- **Period (raw):** {span}")
    parts.append(f"- **End date mode:** {mode}")
    parts.append(f"- **Resolved end date:** `{freeze_date}`")
    parts.append(f"- **yfinance version:** `{_yfinance_version()}`")
    parts.append(f"- **Tickers:** {', '.join(raw_df.columns)}")
    parts.append(f"- **Generated at:** {ts}")
    parts.append("")
    parts.append("## 1. Coverage (Observed vs. Expected Trading Days)")
    parts.append(cov.to_markdown())
    parts.append("")
    parts.append(
        "_Note: expected trading days from `bdate_range` (Mon-Fri incl. holidays). "
        "~96-97% is the holiday-induced lower bound, not data loss._"
    )
    parts.append("")
    parts.append("## 2. Missing Values (Raw Frame, Before ffill/dropna)")
    parts.append(miss.to_markdown())
    parts.append("")
    parts.append("## 3. Adjustment Plausibility (Daily Jumps of the Price Series)")
    parts.append(adjustment_jump_report(raw_df, price_tickers).to_markdown())
    parts.append("")
    parts.append("## 4. Largest Daily Moves (Crisis Plausibility)")
    parts.append(worst_moves_report(raw_df, price_tickers).to_markdown())
    parts.append("")
    parts.append("## 5. Effect of Cleaning (Bronze → Silver)")
    parts.append(cleaning_impact_report(raw_df, preprocessed_df).to_markdown())
    parts.append("")
    return "\n".join(parts)
