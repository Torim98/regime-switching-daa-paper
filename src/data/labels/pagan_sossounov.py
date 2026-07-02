"""
Pagan-Sossounov (2003) bull/bear market labeling.

Reference
---------
Pagan, A. R. & Sossounov, K. A. (2003). "A Simple Framework for Analysing
Bull and Bear Markets." Journal of Applied Econometrics, 18(1), 23-46.
DOI: 10.1002/jae.664

Algorithm (adaptation of the Bry-Boschan procedure for equity markets)
----------------------------------------------------------------------
1. Identify local extrema in a rolling window of ±`window_months`.
2. Enforce alternation (peak -> trough -> peak ...).
3. Duration filter: every phase must last >= `min_phase_months`.
4. Cycle filter: peak->peak and trough->trough distance >= `min_cycle_months`.
5. Amplitude filter: |log return| between adjacent extrema >= `amplitude_threshold`.
6. Bear = 1 between a peak and the following trough, otherwise bull = 0.

Determinism
-----------
Same price input -> same label. No random seeds, no global state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trading days per month (approximation, US market)
TRADING_DAYS_PER_MONTH = 21


def label_pagan_sossounov(
    prices: pd.Series,
    window_months: int = 8,
    min_phase_months: int = 4,
    min_cycle_months: int = 16,
    amplitude_threshold: float = 0.20,
) -> pd.Series:
    """
    Produces a binary regime label (0 = bull, 1 = bear) per Pagan & Sossounov (2003).

    Parameters
    ----------
    prices : pd.Series
        Close prices with a monotonically increasing DatetimeIndex. No NaN allowed.
    window_months : int, default 8
        Window half-width (in months) for identifying local extrema.
    min_phase_months : int, default 4
        Minimum duration of a bull or bear phase (in months).
    min_cycle_months : int, default 16
        Minimum duration of a full cycle peak->peak or trough->trough.
    amplitude_threshold : float, default 0.20
        Minimum |log return| between adjacent extrema (e.g. 0.20 = 20%).

    Returns
    -------
    pd.Series
        int8 series (0/1) with the same DatetimeIndex as `prices`.
        Name: "PagSoss_Signal".
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pd.Series.")
    if prices.isna().any():
        raise ValueError("prices contains NaN values.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices.index must be monotonically increasing.")
    if len(prices) < 2 * window_months * TRADING_DAYS_PER_MONTH:
        raise ValueError(
            f"Too few price observations ({len(prices)}) for window_months={window_months}."
        )

    window_days = window_months * TRADING_DAYS_PER_MONTH
    min_phase_days = min_phase_months * TRADING_DAYS_PER_MONTH
    min_cycle_days = min_cycle_months * TRADING_DAYS_PER_MONTH

    # Step 1: find local extrema within ±window_days
    extrema = _find_local_extrema(prices, window_days)

    # Step 2: enforce alternation
    extrema = _enforce_alternation(extrema)

    # Step 3: duration filter (every phase >= min_phase_days)
    extrema = _apply_phase_filter(extrema, min_phase_days)
    extrema = _enforce_alternation(extrema)

    # Step 4: cycle filter (peak-peak and trough-trough >= min_cycle_days)
    extrema = _apply_cycle_filter(extrema, min_cycle_days)
    extrema = _enforce_alternation(extrema)

    # Step 5: amplitude filter (|log return| >= amplitude_threshold)
    extrema = _apply_amplitude_filter(extrema, amplitude_threshold)
    extrema = _enforce_alternation(extrema)

    # Step 6: build the binary label series from the alternating extrema
    labels = _build_label_series(prices.index, extrema)

    return labels.rename("PagSoss_Signal").astype("int8")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_local_extrema(prices: pd.Series, window_days: int) -> list[tuple]:
    """
    Identifies local maxima and minima within a window of ±window_days.

    A point t is a local maximum if prices[t] = max(prices[t-w : t+w+1]).
    Analogously for minima.

    Returns
    -------
    list[tuple[pd.Timestamp, float, str]]
        Sorted list of (timestamp, price, type) with type in {"P", "T"}.
    """
    values = prices.values
    index = prices.index
    n = len(values)
    extrema = []

    # Rolling max/min via a centered window
    # (pandas' min_periods=1 ensures the edge regions are covered)
    series = prices
    roll_max = series.rolling(window=2 * window_days + 1, center=True, min_periods=1).max()
    roll_min = series.rolling(window=2 * window_days + 1, center=True, min_periods=1).min()

    for i in range(n):
        # Excluding the outer window_days/2 could be considered; here we
        # include edge extrema (important for fold labels, since the current
        # market phase often lies at the edge).
        if values[i] == roll_max.iloc[i]:
            extrema.append((index[i], values[i], "P"))
        elif values[i] == roll_min.iloc[i]:
            extrema.append((index[i], values[i], "T"))

    # Sort by time (deduplicates if price == max == min in a flat window)
    # On ties, prefer "P" over "T"; alternation handles the rest.
    extrema.sort(key=lambda x: (x[0], 0 if x[2] == "P" else 1))
    return extrema


def _enforce_alternation(extrema: list[tuple]) -> list[tuple]:
    """
    Enforces alternating peaks and troughs.

    For two consecutive peaks: remove the lower one.
    For two consecutive troughs: remove the higher one.
    """
    if len(extrema) < 2:
        return extrema

    cleaned: list[tuple] = []
    for ext in extrema:
        if not cleaned:
            cleaned.append(ext)
            continue

        prev = cleaned[-1]
        if ext[2] == prev[2]:
            # Same type -> keep the more extreme value
            if ext[2] == "P":
                if ext[1] >= prev[1]:
                    cleaned[-1] = ext
            else:  # "T"
                if ext[1] <= prev[1]:
                    cleaned[-1] = ext
        else:
            cleaned.append(ext)

    return cleaned


def _apply_phase_filter(extrema: list[tuple], min_phase_days: int) -> list[tuple]:
    """
    Removes extrema that belong to a phase that is too short (< min_phase_days).

    Strategy: iterate over adjacent extrema; if their distance is below
    the threshold, remove the "less extreme" of the two.
    """
    if len(extrema) < 2:
        return extrema

    changed = True
    while changed:
        changed = False
        for i in range(len(extrema) - 1):
            t0, _, _ = extrema[i]
            t1, _, _ = extrema[i + 1]
            duration = (t1 - t0).days
            if duration < min_phase_days:
                # Remove the middle, less pronounced candidate
                # Heuristic: remove the weaker extremum
                e0, e1 = extrema[i], extrema[i + 1]
                if e0[2] == "P" and e1[2] == "T":
                    # Peak followed by trough: remove both if both are weak,
                    # otherwise the weaker one. Pragmatic choice: remove both.
                    del extrema[i:i + 2]
                elif e0[2] == "T" and e1[2] == "P":
                    del extrema[i:i + 2]
                else:
                    # Same type (should not happen after alternation)
                    del extrema[i + 1]
                changed = True
                break
    return extrema


def _apply_cycle_filter(extrema: list[tuple], min_cycle_days: int) -> list[tuple]:
    """
    Removes extrema that lead to a cycle that is too short.

    A cycle is peak->peak or trough->trough (distance of 2 indices).
    """
    if len(extrema) < 3:
        return extrema

    changed = True
    while changed:
        changed = False
        for i in range(len(extrema) - 2):
            t0, _, type0 = extrema[i]
            t2, _, type2 = extrema[i + 2]
            if type0 != type2:
                continue  # effectively impossible after alternation
            cycle_days = (t2 - t0).days
            if cycle_days < min_cycle_days:
                # Remove the middle extremum + one of the two endpoints
                # (the weaker of the same-type endpoints).
                _, p0, _ = extrema[i]
                _, p2, _ = extrema[i + 2]
                if type0 == "P":
                    weaker_idx = i if p0 <= p2 else i + 2
                else:  # "T"
                    weaker_idx = i if p0 >= p2 else i + 2
                middle_idx = i + 1
                for idx in sorted([weaker_idx, middle_idx], reverse=True):
                    del extrema[idx]
                changed = True
                break
    return extrema


def _apply_amplitude_filter(
    extrema: list[tuple],
    amplitude_threshold: float,
) -> list[tuple]:
    """
    Removes extrema pairs whose |log return| is below the threshold.
    """
    if len(extrema) < 2:
        return extrema

    changed = True
    while changed:
        changed = False
        for i in range(len(extrema) - 1):
            _, p0, _ = extrema[i]
            _, p1, _ = extrema[i + 1]
            amplitude = abs(np.log(p1 / p0))
            if amplitude < amplitude_threshold:
                # Remove both extrema -> the surrounding phase is merged
                del extrema[i:i + 2]
                changed = True
                break
    return extrema


def _build_label_series(
    index: pd.DatetimeIndex,
    extrema: list[tuple],
) -> pd.Series:
    """
    Constructs a 0/1 series from alternating peaks/troughs.

    - Before the first extremum: initialize with the *opposite* state
      (before a peak the phase is bull, before a trough it is bear).
    - Between peak_t and the following trough: 1 (bear).
    - Between trough_t and the following peak: 0 (bull).
    - After the last extremum: the last state persists.
    """
    labels = pd.Series(0, index=index, dtype="int8")

    if not extrema:
        return labels

    # Before the first extremum
    first_ts, _, first_type = extrema[0]
    if first_type == "P":
        labels.loc[:first_ts] = 0  # phase before a peak = bull
    else:
        labels.loc[:first_ts] = 1  # phase before a trough = bear

    # Between extrema
    for i in range(len(extrema) - 1):
        t0, _, type0 = extrema[i]
        t1, _, _ = extrema[i + 1]
        segment = (index > t0) & (index <= t1)
        labels.loc[segment] = 1 if type0 == "P" else 0

    # After the last extremum
    last_ts, _, last_type = extrema[-1]
    labels.loc[index > last_ts] = 1 if last_type == "P" else 0

    return labels
