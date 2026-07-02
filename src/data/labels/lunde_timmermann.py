"""
Lunde & Timmermann (2004): duration-dependence-based regime labeling.

Reference
---------
Lunde, A. & Timmermann, A. (2004). "Duration Dependence in Stock Prices."
JBES, 22(3), 253-273. DOI: 10.1198/073500104000000136

Algorithm (simplified variant)
------------------------------
Asymmetric threshold: lambda_bull for upward moves, lambda_bear for
downward moves. A transition is triggered once the cumulative
counter-trend since the last extremum exceeds the respective threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_lunde_timmermann(
    prices: pd.Series,
    lambda_bull: float = 0.20,
    lambda_bear: float = 0.15,
) -> pd.Series:
    """
    0/1 label (1 = bear) per Lunde-Timmermann.

    Parameters
    ----------
    prices : pd.Series
        Close prices with a DatetimeIndex.
    lambda_bull : float, default 0.20
        Threshold: drawdown from the high that triggers a bull -> bear switch.
    lambda_bear : float, default 0.15
        Threshold: rise from the low that triggers a bear -> bull switch.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pd.Series.")
    if prices.isna().any():
        raise ValueError("prices contains NaN values.")

    n = len(prices)
    values = prices.values
    labels = np.zeros(n, dtype=np.int8)

    state = "bull"
    peak = values[0]
    trough = values[0]
    peak_idx = 0
    trough_idx = 0

    for i in range(1, n):
        p = values[i]

        if state == "bull":
            if p > peak:
                peak = p
                peak_idx = i
            drawdown = (p - peak) / peak
            if drawdown <= -lambda_bull:
                labels[peak_idx + 1:i + 1] = 1
                state = "bear"
                trough = p
                trough_idx = i
        else:  # bear
            if p < trough:
                trough = p
                trough_idx = i
                labels[i] = 1
            else:
                labels[i] = 1
            runup = (p - trough) / trough
            if runup >= lambda_bear:
                labels[trough_idx + 1:i + 1] = 0
                state = "bull"
                peak = p
                peak_idx = i

    return pd.Series(labels, index=prices.index, name="LundeT_Signal", dtype="int8")
