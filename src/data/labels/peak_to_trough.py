"""
Peak-to-trough rule: classic 20% definition of a bear market.

Reference
---------
Industry convention (S&P Global, Ned Davis Research). A bear market
begins once the index has fallen by >= `threshold` (default 20%) from
its last high and ends once it has risen by >= `threshold` from its
last low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_peak_to_trough(
    prices: pd.Series,
    threshold: float = 0.20,
) -> pd.Series:
    """
    Produces a 0/1 label (1 = bear) per the peak-to-trough rule.

    State machine:
    - State "bull": track the running max. If price <= (1 - threshold) * max
      -> bear start, backdated to the max.
    - State "bear": track the running min. If price >= (1 + threshold) * min
      -> bull start, backdated to the min.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pd.Series.")
    if prices.isna().any():
        raise ValueError("prices contains NaN values.")

    n = len(prices)
    values = prices.values
    labels = np.zeros(n, dtype=np.int8)

    state = "bull"
    extreme_price = values[0]
    extreme_idx = 0

    for i in range(1, n):
        p = values[i]

        if state == "bull":
            if p > extreme_price:
                extreme_price = p
                extreme_idx = i
            elif p <= (1 - threshold) * extreme_price:
                # Bear signal: backdate from the last peak
                labels[extreme_idx + 1:i + 1] = 1
                state = "bear"
                extreme_price = p
                extreme_idx = i
            else:
                labels[i] = 0
        else:  # bear
            if p < extreme_price:
                extreme_price = p
                extreme_idx = i
                labels[i] = 1
            elif p >= (1 + threshold) * extreme_price:
                labels[extreme_idx + 1:i + 1] = 0
                state = "bull"
                extreme_price = p
                extreme_idx = i
            else:
                labels[i] = 1

    return pd.Series(labels, index=prices.index, name="P2T_Signal", dtype="int8")
