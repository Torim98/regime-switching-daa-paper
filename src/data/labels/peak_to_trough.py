"""
Peak-to-Trough-Rule: klassische 20%-Definition eines Bärenmarktes.

Referenz
--------
Industriekonvention (S&P Global, Ned Davis Research). Ein Bärenmarkt
beginnt, sobald der Index vom letzten Hoch um >= `threshold` (default 20%)
gefallen ist, und endet, sobald er vom letzten Tief um >= `threshold`
gestiegen ist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_peak_to_trough(
    prices: pd.Series,
    threshold: float = 0.20,
) -> pd.Series:
    """
    Erzeugt 0/1-Label (1 = Bear) nach Peak-to-Trough-Regel.

    Zustandsautomat:
    - State "Bull": track running max. Wenn Preis <= (1 - threshold) * max
      -> Bear-Start, Rückdatierung bis zum Max.
    - State "Bear": track running min. Wenn Preis >= (1 + threshold) * min
      -> Bull-Start, Rückdatierung bis zum Min.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices muss pd.Series sein.")
    if prices.isna().any():
        raise ValueError("prices enthält NaN-Werte.")

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
                # Bear-Signal: Rückdatierung ab letztem Peak
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