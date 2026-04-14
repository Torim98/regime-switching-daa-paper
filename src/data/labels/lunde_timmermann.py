"""
Lunde & Timmermann (2004): Duration-Dependence-basiertes Regime-Labeling.

Referenz
--------
Lunde, A. & Timmermann, A. (2004). "Duration Dependence in Stock Prices."
JBES, 22(3), 253-273. DOI: 10.1198/073500104000000136

Algorithmus (vereinfachte Variante)
-----------------------------------
Asymmetrischer Schwellwert: lambda_bull fuer Aufwaertsbewegungen, lambda_bear
fuer Abwaertsbewegungen. Uebergang ausgeloest, sobald der kumulierte
Gegentrend seit dem letzten Extrempunkt den jeweiligen Schwellwert ueberschreitet.
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
    0/1-Label (1 = Bear) nach Lunde-Timmermann.

    Parameter
    ---------
    prices : pd.Series
        Close-Preise mit DatetimeIndex.
    lambda_bull : float, default 0.20
        Schwellwert: Drawdown vom Hoch aus, der einen Bull -> Bear-Wechsel auslöst.
    lambda_bear : float, default 0.15
        Schwellwert: Anstieg vom Tief aus, der einen Bear -> Bull-Wechsel auslöst.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices muss pd.Series sein.")
    if prices.isna().any():
        raise ValueError("prices enthält NaN-Werte.")

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