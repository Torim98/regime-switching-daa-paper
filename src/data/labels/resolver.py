"""Central resolution of the supervised label source from the config."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_supervised_labels(df: pd.DataFrame, cfg) -> pd.Series:
    """
    Produces the external regime label for LSTM/Transformer training.

    Returns
    -------
    pd.Series (int8 or NaN)
        "Supervised_Label" series with the same index as df.
    """
    source = cfg.labels.supervised_label_source
    prices = df["Cumulative_Returns"]

    if source == "pagan_sossounov":
        from src.data.labels.pagan_sossounov import label_pagan_sossounov
        return label_pagan_sossounov(prices, **vars(cfg.labels.pagan_sossounov))
    elif source == "peak_to_trough":
        from src.data.labels.peak_to_trough import label_peak_to_trough
        return label_peak_to_trough(prices, **vars(cfg.labels.peak_to_trough))
    elif source == "hmm":
        return pd.Series(np.nan, index=df.index)
    else:
        raise ValueError(f"Unknown supervised_label_source: {source}")


def resolve_label_col(cfg) -> str:
    """
    Returns the column name that LSTM/Transformer use as `labels_col`.

    - "pagan_sossounov" / "peak_to_trough" -> "Supervised_Label"
    - "hmm"                                -> "HMM_Signal" (fallback, legacy)
    """
    if cfg.labels.supervised_label_source == "hmm":
        return "HMM_Signal"
    return "Supervised_Label"
