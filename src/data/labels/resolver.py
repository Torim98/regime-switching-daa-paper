"""Zentrale Auflösung der Supervised-Label-Quelle aus der Config."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_supervised_labels(df: pd.DataFrame, cfg) -> pd.Series:
    """
    Erzeugt externes Regime-Label fuer LSTM/Transformer-Training.

    Rueckgabe
    ---------
    pd.Series (int8 oder NaN)
        "Supervised_Label"-Serie mit gleichem Index wie df.
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
        raise ValueError(f"Unbekannte supervised_label_source: {source}")


def resolve_label_col(cfg) -> str:
    """
    Liefert den Spaltennamen, den LSTM/Transformer als `labels_col` nutzen.

    - "pagan_sossounov" / "peak_to_trough" -> "Supervised_Label"
    - "hmm"                                -> "HMM_Signal" (Fallback, Legacy)
    """
    if cfg.labels.supervised_label_source == "hmm":
        return "HMM_Signal"
    return "Supervised_Label"