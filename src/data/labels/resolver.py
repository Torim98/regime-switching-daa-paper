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


def compute_supervised_labels_asof(
    df: pd.DataFrame,
    train_index: pd.DatetimeIndex,
    cfg,
) -> pd.Series:
    """Compute external labels using only information available at a fold cutoff.

    Turning-point labelers such as Pagan-Sossounov are ex-post algorithms: a
    centered extrema window can revise recent labels when later observations
    arrive.  Computing the label once on the complete data set and then slicing
    it into walk-forward folds would therefore leak test-period prices into the
    training targets.

    This helper evaluates the labeler on the complete history *through the last
    training observation only* and returns the labels aligned to ``train_index``.
    Keeping the pre-window history avoids artificial left-edge effects in a
    rolling training window while the cutoff prevents any test observation from
    influencing the targets.
    """
    if len(train_index) == 0:
        raise ValueError("train_index must not be empty.")

    train_index = pd.DatetimeIndex(train_index)
    cutoff = train_index.max()
    history = df.loc[df.index <= cutoff]
    if history.empty:
        raise ValueError(f"No observations available through cutoff {cutoff}.")

    labels = compute_supervised_labels(history, cfg).reindex(train_index)
    if labels.isna().any():
        missing = int(labels.isna().sum())
        raise ValueError(
            f"As-of supervised labels are missing for {missing} training rows "
            f"at cutoff {cutoff}."
        )
    return labels


def resolve_label_col(cfg) -> str:
    """
    Returns the column name that LSTM/Transformer use as `labels_col`.

    - "pagan_sossounov" / "peak_to_trough" -> "Supervised_Label"
    - "hmm"                                -> "HMM_Signal" (fallback, legacy)
    """
    if cfg.labels.supervised_label_source == "hmm":
        return "HMM_Signal"
    return "Supervised_Label"
