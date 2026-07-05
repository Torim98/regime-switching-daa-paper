"""Signal smoothing / hysteresis layer (Issue #10 ablation).

Post-processes the raw walk-forward regime signals to reduce high-frequency
churning without touching the models themselves. Two independent, causal
mechanisms that can be combined:

- Minimum holding period (dwell-time lock): a regime change is adopted
  immediately (so a bear signal still de-risks without delay, preserving the
  crisis response), but the new regime is then held for at least
  ``min_holding_days`` observations before another switch is accepted. This cuts
  the rapid flip-flopping that dominates the churning cost. An isolated one-day
  spike is extended to the holding length rather than absorbed; use the
  confidence buffer to suppress such low-conviction spikes at the source.
- Confidence buffer band (Schmitt trigger): the signal only turns ON when the
  regime probability rises to ``threshold + confidence_buffer`` and only turns
  OFF when it falls below ``threshold - confidence_buffer``; between the two
  bounds it keeps its previous state. This damps flip-flopping around a
  threshold that the probability hugs.

Both mechanisms are strictly causal (they only look at past and current
observations), so applying them cannot introduce look-ahead bias.

Set ``backtesting.signal_smoothing.enabled: false`` in config.yaml to restore
the un-smoothed baseline; the raw walk-forward signals are then passed through
unchanged.
"""

import numpy as np
import pandas as pd


# Mapping of the public model names (as used in the *_Signal columns) to the
# cfg.models.<key> namespace that carries the per-model decision threshold.
_MODEL_CFG_KEY = {
    "MSM": "msm",
    "HMM": "hmm",
    "HMM_Uni": "hmm_uni",
    "LSTM": "lstm",
    "Transformer": "transformer",
}


def _apply_confidence_buffer(
    prob: np.ndarray,
    seed_state: int,
    threshold: float,
    buffer: float,
) -> np.ndarray:
    """Schmitt trigger on the probability series.

    Turns ON at ``threshold + buffer`` and OFF at ``threshold - buffer``; holds
    the previous state in between. ``seed_state`` initializes the state for the
    first observation.
    """
    upper = threshold + buffer
    lower = threshold - buffer
    out = np.empty(len(prob), dtype=int)
    state = int(seed_state)
    for i in range(len(prob)):
        p = prob[i]
        if p >= upper:
            state = 1
        elif p < lower:
            state = 0
        # otherwise: keep the previous state
        out[i] = state
    return out


def _enforce_min_holding(signal: np.ndarray, min_holding_days: int) -> np.ndarray:
    """Dwell-time lock: enforce a minimum time between regime switches.

    Causal: a switch is adopted immediately, then the new state is locked for
    ``min_holding_days`` observations before another switch is accepted. Regime
    changes are never delayed (crisis response is preserved); only rapid
    switch-backs within the holding window are suppressed.
    """
    if min_holding_days <= 1 or len(signal) == 0:
        return signal.astype(int)

    out = np.empty(len(signal), dtype=int)
    current = int(signal[0])
    out[0] = current
    hold = min_holding_days  # allow the first switch immediately
    for i in range(1, len(signal)):
        raw = int(signal[i])
        if raw != current and hold >= min_holding_days:
            current = raw
            hold = 1
        else:
            hold += 1
        out[i] = current
    return out


def apply_hysteresis(
    signal: pd.Series,
    prob: pd.Series | None = None,
    threshold: float | None = None,
    min_holding_days: int = 0,
    confidence_buffer: float = 0.0,
) -> pd.Series:
    """Apply the hysteresis mechanisms to a single binary regime signal.

    NaN entries (e.g. DL warm-up rows at fold starts) are preserved in place;
    smoothing operates only on the contiguous run-length of valid observations.

    Parameters
    ----------
    signal : pd.Series
        Raw binary regime signal (1 = bear/defensive, 0 = invested). May contain NaN.
    prob : pd.Series, optional
        Regime probability aligned to ``signal``. Required only when
        ``confidence_buffer > 0``.
    threshold : float, optional
        Decision threshold the raw signal was derived from. Required only when
        ``confidence_buffer > 0``.
    min_holding_days : int
        Minimum holding period in observations (0 or 1 = disabled).
    confidence_buffer : float
        Half-width of the Schmitt-trigger buffer band (0.0 = disabled).

    Returns
    -------
    pd.Series
        Smoothed signal, same index and NaN positions as the input.
    """
    result = signal.copy()
    valid_mask = signal.notna()
    if not valid_mask.any():
        return result

    sig_valid = signal[valid_mask].astype(int).to_numpy()

    # 1. Confidence buffer band (needs prob + threshold)
    if confidence_buffer and confidence_buffer > 0.0:
        if prob is None or threshold is None:
            raise ValueError(
                "confidence_buffer > 0 requires both `prob` and `threshold`."
            )
        prob_valid = prob[valid_mask].astype(float).to_numpy()
        sig_valid = _apply_confidence_buffer(
            prob_valid, seed_state=sig_valid[0],
            threshold=threshold, buffer=confidence_buffer,
        )

    # 2. Minimum holding period
    if min_holding_days and min_holding_days > 1:
        sig_valid = _enforce_min_holding(sig_valid, min_holding_days)

    result.loc[valid_mask] = sig_valid.astype(float)
    return result


def smooth_signal_columns(
    test_df: pd.DataFrame,
    cfg,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Apply the configured hysteresis to every ``<model>_Signal`` column.

    Reads ``cfg.backtesting.signal_smoothing``. If smoothing is disabled (or the
    block is absent), the DataFrame is returned unchanged. Per-model decision
    thresholds for the confidence buffer band are taken from
    ``cfg.models.<key>.threshold``.

    Returns a new DataFrame; the input is not mutated.
    """
    smoothing = getattr(cfg.backtesting, "signal_smoothing", None)
    if smoothing is None or not getattr(smoothing, "enabled", False):
        return test_df

    min_holding_days = int(getattr(smoothing, "min_holding_days", 0) or 0)
    confidence_buffer = float(getattr(smoothing, "confidence_buffer", 0.0) or 0.0)

    if min_holding_days <= 1 and confidence_buffer <= 0.0:
        # Nothing to do: enabled but both mechanisms are no-ops.
        return test_df

    if models is None:
        models = [
            c.rsplit("_", 1)[0]
            for c in test_df.columns if c.endswith("_Signal")
        ]

    out = test_df.copy()
    for m in models:
        sig_col = f"{m}_Signal"
        if sig_col not in out.columns:
            continue

        prob_col = f"{m}_Prob"
        prob = out[prob_col] if prob_col in out.columns else None

        threshold = None
        cfg_key = _MODEL_CFG_KEY.get(m)
        if cfg_key is not None:
            model_cfg = getattr(cfg.models, cfg_key, None)
            if model_cfg is not None:
                threshold = getattr(model_cfg, "threshold", None)

        out[sig_col] = apply_hysteresis(
            out[sig_col],
            prob=prob,
            threshold=threshold,
            min_holding_days=min_holding_days,
            confidence_buffer=confidence_buffer,
        )

    return out
