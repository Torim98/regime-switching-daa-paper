"""Unit tests for the signal smoothing / hysteresis layer (Issue #10).

Run: pytest src/backtest/test_smoothing.py

Covers the causal minimum-holding-period whipsaw filter and the confidence
buffer band (Schmitt trigger), plus NaN preservation and the no-op paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.smoothing import (
    apply_hysteresis,
    _enforce_min_holding,
    _apply_confidence_buffer,
)


def test_min_holding_extends_isolated_spike_immediately():
    # Dwell-lock semantics: a bear signal is adopted IMMEDIATELY (preserves
    # crisis response) and then locked for the holding period. An isolated
    # 1-day spike is therefore extended to a 5-day phase, not absorbed. This
    # is the deliberate trade-off; the confidence buffer suppresses such
    # low-conviction spikes at the source instead.
    raw = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=int)
    out = _enforce_min_holding(raw, 5)
    assert out.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 0]


def test_min_holding_allows_phase_at_least_as_long():
    # A 5-day bear phase reaches the holding period and must survive.
    raw = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=int)
    out = _enforce_min_holding(raw, 5)
    assert out.tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0]


def test_min_holding_locks_new_state_for_n_obs():
    # Switch to 1 at idx 2, then an immediate flip back at idx 3 is suppressed;
    # earliest allowed switch-back is idx 2 + 5 = 7.
    raw = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=int)
    out = _enforce_min_holding(raw, 5)
    assert out.tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0]


def test_min_holding_reduces_switch_count():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 2, size=500)
    n_raw = int((np.abs(np.diff(raw)) == 1).sum())
    out = _enforce_min_holding(raw, 5)
    n_out = int((np.abs(np.diff(out)) == 1).sum())
    assert n_out < n_raw


def test_confidence_buffer_holds_state_in_band():
    # threshold 0.5, buffer 0.2 -> ON >= 0.7, OFF < 0.3, hold in [0.3, 0.7).
    prob = np.array([0.9, 0.6, 0.4, 0.25, 0.5, 0.75])
    out = _apply_confidence_buffer(prob, seed_state=0, threshold=0.5, buffer=0.2)
    # 0.9 -> ON; 0.6 hold ON; 0.4 hold ON; 0.25 -> OFF; 0.5 hold OFF; 0.75 -> ON
    assert out.tolist() == [1, 1, 1, 0, 0, 1]


def test_confidence_buffer_reduces_switches_vs_plain_threshold():
    rng = np.random.default_rng(1)
    prob = np.clip(0.5 + rng.normal(0, 0.1, size=1000), 0, 1)
    plain = (prob >= 0.5).astype(int)
    buffered = _apply_confidence_buffer(prob, seed_state=int(plain[0]),
                                        threshold=0.5, buffer=0.1)
    n_plain = int((np.abs(np.diff(plain)) == 1).sum())
    n_buf = int((np.abs(np.diff(buffered)) == 1).sum())
    assert n_buf < n_plain


def test_apply_hysteresis_preserves_nan_positions():
    sig = pd.Series([np.nan, 0.0, 1.0, 0.0, np.nan, 1.0])
    out = apply_hysteresis(sig, min_holding_days=3)
    assert out.isna().tolist() == sig.isna().tolist()


def test_apply_hysteresis_noop_when_both_disabled():
    sig = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0])
    out = apply_hysteresis(sig, min_holding_days=0, confidence_buffer=0.0)
    pd.testing.assert_series_equal(out, sig)


def test_buffer_requires_prob_and_threshold():
    sig = pd.Series([0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        apply_hysteresis(sig, confidence_buffer=0.1)


def test_causality_prefix_invariance():
    # A causal filter: the smoothed value at t must not depend on future values.
    rng = np.random.default_rng(2)
    raw = rng.integers(0, 2, size=200).astype(float)
    full = apply_hysteresis(pd.Series(raw), min_holding_days=5)
    cut = 137
    prefix = apply_hysteresis(pd.Series(raw[:cut]), min_holding_days=5)
    np.testing.assert_array_equal(full.to_numpy()[:cut], prefix.to_numpy())
