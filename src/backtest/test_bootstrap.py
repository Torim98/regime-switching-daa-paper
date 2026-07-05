"""Unit tests for the MCS bootstrap index builders (Issue #7).

Run: pytest src/backtest/test_bootstrap.py

Covers the stationary bootstrap (Politis & Romano 1994) added alongside the
existing fixed-length block bootstrap, plus the shared dispatch helper.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.backtest.evaluation import (
    _block_bootstrap_indices,
    _stationary_bootstrap_indices,
    build_bootstrap_indices,
    compare_bootstrap_methods,
    bootstrap_robustness_summary,
)

N_SOURCE = 500
N_SIMULATIONS = 200
TOTAL_DAYS = 2520          # 10y x 252 trading days, as in the MCS config
BLOCK_SIZE = 20
SEED = 42


def _mean_realized_block_length(indices: np.ndarray, n_source: int) -> float:
    """Mean length of runs of consecutive (circular +1) indices."""
    # A new block starts wherever the step is not exactly +1 modulo n_source.
    steps = (indices[:, 1:] - indices[:, :-1]) % n_source
    restarts = int(np.sum(steps != 1))
    n_blocks = restarts + indices.shape[0]  # one block start per simulation
    return indices.size / n_blocks


@pytest.mark.parametrize("method", ["block", "stationary"])
def test_shape_is_correct(method):
    rng = np.random.default_rng(SEED)
    idx = build_bootstrap_indices(
        method, N_SOURCE, N_SIMULATIONS, TOTAL_DAYS, BLOCK_SIZE, rng,
    )
    assert idx.shape == (N_SIMULATIONS, TOTAL_DAYS)


@pytest.mark.parametrize("method", ["block", "stationary"])
def test_indices_in_bounds(method):
    rng = np.random.default_rng(SEED)
    idx = build_bootstrap_indices(
        method, N_SOURCE, N_SIMULATIONS, TOTAL_DAYS, BLOCK_SIZE, rng,
    )
    assert idx.min() >= 0
    assert idx.max() < N_SOURCE


def test_stationary_mean_block_length_near_target():
    rng = np.random.default_rng(SEED)
    idx = _stationary_bootstrap_indices(
        N_SOURCE, N_SIMULATIONS, TOTAL_DAYS, BLOCK_SIZE, rng,
    )
    mean_len = _mean_realized_block_length(idx, N_SOURCE)
    # Geometric block lengths have mean == block_size; allow a generous band.
    assert 0.5 * BLOCK_SIZE <= mean_len <= 1.5 * BLOCK_SIZE


def test_block_method_is_deterministic_for_seed():
    """Default 'block' behaviour must be reproducible for a fixed seed."""
    a = _block_bootstrap_indices(
        N_SOURCE, N_SIMULATIONS, TOTAL_DAYS, BLOCK_SIZE,
        np.random.default_rng(SEED),
    )
    b = _block_bootstrap_indices(
        N_SOURCE, N_SIMULATIONS, TOTAL_DAYS, BLOCK_SIZE,
        np.random.default_rng(SEED),
    )
    np.testing.assert_array_equal(a, b)


def test_stationary_method_is_deterministic_for_seed():
    a = _stationary_bootstrap_indices(
        N_SOURCE, N_SIMULATIONS, TOTAL_DAYS, BLOCK_SIZE,
        np.random.default_rng(SEED),
    )
    b = _stationary_bootstrap_indices(
        N_SOURCE, N_SIMULATIONS, TOTAL_DAYS, BLOCK_SIZE,
        np.random.default_rng(SEED),
    )
    np.testing.assert_array_equal(a, b)


def test_unknown_method_raises():
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError):
        build_bootstrap_indices(
            "invalid", N_SOURCE, N_SIMULATIONS, TOTAL_DAYS, BLOCK_SIZE, rng,
        )


# ---------------------------------------------------------------------------
# Robustness comparison (block vs. stationary)
# ---------------------------------------------------------------------------
def _toy_finals(depletion_fraction: float, median_value: float, n: int = 1000):
    """One (scenario, strategy) cell: `depletion_fraction` zeros, rest constant."""
    n_ruin = int(round(depletion_fraction * n))
    arr = np.concatenate([np.zeros(n_ruin), np.full(n - n_ruin, median_value)])
    return {("Standard", "HMM"): arr}


def test_compare_bootstrap_methods_columns_and_delta():
    block = _toy_finals(0.10, 100_000.0)
    stationary = _toy_finals(0.12, 90_000.0)
    df = compare_bootstrap_methods(block, stationary)

    assert df.index.tolist() == [("Standard", "HMM")]
    for col in ("Depletion (Block)", "Depletion (Stationary)",
                "Delta Depletion (pp)", "Median Final (Block)",
                "Median Final (Stationary)", "Delta Median (EUR)"):
        assert col in df.columns

    row = df.loc[("Standard", "HMM")]
    assert row["Depletion (Block)"] == "10.00%"
    assert row["Depletion (Stationary)"] == "12.00%"
    # Stationary depletes 2 pp more than block.
    assert row["Delta Depletion (pp)"] == "+2.00"


def test_compare_bootstrap_methods_only_shared_cells():
    block = {("Standard", "HMM"): np.full(100, 1.0),
             ("Standard", "MSM"): np.full(100, 1.0)}
    stationary = {("Standard", "HMM"): np.full(100, 1.0)}
    df = compare_bootstrap_methods(block, stationary)
    assert df.index.tolist() == [("Standard", "HMM")]


def test_robustness_summary_mentions_metrics():
    block = {
        ("Standard", "Buy_Hold"): _toy_finals(0.20, 50_000.0)[("Standard", "HMM")],
        ("Standard", "HMM"): _toy_finals(0.05, 100_000.0)[("Standard", "HMM")],
    }
    stationary = {
        ("Standard", "Buy_Hold"): _toy_finals(0.22, 48_000.0)[("Standard", "HMM")],
        ("Standard", "HMM"): _toy_finals(0.06, 98_000.0)[("Standard", "HMM")],
    }
    summary = bootstrap_robustness_summary(block, stationary)
    assert "pp" in summary
    # HMM beats Buy_Hold under both methods -> lead sign preserved.
    assert "1/1 scenarios" in summary
