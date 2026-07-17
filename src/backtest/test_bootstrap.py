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


# ---------------------------------------------------------------------------
# Common random numbers across strategies (paired Wilcoxon prerequisite)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["block", "stationary"])
def test_common_random_numbers_pairs_strategies(method):
    """Two strategies with identical returns and signals must yield identical
    path outcomes.

    This only holds if every (scenario, strategy) cell shares one bootstrap
    index stream (common random numbers). Under independent per-cell seeds the
    two columns would resample different trading days and their terminal-capital
    arrays would differ. Guards the pairing assumed by test_h1_drawdown and
    test_h2_transformer ("same bootstrap indices -> paired paths").
    """
    import pandas as pd

    from src.backtest.evaluation import run_monte_carlo_simulation

    n_source = 300
    idx = pd.date_range("2000-01-03", periods=n_source, freq="B")
    gen = np.random.default_rng(7)  # independent of the simulation seed
    rets = gen.normal(0.0004, 0.011, size=n_source)
    sigs = gen.integers(0, 2, size=n_source)

    daily_rets = pd.DataFrame({"A": rets, "B": rets}, index=idx)
    test_df = pd.DataFrame({"A_Signal": sigs, "B_Signal": sigs}, index=idx)
    scenarios = {"Standard": {"start": 100_000.0, "withdrawal": 500.0, "fee": 0.001}}

    _, mcs = run_monte_carlo_simulation(
        daily_rets=daily_rets,
        test_df=test_df,
        scenarios=scenarios,
        n_simulations=200,  # < 1000 -> sequential branch, no multiprocessing
        block_size=BLOCK_SIZE,
        random_seed=SEED,
        sim_years=1,
        trading_days_per_year=252,
        bootstrap_method=method,
    )

    np.testing.assert_array_equal(
        mcs.finals[("Standard", "A")], mcs.finals[("Standard", "B")]
    )
    np.testing.assert_array_equal(
        mcs.max_drawdowns[("Standard", "A")], mcs.max_drawdowns[("Standard", "B")]
    )


# ---------------------------------------------------------------------------
# Streaming statistics vs. full-history reference (memory refactoring guard)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["block", "stationary"])
def test_streaming_stats_match_full_history_reference(method):
    """The in-stream statistics of _simulate_strategy (terminal capital,
    path-wise MaxDD, plot subsample) must reproduce exactly what the former
    implementation derived post hoc from the fully materialized
    (n_simulations, total_days) capital matrix.
    """
    from src.backtest.evaluation import _run_strategy_job

    n_source = 300
    n_sim = 50
    total_days = 300
    n_keep = 7
    # 15 withdrawals of 2,500 on 40,000 start: most paths ruin, some survive,
    # so the ruin branch (-1.0 MaxDD, absorbing zero) is exercised.
    start_capital = 40_000.0
    withdrawal = 2_500.0
    fee = 0.001

    gen = np.random.default_rng(11)
    rets_arr = gen.normal(0.0002, 0.015, size=n_source)
    sig_arr = gen.integers(0, 2, size=n_source).astype(float)

    finals, maxdds, samples = _run_strategy_job(
        rets_arr, sig_arr, n_sim, total_days, BLOCK_SIZE,
        start_capital, withdrawal, fee,
        np.random.SeedSequence(SEED), method, n_keep,
    )

    # --- Reference: former implementation with a full capital matrix ---
    rng = np.random.default_rng(np.random.SeedSequence(SEED))
    idx = build_bootstrap_indices(
        method, n_source, n_sim, total_days, BLOCK_SIZE, rng,
    )
    sim_rets = rets_arr[idx]
    sim_sigs = sig_arr[idx]
    capitals = np.full(n_sim, start_capital, dtype=np.float64)
    histories = np.empty((n_sim, total_days), dtype=np.float64)
    ruined = np.zeros(n_sim, dtype=bool)
    for i in range(total_days):
        capitals *= (1 + sim_rets[:, i])
        if i % 21 == 0:
            amt = np.full(n_sim, withdrawal)
            amt[sim_sigs[:, i] == 0] += withdrawal * fee
            capitals -= amt
        newly = (capitals <= 0) & ~ruined
        capitals[newly] = 0.0
        ruined |= newly
        capitals[ruined] = 0.0
        histories[:, i] = capitals

    ref_finals = histories[:, -1]
    cummax = np.maximum.accumulate(histories, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(cummax > 0, histories / cummax - 1, -1.0)
    ref_maxdds = dd.min(axis=1)

    assert np.any(ref_finals <= 0), "test setup should produce ruined paths"
    np.testing.assert_array_equal(finals, ref_finals)
    np.testing.assert_array_equal(maxdds, ref_maxdds)
    assert samples.shape == (n_keep, total_days)
    np.testing.assert_array_equal(samples, histories[:n_keep])
