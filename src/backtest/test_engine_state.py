"""Regression tests for fold-continuous backtest execution state."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import compute_strategy_log_returns


@pytest.mark.parametrize("signal_shift", [0, 1, 2])
def test_folded_returns_equal_one_continuous_backtest(signal_shift):
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    df = pd.DataFrame(
        {
            "Returns": np.linspace(-0.01, 0.012, len(index)),
            "Cash_Returns": np.full(len(index), 0.0001),
        },
        index=index,
    )
    signal = pd.Series([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1], index=index)

    continuous, _ = compute_strategy_log_returns(
        df, signal, signal_shift=signal_shift, fee=0.001,
    )

    state = None
    folded = []
    for positions in (slice(0, 4), slice(4, 9), slice(9, None)):
        fold_returns, state = compute_strategy_log_returns(
            df.iloc[positions], signal.iloc[positions],
            signal_shift=signal_shift, fee=0.001, state=state,
        )
        folded.append(fold_returns)

    np.testing.assert_allclose(np.concatenate(folded), continuous, atol=0, rtol=0)

