"""Regression tests for leakage-free fold label construction."""

import numpy as np
import pandas as pd

from config.config_loader import PipelineConfig
from src.data.labels.resolver import compute_supervised_labels_asof


def test_asof_labels_are_invariant_to_test_period_prices():
    cfg = PipelineConfig()
    index = pd.date_range("2010-01-01", periods=900, freq="B")
    base = np.exp(np.cumsum(0.0002 + 0.01 * np.sin(np.arange(900) / 30)))
    df = pd.DataFrame({"Cumulative_Returns": base}, index=index)
    train_index = index[:650]

    original = compute_supervised_labels_asof(df, train_index, cfg)

    changed_future = df.copy()
    changed_future.loc[index[650]:, "Cumulative_Returns"] *= np.linspace(1, 0.2, 250)
    after_future_change = compute_supervised_labels_asof(
        changed_future, train_index, cfg,
    )

    pd.testing.assert_series_equal(original, after_future_change)

