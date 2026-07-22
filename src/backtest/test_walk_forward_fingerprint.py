"""Cache fingerprint must cover configuration and complete input values."""

from copy import deepcopy

import pandas as pd

from config.config_loader import PipelineConfig
from src.backtest.walk_forward import _walk_forward_fingerprint


def _sample_frame():
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    return pd.DataFrame(
        {"Returns": [0.01, -0.02, 0.03, 0.0], "VIX": [14.0, 18.0, 16.0, 15.0]},
        index=index,
    )


def test_fingerprint_is_deterministic_and_value_sensitive():
    cfg = PipelineConfig()
    df = _sample_frame()
    first = _walk_forward_fingerprint(cfg, df)
    assert first == _walk_forward_fingerprint(cfg, df.copy())

    changed = df.copy()
    changed.iloc[1, 0] += 1e-12
    assert _walk_forward_fingerprint(cfg, changed) != first


def test_fingerprint_covers_previously_missing_model_parameters():
    cfg = PipelineConfig()
    df = _sample_frame()
    first = _walk_forward_fingerprint(cfg, df)

    changed = deepcopy(cfg)
    changed.models.lstm.learning_rate *= 2
    assert _walk_forward_fingerprint(changed, df) != first

    changed = deepcopy(cfg)
    changed.models.transformer.n_heads *= 2
    assert _walk_forward_fingerprint(changed, df) != first

