import numpy as np, pandas as pd
from types import SimpleNamespace
from services.model_service.routes import _walk_forward_status

KEYS = ["msm", "hmm", "hmm_uni", "lstm", "transformer"]


def _cfg(cache_path):
    """Minimal stand-in for PipelineConfig exposing only data_path()."""
    return SimpleNamespace(data_path=lambda key: str(cache_path))


def test_missing_cache_marks_all_untrained(tmp_path):
    cfg = _cfg(tmp_path / "wf_cache.parquet")  # file not created
    assert _walk_forward_status(cfg, KEYS) == {k: False for k in KEYS}


def test_populated_signal_column_marks_trained(tmp_path):
    cache = tmp_path / "wf_cache.parquet"
    # MSM fully trained, HMM present but every fold failed (all-NaN),
    # the remaining models never ran (columns absent).
    pd.DataFrame({
        "MSM_Signal": [0.0, 1.0, np.nan],
        "HMM_Signal": [np.nan, np.nan, np.nan],
    }).to_parquet(cache)

    status = _walk_forward_status(_cfg(cache), KEYS)
    assert status == {
        "msm": True,
        "hmm": False,
        "hmm_uni": False,
        "lstm": False,
        "transformer": False,
    }
