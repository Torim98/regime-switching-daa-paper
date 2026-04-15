import pandas as pd, numpy as np, pytest
from pathlib import Path
from config.config_loader import PipelineConfig

@pytest.mark.slow
def test_cpu_models_match_baseline_snapshot():
    """Parallel-Run muss bit-identische MSM/HMM-Outputs zum Baseline-Snapshot liefern."""
    cfg = PipelineConfig()
    baseline_path = Path(cfg.data_path("walk_forward_cache").replace(
        "wf_cache.parquet", "wf_cache.baseline.parquet",
    ))
    if not baseline_path.exists():
        pytest.skip("Kein Baseline-Snapshot vorhanden.")

    baseline = pd.read_parquet(baseline_path)
    current  = pd.read_parquet(cfg.data_path("walk_forward_cache"))

    # MSM + HMM sind deterministisch (fester random_state) → bit-identisch
    for m in ("MSM", "HMM"):
        pd.testing.assert_series_equal(
            baseline[f"{m}_Signal"].dropna(),
            current[f"{m}_Signal"].dropna(),
            check_names=False,
        )
        np.testing.assert_allclose(
            baseline[f"{m}_Prob"].dropna().values,
            current[f"{m}_Prob"].dropna().values,
            rtol=1e-10, atol=1e-10,
        )

    # LSTM/Transformer: GPU-Nichtdeterminismus → nur grobe Konsistenz pruefen
    for m in ("LSTM", "Transformer"):
        sig_diff = (
            baseline[f"{m}_Signal"].dropna().astype(int)
            != current[f"{m}_Signal"].dropna().astype(int)
        ).mean()
        assert sig_diff < 0.05, (
            f"{m}: >5% Signal-Flips zwischen Baseline und Current — "
            f"Regression oder Label-Quelle geaendert?"
        )

@pytest.mark.slow
def test_parallel_reproducible_across_n_jobs():
    cfg = PipelineConfig()
    df = pd.read_parquet(cfg.data_path("feature_engineered"))
    from src.backtest.walk_forward import walk_forward_splits, run_walk_forward
    splits = walk_forward_splits(
        df.index, cfg.walk_forward.mode, cfg.walk_forward.train_window_years,
        cfg.walk_forward.test_window_months, cfg.walk_forward.step_months,
        cfg.walk_forward.min_train_years,
    )[:5]

    cfg.walk_forward.n_jobs = 1
    a = run_walk_forward(df, splits, cfg, ["MSM", "HMM"])
    cfg.walk_forward.n_jobs = 2
    b = run_walk_forward(df, splits, cfg, ["MSM", "HMM"])

    for m in ("MSM", "HMM"):
        np.testing.assert_allclose(
            a[f"{m}_Prob"].dropna().values,
            b[f"{m}_Prob"].dropna().values,
            rtol=1e-10, atol=1e-10,
        )