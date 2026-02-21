"""
Central Configuration Loader for the Regime-Switching DAA Pipeline.

Usage in any notebook:
    import sys; sys.path.insert(0, "../config")
    from config_loader import cfg

    # Access parameters:
    cfg.data.tickers
    cfg.backtesting.transaction_cost_bps
    cfg.transaction_cost_rate        # convenience: 0.001
    cfg.data_path("preprocessed")    # "../data/01_preprocessed_data.parquet"
    cfg.asset_path("equity_curves")  # "../assets/equity_curves.png"
"""

import yaml
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for dot-access."""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = _dict_to_namespace(value)
        elif isinstance(value, list):
            d[key] = [
                _dict_to_namespace(item) if isinstance(item, dict) else item
                for item in value
            ]
    return SimpleNamespace(**d)


class PipelineConfig:
    """Wrapper around the YAML config with convenience methods."""

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            candidates = [
                Path(__file__).parent / "config.yaml",
                Path("../config/config.yaml"),
                Path("config/config.yaml"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    config_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    "config.yaml not found. Searched: "
                    + ", ".join(str(c) for c in candidates)
                )

        self._path = Path(config_path)
        with open(self._path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

        ns = _dict_to_namespace(self._raw.copy())
        self.dependencies = ns.dependencies
        self.data = ns.data
        self.features = ns.features
        self.portfolio = ns.portfolio
        self.models = ns.models
        self.backtesting = ns.backtesting
        self.evaluation = ns.evaluation
        self.paths = ns.paths
        self.pipeline = ns.pipeline
        self.fast_mode = ns.fast_mode

        # Dynamic end_date
        if self.data.end_date is None:
            self.data.end_date = datetime.now().strftime("%Y-%m-%d")

        # Apply fast_mode overrides
        if self.fast_mode.enabled:
            self.models.lstm.epochs = self.fast_mode.overrides.lstm_epochs
            self.models.lstm_unsupervised.epochs = self.fast_mode.overrides.lstm_unsupervised_epochs
            self.evaluation.mcs.n_paths = self.fast_mode.overrides.mcs_n_paths

    def data_path(self, key: str) -> str:
        """Full path to a data file: cfg.data_path("preprocessed")"""
        filename = getattr(self.paths.files, key)
        return str(Path(self.paths.data_dir) / filename)

    def asset_path(self, key: str) -> str:
        """Full path to an asset file: cfg.asset_path("equity_curves")"""
        filename = getattr(self.paths.assets, key)
        return str(Path(self.paths.assets_dir) / filename)

    @property
    def transaction_cost_rate(self) -> float:
        """Transaction cost as decimal (10 bps -> 0.001)."""
        return self.backtesting.transaction_cost_bps / 10_000

    def __repr__(self) -> str:
        return f"PipelineConfig(source='{self._path}')"


# Singleton — importable from any notebook
cfg = PipelineConfig()