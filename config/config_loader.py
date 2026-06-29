"""
Central Configuration Loader for the Regime-Switching DAA Pipeline.

Usage in services:
    import sys; sys.path.insert(0, "../config")
    from config_loader import cfg

Usage in FastAPI services:
    from config.config_loader import PipelineConfig
    cfg = PipelineConfig()          # auto-detects project root
    # or explicitly for Docker:
    cfg = PipelineConfig(base_dir="/app")

    # Access parameters:
    cfg.data.tickers
    cfg.backtesting.transaction_cost_bps
    cfg.transaction_cost_rate        # convenience: 0.001
    cfg.data_path("preprocessed")    # "<base_dir>/data/silver/02_preprocessed_data.parquet"
    cfg.asset_path("equity_curves")  # "<base_dir>/assets/equity_curves.png"
    cfg.model_path("lstm")           # "<base_dir>/models/lstm_regime_model.keras"
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

    def __init__(
        self,
        config_path: str | Path | None = None,
        base_dir: str | Path | None = None,
    ):
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

        # base_dir = project root for resolving data/assets/models paths
        # Default: parent of config/ directory (works for services and local runs)
        # Override: set explicitly for Docker (e.g. base_dir="/app")
        if base_dir is not None:
            self._base_dir = Path(base_dir)
        else:
            self._base_dir = self._path.parent.parent

        ns = _dict_to_namespace(self._raw.copy())
        self.data = ns.data
        self.features = ns.features
        self.portfolio = ns.portfolio
        self.models = ns.models
        self.optimization = ns.optimization
        self.backtesting = ns.backtesting
        self.walk_forward = ns.walk_forward
        self.evaluation = ns.evaluation
        self.paths = ns.paths
        self.plotting = ns.plotting
        self.fast_mode = ns.fast_mode
        self.model_persistence = ns.model_persistence
        self.labels = ns.labels

        # --- End-Date-Auflösung ---------------------------------------------
        # Normalisierung: None / "" / whitespace  -> dynamisch (heute)
        # Expliziter "YYYY-MM-DD"-String          -> thesis-freeze (inklusiv)
        raw_end = self.data.end_date
        if raw_end is None or (isinstance(raw_end, str) and not raw_end.strip()):
            self.data.end_date = datetime.now().strftime("%Y-%m-%d")
            self.data.end_date_is_frozen = False
        else:
            self.data.end_date = str(raw_end).strip()
            self.data.end_date_is_frozen = True

        # Apply fast_mode overrides
        if self.fast_mode.enabled:
            self.models.lstm.epochs = self.fast_mode.overrides.lstm_epochs
            self.models.transformer.epochs = self.fast_mode.overrides.transformer_epochs
            self.evaluation.mcs.n_paths = self.fast_mode.overrides.mcs_n_paths
            self.walk_forward.test_window_months = self.fast_mode.overrides.walk_forward_test_window_months
            self.walk_forward.step_months = self.fast_mode.overrides.walk_forward_step_months

    def data_path(self, key: str) -> str:
        """Full path to a data file: cfg.data_path("preprocessed")"""
        filename = getattr(self.paths.files, key)
        return str(self._base_dir / "data" / filename)

    def asset_path(self, key: str) -> str:
        """Full path to an asset file: cfg.asset_path("equity_curves")"""
        filename = getattr(self.paths.assets, key)
        return str(self._base_dir / "assets" / filename)

    def model_path(self, key: str) -> str:
        """Full path to a model file: cfg.model_path("lstm")"""
        filename = getattr(self.model_persistence.files, key)
        return str(self._base_dir / "models" / filename)

    @property
    def transaction_cost_rate(self) -> float:
        """Transaction cost as decimal (10 bps -> 0.001)."""
        return self.backtesting.transaction_cost_bps / 10_000

    @property
    def color_map(self) -> dict:
        """Model → color mapping from config.plotting.colors."""
        return {k: v for k, v in vars(self.plotting.colors).items()}

    def __repr__(self) -> str:
        return f"PipelineConfig(source='{self._path}', base_dir='{self._base_dir}')"


# Singleton — importable from any service
cfg = PipelineConfig()