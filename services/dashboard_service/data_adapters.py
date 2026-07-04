"""Data adapters: Parquet/MD → Plotly JSON for the dashboard.

All endpoints are GETs under /api/*. The UI loads them via fetch() and
renders client-side with Plotly.js. No recomputation: the values come
from the artifacts written by the pipeline.
"""
from pathlib import Path
from functools import wraps
from typing import Callable
import hashlib
import json
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse

from config.config_loader import PipelineConfig

router = APIRouter(prefix="/api", tags=["data"])
logger = logging.getLogger("dashboard_service")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> PipelineConfig:
    return PipelineConfig()


def _read_parquet_or_404(
    path: str, hint: str, *, columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load Parquet or 404. With `columns`, read only the desired columns.
    Saves many seconds per request for large, column-rich artifacts
    (especially mcs_data, 3.7 GB)."""
    try:
        return pd.read_parquet(path, columns=columns)
    except FileNotFoundError:
        raise HTTPException(404, f"Artifact missing: {path} ({hint})")


def _list_parquet_columns_or_404(path: str, hint: str) -> list[str]:
    """Reads only the column names from the Parquet footer (virtually free)."""
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema_arrow.names)
    except (FileNotFoundError, OSError):
        raise HTTPException(404, f"Artifact missing: {path} ({hint})")


def _fig_to_json(fig: go.Figure) -> dict:
    """Plotly figure → dict. We use to_json() and parse it back so that
    numpy/datetime values are serialized correctly."""
    return json.loads(pio.to_json(fig))


# ---------------------------------------------------------------------------
# Chart JSON cache
#
# In-memory cache with mtime invalidation. Each chart endpoint refetches
# the artifacts only if at least one of the source Parquets has changed
# since the last call. The cache only survives the process lifecycle;
# after a container restart it is empty.
#
# Why: some Plotly figures (especially MCS) are based on Parquets of
# several GB; repeated loading + aggregation per user request was the
# dominant overhead. The Plotly JSON itself is negligible at < 2 MB.
# ---------------------------------------------------------------------------

_FIG_CACHE: dict[str, tuple[float, dict]] = {}
_FIG_CACHE_MAX = 128


def _resolve_source_path(cfg: PipelineConfig, source_key: str) -> Path:
    """`"data_key"` → data_path, `"model:key"` → model_path, `"asset:file"` → assets/<file>."""
    if source_key.startswith("model:"):
        return Path(cfg.model_path(source_key[6:]))
    if source_key.startswith("asset:"):
        return cfg._base_dir / "assets" / source_key[6:]
    return Path(cfg.data_path(source_key))


def _mtime_fingerprint(paths: list[Path]) -> float:
    """Max mtime of all sources; changes exactly when a file is
    rewritten."""
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def _cache_chart(*source_keys: str) -> Callable:
    """Decorator: memoizes the JSON result of a chart endpoint until one
    of the source artifacts changes. Query parameters are part of the key."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            cfg = _cfg()
            paths = [_resolve_source_path(cfg, k) for k in source_keys]
            fp = _mtime_fingerprint(paths)
            key_raw = json.dumps(
                {"fn": fn.__name__, "args": args, "kwargs": kwargs},
                sort_keys=True, default=str,
            )
            key = f"{fn.__name__}:{hashlib.md5(key_raw.encode()).hexdigest()}"
            cached = _FIG_CACHE.get(key)
            if cached and cached[0] == fp:
                return cached[1]
            payload = fn(*args, **kwargs)
            # LRU-like: evict the oldest entry when full
            if len(_FIG_CACHE) >= _FIG_CACHE_MAX:
                _FIG_CACHE.pop(next(iter(_FIG_CACHE)))
            _FIG_CACHE[key] = (fp, payload)
            return payload
        return wrapper
    return decorator


@router.get("/cache/status")
def cache_status():
    """Number of cached chart responses + key list, for diagnostics."""
    return {
        "entries": len(_FIG_CACHE),
        "max_entries": _FIG_CACHE_MAX,
        "keys": list(_FIG_CACHE.keys()),
    }


@router.post("/cache/clear")
def cache_clear():
    """Clears the chart JSON cache manually (usually unnecessary: the mtime-
    based invalidation kicks in automatically after pipeline runs)."""
    n = len(_FIG_CACHE)
    _FIG_CACHE.clear()
    return {"cleared": n}


# ---------------------------------------------------------------------------
# Status / pipeline artifacts
# ---------------------------------------------------------------------------

@router.get("/status")
def status():
    """Which pipeline artifacts already exist? (For the overview tiles.)"""
    cfg = _cfg()
    keys = [
        ("raw", "data"),
        ("preprocessed", "data"),
        ("feature_engineered", "data"),
        ("test_data", "data"),
        ("backtesting_results", "data"),
        ("mcs_data", "data"),
    ]
    files = {}
    for key, kind in keys:
        path = Path(cfg.data_path(key))
        files[key] = {
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / 1_048_576, 2) if path.exists() else None,
            "mtime": (
                pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
                if path.exists() else None
            ),
        }

    assets_dir = cfg._base_dir / "assets"
    assets = {}
    if assets_dir.exists():
        for p in sorted(assets_dir.iterdir()):
            if p.is_file():
                assets[p.name] = {
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "mtime": pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
                }

    return {
        "data": files,
        "assets_count": len(assets),
        "assets": assets,
        "end_date": cfg.data.end_date,
        "end_date_frozen": getattr(cfg.data, "end_date_is_frozen", False),
        "walk_forward": cfg.walk_forward.enabled,
        "fast_mode": cfg.fast_mode.enabled,
    }


# ---------------------------------------------------------------------------
# Asset gallery (PNG + MD)
# ---------------------------------------------------------------------------

@router.get("/asset/{name}")
def get_asset(name: str):
    """Serve PNG or MD from assets/ (read-only)."""
    cfg = _cfg()
    path = cfg._base_dir / "assets" / name
    # Simple path traversal protection
    if ".." in name or not path.exists() or not path.is_file():
        raise HTTPException(404, f"Asset not found: {name}")
    if name.endswith(".md"):
        return PlainTextResponse(path.read_text(encoding="utf-8"))
    return FileResponse(str(path))


@router.get("/markdown/{name}")
def get_markdown(name: str):
    """MD file as JSON payload (for client-side rendering with marked.js)."""
    cfg = _cfg()
    path = cfg._base_dir / "assets" / name
    if not path.exists() or not name.endswith(".md"):
        raise HTTPException(404, f"Markdown not found: {name}")
    return {"name": name, "content": path.read_text(encoding="utf-8")}


@router.get("/optuna/best-params")
def optuna_best_params():
    """
    Best hyperparameters per model from the Optuna SQLite storage.

    Enables the dashboard UI to show a structured card layout instead of
    Markdown rendering, incl. best score, trial counts, and parameter map.
    """
    cfg = _cfg()
    db_path = Path(cfg.model_path("optuna_db"))
    if not db_path.exists():
        raise HTTPException(
            404, "No Optuna DB found: optimization not yet performed."
        )

    try:
        import optuna
    except ImportError:
        raise HTTPException(500, "optuna not installed.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{db_path}"

    # Study names carry the configured suffix (e.g. opt_LSTM_v2_martin), so a
    # changed search space / metric never resumes into an old study.
    suffix = getattr(cfg.optimization, "study_suffix", None)
    metric = getattr(cfg.optimization, "metric", "sharpe")

    results = []
    study_metric = None
    for model_name in ["MSM", "HMM", "HMM_Uni", "LSTM", "Transformer"]:
        study_name = f"opt_{model_name}" + (f"_{suffix}" if suffix else "")
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except KeyError:
            continue

        try:
            best_value = float(study.best_value)
            best_params = dict(study.best_params)
        except ValueError:
            continue  # no completed trials

        study_metric = study.user_attrs.get("metric", study_metric)

        n_complete = sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        n_pruned = sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        )

        results.append({
            "model": model_name,
            "best_score": best_value,
            "best_params": best_params,
            "n_trials_total": len(study.trials),
            "n_trials_complete": n_complete,
            "n_trials_pruned": n_pruned,
        })

    if not results:
        raise HTTPException(404, "No Optuna studies found in the DB.")

    return {
        "metric": f"{study_metric or metric} (pooled OOS)",
        "results": results,
    }


_OPTUNA_PLOT_TITLES = {
    "history":    "Optimization History",
    "importance": "Hyperparameter Importances",
    "slice":      "Slice Plot",
    "contour":    "Contour Plot",
}


@router.get("/chart/optuna")
@_cache_chart("model:optuna_db", "asset:optuna_importance_values.json")
def chart_optuna(
    model: str = Query(..., pattern="^(MSM|HMM|LSTM|Transformer)$"),
    plot: str = Query(..., pattern="^(history|importance|slice|contour)$"),
):
    """Interactive Optuna visualization (history · importance · slice · contour).

    Uses optuna.visualization directly → native Plotly figure, 1:1 to the PNG counterpart.
    Contour requires >= 2 hyperparameters, otherwise rejected with 400.
    """
    cfg = _cfg()
    db_path = Path(cfg.model_path("optuna_db"))
    if not db_path.exists():
        raise HTTPException(404, "No Optuna DB found.")

    try:
        import optuna
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_contour,
            plot_slice,
        )
        from optuna.importance import FanovaImportanceEvaluator
    except ImportError:
        raise HTTPException(500, "optuna not installed.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{db_path}"

    suffix = getattr(cfg.optimization, "study_suffix", None)
    study_name = f"opt_{model}" + (f"_{suffix}" if suffix else "")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        raise HTTPException(404, f"No Optuna study for {model}.")

    try:
        _ = study.best_value  # fails if there are no COMPLETE trials
    except ValueError:
        raise HTTPException(400, f"{model}: no completed trials.")

    n_params = max((len(t.params) for t in study.trials if t.params), default=1)
    if plot == "contour" and n_params < 2:
        raise HTTPException(400, f"{model}: contour requires >= 2 hyperparameters "
                                  f"(found: {n_params}).")

    # fANOVA is stochastic. For a 1:1 match with the pipeline PNG, we read
    # the values that actually went into the PNG from
    # assets/optuna_importance_values.json (written by save_optuna_plots).
    # If the cache is missing → live computation with a fixed seed as fallback.
    def _build_importance_fig() -> go.Figure:
        cache_path = cfg._base_dir / "assets" / "optuna_importance_values.json"
        cached = None
        if cache_path.exists():
            import json as _json
            try:
                data = _json.loads(cache_path.read_text(encoding="utf-8"))
                cached = (data.get("studies") or {}).get(model)
            except Exception as e:
                logger.warning("optuna_importance_values.json unreadable: %s", e)

        if cached:
            items = sorted(cached.items(), key=lambda kv: kv[1])
            names = [k for k, _ in items]
            values = [float(v) for _, v in items]
            bar = go.Figure(go.Bar(
                x=values, y=names, orientation="h",
                text=[f"{v:.2f}" for v in values], textposition="outside",
                marker=dict(color="rgb(99,110,250)"),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            ))
            bar.update_layout(
                xaxis_title="Hyperparameter Importance",
                yaxis_title="Hyperparameter",
            )
            return bar

        evaluator = FanovaImportanceEvaluator(seed=42)
        return plot_param_importances(study, evaluator=evaluator)

    builders = {
        "history":    lambda: plot_optimization_history(study),
        "importance": _build_importance_fig,
        "slice":      lambda: plot_slice(study),
        "contour":    lambda: plot_contour(study),
    }
    fig = builders[plot]()

    # Uniform theme (like all other charts). Slice (one row with n panels)
    # and contour (n×n matrix) scale with n_params, otherwise the axis
    # labels get squeezed illegibly; see the PNG pipeline in
    # src/backtest/plots.py.
    heights = {
        "history":    360,
        "importance": max(280, 60 * n_params + 120),
        "slice":      420,
        "contour":    max(600, 180 * n_params),
    }
    layout_updates = dict(
        title=f"{_OPTUNA_PLOT_TITLES[plot]}: {model}",
        template="plotly_white",
        height=heights[plot],
        margin=dict(l=60, r=30, t=60, b=50),
        font=dict(size=11),
    )
    # Slice / contour: set the Plotly-internal width explicitly, otherwise
    # many subplots get squeezed into a 600-800 px container and overlap
    # with neighboring panels.
    if plot == "slice":
        layout_updates["width"] = max(900, 260 * n_params)
    elif plot == "contour":
        layout_updates["width"] = max(900, 180 * n_params)

    fig.update_layout(**layout_updates)
    fig.update_xaxes(automargin=True, title_standoff=8)
    fig.update_yaxes(automargin=True, title_standoff=8)
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# EDA charts
# ---------------------------------------------------------------------------

@router.get("/chart/returns")
@_cache_chart("feature_engineered")
def chart_returns(
    col: str = Query("Returns", description="Column from feature_engineered"),
    smoothing: int = Query(0, ge=0, le=252),
):
    """Interactive return time series with zoom/hover."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("feature_engineered"), "run /data/ingest first")
    if col not in df.columns:
        raise HTTPException(400, f"Column '{col}' not available. Choices: {list(df.columns)}")

    s = df[col].dropna()
    if smoothing > 0:
        s = s.rolling(smoothing).mean()

    fig = go.Figure(go.Scatter(
        x=s.index, y=s.values, mode="lines", name=col,
        line=dict(width=1), hovertemplate="%{x|%Y-%m-%d}<br>%{y:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{col}" + (f" (MA{smoothing})" if smoothing else ""),
        xaxis_title="Date", yaxis_title=col,
        template="plotly_white", margin=dict(l=40, r=20, t=50, b=40),
        height=420,
    )
    return _fig_to_json(fig)


@router.get("/chart/feature-correlation")
@_cache_chart("feature_engineered")
def chart_feature_correlation():
    """Correlation matrix of the model features (lower triangle, red=+1, blue=-1)."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("feature_engineered"), "run /data/ingest first")
    cols = [c for c in cfg.features.model_features if c in df.columns]
    corr = df[cols].corr()

    upper = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
    z = corr.values.astype(float).copy()
    z[upper] = np.nan
    text = np.where(upper, "", np.round(corr.values, 2).astype(str))

    fig = go.Figure(go.Heatmap(
        z=z, x=corr.columns, y=corr.index,
        colorscale="RdBu", reversescale=True,
        zmin=-1, zmax=1,
        text=text, texttemplate="%{text}",
        hovertemplate="%{y} ↔ %{x}<br>ρ = %{z:.2f}<extra></extra>",
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(
        title="Feature Correlation Matrix (Pearson)",
        template="plotly_white", height=520,
        margin=dict(l=130, r=20, t=50, b=80),
        xaxis=dict(tickangle=-30, automargin=True),
        yaxis=dict(automargin=True, autorange="reversed"),
    )
    return _fig_to_json(fig)


@router.get("/chart/volatility-clusters")
@_cache_chart("feature_engineered")
def chart_volatility_clusters():
    """S&P 500 returns + ACF of the squared returns (GARCH effects)."""
    from plotly.subplots import make_subplots
    from statsmodels.tsa.stattools import acf

    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("feature_engineered"), "run /data/ingest first")
    if "Returns_GSPC" not in df.columns:
        raise HTTPException(400, "Returns_GSPC missing in the DataFrame")

    rets = df["Returns_GSPC"].dropna()
    squared = (rets ** 2).dropna()

    nlags = 40
    ac_values, confint = acf(squared.values, nlags=nlags, alpha=0.05, fft=True)
    ci_low = confint[:, 0] - ac_values
    ci_high = confint[:, 1] - ac_values
    lags = np.arange(nlags + 1)

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.12,
        subplot_titles=(
            "S&P 500 Daily Returns: Visualization of Volatility Clusters",
            "Autocorrelation of the Squared Returns: Evidence of GARCH Effects",
        ),
    )

    fig.add_trace(go.Scatter(
        x=rets.index, y=rets.values, mode="lines",
        name="Returns_GSPC",
        line=dict(color="#1f77b4", width=0.8),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # 95% confidence band around 0
    fig.add_trace(go.Scatter(
        x=np.concatenate([lags, lags[::-1]]),
        y=np.concatenate([ci_high, ci_low[::-1]]),
        fill="toself",
        fillcolor="rgba(100,149,237,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="95% CI",
    ), row=2, col=1)

    # Stems (vertical lines from the zero level to the ACF value)
    stem_x, stem_y = [], []
    for lag, val in zip(lags, ac_values):
        stem_x.extend([lag, lag, None])
        stem_y.extend([0, val, None])
    fig.add_trace(go.Scatter(
        x=stem_x, y=stem_y, mode="lines",
        line=dict(color="#1f77b4", width=1.4),
        hoverinfo="skip", showlegend=False,
    ), row=2, col=1)

    # ACF markers
    fig.add_trace(go.Scatter(
        x=lags, y=ac_values, mode="markers",
        marker=dict(color="#d62728", size=7),
        hovertemplate="Lag %{x}<br>ρ = %{y:.3f}<extra></extra>",
        showlegend=False, name="ACF",
    ), row=2, col=1)

    fig.add_hline(y=0, row=2, col=1, line_color="#d62728", line_width=1)

    fig.update_xaxes(title_text="Date", row=1, col=1, hoverformat="%Y-%m-%d")
    fig.update_yaxes(title_text="Return", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_yaxes(title_text="Autocorrelation", row=2, col=1, range=[-1, 1])

    fig.update_layout(
        template="plotly_white", height=720,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return _fig_to_json(fig)


def _top_n_drawdown_periods(drawdowns: pd.Series, n: int = 5) -> list[dict]:
    """Identifies the n deepest drawdown periods (peak → trough → recovery)."""
    periods = []
    in_period = False
    start_date = None
    for date, dd in drawdowns.items():
        if dd < 0 and not in_period:
            start_date = date
            in_period = True
        elif dd >= 0 and in_period:
            seg = drawdowns.loc[start_date:date]
            periods.append({
                "start": start_date,
                "trough": seg.idxmin(),
                "end": date,
                "magnitude": float(seg.min()),
            })
            in_period = False
    if in_period:
        seg = drawdowns.loc[start_date:]
        periods.append({
            "start": start_date,
            "trough": seg.idxmin(),
            "end": drawdowns.index[-1],
            "magnitude": float(seg.min()),
        })
    periods.sort(key=lambda p: p["magnitude"])
    return periods[:n]


@router.get("/chart/historical-drawdowns")
@_cache_chart("feature_engineered")
def chart_historical_drawdowns():
    """Historical drawdowns of the 60/40 portfolio, with the top 5 episodes marked."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("feature_engineered"), "run /data/ingest first")
    if "Returns" not in df.columns:
        raise HTTPException(400, "Returns missing in the DataFrame")

    rets = df["Returns"].fillna(0)
    cum_returns = np.exp(rets.cumsum())
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1.0

    top5 = _top_n_drawdown_periods(drawdowns, n=5)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdowns.index, y=drawdowns.values * 100,
        mode="lines",
        name="Drawdown",
        line=dict(color="#8b0000", width=1.2),
        fill="tozeroy", fillcolor="rgba(220,20,60,0.25)",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
    ))

    shapes = []
    for i, p in enumerate(top5, 1):
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=p["start"], x1=p["end"], y0=0, y1=1,
            fillcolor="rgba(255,0,0,0.08)",
            line=dict(width=0), layer="below",
        ))
        fig.add_trace(go.Scatter(
            x=[p["trough"]],
            y=[p["magnitude"] * 100],
            mode="markers+text",
            marker=dict(color="red", size=11, symbol="x-thin",
                        line=dict(color="darkred", width=2)),
            text=[f"#{i}"],
            textposition="bottom center",
            textfont=dict(color="darkred", size=11),
            name=f"Top-{i}: {p['magnitude']*100:.1f}%",
            hovertemplate=(
                f"<b>Top-{i} Drawdown</b><br>"
                f"Peak: {p['start'].strftime('%Y-%m-%d')}<br>"
                "Trough: %{x|%Y-%m-%d}<br>"
                f"Recovery: {p['end'].strftime('%Y-%m-%d')}<br>"
                "Depth: %{y:.2f}%<extra></extra>"
            ),
            showlegend=True,
        ))

    fig.update_layout(
        title="Historical Drawdowns of the 60/40 Portfolio: Top 5 Marked",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_white", height=460,
        margin=dict(l=60, r=20, t=50, b=40),
        shapes=shapes,
        xaxis=dict(hoverformat="%Y-%m-%d", automargin=True),
        legend=dict(orientation="h", y=-0.18, x=0, xanchor="left"),
    )
    return _fig_to_json(fig)


@router.get("/chart/capital-curve")
@_cache_chart("feature_engineered")
def chart_capital_curve():
    """60/40 benchmark capital curve in € (Standard scenario initial_capital)."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("feature_engineered"), "run /data/ingest first")
    if "Cumulative_Returns" in df.columns:
        s = df["Cumulative_Returns"].dropna()
    elif "Returns" in df.columns:
        s = (1 + df["Returns"].fillna(0)).cumprod()
    else:
        raise HTTPException(400, "Returns/Cumulative_Returns missing in the DataFrame")

    initial_capital = float(cfg.backtesting.sorr.scenarios.Standard.initial_capital)
    s = s / s.iloc[0] * initial_capital

    fig = go.Figure(go.Scatter(
        x=s.index, y=s.values, mode="lines", name="60/40 Benchmark",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} €<extra></extra>",
    ))
    fig.update_layout(
        title=f"60/40 Benchmark Capital Curve (Initial Capital {initial_capital:,.0f} €)",
        xaxis_title="Date", yaxis_title="Capital (€)",
        template="plotly_white", height=420,
        margin=dict(l=110, r=20, t=50, b=40),
        xaxis=dict(automargin=True),
        yaxis=dict(tickformat=",.0f", automargin=True, title_standoff=18),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Backtest charts
# ---------------------------------------------------------------------------

@router.get("/chart/equity-curves")
@_cache_chart("backtesting_results")
def chart_equity_curves():
    """Equity curves of all strategies in € (initial capital = Standard scenario)."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("backtesting_results"), "run /backtest/run first")
    colors = cfg.color_map
    initial_capital = float(cfg.backtesting.sorr.scenarios.Standard.initial_capital)

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col] * initial_capital, mode="lines", name=col,
            line=dict(color=_plotly_color(colors.get(col)), width=1.8),
            hovertemplate=f"<b>{col}</b>: %{{y:,.0f}} €<extra></extra>",
        ))
    fig.update_layout(
        title=f"Equity Curves (OOS): Initial Capital {initial_capital:,.0f} €",
        xaxis_title="Date", yaxis_title="Capital (€)",
        template="plotly_white", height=500,
        hovermode="x unified",
        xaxis=dict(hoverformat="%Y-%m-%d"),
        yaxis=dict(tickformat=",.0f", automargin=True, title_standoff=18),
        margin=dict(l=110, r=20, t=50, b=40),
    )
    return _fig_to_json(fig)


@router.get("/chart/drawdown")
@_cache_chart("backtesting_results")
def chart_drawdown():
    """Drawdown paths of all strategies."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("backtesting_results"), "run /backtest/run first")
    colors = cfg.color_map

    fig = go.Figure()
    for col in df.columns:
        peak = df[col].cummax()
        dd = (df[col] - peak) / peak
        fig.add_trace(go.Scatter(
            x=df.index, y=dd * 100, mode="lines", name=col,
            line=dict(color=_plotly_color(colors.get(col)), width=1.5),
            hovertemplate=f"<b>{col}</b>: %{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        title="Drawdown Paths (OOS)",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_white", height=420,
        hovermode="x unified",
        xaxis=dict(hoverformat="%Y-%m-%d"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return _fig_to_json(fig)


@router.get("/chart/transaction-costs")
@_cache_chart("backtesting_costs")
def chart_transaction_costs():
    """Cumulative transaction costs per strategy (in %), without Buy_Hold."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("backtesting_costs"), "run /backtest/run first")
    colors = cfg.color_map
    fee_rate = float(cfg.transaction_cost_rate)

    fig = go.Figure()
    for col in df.columns:
        if col == "Buy_Hold":
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col] * 100, mode="lines",
            name=f"Costs: {col.replace('_', ' ')}",
            line=dict(color=_plotly_color(colors.get(col)), width=1.5),
            hovertemplate=f"<b>{col}</b>: %{{y:.3f}}%<extra></extra>",
        ))
    fig.update_layout(
        title=f"Cumulative Transaction Costs Over Time (Fee: {fee_rate*100:g}%)",
        xaxis_title="Date", yaxis_title="Costs in %",
        template="plotly_white", height=420,
        hovermode="x unified",
        xaxis=dict(hoverformat="%Y-%m-%d", automargin=True),
        yaxis=dict(tickformat=".2f", automargin=True),
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation="v", x=0.01, y=0.99, xanchor="left", yanchor="top"),
    )
    return _fig_to_json(fig)


@router.get("/chart/sorr-scenario")
@_cache_chart("backtesting_sorr")
def chart_sorr_scenario(
    scenario: str = Query("Standard", pattern="^(Standard|Aggressive|Low_Capital)$"),
):
    """SORR capital path for one scenario, 1:1 to the pipeline PNG.

    Data source: backtesting_sorr (columns = "{scenario}_{strategy}").
    Parameters (start, withdrawal) come from cfg.backtesting.sorr.scenarios.
    """
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("backtesting_sorr"), "run /backtest/run first")
    colors = cfg.color_map

    scenarios_cfg = cfg.backtesting.sorr.scenarios
    if not hasattr(scenarios_cfg, scenario):
        raise HTTPException(400, f"Scenario '{scenario}' not in the config.")
    sc = getattr(scenarios_cfg, scenario)
    start = float(sc.initial_capital)
    monthly_withdrawal = start * float(sc.annual_withdrawal_rate) / 12

    prefix = f"{scenario}_"
    strat_cols = [c for c in df.columns if c.startswith(prefix)]
    if not strat_cols:
        available = sorted({c.split("_", 1)[0] for c in df.columns if "_" in c})
        raise HTTPException(400, f"Scenario '{scenario}' not in the Parquet. "
                                  f"Available: {available}")

    fig = go.Figure()
    for col in strat_cols:
        strat = col[len(prefix):]
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines",
            name=strat.replace("_", " "),
            line=dict(color=_plotly_color(colors.get(strat)), width=1.6),
            hovertemplate=f"<b>{strat}</b>: %{{y:,.0f}} €<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="#000", line_width=1)

    fig.update_layout(
        title=(f"SORR Scenario {scenario.replace('_', ' ')}: "
               f"Start {start:,.0f} €, Withdrawal {monthly_withdrawal:,.0f} €/Month"),
        xaxis_title="Date", yaxis_title="Capital (€)",
        template="plotly_white", height=460,
        hovermode="x unified",
        xaxis=dict(hoverformat="%Y-%m-%d", automargin=True),
        yaxis=dict(tickformat=",.0f", automargin=True, title_standoff=18),
        margin=dict(l=110, r=140, t=50, b=40),
        legend=dict(orientation="v", x=1.02, y=1.0, xanchor="left", yanchor="top"),
    )
    return _fig_to_json(fig)


@router.get("/chart/rolling-sharpe")
@_cache_chart("backtesting_results")
def chart_rolling_sharpe(window: int = Query(252, ge=21, le=1260)):
    """Rolling Sharpe over a configurable window.

    Uses the same `calculate_rolling_sharpe` function as the pipeline plot
    (low-vol windows → 0, cap ±10), so that both visualizations match 1:1
    and HMM spikes from cash-only phases are suppressed.
    """
    from src.backtest.engine import calculate_rolling_sharpe

    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("backtesting_results"), "run /backtest/run first")
    rolling_sharpe = calculate_rolling_sharpe(df, window_days=window)

    fig = go.Figure()
    for col in rolling_sharpe.columns:
        series = rolling_sharpe[col].dropna()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines", name=col,
            line=dict(color=_plotly_color(cfg.color_map.get(col)), width=1.4),
            hovertemplate=f"<b>{col}</b>: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title=f"Rolling Sharpe (Window = {window} Days, Cap ±10, Low-Vol → 0)",
        xaxis_title="Date", yaxis_title="Sharpe ratio (annualized)",
        template="plotly_white", height=420,
        hovermode="x unified",
        xaxis=dict(hoverformat="%Y-%m-%d"),
        yaxis=dict(range=[-3, 5]),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Regime chart
# ---------------------------------------------------------------------------

@router.get("/chart/regime-overlay")
@_cache_chart("test_data")
def chart_regime_overlay(model: str = Query("MSM", pattern="^(MSM|HMM|LSTM|Transformer)$")):
    """Price + bear probability + signal overlay for one model (OOS)."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("test_data"),
                              "run /models/train-all + /backtest/run first")

    prob_col = f"{model}_Prob"
    sig_col = f"{model}_Signal"
    if prob_col not in df.columns or sig_col not in df.columns:
        raise HTTPException(400, f"{prob_col}/{sig_col} missing in test_data")

    price_col = "Cumulative_Returns" if "Cumulative_Returns" in df.columns else None
    color = _plotly_color(cfg.color_map.get(model, "steelblue"))

    fig = go.Figure()
    if price_col:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[price_col], mode="lines", name="60/40 Price",
            line=dict(color="#444", width=1.2),
            yaxis="y2",
            # Unified hover shows the date once globally → only y here.
            hovertemplate="Price %{y:.3f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df[prob_col], mode="lines",
        name=f"{model} Bear-Prob",
        line=dict(color=color, width=1.4),
        fill="tozeroy", fillcolor=_rgba(color, 0.15),
        hovertemplate="Prob %{y:.3f}<extra></extra>",
    ))

    # Bear signal bands as shapes
    signal = df[sig_col].fillna(0).astype(int)
    changes = np.where(np.diff(signal.values, prepend=0) != 0)[0]
    shapes = []
    in_bear = False
    start = None
    for idx, val in zip(df.index, signal.values):
        if val == 1 and not in_bear:
            start = idx; in_bear = True
        elif val == 0 and in_bear:
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=start, x1=idx, y0=0, y1=1,
                               fillcolor=_rgba("#e53935", 0.28),
                               line=dict(color=_rgba("#e53935", 0.75), width=1),
                               layer="below"))
            in_bear = False
    if in_bear and start is not None:
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=start, x1=df.index[-1], y0=0, y1=1,
                           fillcolor=_rgba("#e53935", 0.28),
                           line=dict(color=_rgba("#e53935", 0.75), width=1),
                           layer="below"))

    fig.update_layout(
        title=f"Regime Overlay: {model}",
        template="plotly_white", height=500,
        hovermode="x unified",
        # Uniform date format in the unified hover header.
        xaxis=dict(hoverformat="%Y-%m-%d"),
        shapes=shapes,
        yaxis=dict(title="Bear probability", range=[0, 1]),
        yaxis2=dict(title="Price", overlaying="y", side="right", showgrid=False),
        margin=dict(l=40, r=60, t=50, b=40),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Label analysis (concordance, kappa, timeline)
# ---------------------------------------------------------------------------

def _build_label_dict(test_df: pd.DataFrame) -> dict:
    """MSM/HMM/HMM_Uni from test_df + price-based/macro alternatives (computed on the fly)."""
    from src.data.labels import (
        label_pagan_sossounov, label_peak_to_trough,
        label_lunde_timmermann, load_nber_recession,
    )
    prices = test_df["Cumulative_Returns"]
    return {
        "MSM":     test_df["MSM_Signal"].astype("int8"),
        "HMM":     test_df["HMM_Signal"].astype("int8"),
        "HMM_Uni": test_df["HMM_Uni_Signal"].astype("int8"),
        "PagSoss": label_pagan_sossounov(prices),
        "P2T":     label_peak_to_trough(prices, threshold=0.20),
        "LundeT":  label_lunde_timmermann(prices),
        "NBER":    load_nber_recession(test_df.index),
    }


@router.get("/chart/label-agreement")
@_cache_chart("test_data")
def chart_label_agreement():
    """Label agreement as heatmap with toggle: concordance ↔ Cohen's κ."""
    from src.data.labels import compute_concordance_matrix, compute_kappa_matrix

    cfg = _cfg()
    test_df = _read_parquet_or_404(cfg.data_path("test_data"),
                                   "run /models/train-all first")

    labels = _build_label_dict(test_df)
    concordance = compute_concordance_matrix(labels)
    kappa = compute_kappa_matrix(labels)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=concordance.values, x=list(concordance.columns), y=list(concordance.index),
        colorscale="RdYlGn", zmin=0.5, zmax=1.0,
        text=np.round(concordance.values, 2), texttemplate="%{text:.2f}",
        textfont=dict(size=12),
        hovertemplate="%{y} ↔ %{x}<br>Concordance: %{z:.3f}<extra></extra>",
        colorbar=dict(title="ρ", thickness=14),
        visible=True, name="Concordance",
    ))
    fig.add_trace(go.Heatmap(
        z=kappa.values, x=list(kappa.columns), y=list(kappa.index),
        colorscale="RdYlGn", zmin=-0.2, zmax=1.0,
        text=np.round(kappa.values, 2), texttemplate="%{text:.2f}",
        textfont=dict(size=12),
        hovertemplate="%{y} ↔ %{x}<br>κ: %{z:.3f}<extra></extra>",
        colorbar=dict(title="κ", thickness=14),
        visible=False, name="Cohen's κ",
    ))

    fig.update_layout(
        title="Label Concordance (Share of Matching Days)",
        template="plotly_white", height=540,
        margin=dict(l=80, r=60, t=100, b=80),
        xaxis=dict(automargin=True, tickangle=-30, side="bottom"),
        yaxis=dict(automargin=True, autorange="reversed"),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            showactive=True,
            x=0.0, xanchor="left",
            y=1.16, yanchor="top",
            pad=dict(t=4, b=4, l=8, r=8),
            buttons=[
                dict(label="Concordance", method="update",
                     args=[{"visible": [True, False]},
                           {"title": "Label Concordance (Share of Matching Days)"}]),
                dict(label="Cohen's κ", method="update",
                     args=[{"visible": [False, True]},
                           {"title": "Cohen's κ: Label Concordance (Chance-Corrected)"}]),
            ],
        )],
    )
    return _fig_to_json(fig)


@router.get("/chart/label-timeline")
@_cache_chart("test_data", "raw")
def chart_label_timeline():
    """Multi-panel: S&P 500 price + horizontal bear bands per labeling method."""
    from plotly.subplots import make_subplots

    cfg = _cfg()
    test_df = _read_parquet_or_404(cfg.data_path("test_data"),
                                   "run /models/train-all first")
    raw_df = _read_parquet_or_404(cfg.data_path("raw"),
                                  "run /data/ingest first")

    plot_prices = raw_df["^GSPC"].reindex(test_df.index).ffill()
    labels = _build_label_dict(test_df)

    n = len(labels)
    # Price panel 3× as tall as one label strip
    heights = [3] + [1] * n
    total = sum(heights)
    row_heights = [h / total for h in heights]

    fig = make_subplots(
        rows=n + 1, cols=1, shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.012,
    )

    # Price (panel 1)
    fig.add_trace(go.Scatter(
        x=plot_prices.index, y=plot_prices.values,
        mode="lines", line=dict(color="#111", width=1.1),
        name="S&P 500", showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}<br>S&P 500: %{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1, tickformat=",.0f",
                     automargin=True)

    # Label strips (panels 2..n+1)
    for i, (name, series) in enumerate(labels.items(), start=2):
        s = series.reindex(test_df.index).fillna(0).astype(int)
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values.astype(float),
            mode="lines",
            line=dict(width=0, shape="hv"),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.55)",
            showlegend=False,
            hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>Bear = %{{y:.0f}}<extra></extra>",
        ), row=i, col=1)
        fig.update_yaxes(
            range=[0, 1], showticklabels=False, ticks="",
            title_text=name, title_standoff=6,
            row=i, col=1,
        )

    fig.update_xaxes(hoverformat="%Y-%m-%d", row=n + 1, col=1,
                     title_text="Date")

    fig.update_layout(
        title="S&P 500 with Regime Labels (red = Bear)",
        template="plotly_white",
        height=160 + 70 * n,
        margin=dict(l=80, r=20, t=60, b=50),
        hovermode="x unified",
        showlegend=False,
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Walk-forward schema
# ---------------------------------------------------------------------------

@router.get("/chart/walk-forward-schema")
@_cache_chart("feature_engineered")
def chart_walk_forward_schema():
    """Gantt-style view of the rolling train/test windows."""
    from src.backtest.walk_forward import walk_forward_splits, summarize_splits

    cfg = _cfg()
    wf = cfg.walk_forward

    fe = _read_parquet_or_404(cfg.data_path("feature_engineered"),
                              "run /data/ingest first")

    splits = walk_forward_splits(
        index=fe.index,
        mode=wf.mode,
        train_window_years=wf.train_window_years,
        test_window_months=wf.test_window_months,
        step_months=wf.step_months,
        min_train_years=wf.min_train_years,
    )
    summary = summarize_splits(splits)
    n_folds = len(summary)
    if n_folds == 0:
        raise HTTPException(400, "No walk-forward splits can be generated.")

    train_color = "#4C72B0"
    test_color = "#DD8452"

    fig = go.Figure()
    for fold_id, row in summary.iterrows():
        train_dur_ms = (row["train_end"] - row["train_start"]).total_seconds() * 1000
        test_dur_ms = (row["test_end"] - row["test_start"]).total_seconds() * 1000

        fig.add_trace(go.Bar(
            x=[train_dur_ms], y=[fold_id],
            base=[row["train_start"]],
            orientation="h", width=0.72,
            marker=dict(color=train_color, line=dict(width=0)),
            name="Train", legendgroup="train",
            showlegend=bool(fold_id == summary.index[0]),
            hovertemplate=(
                f"<b>Fold {fold_id} · Train</b><br>"
                f"{row['train_start']:%Y-%m-%d} → {row['train_end']:%Y-%m-%d}<br>"
                f"{row['n_train']} trading days<extra></extra>"
            ),
        ))
        fig.add_trace(go.Bar(
            x=[test_dur_ms], y=[fold_id],
            base=[row["test_start"]],
            orientation="h", width=0.72,
            marker=dict(color=test_color, line=dict(width=0)),
            name="Test (OOS)", legendgroup="test",
            showlegend=bool(fold_id == summary.index[0]),
            hovertemplate=(
                f"<b>Fold {fold_id} · Test (OOS)</b><br>"
                f"{row['test_start']:%Y-%m-%d} → {row['test_end']:%Y-%m-%d}<br>"
                f"{row['n_test']} trading days<extra></extra>"
            ),
        ))

    subtitle_parts = [f"Mode: <b>{wf.mode}</b>"]
    if wf.train_window_years:
        subtitle_parts.append(f"Train: {wf.train_window_years}&nbsp;y")
    if wf.test_window_months:
        subtitle_parts.append(f"Test: {wf.test_window_months}&nbsp;m")
    subtitle_parts.append(f"{n_folds}&nbsp;folds")

    fig.update_layout(
        title=dict(
            text=(
                "Walk-Forward Schema: Train/Test Windows Over Time"
                f"<br><sub style='color:#64748b'>{' &nbsp;·&nbsp; '.join(subtitle_parts)}</sub>"
            ),
            x=0.02, xanchor="left",
        ),
        template="plotly_white",
        height=max(440, 26 * n_folds + 140),
        barmode="overlay", bargap=0,
        xaxis=dict(title="Date", type="date",
                   showgrid=True, gridcolor="#e2e8f0",
                   hoverformat="%Y-%m-%d", automargin=True),
        yaxis=dict(title="Fold", autorange="reversed", dtick=1,
                   showgrid=False, automargin=True),
        margin=dict(l=50, r=20, t=90, b=50),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)"),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# MCS charts
# ---------------------------------------------------------------------------

@router.get("/chart/mcs-distribution")
@_cache_chart("mcs_data")
def chart_mcs_distribution(
    scenario: str = Query("Standard", pattern="^(Standard|Aggressive|Low_Capital)$"),
    kind: str = Query("boxplot", pattern="^(boxplot|violin)$"),
):
    """Distribution of the MCS terminal capital as box or violin plot.

    1:1 counterpart to the pipeline PNGs (mcs_boxplot_{sc}.png / mcs_violin_{sc}.png):
    per strategy, the terminal capital of all paths (last value of the simulation).
    """
    cfg = _cfg()
    mcs_path = cfg.data_path("mcs_data")
    hint = "run /backtest/evaluate first"
    colors = cfg.color_map

    # Read the Parquet footer (cheap), then load only the path columns
    # relevant for this scenario. With ~150k total columns in the 3.7 GB
    # Parquet, this saves double-digit seconds per request; the rest of the
    # file stays on disk.
    all_cols = _list_parquet_columns_or_404(mcs_path, hint)
    sc_prefix = f"{scenario}_"
    wanted = [c for c in all_cols if c.startswith(sc_prefix) and "_path_" in c]
    if not wanted:
        available = sorted({c.rsplit("_path_", 1)[0] for c in all_cols if "_path_" in c})
        raise HTTPException(400, f"No data for scenario '{scenario}'. "
                                  f"Available: {available}")
    df = _read_parquet_or_404(mcs_path, hint, columns=wanted)

    # Strategy order as in the pipeline (B&H left, models right).
    strategies = ["Buy_Hold", "MSM", "HMM", "LSTM", "Transformer"]
    finals: dict[str, np.ndarray] = {}
    for strat in strategies:
        prefix = f"{scenario}_{strat}_path_"
        path_cols = [c for c in df.columns if c.startswith(prefix)]
        if path_cols:
            # Last value per path = terminal capital after sim_years years
            finals[strat] = df[path_cols].iloc[-1].to_numpy(dtype=float)

    if not finals:
        available = sorted({c.rsplit("_path_", 1)[0] for c in df.columns if "_path_" in c})
        raise HTTPException(400, f"No data for scenario '{scenario}'. "
                                  f"Available: {available}")

    sc_cfg = getattr(cfg.backtesting.sorr.scenarios, scenario)
    start = float(sc_cfg.initial_capital)
    sim_years = int(cfg.evaluation.mcs.sim_years)
    n_paths = len(next(iter(finals.values())))

    fig = go.Figure()

    if kind == "boxplot":
        for strat, arr in finals.items():
            base = _plotly_color(colors.get(strat))
            fig.add_trace(go.Box(
                y=arr, name=strat.replace("_", " "),
                boxpoints=False,  # 10k outlier points would be illegible
                marker_color=base,
                line=dict(color=base, width=1.2),
                fillcolor=_rgba(base, 0.25),
                hovertemplate=(
                    f"<b>{strat}</b><br>"
                    "Max: %{upperfence:,.0f} €<br>"
                    "Q3: %{q3:,.0f} €<br>"
                    "Median: %{median:,.0f} €<br>"
                    "Q1: %{q1:,.0f} €<br>"
                    "Min: %{lowerfence:,.0f} €<extra></extra>"
                ),
            ))
            med = float(np.median(arr))
            fig.add_annotation(
                x=strat.replace("_", " "), y=med,
                xshift=36, showarrow=False,
                text=f"{med:,.0f}€",
                font=dict(color="#d62728", size=10),
            )
        title = (f"MCS {scenario}: Distribution of Terminal Capital "
                 f"(n={n_paths:,}, start: {start:,.0f}€)")
        y_title = f"Terminal capital after {sim_years} years in €"
    else:  # violin
        for strat, arr in finals.items():
            base = _plotly_color(colors.get(strat))
            fig.add_trace(go.Violin(
                y=arr, name=strat.replace("_", " "),
                box_visible=True, meanline_visible=False,
                points=False, spanmode="hard",
                fillcolor=_rgba(base, 0.55),
                line=dict(color=base, width=1.2),
                box=dict(width=0.15, fillcolor=_rgba(base, 0.85),
                         line=dict(color="#1f2937", width=1)),
                hovertemplate=(
                    f"<b>{strat}</b><br>"
                    "%{y:,.0f} €<extra></extra>"
                ),
            ))
        title = (f"MCS Terminal Wealth: Scenario {scenario} "
                 f"(n={n_paths:,}, start: {start:,.0f}€)")
        y_title = f"Terminal capital after {sim_years} years in €"

    fig.add_hline(y=0, line_dash="dash", line_color="#d62728",
                  line_width=1.2, opacity=0.7)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=480,
        showlegend=False,
        margin=dict(l=90, r=40, t=60, b=60),
        yaxis=dict(title=y_title, tickformat=",.0f",
                   automargin=True, title_standoff=14),
        xaxis=dict(automargin=True),
    )
    return _fig_to_json(fig)


@router.get("/chart/mcs-quantiles")
@_cache_chart("mcs_data")
def chart_mcs_quantiles(scenario: str = Query("Standard"), strategy: str = Query("Transformer")):
    """Quantile fan (5 / 25 / 50 / 75 / 95%) of the MCS paths."""
    cfg = _cfg()
    mcs_path = cfg.data_path("mcs_data")
    hint = "run /backtest/evaluate first"

    # Columns follow the schema: {scenario}_{strategy}_path_{N:03d}
    # e.g. "Standard_Transformer_path_000", "Standard_Transformer_path_001", ...
    # Read the Parquet footer, then load only the relevant ~10k columns.
    prefix = f"{scenario}_{strategy}_path_"
    all_cols = _list_parquet_columns_or_404(mcs_path, hint)
    path_cols = [c for c in all_cols if c.startswith(prefix)]
    if not path_cols:
        available = sorted({c.rsplit("_path_", 1)[0] for c in all_cols if "_path_" in c})
        raise HTTPException(400, f"Combination ({scenario}, {strategy}) not found. "
                                  f"Available combinations: {available}")
    df = _read_parquet_or_404(mcs_path, hint, columns=path_cols)

    # matrix: (n_paths, n_days); each column is a path, each row a day
    matrix = df[path_cols].values.T      # transpose: rows=paths, columns=days

    q = np.quantile(matrix, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    days = np.arange(matrix.shape[1])
    color = _plotly_color(cfg.color_map.get(strategy, "steelblue"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=q[4], mode="lines", name="5-95%",
                              line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=q[0], mode="lines", name="5-95%",
                              fill="tonexty", fillcolor=_rgba(color, 0.15),
                              line=dict(width=0)))
    fig.add_trace(go.Scatter(x=days, y=q[3], mode="lines", name="25-75%",
                              line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=q[1], mode="lines", name="25-75%",
                              fill="tonexty", fillcolor=_rgba(color, 0.3),
                              line=dict(width=0)))
    fig.add_trace(go.Scatter(x=days, y=q[2], mode="lines", name="Median",
                              line=dict(color=color, width=2)))
    fig.update_layout(
        title=f"MCS Quantiles: {scenario} / {strategy}",
        xaxis_title="Trading days", yaxis_title="Capital",
        template="plotly_white", height=460,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Classification vs. NBER (confusion matrix · ROC · PR)
# ---------------------------------------------------------------------------

def _load_classification_truth(cfg: PipelineConfig) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """test_df + y_true (NBER) + list of models to plot: shared basis."""
    from src.data.labels import load_nber_recession

    test_df = _read_parquet_or_404(
        cfg.data_path("test_data"),
        "run /models/train-all + /backtest/run first",
    )
    models = [m for m in cfg.evaluation.extended.f1_models]

    try:
        nber = load_nber_recession(
            test_df.index,
            source=cfg.evaluation.extended.nber_source,
        )
    except Exception as e:
        raise HTTPException(502, f"NBER data not loadable: {e}")

    y_true = nber.reindex(test_df.index).fillna(0).astype(int).values
    return test_df, y_true, models


@router.get("/chart/confusion-matrices")
@_cache_chart("test_data")
def chart_confusion_matrices():
    """Confusion matrices (MSM · HMM · LSTM · Transformer) vs. NBER as a subplot grid."""
    from plotly.subplots import make_subplots
    from sklearn.metrics import confusion_matrix

    cfg = _cfg()
    test_df, y_true, models = _load_classification_truth(cfg)

    available = [(m, f"{m}_Signal") for m in models if f"{m}_Signal" in test_df.columns]
    if not available:
        raise HTTPException(400, "No *_Signal columns found in test_data.")

    n = len(available)
    fig = make_subplots(
        rows=1, cols=n, horizontal_spacing=0.10,
        subplot_titles=[m for m, _ in available],
    )

    labels = ["No-Rec", "Rec"]
    for i, (model, sig_col) in enumerate(available, start=1):
        y_pred = test_df[sig_col].fillna(0).astype(int).values
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        # Rows = ground truth (NBER), columns = predicted
        # autorange='reversed' ensures y="No-Rec" is at the top and y="Rec" at the bottom.
        fig.add_trace(
            go.Heatmap(
                z=cm, x=labels, y=labels,
                colorscale="Blues", showscale=(i == n),
                zmin=0, zmax=float(cm.max()) if cm.max() > 0 else 1.0,
                text=cm.astype(int), texttemplate="%{text}",
                textfont=dict(size=14),
                hovertemplate=(
                    f"<b>{model}</b><br>NBER: %{{y}}<br>"
                    "Predicted: %{x}<br>Count: %{z}<extra></extra>"
                ),
                colorbar=dict(title="Count", thickness=12) if i == n else None,
            ),
            row=1, col=i,
        )
        fig.update_xaxes(title_text="Predicted", row=1, col=i, automargin=True)
        fig.update_yaxes(
            title_text="NBER (ground truth)" if i == 1 else "",
            autorange="reversed", row=1, col=i, automargin=True,
        )

    fig.update_layout(
        title="Confusion Matrices (vs. NBER)",
        template="plotly_white",
        height=380,
        margin=dict(l=70, r=40, t=70, b=50),
    )
    return _fig_to_json(fig)


@router.get("/chart/roc-pr-curves")
@_cache_chart("test_data")
def chart_roc_pr_curves(kind: str = Query("roc", pattern="^(roc|pr)$")):
    """ROC or PR curves of all models vs. NBER; uses *_Prob (threshold-independent)."""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    cfg = _cfg()
    test_df, y_true, models = _load_classification_truth(cfg)
    colors = cfg.color_map

    available = [(m, f"{m}_Prob") for m in models if f"{m}_Prob" in test_df.columns]
    if not available:
        raise HTTPException(400, "No *_Prob columns found in test_data.")

    fig = go.Figure()

    if kind == "roc":
        # Random baseline (diagonal)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="rgba(100,100,100,0.6)", width=1),
            name="Random", hoverinfo="skip",
        ))

    for model, prob_col in available:
        y_score = test_df[prob_col].fillna(0).values
        color = _plotly_color(colors.get(model))

        if kind == "roc":
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_value = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{model} (AUC={auc_value:.2f})",
                line=dict(color=color, width=1.8),
                hovertemplate=(
                    f"<b>{model}</b><br>"
                    "FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
                ),
            ))
        else:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            auc_value = auc(rec, prec)
            fig.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines",
                name=f"{model} (AUC={auc_value:.2f})",
                line=dict(color=color, width=1.8),
                hovertemplate=(
                    f"<b>{model}</b><br>"
                    "Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
                ),
            ))

    if kind == "roc":
        title = "ROC Curves (vs. NBER)"
        x_title, y_title = "False Positive Rate", "True Positive Rate"
        legend_loc = dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom")
    else:
        title = "Precision-Recall Curves (vs. NBER)"
        x_title, y_title = "Recall", "Precision"
        legend_loc = dict(x=0.02, y=0.02, xanchor="left", yanchor="bottom")

    fig.update_layout(
        title=title,
        xaxis=dict(title=x_title, range=[0, 1], automargin=True),
        yaxis=dict(title=y_title, range=[0, 1.02], automargin=True,
                   scaleanchor="x", scaleratio=1),
        template="plotly_white", height=520,
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(orientation="v", bgcolor="rgba(255,255,255,0.0)", **legend_loc),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Regime probability heatmap
# ---------------------------------------------------------------------------

@router.get("/chart/regime-probability-heatmap")
@_cache_chart("test_data")
def chart_regime_probability_heatmap():
    """Bear probabilities of all models as heatmap (y=model, x=time).

    1:1 counterpart to the pipeline PNG (src.backtest.evaluation.plot_regime_probability_heatmap),
    reads the `{model}_Prob` columns directly from test_data.
    """
    cfg = _cfg()
    test_df = _read_parquet_or_404(
        cfg.data_path("test_data"),
        "run /models/train-all + /backtest/run first",
    )

    models = [m for m in cfg.evaluation.extended.f1_models
              if f"{m}_Prob" in test_df.columns]
    if not models:
        raise HTTPException(400, "No *_Prob columns found in test_data.")

    probs = pd.DataFrame({m: test_df[f"{m}_Prob"] for m in models})

    # Heatmap matrix: rows = models, columns = time points.
    z = probs.T.values.astype(float)
    x = probs.index  # DatetimeIndex

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=models,
        colorscale="RdYlGn", reversescale=True,
        zmin=0.0, zmax=1.0,
        colorbar=dict(title="P(Bear)", thickness=14),
        hovertemplate=(
            "<b>%{y}</b><br>%{x|%Y-%m-%d}<br>"
            "P(Bear) = %{z:.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Regime Bear Probabilities Over the OOS Period",
        template="plotly_white",
        height=max(280, 90 * len(models) + 140),
        margin=dict(l=110, r=40, t=60, b=60),
        xaxis=dict(title="Date", hoverformat="%Y-%m-%d",
                   automargin=True, tickangle=-30),
        yaxis=dict(automargin=True, autorange="reversed"),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Break-even transaction costs
# ---------------------------------------------------------------------------

@router.get("/chart/break-even-costs")
@_cache_chart("test_data", "backtesting_results")
def chart_break_even_costs():
    """Final wealth vs. cost rate per model: 1:1 counterpart to the pipeline PNG.

    Computes the break-even curves on demand (analogous to the pipeline in
    `/backtest/evaluate`): for each fee level from `fee_grid_bps`, the
    backtest is re-evaluated per model; the last equity per run provides
    the curve point. B&H reference line = last value from `backtesting_results`.
    """
    from src.backtest.evaluation import break_even_transaction_cost
    from src.backtest.engine import backtest

    cfg = _cfg()
    test_df = _read_parquet_or_404(
        cfg.data_path("test_data"),
        "run /models/train-all + /backtest/run first",
    )
    results = _read_parquet_or_404(
        cfg.data_path("backtesting_results"),
        "run /backtest/run first",
    )
    if "Buy_Hold" not in results.columns:
        raise HTTPException(400, "Buy_Hold missing in backtesting_results.")
    bh_series = results["Buy_Hold"]
    bh_final = float(bh_series.iloc[-1])

    ext = cfg.evaluation.extended
    models = [m for m in ext.f1_models if f"{m}_Signal" in test_df.columns]
    if not models:
        raise HTTPException(400, "No *_Signal columns found in test_data.")

    _, be_curves = break_even_transaction_cost(
        test_df, backtest, bh_series, models,
        list(ext.fee_grid_bps), cfg.backtesting.signal_shift,
    )

    curves = pd.DataFrame(be_curves).sort_index()
    fee_bps = curves.index.astype(float).tolist()
    colors = cfg.color_map

    fig = go.Figure()
    for model in curves.columns:
        fig.add_trace(go.Scatter(
            x=fee_bps, y=curves[model].values,
            mode="lines+markers", name=model,
            line=dict(color=_plotly_color(colors.get(model)), width=1.8),
            marker=dict(size=7),
            hovertemplate=(
                f"<b>{model}</b><br>"
                "Costs: %{x:.0f} bps<br>"
                "Final Wealth: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.add_hline(
        y=bh_final, line_color="#111", line_dash="dash", line_width=1.2,
        annotation_text=f"Buy & Hold ({bh_final:.2f})",
        annotation_position="top right",
        annotation_font=dict(size=11),
    )

    fig.update_layout(
        title="Break-Even Analysis: Cost Rate vs. Terminal Wealth",
        xaxis_title="Transaction costs (bps)",
        yaxis_title="Final wealth (cumulative)",
        template="plotly_white", height=460,
        margin=dict(l=70, r=30, t=60, b=50),
        hovermode="x unified",
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True),
        legend=dict(orientation="v", x=0.98, y=0.98,
                    xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.0)"),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Helpers: colors
# ---------------------------------------------------------------------------

def _plotly_color(c) -> str:
    """Converts matplotlib names ('tab:blue', 'darkorange', ...) to hex."""
    if c is None:
        return "#1f77b4"
    tab_map = {
        "tab:blue": "#1f77b4", "tab:orange": "#ff7f0e", "tab:green": "#2ca02c",
        "tab:red": "#d62728", "tab:purple": "#9467bd", "tab:brown": "#8c564b",
        "tab:pink": "#e377c2", "tab:gray": "#7f7f7f", "tab:olive": "#bcbd22",
        "tab:cyan": "#17becf",
    }
    return tab_map.get(c, c)


def _rgba(hex_or_name: str, alpha: float) -> str:
    hex_str = _plotly_color(hex_or_name).lstrip("#")
    if len(hex_str) != 6:
        return f"rgba(100,100,100,{alpha})"
    r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
