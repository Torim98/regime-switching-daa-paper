"""Data-Adapter: Parquet/MD → Plotly-JSON für das Dashboard.

Alle Endpoints sind GETs unter /api/*. Die UI lädt sie via fetch() und
rendert clientseitig mit Plotly.js. Keine Neuberechnungen — die Werte
stammen aus den Artefakten, die von der Pipeline geschrieben wurden.
"""
from pathlib import Path
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


def _read_parquet_or_404(path: str, hint: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        raise HTTPException(404, f"Artefakt fehlt: {path} — {hint}")


def _fig_to_json(fig: go.Figure) -> dict:
    """Plotly-Figure → dict. Wir nutzen to_json() und parsen zurück,
    damit numpy/datetime-Werte korrekt serialisiert werden."""
    import json
    return json.loads(pio.to_json(fig))


# ---------------------------------------------------------------------------
# Status / Pipeline-Artefakte
# ---------------------------------------------------------------------------

@router.get("/status")
def status():
    """Welche Pipeline-Artefakte existieren bereits? (Für Overview-Kacheln.)"""
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
# Asset-Gallery (PNG + MD)
# ---------------------------------------------------------------------------

@router.get("/asset/{name}")
def get_asset(name: str):
    """PNG oder MD aus assets/ ausliefern (read-only)."""
    cfg = _cfg()
    path = cfg._base_dir / "assets" / name
    # Simple Path-Traversal-Schutz
    if ".." in name or not path.exists() or not path.is_file():
        raise HTTPException(404, f"Asset nicht gefunden: {name}")
    if name.endswith(".md"):
        return PlainTextResponse(path.read_text(encoding="utf-8"))
    return FileResponse(str(path))


@router.get("/markdown/{name}")
def get_markdown(name: str):
    """MD-Datei als JSON-Payload (für clientseitiges Rendering mit marked.js)."""
    cfg = _cfg()
    path = cfg._base_dir / "assets" / name
    if not path.exists() or not name.endswith(".md"):
        raise HTTPException(404, f"Markdown nicht gefunden: {name}")
    return {"name": name, "content": path.read_text(encoding="utf-8")}


# ---------------------------------------------------------------------------
# EDA-Charts
# ---------------------------------------------------------------------------

@router.get("/chart/returns")
def chart_returns(
    col: str = Query("Returns", description="Spalte aus feature_engineered"),
    smoothing: int = Query(0, ge=0, le=252),
):
    """Interaktive Renditen-Zeitreihe mit Zoom/Hover."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("feature_engineered"), "/data/ingest zuerst")
    if col not in df.columns:
        raise HTTPException(400, f"Spalte '{col}' nicht verfügbar. Auswahl: {list(df.columns)}")

    s = df[col].dropna()
    if smoothing > 0:
        s = s.rolling(smoothing).mean()

    fig = go.Figure(go.Scatter(
        x=s.index, y=s.values, mode="lines", name=col,
        line=dict(width=1), hovertemplate="%{x|%Y-%m-%d}<br>%{y:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{col}" + (f" (MA{smoothing})" if smoothing else ""),
        xaxis_title="Datum", yaxis_title=col,
        template="plotly_white", margin=dict(l=40, r=20, t=50, b=40),
        height=420,
    )
    return _fig_to_json(fig)


@router.get("/chart/feature-correlation")
def chart_feature_correlation():
    """Korrelationsmatrix der Modell-Features (unteres Dreieck, rot=+1, blau=-1)."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("feature_engineered"), "/data/ingest zuerst")
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
        title="Feature-Korrelationsmatrix (Pearson)",
        template="plotly_white", height=520,
        margin=dict(l=130, r=20, t=50, b=80),
        xaxis=dict(tickangle=-30, automargin=True),
        yaxis=dict(automargin=True, autorange="reversed"),
    )
    return _fig_to_json(fig)


@router.get("/chart/capital-curve")
def chart_capital_curve():
    """60/40-Benchmark Kapitalkurve in € (Standard-Szenario initial_capital)."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("feature_engineered"), "/data/ingest zuerst")
    if "Cumulative_Returns" in df.columns:
        s = df["Cumulative_Returns"].dropna()
    elif "Returns" in df.columns:
        s = (1 + df["Returns"].fillna(0)).cumprod()
    else:
        raise HTTPException(400, "Returns/Cumulative_Returns fehlt im DataFrame")

    initial_capital = float(cfg.backtesting.sorr.scenarios.Standard.initial_capital)
    s = s / s.iloc[0] * initial_capital

    fig = go.Figure(go.Scatter(
        x=s.index, y=s.values, mode="lines", name="60/40 Benchmark",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} €<extra></extra>",
    ))
    fig.update_layout(
        title=f"60/40 Benchmark-Kapitalkurve (Startkapital {initial_capital:,.0f} €)",
        xaxis_title="Datum", yaxis_title="Kapital (€)",
        template="plotly_white", height=420,
        margin=dict(l=110, r=20, t=50, b=40),
        xaxis=dict(automargin=True),
        yaxis=dict(tickformat=",.0f", automargin=True, title_standoff=18),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Backtest-Charts
# ---------------------------------------------------------------------------

@router.get("/chart/equity-curves")
def chart_equity_curves():
    """Equity Curves aller Strategien in € (Startkapital = Standard-Szenario)."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("backtesting_results"), "/backtest/run zuerst")
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
        title=f"Equity Curves (OOS) — Startkapital {initial_capital:,.0f} €",
        xaxis_title="Datum", yaxis_title="Kapital (€)",
        template="plotly_white", height=500,
        hovermode="x unified",
        xaxis=dict(hoverformat="%Y-%m-%d"),
        yaxis=dict(tickformat=",.0f", automargin=True, title_standoff=18),
        margin=dict(l=110, r=20, t=50, b=40),
    )
    return _fig_to_json(fig)


@router.get("/chart/drawdown")
def chart_drawdown():
    """Drawdown-Verlauf aller Strategien."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("backtesting_results"), "/backtest/run zuerst")
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
        title="Drawdown-Verlauf (OOS)",
        xaxis_title="Datum", yaxis_title="Drawdown (%)",
        template="plotly_white", height=420,
        hovermode="x unified",
        xaxis=dict(hoverformat="%Y-%m-%d"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return _fig_to_json(fig)


@router.get("/chart/rolling-sharpe")
def chart_rolling_sharpe(window: int = Query(252, ge=21, le=1260)):
    """Rolling Sharpe über konfigurierbares Fenster."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("backtesting_results"), "/backtest/run zuerst")
    rets = df.pct_change().dropna()

    fig = go.Figure()
    for col in rets.columns:
        rolling = rets[col].rolling(window).mean() / rets[col].rolling(window).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values, mode="lines", name=col,
            line=dict(color=_plotly_color(cfg.color_map.get(col)), width=1.4),
            hovertemplate=f"<b>{col}</b>: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title=f"Rolling Sharpe (Fenster = {window} Tage)",
        xaxis_title="Datum", yaxis_title="Sharpe Ratio (annualisiert)",
        template="plotly_white", height=420,
        hovermode="x unified",
        xaxis=dict(hoverformat="%Y-%m-%d"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Regime-Chart
# ---------------------------------------------------------------------------

@router.get("/chart/regime-overlay")
def chart_regime_overlay(model: str = Query("MSM", pattern="^(MSM|HMM|LSTM|Transformer)$")):
    """Preis + Bear-Probability + Signal-Overlay für ein Modell (OOS)."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("test_data"),
                              "/models/train-all + /backtest/run zuerst")

    prob_col = f"{model}_Prob"
    sig_col = f"{model}_Signal"
    if prob_col not in df.columns or sig_col not in df.columns:
        raise HTTPException(400, f"{prob_col}/{sig_col} fehlt in test_data")

    price_col = "Cumulative_Returns" if "Cumulative_Returns" in df.columns else None
    color = _plotly_color(cfg.color_map.get(model, "steelblue"))

    fig = go.Figure()
    if price_col:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[price_col], mode="lines", name="60/40 Kurs",
            line=dict(color="#444", width=1.2),
            yaxis="y2",
            # Unified-Hover blendet das Datum einmal global ein → hier nur y.
            hovertemplate="Kurs %{y:.3f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df[prob_col], mode="lines",
        name=f"{model} Bear-Prob",
        line=dict(color=color, width=1.4),
        fill="tozeroy", fillcolor=_rgba(color, 0.15),
        hovertemplate="Prob %{y:.3f}<extra></extra>",
    ))

    # Bear-Signal-Bänder als shapes
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
        title=f"Regime-Overlay – {model}",
        template="plotly_white", height=500,
        hovermode="x unified",
        # Einheitliches Datums-Format im unified-Hover-Header.
        xaxis=dict(hoverformat="%Y-%m-%d"),
        shapes=shapes,
        yaxis=dict(title="Bear-Probability", range=[0, 1]),
        yaxis2=dict(title="Kurs", overlaying="y", side="right", showgrid=False),
        margin=dict(l=40, r=60, t=50, b=40),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# MCS-Chart
# ---------------------------------------------------------------------------

@router.get("/chart/mcs-quantiles")
def chart_mcs_quantiles(scenario: str = Query("Standard"), strategy: str = Query("Transformer")):
    """Quantil-Fächer (5 / 25 / 50 / 75 / 95 %) der MCS-Pfade."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("mcs_data"), "/backtest/evaluate zuerst")

    # Spalten folgen dem Schema: {scenario}_{strategy}_path_{N:03d}
    # z.B. "Standard_Transformer_path_000", "Standard_Transformer_path_001", ...
    prefix = f"{scenario}_{strategy}_path_"
    path_cols = [c for c in df.columns if c.startswith(prefix)]
    if not path_cols:
        available = sorted({c.rsplit("_path_", 1)[0] for c in df.columns if "_path_" in c})
        raise HTTPException(400, f"Kombination ({scenario}, {strategy}) nicht gefunden. "
                                  f"Verfügbare Kombinationen: {available}")

    # matrix: (n_paths, n_days) — jede Spalte ist ein Pfad, jede Zeile ein Tag
    matrix = df[path_cols].values.T      # Transponieren: Zeilen=Pfade, Spalten=Tage

    q = np.quantile(matrix, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    days = np.arange(matrix.shape[1])
    color = _plotly_color(cfg.color_map.get(strategy, "steelblue"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=q[4], mode="lines", name="95%",
                              line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=q[0], mode="lines", name="5-95%",
                              fill="tonexty", fillcolor=_rgba(color, 0.15),
                              line=dict(width=0)))
    fig.add_trace(go.Scatter(x=days, y=q[3], mode="lines",
                              line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=q[1], mode="lines", name="25-75%",
                              fill="tonexty", fillcolor=_rgba(color, 0.3),
                              line=dict(width=0)))
    fig.add_trace(go.Scatter(x=days, y=q[2], mode="lines", name="Median",
                              line=dict(color=color, width=2)))
    fig.update_layout(
        title=f"MCS-Quantile – {scenario} / {strategy}",
        xaxis_title="Handelstage", yaxis_title="Kapital",
        template="plotly_white", height=460,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Helpers — Farben
# ---------------------------------------------------------------------------

def _plotly_color(c) -> str:
    """Konvertiert matplotlib-Namen ('tab:blue', 'darkorange', ...) in hex."""
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
