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


@router.get("/optuna/best-params")
def optuna_best_params():
    """
    Beste Hyperparameter pro Modell aus dem Optuna-SQLite-Storage.

    Ermöglicht der Dashboard-UI eine strukturierte Card-Darstellung statt
    Markdown-Rendering — inkl. Best Score, Trial-Counts und Parameter-Map.
    """
    cfg = _cfg()
    db_path = Path(cfg.model_path("optuna_db"))
    if not db_path.exists():
        raise HTTPException(
            404, "Keine Optuna-DB gefunden — Optimierung noch nicht durchgeführt."
        )

    try:
        import optuna
    except ImportError:
        raise HTTPException(500, "optuna nicht installiert.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{db_path}"

    results = []
    for model_name in ["MSM", "HMM", "LSTM", "Transformer"]:
        try:
            study = optuna.load_study(
                study_name=f"opt_{model_name}", storage=storage,
            )
        except KeyError:
            continue

        try:
            best_value = float(study.best_value)
            best_params = dict(study.best_params)
        except ValueError:
            continue  # keine abgeschlossenen Trials

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
        raise HTTPException(404, "Keine Optuna-Studies in der DB gefunden.")

    return {
        "metric": "Sharpe (Median OOS)",
        "results": results,
    }


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


@router.get("/chart/volatility-clusters")
def chart_volatility_clusters():
    """S&P 500 Renditen + ACF der quadrierten Renditen (GARCH-Effekte)."""
    from plotly.subplots import make_subplots
    from statsmodels.tsa.stattools import acf

    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("feature_engineered"), "/data/ingest zuerst")
    if "Returns_GSPC" not in df.columns:
        raise HTTPException(400, "Returns_GSPC fehlt im DataFrame")

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
            "S&P 500 Tägliche Renditen — Visualisierung von Volatilitätsclustern",
            "Autokorrelation der quadrierten Renditen — Nachweis von GARCH-Effekten",
        ),
    )

    fig.add_trace(go.Scatter(
        x=rets.index, y=rets.values, mode="lines",
        name="Returns_GSPC",
        line=dict(color="#1f77b4", width=0.8),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # 95%-Konfidenzband um 0
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

    # Stems (vertikale Linien vom Nullniveau zum ACF-Wert)
    stem_x, stem_y = [], []
    for lag, val in zip(lags, ac_values):
        stem_x.extend([lag, lag, None])
        stem_y.extend([0, val, None])
    fig.add_trace(go.Scatter(
        x=stem_x, y=stem_y, mode="lines",
        line=dict(color="#1f77b4", width=1.4),
        hoverinfo="skip", showlegend=False,
    ), row=2, col=1)

    # ACF-Marker
    fig.add_trace(go.Scatter(
        x=lags, y=ac_values, mode="markers",
        marker=dict(color="#d62728", size=7),
        hovertemplate="Lag %{x}<br>ρ = %{y:.3f}<extra></extra>",
        showlegend=False, name="ACF",
    ), row=2, col=1)

    fig.add_hline(y=0, row=2, col=1, line_color="#d62728", line_width=1)

    fig.update_xaxes(title_text="Datum", row=1, col=1, hoverformat="%Y-%m-%d")
    fig.update_yaxes(title_text="Rendite", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_yaxes(title_text="Autokorrelation", row=2, col=1, range=[-1, 1])

    fig.update_layout(
        template="plotly_white", height=720,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return _fig_to_json(fig)


def _top_n_drawdown_periods(drawdowns: pd.Series, n: int = 5) -> list[dict]:
    """Identifiziert die n tiefsten Drawdown-Perioden (Peak → Trough → Recovery)."""
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
def chart_historical_drawdowns():
    """Historische Drawdowns des 60/40 Portfolios — mit markierten Top-5-Episoden."""
    cfg = _cfg()
    df = _read_parquet_or_404(cfg.data_path("feature_engineered"), "/data/ingest zuerst")
    if "Returns" not in df.columns:
        raise HTTPException(400, "Returns fehlt im DataFrame")

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
                "Tiefe: %{y:.2f}%<extra></extra>"
            ),
            showlegend=True,
        ))

    fig.update_layout(
        title="Historische Drawdowns des 60/40 Portfolios — Top 5 markiert",
        xaxis_title="Datum", yaxis_title="Drawdown (%)",
        template="plotly_white", height=460,
        margin=dict(l=60, r=20, t=50, b=40),
        shapes=shapes,
        xaxis=dict(hoverformat="%Y-%m-%d", automargin=True),
        legend=dict(orientation="h", y=-0.18, x=0, xanchor="left"),
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
