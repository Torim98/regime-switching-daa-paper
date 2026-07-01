"""Datenqualitätsprüfung (Issue #2): Coverage, Missing Values, Adjustment-Plausibilität.

Erzeugt einen Markdown-Report über die Rohdaten (Bronze, VOR ffill/dropna) und
die Auswirkung der Bereinigung (Silver). Reine Funktionen ohne Config-Abhängigkeit;
die aufrufende Route (data_service) übergibt Freeze-Metadaten und schreibt die Datei.

Analog zu src/data/eda.py gehalten: DataFrame-basierte Teil-Reports, die im
Master-Report als Markdown-Tabellen zusammengesetzt werden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError


# Preis-basierte Ticker (Log-Return-Plausibilität) vs. Level-Serien (VIX/Zinsen)
PRICE_TICKERS = ["^GSPC", "VUSTX"]


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def _yfinance_version() -> str:
    """Installierte yfinance-Version für die Freeze-Dokumentation."""
    try:
        return version("yfinance")
    except PackageNotFoundError:
        return "unknown"


def _fmt_date(ts) -> str:
    """Timestamp -> 'YYYY-MM-DD' (robust gegen None/NaT)."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)) or pd.isna(ts):
        return "n/a"
    return pd.Timestamp(ts).date().isoformat()


def _longest_nan_run(series: pd.Series) -> int:
    """Längste zusammenhängende Folge fehlender Werte (in Beobachtungen)."""
    isna = series.isna().to_numpy()
    if not isna.any():
        return 0
    max_run = run = 0
    for flag in isna:
        run = run + 1 if flag else 0
        if run > max_run:
            max_run = run
    return int(max_run)


def _robust_z(x: pd.Series) -> pd.Series:
    """MAD-basierter robuster z-Score (fat-tail-tauglich, konsistent zu RobustScaler)."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - med) / (1.4826 * mad)


# --------------------------------------------------------------------------- #
# Teil-Reports
# --------------------------------------------------------------------------- #
def coverage_report(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Abdeckung je Ticker: Zeitraum, beobachtete vs. erwartete Handelstage.

    Erwartete Handelstage werden über `pd.bdate_range` (Mo–Fr) approximiert.
    Börsenfeiertage sind darin enthalten, daher ist die Coverage eine konservative
    Untergrenze (auch bei lückenlosen Daten < 100 % wegen Feiertagen).
    """
    rows = []
    for col in raw_df.columns:
        valid = raw_df[col].dropna()
        if valid.empty:
            rows.append({
                "Ticker": col, "Von": "n/a", "Bis": "n/a",
                "Beob. Tage": 0, "Erw. Bd (Mo–Fr)": 0, "Coverage %": "n/a",
            })
            continue
        first, last = valid.index.min(), valid.index.max()
        expected = len(pd.bdate_range(first, last))
        observed = int(valid.shape[0])
        rows.append({
            "Ticker": col,
            "Von": _fmt_date(first),
            "Bis": _fmt_date(last),
            "Beob. Tage": observed,
            "Erw. Bd (Mo–Fr)": expected,
            "Coverage %": f"{100 * observed / expected:.2f}" if expected else "n/a",
        })
    return pd.DataFrame(rows).set_index("Ticker")


def missing_value_report(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Fehlende Werte je Ticker im ROH-Frame (vor ffill/dropna).

    Macht den in preprocessing.fill_missing_values() stillen Datenverlust sichtbar.
    """
    rows = []
    n = len(raw_df)
    for col in raw_df.columns:
        s = raw_df[col]
        n_missing = int(s.isna().sum())
        rows.append({
            "Ticker": col,
            "NaN (roh)": n_missing,
            "NaN %": f"{100 * n_missing / n:.3f}" if n else "n/a",
            "Längste Lücke (Tage)": _longest_nan_run(s),
            "Erster Wert": _fmt_date(s.first_valid_index()),
            "Letzter Wert": _fmt_date(s.last_valid_index()),
        })
    return pd.DataFrame(rows).set_index("Ticker")


def adjustment_jump_report(
    raw_df: pd.DataFrame,
    price_tickers: list[str] = PRICE_TICKERS,
    z_thresh: float = 8.0,
) -> pd.DataFrame:
    """Sprung-/Ausreißer-Check auf Log-Renditen der Preis-Serien.

    Proxy für Adjustment-Fehler: eine fehlerhafte Split-/Dividenden-Bereinigung
    zeigt sich als unplausibler Tagessprung. Gemeldet werden je Ticker die Anzahl
    Tage mit |robust-z| > z_thresh sowie die größte Tagesbewegung.
    """
    rows = []
    for col in price_tickers:
        if col not in raw_df.columns:
            continue
        s = raw_df[col].dropna()
        logret = np.log(s / s.shift(1)).dropna()
        if logret.empty:
            continue
        z = _robust_z(logret)
        rows.append({
            "Ticker": col,
            "Max. abs. Tagesrendite": f"{logret.abs().max():.4f}",
            f"Ausreißertage (z>{z_thresh:g})": int((z.abs() > z_thresh).sum()),
            "Größter Sprung (Datum)": _fmt_date(logret.abs().idxmax()),
        })
    return pd.DataFrame(rows).set_index("Ticker")


def worst_moves_report(
    raw_df: pd.DataFrame,
    price_tickers: list[str] = PRICE_TICKERS,
    top_n: int = 5,
) -> pd.DataFrame:
    """Top-N größte absolute Tagesbewegungen je Preis-Serie (Plausibilitätssicht).

    Dient dem Abgleich mit bekannten Krisentagen (z. B. 2008-10, 2020-03):
    Sprünge an bekannten Terminen sind plausibel, isolierte Sprünge nicht.
    """
    rows = []
    for col in price_tickers:
        if col not in raw_df.columns:
            continue
        s = raw_df[col].dropna()
        logret = np.log(s / s.shift(1)).dropna()
        if logret.empty:
            continue
        top = logret.reindex(logret.abs().sort_values(ascending=False).index).head(top_n)
        for rank, (date, val) in enumerate(top.items(), start=1):
            rows.append({
                "Ticker": col,
                "Rang": rank,
                "Datum": _fmt_date(date),
                "Log-Rendite": f"{val:+.4f}",
            })
    return pd.DataFrame(rows).set_index("Ticker")


def cleaning_impact_report(
    raw_df: pd.DataFrame,
    preprocessed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Auswirkung der Bereinigung: Zeilen roh vs. preprocessed.

    Die Differenz umfasst per dropna() entfernte Start-Zeilen (kein ffill-Anker)
    sowie die eine durch die Log-Return-Bildung (shift) verlorene Zeile.
    """
    n_raw = len(raw_df)
    n_pre = len(preprocessed_df)
    dropped = n_raw - n_pre
    rows = [
        {"Kennzahl": "Zeilen roh (Bronze)", "Wert": n_raw},
        {"Kennzahl": "Zeilen bereinigt (Silver)", "Wert": n_pre},
        {"Kennzahl": "Entfernt (dropna + Return-Shift)", "Wert": dropped},
        {"Kennzahl": "Entfernt %", "Wert": f"{100 * dropped / n_raw:.3f}" if n_raw else "n/a"},
    ]
    return pd.DataFrame(rows).set_index("Kennzahl")


# --------------------------------------------------------------------------- #
# Master-Report
# --------------------------------------------------------------------------- #
def build_data_quality_report(
    raw_df: pd.DataFrame,
    preprocessed_df: pd.DataFrame,
    *,
    freeze_date: str,
    is_frozen: bool,
    price_tickers: list[str] = PRICE_TICKERS,
) -> str:
    """Vollständigen Data-Quality-Report als Markdown-String zusammensetzen.

    Parameter
    ---------
    raw_df : Bronze-Frame (Ticker-Spalten, VOR ffill/dropna).
    preprocessed_df : Silver-Frame (nach preprocess_pipeline).
    freeze_date : cfg.data.end_date (aufgelöstes Enddatum).
    is_frozen : cfg.data.end_date_is_frozen (False = dynamisch/rolling).
    """
    mode = "Freeze (fester Cutoff)" if is_frozen else "Rolling (dynamisch = letzter Handelstag)"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    idx = raw_df.index
    span = f"{_fmt_date(idx.min())} – {_fmt_date(idx.max())}" if len(idx) else "n/a"

    # Teil-Reports einmal berechnen (für Verdict UND Sektionen wiederverwendet)
    cov = coverage_report(raw_df)
    miss = missing_value_report(raw_df)
    cov_min = pd.to_numeric(cov["Coverage %"], errors="coerce").min()
    gap_max = int(miss["Längste Lücke (Tage)"].max()) if len(miss) else 0
    verdict = (
        f"Coverage ≥ {cov_min:.1f} % · max. Lücke {gap_max} Tage"
        if pd.notna(cov_min) else "n/a"
    )

    parts: list[str] = []
    parts.append("# Data Quality Report")
    parts.append("")
    parts.append(f"- **Status:** {verdict}")
    parts.append(f"- **Zeitraum (roh):** {span}")
    parts.append(f"- **End-Datum-Modus:** {mode}")
    parts.append(f"- **Aufgelöstes Enddatum:** `{freeze_date}`")
    parts.append(f"- **yfinance-Version:** `{_yfinance_version()}`")
    parts.append(f"- **Ticker:** {', '.join(raw_df.columns)}")
    parts.append(f"- **Erzeugt am:** {ts}")
    parts.append("")
    parts.append("## 1. Coverage (beobachtete vs. erwartete Handelstage)")
    parts.append(cov.to_markdown())
    parts.append("")
    parts.append(
        "_Hinweis: Erwartete Handelstage aus `bdate_range` (Mo–Fr inkl. Feiertage). "
        "~96–97 % sind die feiertagsbedingte Untergrenze, kein Datenverlust._"
    )
    parts.append("")
    parts.append("## 2. Fehlende Werte (Roh-Frame, vor ffill/dropna)")
    parts.append(miss.to_markdown())
    parts.append("")
    parts.append("## 3. Adjustment-Plausibilität (Tagessprünge Preis-Serien)")
    parts.append(adjustment_jump_report(raw_df, price_tickers).to_markdown())
    parts.append("")
    parts.append("## 4. Größte Tagesbewegungen (Krisen-Plausibilität)")
    parts.append(worst_moves_report(raw_df, price_tickers).to_markdown())
    parts.append("")
    parts.append("## 5. Auswirkung der Bereinigung (Bronze → Silver)")
    parts.append(cleaning_impact_report(raw_df, preprocessed_df).to_markdown())
    parts.append("")
    return "\n".join(parts)