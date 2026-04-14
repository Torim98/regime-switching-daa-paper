"""
NBER-Rezessions-Dates ueber FRED (Serie USREC).

Referenz
--------
Burns & Mitchell (1946); NBER Business Cycle Dating Committee.
https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions

Hinweis
-------
USREC ist monatlich (0/1). Wir resamplen auf Tagesfrequenz via forward-fill
und schneiden auf den uebergebenen Index zu.

Fallback ohne `fredapi`: statische URL der FRED-CSV-Exports.
"""

from __future__ import annotations

import pandas as pd


FRED_USREC_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"


def load_nber_recession(
    index: pd.DatetimeIndex,
    source: str = "fred_csv",
) -> pd.Series:
    """
    Laedt NBER-Rezessionsflagge und projiziert sie auf `index`.

    Parameter
    ---------
    index : pd.DatetimeIndex
        Ziel-Index (Handelstage), auf den die monatliche Serie gemappt wird.
    source : {"fred_csv", "local"}
        "fred_csv" laedt via CSV-URL (kein API-Key noetig).
        "local" liest aus data/bronze/usrec.csv (falls offline benoetigt).
    """
    if source == "fred_csv":
        df = pd.read_csv(FRED_USREC_URL, parse_dates=["observation_date"])
        df = df.rename(columns={"observation_date": "date", "USREC": "recession"})
    elif source == "local":
        df = pd.read_csv("data/bronze/usrec.csv", parse_dates=["date"])
    else:
        raise ValueError(f"Unbekannte source: {source}")

    df = df.set_index("date").sort_index()
    # Monatlich -> taeglich per forward-fill, dann auf Ziel-Index
    daily = df["recession"].resample("D").ffill()
    aligned = daily.reindex(index, method="ffill").fillna(0).astype("int8")
    aligned.name = "NBER_Signal"
    return aligned