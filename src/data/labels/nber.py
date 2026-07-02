"""
NBER recession dates via FRED (USREC series).

Reference
---------
Burns & Mitchell (1946); NBER Business Cycle Dating Committee.
https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions

Note
----
USREC is monthly (0/1). We resample to daily frequency via forward fill
and trim to the provided index.

Fallback without `fredapi`: static URL of the FRED CSV export.
"""

from __future__ import annotations

import pandas as pd


FRED_USREC_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"


def load_nber_recession(
    index: pd.DatetimeIndex,
    source: str = "fred_csv",
) -> pd.Series:
    """
    Loads the NBER recession flag and projects it onto `index`.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Target index (trading days) onto which the monthly series is mapped.
    source : {"fred_csv", "local"}
        "fred_csv" loads via CSV URL (no API key required).
        "local" reads from data/bronze/usrec.csv (if needed offline).
    """
    if source == "fred_csv":
        df = pd.read_csv(FRED_USREC_URL, parse_dates=["observation_date"])
        df = df.rename(columns={"observation_date": "date", "USREC": "recession"})
    elif source == "local":
        df = pd.read_csv("data/bronze/usrec.csv", parse_dates=["date"])
    else:
        raise ValueError(f"Unknown source: {source}")

    df = df.set_index("date").sort_index()
    # Monthly -> daily via forward fill, then onto the target index
    daily = df["recession"].resample("D").ffill()
    aligned = daily.reindex(index, method="ffill").fillna(0).astype("int8")
    aligned.name = "NBER_Signal"
    return aligned
