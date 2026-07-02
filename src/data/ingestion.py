"""Yahoo Finance data download and raw-data persistence."""

import time
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def _extract_close_frame(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Schema-robust extraction of the (Adj) Close columns from the raw yfinance frame.

    Since yfinance 0.2.40+, the MultiIndex can vary depending on call/version
    ("Field"/"Ticker" vs. "Ticker"/"Field", with/without "Adj Close"). This function
    covers all known layouts and returns a flat DataFrame in which the columns
    correspond to the ticker names.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        lvl1 = raw.columns.get_level_values(1)
        if "Adj Close" in lvl0:
            return raw["Adj Close"].copy()
        if "Close" in lvl0:
            return raw["Close"].copy()
        if "Adj Close" in lvl1:
            return raw.xs("Adj Close", axis=1, level=1).copy()
        if "Close" in lvl1:
            return raw.xs("Close", axis=1, level=1).copy()
        raise RuntimeError(f"No Close/Adj Close in columns: {raw.columns}")

    # Flat columns (single-ticker case)
    col = "Adj Close" if "Adj Close" in raw.columns else "Close"
    return raw[[col]].rename(columns={col: tickers[0]})


def _resolve_end_exclusive(end_date):
    """
    yfinance treats `end` as exclusive. We add +1 day so that
    `end_date` is inclusive from the user's perspective (important in thesis freeze mode).
    None/empty/whitespace -> None (yfinance default: up to today).
    """
    if end_date is None:
        return None
    if isinstance(end_date, str) and not end_date.strip():
        return None
    return (
        datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")


def _download_once(
    tickers: list[str],
    start_date: str,
    end_exclusive,
    threads: bool = True,
) -> pd.DataFrame:
    """Single download with an enforced classic schema."""
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_exclusive,
        auto_adjust=False,       # enforces the classic schema with "Adj Close"
        progress=False,
        group_by="column",       # (Field, Ticker) order
        threads=threads,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned empty frame")
    data = _extract_close_frame(raw, tickers)
    keep = [t for t in tickers if t in data.columns]
    return data[keep].copy()


def download_market_data(
    tickers: list[str],
    start_date: str,
    end_date,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Download market data from Yahoo Finance (robust against yfinance schema
    changes and occasional Yahoo outages for mutual-fund tickers such as VUSTX).

    Strategy:
    1. Bulk download of all tickers in one go (threads=True).
    2. Missing/empty columns are reloaded individually (threads=False,
       sequentially), which mitigates Yahoo rate limits for single tickers.
    3. The whole process is repeated up to `max_retries` times if tickers
       are still missing at the end.

    Note: yfinance treats `end` as exclusive. We add +1 day so that
    `end_date` is inclusive from the user's perspective (important in thesis freeze mode).

    ^GSPC = S&P 500 | VUSTX = long bonds | ^VIX = volatility
    ^IRX  = 3-month rate | ^TNX = 10-year rate
    """
    end_exclusive = _resolve_end_exclusive(end_date)
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            data = _download_once(
                tickers, start_date, end_exclusive, threads=True,
            )

            # Identify missing or completely empty tickers
            missing = [
                t for t in tickers
                if t not in data.columns or data[t].dropna().empty
            ]

            # Individual reload for missing tickers (e.g. VUSTX during a Yahoo outage)
            for t in missing:
                try:
                    single = _download_once(
                        [t], start_date, end_exclusive, threads=False,
                    )
                    if t in single.columns and not single[t].dropna().empty:
                        data[t] = single[t]
                except Exception as e:
                    last_err = e
                    continue

            # Final check after bulk + single retry
            still_missing = [
                t for t in tickers
                if t not in data.columns or data[t].dropna().empty
            ]
            if still_missing:
                raise RuntimeError(
                    f"No data for tickers after bulk + single retry: "
                    f"{still_missing}. Yahoo may be rate-limiting or the "
                    f"tickers are temporarily unavailable."
                )

            # Order columns as in the input (important downstream)
            return data[tickers].copy()

        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)   # 1s, 2s, 4s
                continue
            raise RuntimeError(
                f"yfinance download failed after {max_retries} attempts: "
                f"{last_err}"
            ) from last_err


def save_raw_data(data: pd.DataFrame, output_path: str) -> None:
    """Persist raw data in the Bronze layer (before any cleaning)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path)
