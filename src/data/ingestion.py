"""Yahoo Finance Datendownload und Rohdaten-Persistierung."""

import time
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def _extract_close_frame(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Schema-robuste Extraktion der (Adj-)Close-Spalten aus dem yfinance-Rohframe.

    Seit yfinance 0.2.40+ kann der MultiIndex je nach Aufruf/Version variieren
    ("Field"/"Ticker" vs. "Ticker"/"Field", mit/ohne "Adj Close"). Diese Funktion
    deckt alle bekannten Layouts ab und gibt eine flache DataFrame zurück,
    in der die Spalten den Ticker-Namen entsprechen.
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

    # Flache Spalten (Single-Ticker-Fall)
    col = "Adj Close" if "Adj Close" in raw.columns else "Close"
    return raw[[col]].rename(columns={col: tickers[0]})


def _resolve_end_exclusive(end_date):
    """
    yfinance behandelt `end` exklusiv. Wir addieren +1 Tag, damit
    `end_date` aus Nutzersicht inklusiv ist (wichtig im Thesis-Freeze-Mode).
    None/empty/whitespace -> None (yfinance-Standard: bis heute).
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
    """Einzel-Download mit erzwungenem klassischem Schema."""
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_exclusive,
        auto_adjust=False,       # erzwingt klassisches Schema mit "Adj Close"
        progress=False,
        group_by="column",       # (Field, Ticker) Reihenfolge
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
    Marktdaten von Yahoo Finance herunterladen (robust gegen yfinance-Schema-
    Wechsel und gelegentliche Yahoo-Ausfaelle bei Mutual-Fund-Tickern wie VUSTX).

    Strategie:
    1. Bulk-Download aller Ticker in einem Rutsch (threads=True).
    2. Fehlen/leere Spalten werden einzeln (threads=False, sequenziell)
       nachgeladen — entschaerft Yahoo-Ratelimits bei einzelnen Tickern.
    3. Der gesamte Prozess wird bis zu `max_retries` wiederholt, wenn am
       Ende noch Ticker fehlen.

    Hinweis: yfinance behandelt `end` exklusiv. Wir addieren +1 Tag, damit
    `end_date` aus Nutzersicht inklusiv ist (wichtig im Thesis-Freeze-Mode).

    ^GSPC = S&P 500 | VUSTX = Long Bonds | ^VIX = Volatilitaet
    ^IRX  = 3-Monats-Zins | ^TNX = 10-Jahres-Zins
    """
    end_exclusive = _resolve_end_exclusive(end_date)
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            data = _download_once(
                tickers, start_date, end_exclusive, threads=True,
            )

            # Fehlende oder komplett leere Ticker identifizieren
            missing = [
                t for t in tickers
                if t not in data.columns or data[t].dropna().empty
            ]

            # Einzel-Nachladen fuer fehlende Ticker (z.B. VUSTX bei Yahoo-Ausfall)
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

            # Finale Pruefung nach Bulk + Einzel-Retry
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

            # Spalten in Input-Reihenfolge bringen (wichtig fuer downstream)
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
    """Rohdaten im Bronze-Layer sichern (vor jeglicher Bereinigung)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path)
