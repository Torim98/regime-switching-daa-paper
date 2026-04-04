"""Yahoo Finance Datendownload und Rohdaten-Persistierung."""

import yfinance as yf
import pandas as pd
from pathlib import Path


def download_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Marktdaten von Yahoo Finance herunterladen.
    ^GSPC = S&P 500 | VUSTX = Long Bonds | ^VIX = Volatilität | ^IRX = 3-Monats-Zins | ^TNX = 10-Jahres-Zins

    Wählt 'Adj Close' wenn verfügbar, sonst 'Close'.
    Gibt DataFrame mit einer Spalte pro Ticker (tägliche Kurse) zurück.
    """
    # Alles ohne Index-Zugriff runterladen
    # Enddatum: yfinance lädt exklusive Enddatum, d.h. bis gestern inklusive
    raw_data = yf.download(tickers, start=start_date, end=end_date)

    # --- ROBUSTER MULTI-INDEX FIX (Keyerror-Fix) ---
    # Wir prüfen, welcher Preis-Typ verfügbar ist ('Adj Close' bevorzugt, sonst 'Close')
    # In neueren yfinance-Versionen wurde 'Adj Close' oft durch 'Close' ersetzt,
    # wenn die Daten bereits bereinigt sind. Prüfung beider Cases.
    if "Adj Close" in raw_data.columns.get_level_values(0):
        data = raw_data["Adj Close"].copy()
    else:
        data = raw_data["Close"].copy()

    return data


def save_raw_data(data: pd.DataFrame, output_path: str) -> None:
    """Rohdaten im Bronze-Layer sichern (vor jeglicher Bereinigung)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path)