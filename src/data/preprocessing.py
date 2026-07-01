"""Portfolio-Konstruktion, Renditeberechnung und Datenbereinigung."""

import pandas as pd
import numpy as np

def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fehlende Werte behandeln.
    Level-Serien (VIX, IRX, TNX): Forward-Fill, da Volatilität/Zinsen sich nur
    an Handelstagen aktualisieren und Feiertags-/Meldelücken den letzten bekannten
    Wert fortschreiben. Preis-Serien (GSPC, VUSTX) werden bewusst NICHT ge-ffillt,
    da ein fortgeschriebener Kurs eine künstliche 0-Rendite erzeugen würde.
    Verbleibende NaNs am Anfang der Serie entfernen (kein ffill-Anker vorhanden).
    """
    data = data.copy()
    level_tickers = ["^VIX", "^IRX", "^TNX"]
    data[level_tickers] = data[level_tickers].ffill()
    data = data.dropna()
    return data

def calculate_log_returns(
    data: pd.DataFrame,
    price_tickers: list[str],
) -> pd.DataFrame:
    """
    Log-Renditen (stetige Renditen) berechnen.
    r_t = ln(P_t / P_{t-1}) — additiv, symmetrisch, näher an Normalverteilung.
    Log-Renditen nur für Preis-basierte Assets (nicht für Zins-/Volatilitäts-Levels).
    """
    price_ratio = (data[price_tickers] / data[price_tickers].shift(1)).dropna()
    return np.log(price_ratio)


def construct_portfolio(
    log_returns: pd.DataFrame,
    weight_equity: float,
    weight_bonds: float,
) -> pd.Series:
    """
    Portfolio Erstellung (z.B. 60% S&P 500, 40% Long Term Bonds).
    Gewichtete Summe der Log-Renditen.
    """
    weights = np.array([weight_equity, weight_bonds])
    return (log_returns[["^GSPC", "VUSTX"]] * weights).sum(axis=1)


def build_preprocessed_dataframe(
    data: pd.DataFrame,
    log_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    """
    Finalen DataFrame mit allen Spalten zusammenstellen:
    - Returns_GSPC, Returns_VUSTX: Einzel-Returns von S&P 500 und Bonds
    - Returns: Gewichtete Portfolio-Rendite
    - Cumulative_Returns: Kumulative Rendite (bei Log-Returns über exp(cumsum))
    - Cash_Returns: ^IRX gibt die jährliche Rendite in % an.
      Umrechnung in tägliche Rendite: (Wert / 100) / 252 Handelstage.
      Level-Daten: kein Log, direkter Zugriff (Pandas aligned automatisch auf den Index)
    - VIX, TNX_10Y, IRX_3M: Feature-Spalten
    """
    df = pd.DataFrame(index=portfolio_returns.index)

    # Einzel-Returns von S&P 500 und Bonds
    df["Returns_GSPC"] = log_returns["^GSPC"]
    df["Returns_VUSTX"] = log_returns["VUSTX"]

    df["Returns"] = portfolio_returns
    # Kumulative Rendite: bei Log-Returns über exp(cumsum)
    df["Cumulative_Returns"] = np.exp(df["Returns"].cumsum())

    # --- CASH-RENDITE INTEGRATION ---
    # ^IRX gibt die jährliche Rendite in % an. Umrechnung in tägliche Rendite:
    # (Wert / 100) / 252 Handelstage
    # Level-Daten: kein Log, direkter Zugriff (Pandas aligned automatisch auf den Index)
    df["Cash_Returns"] = np.log(1 + (data["^IRX"] / 100) / 252)
    df["VIX"] = data["^VIX"]
    df["TNX_10Y"] = data["^TNX"]
    df["IRX_3M"] = data["^IRX"]

    return df


def preprocess_pipeline(
    data: pd.DataFrame,
    weight_equity: float,
    weight_bonds: float,
) -> pd.DataFrame:
    """
    Orchestriert den gesamten Preprocessing-Flow:
    1. Fehlende Werte behandeln (Forward-Fill für IRX/VIX)
    2. Log-Renditen berechnen (nur Preis-basierte Assets)
    3. Portfolio konstruieren (gewichtete Summe)
    4. Finalen DataFrame zusammenstellen
    """
    # Fehlende Werte behandeln
    data = fill_missing_values(data)

    # Log-Renditen nur für Preis-basierte Assets
    price_tickers = ["^GSPC", "VUSTX"]
    log_returns = calculate_log_returns(data, price_tickers)

    # Portfolio Erstellung
    portfolio_returns = construct_portfolio(log_returns, weight_equity, weight_bonds)

    # Finalen DataFrame zusammenstellen
    df = build_preprocessed_dataframe(data, log_returns, portfolio_returns)

    return df