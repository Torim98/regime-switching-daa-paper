"""Rolling-Window Feature-Konstruktion für Regime-Erkennung."""

import pandas as pd


def engineer_features(
    df: pd.DataFrame,
    volatility_window: int,
    sma_window: int,
    momentum_window: int,
) -> pd.DataFrame:
    """
    Berechnet alle Features aus dem preprocessed DataFrame:
    - Vol_20: Rollierende Standardabweichung der Portfolio-Renditen
    - SMA_200: Rollierender Mittelwert der kumulativen Renditen
      (Wir nutzen 'Cumulative_Returns' als unseren "Preis",
      da dies den Wert des 60/40 Portfolios über die Zeit darstellt)
    - Distance_SMA: Relative Abweichung vom gleitenden Durchschnitt
    - Momentum: Rollierender Mittelwert der Renditen
    - Yield_Spread: Renditestrukturkurve (10Y - 3M Spread)
      Ein inverser Spread (3M > 10Y) ist ein klassischer Rezessionsindikator

    Zeilen mit NaN-Werten (durch rolling) werden entfernt.
    """
    result = df.copy()

    result["Vol_20"] = result["Returns"].rolling(volatility_window).std()
    result["SMA_200"] = result["Cumulative_Returns"].rolling(sma_window).mean()
    result["Distance_SMA"] = (
        (result["Cumulative_Returns"] - result["SMA_200"]) / result["SMA_200"]
    )
    result["Momentum"] = result["Returns"].rolling(momentum_window).mean()
    # Renditestrukturkurve (10Y - 3M Spread) - Ein inverser Spread (3M > 10Y) ist ein klassischer Rezessionsindikator
    result["Yield_Spread"] = result["TNX_10Y"] - result["IRX_3M"]

    # Zeilen mit NaN-Werten (durch rolling) entfernen
    result = result.dropna()

    return result