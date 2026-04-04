"""Explorative Datenanalyse (EDA) — Deskriptive Statistik und Stationaritätstests."""

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller


def calculate_descriptive_stats(
    data_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Deskriptive Statistik-Tabelle berechnen.
    Pro Spalte: Mittelwert, Standardabweichung, Min, Max, Schiefe, Kurtosis.
    """
    stats_list = []
    for col in columns:
        series = data_df[col].dropna()
        stats_list.append({
            "Zeitreihe": col,
            "Mittelwert (tägl.)": f"{series.mean():.6f}",
            "Std.Abw. (tägl.)": f"{series.std():.6f}",
            "Min": f"{series.min():.4f}",
            "Max": f"{series.max():.4f}",
            "Schiefe (Skew)": f"{stats.skew(series):.4f}",
            "Kurtosis": f"{stats.kurtosis(series):.4f}",
        })
    return pd.DataFrame(stats_list).set_index("Zeitreihe")


def run_adf_test(
    data_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Stationaritätstest (Augmented Dickey-Fuller) pro Spalte.
    Gibt ADF-Statistik, p-Wert, kritischen Wert (5%) und Stationaritäts-Urteil zurück.
    """
    adf_results = []
    for col in columns:
        series = data_df[col].dropna()
        result = adfuller(series)
        adf_results.append({
            "Zeitreihe": col,
            "ADF-Statistik": f"{result[0]:.4f}",
            "p-Wert": f"{result[1]:.4e}",
            "Krit. Wert (5%)": f"{result[4]['5%']:.4f}",
            "Stationär?": "Ja" if result[1] < 0.05 else "Nein",
        })
    return pd.DataFrame(adf_results).set_index("Zeitreihe")