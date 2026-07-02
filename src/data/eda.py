"""Exploratory data analysis (EDA): descriptive statistics and stationarity tests."""

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller


def calculate_descriptive_stats(
    data_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Compute the descriptive statistics table.
    Per column: mean, standard deviation, min, max, skewness, kurtosis.
    """
    stats_list = []
    for col in columns:
        series = data_df[col].dropna()
        stats_list.append({
            "Time Series": col,
            "Mean (daily)": f"{series.mean():.6f}",
            "Std. Dev. (daily)": f"{series.std():.6f}",
            "Min": f"{series.min():.4f}",
            "Max": f"{series.max():.4f}",
            "Skewness": f"{stats.skew(series):.4f}",
            "Kurtosis": f"{stats.kurtosis(series):.4f}",
        })
    return pd.DataFrame(stats_list).set_index("Time Series")


def run_adf_test(
    data_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Stationarity test (augmented Dickey-Fuller) per column.
    Returns the ADF statistic, p-value, critical value (5%), and stationarity verdict.
    """
    adf_results = []
    for col in columns:
        series = data_df[col].dropna()
        result = adfuller(series)
        adf_results.append({
            "Time Series": col,
            "ADF Statistic": f"{result[0]:.4f}",
            "p-Value": f"{result[1]:.4e}",
            "Crit. Value (5%)": f"{result[4]['5%']:.4f}",
            "Stationary?": "Yes" if result[1] < 0.05 else "No",
        })
    return pd.DataFrame(adf_results).set_index("Time Series")
