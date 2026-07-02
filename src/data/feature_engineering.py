"""Rolling-window feature construction for regime detection."""

import pandas as pd


def engineer_features(
    df: pd.DataFrame,
    volatility_window: int,
    sma_window: int,
    momentum_window: int,
) -> pd.DataFrame:
    """
    Computes all features from the preprocessed DataFrame:
    - Vol_20: rolling standard deviation of the portfolio returns
    - SMA_200: rolling mean of the cumulative returns
      (we use 'Cumulative_Returns' as our "price",
      since it represents the value of the 60/40 portfolio over time)
    - Distance_SMA: relative deviation from the moving average
    - Momentum: rolling mean of the returns
    - Yield_Spread: yield curve (10Y - 3M spread)
      An inverted spread (3M > 10Y) is a classic recession indicator

    Rows with NaN values (caused by rolling) are removed.
    """
    result = df.copy()

    result["Vol_20"] = result["Returns"].rolling(volatility_window).std()
    result["SMA_200"] = result["Cumulative_Returns"].rolling(sma_window).mean()
    result["Distance_SMA"] = (
        (result["Cumulative_Returns"] - result["SMA_200"]) / result["SMA_200"]
    )
    result["Momentum"] = result["Returns"].rolling(momentum_window).mean()
    # Yield curve (10Y - 3M spread): an inverted spread (3M > 10Y) is a classic recession indicator
    result["Yield_Spread"] = result["TNX_10Y"] - result["IRX_3M"]

    # Remove rows with NaN values (caused by rolling)
    result = result.dropna()

    return result
