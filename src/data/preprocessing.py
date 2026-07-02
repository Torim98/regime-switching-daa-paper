"""Portfolio construction, return calculation, and data cleaning."""

import pandas as pd
import numpy as np

def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values.
    Level series (VIX, IRX, TNX): forward fill, since volatility/interest rates
    only update on trading days and holiday/reporting gaps carry the last known
    value forward. Price series (GSPC, VUSTX) are deliberately NOT forward-filled,
    since a carried-forward price would create an artificial zero return.
    Remaining NaNs at the start of the series are removed (no ffill anchor available).
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
    Compute log returns (continuously compounded returns).
    r_t = ln(P_t / P_{t-1}): additive, symmetric, closer to a normal distribution.
    Log returns only for price-based assets (not for interest rate/volatility levels).
    """
    price_ratio = (data[price_tickers] / data[price_tickers].shift(1)).dropna()
    return np.log(price_ratio)


def construct_portfolio(
    log_returns: pd.DataFrame,
    weight_equity: float,
    weight_bonds: float,
) -> pd.Series:
    """
    Portfolio construction (e.g. 60% S&P 500, 40% long-term bonds).
    Weighted sum of the log returns.
    """
    weights = np.array([weight_equity, weight_bonds])
    return (log_returns[["^GSPC", "VUSTX"]] * weights).sum(axis=1)


def build_preprocessed_dataframe(
    data: pd.DataFrame,
    log_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    """
    Assemble the final DataFrame with all columns:
    - Returns_GSPC, Returns_VUSTX: individual returns of the S&P 500 and bonds
    - Returns: weighted portfolio return
    - Cumulative_Returns: cumulative return (for log returns via exp(cumsum))
    - Cash_Returns: ^IRX quotes the annual yield in %.
      Conversion to a daily return: (value / 100) / 252 trading days.
      Level data: no log, direct access (pandas aligns automatically on the index)
    - VIX, TNX_10Y, IRX_3M: feature columns
    """
    df = pd.DataFrame(index=portfolio_returns.index)

    # Individual returns of the S&P 500 and bonds
    df["Returns_GSPC"] = log_returns["^GSPC"]
    df["Returns_VUSTX"] = log_returns["VUSTX"]

    df["Returns"] = portfolio_returns
    # Cumulative return: for log returns via exp(cumsum)
    df["Cumulative_Returns"] = np.exp(df["Returns"].cumsum())

    # --- CASH RETURN INTEGRATION ---
    # ^IRX quotes the annual yield in %. Conversion to a daily return:
    # (value / 100) / 252 trading days
    # Level data: no log, direct access (pandas aligns automatically on the index)
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
    Orchestrates the entire preprocessing flow:
    1. Handle missing values (forward fill for IRX/VIX)
    2. Compute log returns (price-based assets only)
    3. Construct the portfolio (weighted sum)
    4. Assemble the final DataFrame
    """
    # Handle missing values
    data = fill_missing_values(data)

    # Log returns only for price-based assets
    price_tickers = ["^GSPC", "VUSTX"]
    log_returns = calculate_log_returns(data, price_tickers)

    # Portfolio construction
    portfolio_returns = construct_portfolio(log_returns, weight_equity, weight_bonds)

    # Assemble the final DataFrame
    df = build_preprocessed_dataframe(data, log_returns, portfolio_returns)

    return df
