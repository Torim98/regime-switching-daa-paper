"""Backtesting engine: cumulative return calculation with transaction costs."""

import pandas as pd
import numpy as np


def backtest(
    df: pd.DataFrame,
    signal_col: str,
    signal_shift: int,
    fee: float,
) -> pd.Series:
    """
    Computes the cumulative return taking transaction costs into account.
    fee: cost of a full switch (e.g. 0.1% = 0.001).

    Logic:
    - Shift the signal by a configurable number of days to prevent look-ahead bias
    - Identify trades: where does today's signal differ from yesterday's?
    - If signal 0 → portfolio return, otherwise cash return
    - Subtract transaction costs
    - Compute the cumulative return
    """
    # Shift the signal by a configurable number of days to prevent look-ahead bias
    trading_signal = df[signal_col].shift(signal_shift).fillna(0)

    # Identify trades: where does today's signal differ from yesterday's?
    trades = trading_signal.diff().fillna(0).abs()

    # Logic: if signal 0 -> portfolio return, otherwise cash return
    strategy_returns = np.where(
        trading_signal == 0,
        df["Returns"],
        df["Cash_Returns"],
    )

    # Subtract transaction costs
    net_strategy_returns = strategy_returns - (trades * fee)

    # Compute the cumulative return
    return pd.Series(np.exp(net_strategy_returns.cumsum()), index=df.index)


def run_all_backtests(
    test_df: pd.DataFrame,
    fee_rate: float,
    signal_shift: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dynamically identify all available models (via the _Signal suffix)
    and run the backtesting.

    Benchmark: buy and hold of the 60/40 portfolio (0 transaction costs, since never reallocated).

    Returns (backtesting_results, backtesting_transaction_costs).
    """
    # Dynamically identify all available models (via the _Signal suffix)
    signal_cols = [col for col in test_df.columns if col.endswith("_Signal")]

    # Initialize the results DataFrame
    backtesting_results = pd.DataFrame(index=test_df.index)
    # DataFrame for the transaction costs over time
    backtesting_transaction_costs = pd.DataFrame(index=test_df.index)

    # Compute the benchmark (buy and hold of the 60/40 portfolio)
    backtesting_results["Buy_Hold"] = np.exp(test_df["Returns"].cumsum())
    # Buy and hold has 0 transaction costs, since we never reallocate
    backtesting_transaction_costs["Buy_Hold"] = 0.0

    # Dynamically backtest all detected models
    for sig_col in signal_cols:
        model_name = sig_col.rsplit("_", 1)[0]

        print(f"Running backtest for {model_name} with {fee_rate*100}% costs...")
        backtesting_results[model_name] = backtest(
            test_df, sig_col, signal_shift=signal_shift, fee=fee_rate,
        )

        # Compute transaction costs over time
        trading_signal = test_df[sig_col].shift(signal_shift).fillna(0)
        trades = trading_signal.diff().fillna(0).abs()
        backtesting_transaction_costs[model_name] = (trades * fee_rate).cumsum()

    return backtesting_results, backtesting_transaction_costs


def calculate_performance_summary(
    backtesting_results: pd.DataFrame,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """
    Compute the performance & drawdown summary.
    Per strategy: final wealth (in €), total return, max drawdown.
    """
    summary_stats = []

    for col in backtesting_results.columns:
        series = backtesting_results[col]

        # Normalized final value (start = 1.0) and total return remain scale-invariant
        final_norm = series.iloc[-1]
        total_ret = (final_norm - 1) * 100
        final_eur = final_norm * initial_capital

        # Compute max drawdown
        roll_max = series.cummax()
        drawdown = series / roll_max - 1.0
        mdd = drawdown.min() * 100

        summary_stats.append({
            "Strategy": col,
            "Final Wealth": f"{final_eur:,.0f} €",
            "Total Return": f"{total_ret:+.2f}%",
            "Max Drawdown": f"{mdd:.2f}%",
        })

    return pd.DataFrame(summary_stats).set_index("Strategy")

def calculate_annualized_metrics(
    backtesting_results: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Annualized performance metrics for all strategies.

    Convention (canonical per Sharpe, 1966; consistent with
    src/backtest/evaluation.py::evaluate_strategies and
    src/backtest/optimize.py::compute_oos_metrics):

    - Daily simple returns via pct_change (NOT log returns).
    - Sharpe / Sortino based on the arithmetic mean of the daily returns
      (AM × √252 / σ and AM × √252 / σ_downside), not CAGR-based.
      The CAGR variant (cagr / σ) systematically underestimates Sharpe by the
      volatility drag (½σ²) and is not the Sharpe (1966) definition.

    Computes per strategy:
    - Annualized return (CAGR; still geometric for return reporting)
    - Annualized volatility (σ × √252 from simple daily returns)
    - Sharpe ratio (AM × √252 / σ; rf=0, since cash is already priced into the strategy)
    - Sortino ratio (AM × √252 / σ_downside)
    - Max drawdown
    - Calmar ratio (CAGR / |max DD|)
    - OOS days / OOS years
    """
    summary = []

    for col in backtesting_results.columns:
        equity = backtesting_results[col]
        # Daily simple returns (consistent with evaluate_strategies)
        daily_rets = equity.pct_change().dropna()

        n_days = len(daily_rets)
        n_years = n_days / trading_days_per_year

        # CAGR (geometric annual return; only for return reporting & Calmar)
        total_return = equity.iloc[-1] / equity.iloc[0]
        cagr = total_return ** (1 / n_years) - 1 if n_years > 0 else 0

        # Volatility (annualized)
        ann_vol = daily_rets.std() * np.sqrt(trading_days_per_year)

        # Sharpe ratio, AM-based (Sharpe, 1966)
        std_daily = daily_rets.std()
        sharpe = (
            (daily_rets.mean() / std_daily) * np.sqrt(trading_days_per_year)
            if std_daily > 0 else 0
        )

        # Sortino ratio, AM-based (downside vol only)
        downside = daily_rets[daily_rets < 0]
        downside_std_daily = downside.std() if len(downside) > 0 else 0
        sortino = (
            (daily_rets.mean() / downside_std_daily) * np.sqrt(trading_days_per_year)
            if downside_std_daily > 0 else 0
        )

        # Max drawdown
        roll_max = equity.cummax()
        drawdown = equity / roll_max - 1.0
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        summary.append({
            "Strategy": col,
            "CAGR": f"{cagr*100:+.2f}%",
            "Ann. Volatility": f"{ann_vol*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Sortino Ratio": f"{sortino:.3f}",
            "Max Drawdown": f"{max_dd*100:.2f}%",
            "Calmar Ratio": f"{calmar:.3f}",
            "OOS Days": n_days,
            "OOS Years": f"{n_years:.1f}",
        })

    return pd.DataFrame(summary).set_index("Strategy")

def calculate_crisis_performance(
    backtesting_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Performance during historical crisis periods.
    Shows return and max drawdown per strategy in each crisis.
    """
    crises = {
        "Dot-Com (2000-03 to 2002-10)":        ("2000-03-01", "2002-10-31"),
        "GFC (2007-10 to 2009-03)":            ("2007-10-01", "2009-03-31"),
        "EU Debt Crisis (2011-07 to 2011-11)": ("2011-07-01", "2011-11-30"),
        "COVID Crash (2020-02 to 2020-03)":    ("2020-02-01", "2020-03-31"),
        "Rate Hikes (2022-01 to 2022-10)":     ("2022-01-01", "2022-10-31"),
    }

    rows = []
    for crisis_name, (start, end) in crises.items():
        mask = (backtesting_results.index >= start) & (backtesting_results.index <= end)
        crisis_data = backtesting_results.loc[mask]

        if len(crisis_data) < 2:
            continue  # Crisis not in the OOS range

        for col in backtesting_results.columns:
            equity = crisis_data[col]
            crisis_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
            roll_max = equity.cummax()
            crisis_dd = (equity / roll_max - 1).min() * 100

            rows.append({
                "Crisis": crisis_name,
                "Strategy": col,
                "Return": f"{crisis_ret:+.2f}%",
                "Max Drawdown": f"{crisis_dd:.2f}%",
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).pivot(
        index="Crisis", columns="Strategy", values=["Return", "Max Drawdown"]
    )

def calculate_rolling_sharpe(
    backtesting_results: pd.DataFrame,
    window_days: int = 252,
    trading_days_per_year: int = 252,
    min_vol_annualized: float = 0.005,  # 0.5% minimum annualized volatility threshold
    cap: float = 10.0,                   # economically plausible maximum
) -> pd.DataFrame:
    """
    Rolling Sharpe ratio (1-year window) for all strategies.

    Stabilization:
    - Windows with annualized volatility below `min_vol_annualized` are
      marked as NaN (avoids division by ~0 in cash-only phases in which
      the strategy sits entirely in the safe haven).
    - Values are capped at ±`cap` (Sharpe > 10 is economically implausible
      and usually an artifact of numerical instability).

    Returns a DataFrame with the same structure as backtesting_results.
    """
    rolling_sharpe = pd.DataFrame(index=backtesting_results.index)

    for col in backtesting_results.columns:
        equity = backtesting_results[col]
        daily_rets = np.log(equity / equity.shift(1))

        roll_mean = daily_rets.rolling(window_days).mean() * trading_days_per_year
        roll_std = daily_rets.rolling(window_days).std() * np.sqrt(trading_days_per_year)

        # Compute the raw value (with division protection)
        sharpe = roll_mean / roll_std.replace(0, np.nan)

        # Set low-vol windows (cash-only phases) to 0 instead of NaN.
        # Rationale: flat phase = no excess return, no risk = Sharpe 0.
        # This keeps the line continuous and avoids division-by-~0 peaks.
        low_vol_mask = roll_std < min_vol_annualized
        sharpe[low_vol_mask] = 0.0

        # Clip to a plausible range (catches remaining numerical outliers)
        sharpe = sharpe.clip(lower=-cap, upper=cap)

        # Remaining NaNs are structural (fold edges, window_size offset
        # for DL models) and are kept as NaN.

        rolling_sharpe[col] = sharpe

    return rolling_sharpe
