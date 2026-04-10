"""Backtesting-Engine — Kumulierte Renditeberechnung mit Transaktionskosten."""

import pandas as pd
import numpy as np


def backtest(
    df: pd.DataFrame,
    signal_col: str,
    signal_shift: int,
    fee: float,
) -> pd.Series:
    """
    Berechnet die kumulierte Rendite unter Berücksichtigung von Transaktionskosten.
    fee: Kosten für einen vollständigen Wechsel (z.B. 0.1% = 0.001).

    Logik:
    - Signal um konfigurierbare Tage verschieben zur Vermeidung von Look-ahead Bias
    - Trades identifizieren: Wo unterscheidet sich das Signal von heute zu gestern?
    - Wenn Signal 0 → Portfolio-Return, sonst Cash-Return
    - Transaktionskosten abziehen
    - Kumulierte Rendite berechnen
    """
    # Signal um konfigurierbare Tage verschieben zur Vermeidung von Look-ahead Bias
    trading_signal = df[signal_col].shift(signal_shift).fillna(0)

    # Trades identifizieren: Wo unterscheidet sich das Signal von heute zu gestern?
    trades = trading_signal.diff().fillna(0).abs()

    # Logik: Wenn Signal 0 -> Portfolio-Return, sonst Cash-Return
    strategy_returns = np.where(
        trading_signal == 0,
        df["Returns"],
        df["Cash_Returns"],
    )

    # Transaktionskosten abziehen
    net_strategy_returns = strategy_returns - (trades * fee)

    # Kumulierte Rendite berechnen
    return pd.Series(np.exp(net_strategy_returns.cumsum()), index=df.index)


def run_all_backtests(
    test_df: pd.DataFrame,
    fee_rate: float,
    signal_shift: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Alle verfügbaren Modelle dynamisch identifizieren (anhand der _Signal Endung)
    und Backtesting durchführen.

    Benchmark: Buy & Hold des 60/40 Portfolios (0 Transaktionskosten, da nie umgeschichtet).

    Gibt (backtesting_results, backtesting_transaction_costs) zurück.
    """
    # Alle verfügbaren Modelle dynamisch identifizieren (anhand der _Signal Endung)
    signal_cols = [col for col in test_df.columns if col.endswith("_Signal")]

    # Ergebnisse-DataFrame initialisieren
    backtesting_results = pd.DataFrame(index=test_df.index)
    # DataFrame für den zeitlichen Verlauf der Transaktionskosten
    backtesting_transaction_costs = pd.DataFrame(index=test_df.index)

    # Benchmark berechnen (Buy & Hold des 60/40 Portfolios)
    backtesting_results["Buy_Hold"] = np.exp(test_df["Returns"].cumsum())
    # Buy & Hold hat 0 Transaktionskosten, da wir nie umschichten
    backtesting_transaction_costs["Buy_Hold"] = 0.0

    # Alle erkannten Modelle dynamisch backtesten
    for sig_col in signal_cols:
        model_name = sig_col.rsplit("_", 1)[0]

        print(f"Berechne Backtest für {model_name} mit {fee_rate*100}% Kosten...")
        backtesting_results[model_name] = backtest(
            test_df, sig_col, signal_shift=signal_shift, fee=fee_rate,
        )

        # Transaktionskosten im zeitlichen Verlauf berechnen
        trading_signal = test_df[sig_col].shift(signal_shift).fillna(0)
        trades = trading_signal.diff().fillna(0).abs()
        backtesting_transaction_costs[model_name] = (trades * fee_rate).cumsum()

    return backtesting_results, backtesting_transaction_costs


def calculate_performance_summary(
    backtesting_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Performance & Drawdown Zusammenfassung berechnen.
    Pro Strategie: Final Wealth, Total Return, Max Drawdown.
    """
    summary_stats = []

    for col in backtesting_results.columns:
        series = backtesting_results[col]

        # Finale Werte berechnen
        final_val = series.iloc[-1]
        total_ret = (final_val - 1) * 100

        # Max Drawdown berechnen
        roll_max = series.cummax()
        drawdown = series / roll_max - 1.0
        mdd = drawdown.min() * 100

        summary_stats.append({
            "Strategie": col,
            "Final Wealth": f"{final_val:.4f}",
            "Total Return": f"{total_ret:+.2f}%",
            "Max Drawdown": f"{mdd:.2f}%",
        })

    return pd.DataFrame(summary_stats).set_index("Strategie")

def calculate_annualized_metrics(
    backtesting_results: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Annualisierte Performance-Metriken für alle Strategien.

    Berechnet pro Strategie:
    - Annualisierte Rendite (CAGR)
    - Annualisierte Volatilität
    - Sharpe Ratio (rf=0, da Cash_Returns bereits in Strategie eingepreist)
    - Sortino Ratio (Downside-Volatilität)
    - Max Drawdown
    - Calmar Ratio (CAGR / |Max DD|)
    - Anzahl Trades (Signal-Wechsel)
    - Zeit im Markt (% der Tage mit Signal=0, also investiert)
    """
    summary = []

    for col in backtesting_results.columns:
        equity = backtesting_results[col]
        # Tägliche Log-Returns aus der Equity-Kurve
        daily_rets = np.log(equity / equity.shift(1)).dropna()

        n_days = len(daily_rets)
        n_years = n_days / trading_days_per_year

        # CAGR
        total_return = equity.iloc[-1] / equity.iloc[0]
        cagr = total_return ** (1 / n_years) - 1 if n_years > 0 else 0

        # Volatilität (annualisiert)
        ann_vol = daily_rets.std() * np.sqrt(trading_days_per_year)

        # Sharpe Ratio
        sharpe = cagr / ann_vol if ann_vol > 0 else 0

        # Sortino Ratio (nur Downside-Vol)
        downside = daily_rets[daily_rets < 0]
        downside_vol = downside.std() * np.sqrt(trading_days_per_year) if len(downside) > 0 else 0
        sortino = cagr / downside_vol if downside_vol > 0 else 0

        # Max Drawdown
        roll_max = equity.cummax()
        drawdown = equity / roll_max - 1.0
        max_dd = drawdown.min()

        # Calmar Ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        summary.append({
            "Strategie": col,
            "CAGR": f"{cagr*100:+.2f}%",
            "Ann. Volatilität": f"{ann_vol*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Sortino Ratio": f"{sortino:.3f}",
            "Max Drawdown": f"{max_dd*100:.2f}%",
            "Calmar Ratio": f"{calmar:.3f}",
            "OOS-Tage": n_days,
            "OOS-Jahre": f"{n_years:.1f}",
        })

    return pd.DataFrame(summary).set_index("Strategie")

def calculate_crisis_performance(
    backtesting_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Performance während historischer Krisenperioden.
    Zeigt Return und Max Drawdown pro Strategie in jeder Krise.
    """
    crises = {
        "Dot-Com (2000-03 – 2002-10)":    ("2000-03-01", "2002-10-31"),
        "GFC (2007-10 – 2009-03)":         ("2007-10-01", "2009-03-31"),
        "EU-Schuldenkrise (2011-07 – 2011-11)": ("2011-07-01", "2011-11-30"),
        "COVID Crash (2020-02 – 2020-03)": ("2020-02-01", "2020-03-31"),
        "Zinsanstieg (2022-01 – 2022-10)": ("2022-01-01", "2022-10-31"),
    }

    rows = []
    for crisis_name, (start, end) in crises.items():
        mask = (backtesting_results.index >= start) & (backtesting_results.index <= end)
        crisis_data = backtesting_results.loc[mask]

        if len(crisis_data) < 2:
            continue  # Krise nicht im OOS-Bereich

        for col in backtesting_results.columns:
            equity = crisis_data[col]
            crisis_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
            roll_max = equity.cummax()
            crisis_dd = (equity / roll_max - 1).min() * 100

            rows.append({
                "Krise": crisis_name,
                "Strategie": col,
                "Return": f"{crisis_ret:+.2f}%",
                "Max Drawdown": f"{crisis_dd:.2f}%",
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).pivot(
        index="Krise", columns="Strategie", values=["Return", "Max Drawdown"]
    )

def calculate_rolling_sharpe(
    backtesting_results: pd.DataFrame,
    window_days: int = 252,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Rollierender Sharpe Ratio (1-Jahres-Fenster) für alle Strategien.
    Gibt DataFrame mit gleicher Struktur wie backtesting_results zurück.
    """
    rolling_sharpe = pd.DataFrame(index=backtesting_results.index)

    for col in backtesting_results.columns:
        equity = backtesting_results[col]
        daily_rets = np.log(equity / equity.shift(1))

        roll_mean = daily_rets.rolling(window_days).mean() * trading_days_per_year
        roll_std = daily_rets.rolling(window_days).std() * np.sqrt(trading_days_per_year)
        rolling_sharpe[col] = roll_mean / roll_std

    return rolling_sharpe