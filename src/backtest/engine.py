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