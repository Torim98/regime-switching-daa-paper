"""Performance-Metriken, Strategie-Evaluation und Monte Carlo Simulation."""

import pandas as pd
import numpy as np


def evaluate_strategies(
    results_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    costs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Umfassende Evaluation aller Strategien.
    Berechnet pro Strategie:
    1. Total Return & CAGR (Annualisierte Rendite)
    2. Volatilität (annualisiert)
    3. Sharpe Ratio (Annahme: Risk-Free Rate = 0, da Cash bereits in der Strategie steckt)
    4. Maximum Drawdown
    5. Sortino Ratio (Fokus auf Downside-Risiko)
    6. Calmar Ratio (Verhältnis Rendite zu Max Drawdown)
    7. Anzahl der Trades (Regime-Wechsel)
    8. Gesamte Transaktionskosten am Ende des Zeitraums
    """
    stats = []

    for col in results_df.columns:
        equity_curve = results_df[col]
        daily_returns = equity_curve.pct_change().dropna()

        # 1. Total Return & CAGR (Annualisierte Rendite)
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365.25 / days) - 1

        # 2. Volatilität (annualisiert)
        vol = daily_returns.std() * np.sqrt(252)

        # 3. Sharpe Ratio (Annahme: Risk-Free Rate = 0, da Cash bereits in der Strategie steckt)
        sharpe = (
            (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            if daily_returns.std() != 0
            else 0
        )

        # 4. Maximum Drawdown
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve / peak) - 1
        mdd = drawdown.min()

        # 5. Sortino Ratio (Fokus auf Downside-Risiko)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (daily_returns.mean() * 252) / downside_std if downside_std != 0 else np.nan

        # 6. Calmar Ratio (Verhältnis Rendite zu Max Drawdown)
        calmar = cagr / abs(mdd) if mdd != 0 else np.nan

        # 7. Anzahl der Trades (Regime-Wechsel)
        if col in trades_df.columns:
            switches = trades_df[col].diff().abs().sum()
        else:
            switches = 0

        # 8. Gesamte Transaktionskosten am Ende des Zeitraums extrahieren
        if col in costs_df.columns:
            total_fees = costs_df[col].iloc[-1]
        else:
            total_fees = 0.0

        stats.append({
            "Strategie": col.replace("_", " "),
            "Total Return": f"{total_return:.2%}",
            "CAGR (p.a.)": f"{cagr:.2%}",
            "Volatilität": f"{vol:.2%}",
            "Max Drawdown": f"{mdd:.2%}",
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Calmar Ratio": round(calmar, 2),
            "Regime-Wechsel": int(switches),
            "Gesamtkosten (Gebühren)": f"{total_fees:.2%}",
        })

    return pd.DataFrame(stats).set_index("Strategie")


def find_matching_signal_col(
    strategy_name: str,
    test_df_columns: pd.Index,
) -> str | None:
    """
    Dynamische Zuordnung von Strategie zu Signal-Spalte.
    Sucht passende _Signal-Spalte im test_df für eine gegebene Strategie.
    """
    if strategy_name == "Buy_Hold":
        return None
    if f"{strategy_name}_Signal" in test_df_columns:
        return f"{strategy_name}_Signal"
    root_name = strategy_name.split("_")[0]
    potential_cols = [c for c in test_df_columns if root_name in c and "Signal" in c]
    if len(potential_cols) == 1:
        return potential_cols[0]
    for c in potential_cols:
        if strategy_name[:5] in c:
            return c
    return None


def run_monte_carlo_simulation(
    daily_rets: pd.DataFrame,
    test_df: pd.DataFrame,
    scenarios: dict,
    n_simulations: int,
    block_size: int,
    random_seed: int,
    sim_years: int,
    trading_days_per_year: int,
) -> tuple[list[dict], dict]:
    """
    Block-Bootstrap Monte Carlo Simulation (MCS) — Robustness-Check.

    Paired Block Bootstrap: Rendite-Blöcke + Signal-Blöcke werden gemeinsam gezogen,
    um die Korrelation zwischen Renditen und Signalen zu erhalten.

    Entnahme-Simulation: Monatliche Entnahme (alle 21 Handelstage) mit
    Liquiditäts-Fee falls in Bull-Phase investiert.

    Reproduzierbarkeit über random_seed sichergestellt.

    Gibt (all_mc_summaries, mcs_paths_collector) zurück:
    - all_mc_summaries: Liste von Dicts mit Ruin-Wahrscheinlichkeit und Median Endkapital
    - mcs_paths_collector: Dict mit allen simulierten Kapitalpfaden
    """
    total_days = sim_years * trading_days_per_year

    # Reproduzierbarkeit sicherstellen
    np.random.seed(random_seed)

    # Prüfen, ob Renditen vorhanden sind
    if daily_rets.empty:
        raise ValueError("daily_rets ist leer. Prüfe die Datenquelle backtesting_results.")

    all_mc_summaries = []
    mcs_paths_collector = {}

    for sc_name, params in scenarios.items():
        print(f"Starte Monte Carlo Simulation für Szenario: {sc_name}")
        mc_results_scenario = {}

        for strategy in daily_rets.columns:
            final_capitals = []
            sig_col = find_matching_signal_col(strategy, test_df.columns)

            for s in range(n_simulations):
                # --- Paired Block Bootstrap ---
                sim_rets = []
                sim_sigs = []

                while len(sim_rets) < total_days:
                    start_idx = np.random.randint(0, len(daily_rets) - block_size)
                    sim_rets.extend(
                        daily_rets[strategy].iloc[start_idx : start_idx + block_size].values
                    )
                    if sig_col:
                        sim_sigs.extend(
                            test_df[sig_col].iloc[start_idx : start_idx + block_size].values
                        )
                    else:
                        sim_sigs.extend([0] * block_size)

                sim_rets = np.array(sim_rets[:total_days])
                sim_sigs = np.array(sim_sigs[:total_days])

                # --- Entnahme-Simulation ---
                cap = params["start"]
                capital_history = []

                for i in range(total_days):
                    cap *= (1 + sim_rets[i])

                    # Monatliche Entnahme (alle 21 Handelstage)
                    if i % 21 == 0:
                        withdrawal_amt = params["withdrawal"]
                        if sim_sigs[i] == 0:
                            withdrawal_amt += (params["withdrawal"] * params["fee"])
                        cap -= withdrawal_amt

                    if cap <= 0:
                        cap = 0.0
                        capital_history.append(0.0)
                        remaining_days = total_days - len(capital_history)
                        capital_history.extend([0.0] * remaining_days)
                        break
                    else:
                        capital_history.append(cap)

                final_capitals.append(cap)

                path_id = f"{sc_name}_{strategy}_path_{s:03d}"
                mcs_paths_collector[path_id] = capital_history

            mc_results_scenario[strategy] = final_capitals

            if len(final_capitals) > 0:
                ruin_prob = np.mean(np.array(final_capitals) <= 0)
                median_wealth = np.median(final_capitals)

                all_mc_summaries.append({
                    "Szenario": sc_name,
                    "Strategie": strategy.replace("_", " "),
                    "Ruin-Wahrscheinlichkeit": f"{ruin_prob:.2%}",
                    "Median Endkapital": f"{median_wealth:,.2f} €",
                })

    return all_mc_summaries, mcs_paths_collector