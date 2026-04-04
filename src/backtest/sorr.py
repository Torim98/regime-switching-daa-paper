"""Sequence of Returns Risk (SORR) Simulation — Analyse der Entnahmephase."""

import pandas as pd
import numpy as np


def run_sorr_simulation(
    backtest_res: pd.DataFrame,
    test_data: pd.DataFrame,
    start_capital: float,
    monthly_withdrawal: float,
    fee_rate: float,
) -> pd.DataFrame:
    """
    Führt eine SORR-Simulation für alle im DataFrame enthaltenen Strategien durch.

    Berechnet den Kapitalverlauf unter monatlicher Entnahme.
    Logik pro Tag:
    - Kapital wächst/fällt gemäß täglicher Rendite
    - Am Monatsanfang: Entnahme + Liquiditäts-Fee (falls in Bull-Phase investiert)
    - Kapital wird bei 0 gedeckelt (kein Negativ-Kapital)

    Gibt DataFrame mit Kapitalverläufen pro Strategie zurück.
    """
    daily_ret = backtest_res.pct_change().fillna(0)
    path_results = pd.DataFrame(index=daily_ret.index)

    for col in daily_ret.columns:
        capital_path = []
        current_capital = start_capital
        last_month = -1

        # Signalzuordnung für Gebührenlogik
        sig_col = None
        if col == "Buy_Hold":
            sig_col = None
        elif f"{col}_Signal" in test_data.columns:
            sig_col = f"{col}_Signal"
        elif col == "HMM_Based" and "HMM_Regime" in test_data.columns:
            sig_col = "HMM_Regime"

        for date, ret in daily_ret[col].items():
            current_capital *= (1 + ret)

            if date.month != last_month:
                withdrawal = monthly_withdrawal
                is_invested = True
                if sig_col is not None and date in test_data.index:
                    if test_data.at[date, sig_col] == 1:
                        is_invested = False

                if is_invested:
                    withdrawal += (monthly_withdrawal * fee_rate)

                current_capital -= withdrawal
                last_month = date.month

            current_capital = max(0, current_capital)
            capital_path.append(current_capital)

        path_results[col] = capital_path
    return path_results


def build_sorr_scenarios(scenarios_cfg) -> dict:
    """
    Definition der Szenarien aus zentraler Config.
    Berechnet monatliche Entnahme aus Jahresrate:
    withdrawal = initial_capital * annual_withdrawal_rate / 12

    Gibt dict mit Szenario-Name → {start, withdrawal, fee} zurück.
    """
    scenarios = {}
    for scenario_name, scenario_cfg in vars(scenarios_cfg).items():
        scenarios[scenario_name] = {
            "start": scenario_cfg.initial_capital,
            "withdrawal": scenario_cfg.initial_capital * scenario_cfg.annual_withdrawal_rate / 12,
            "fee": scenario_cfg.liquidity_fee,
        }
    return scenarios


def build_sorr_summary(
    sim_results: pd.DataFrame,
    scenario_name: str,
) -> list[dict]:
    """
    Statistische Auswertung für ein SORR-Szenario.
    Pro Strategie: Endkapital und Status (Kapitalerhalt oder Erschöpfungsjahr).

    Gibt Liste von Dicts zurück (für DataFrame-Erstellung).
    """
    summaries = []
    for col in sim_results.columns:
        final_cap = sim_results[col].iloc[-1]
        if final_cap > 0:
            status = "Kapitalerhalt"
        else:
            exhausted_idx = sim_results[sim_results[col] <= 0].index[0]
            status = f"Erschöpft ({exhausted_idx.strftime('%Y')})"
        summaries.append({
            "Szenario": scenario_name,
            "Strategie": col.replace("_", " "),
            "Endkapital": f"{final_cap:,.2f} €",
            "Status": status,
        })
    return summaries