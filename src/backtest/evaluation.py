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


def _simulate_strategy(
    rets_arr: np.ndarray,
    sig_arr: np.ndarray,
    n_simulations: int,
    total_days: int,
    block_size: int,
    start_capital: float,
    withdrawal: float,
    fee: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vektorisierte MCS für eine einzelne Strategie + Szenario-Kombination.

    1. Paired Block Bootstrap: Alle Pfade gleichzeitig via vorberechneter
       Block-Indizes (Renditen + Signale bleiben korreliert).
    2. Capital Evolution: Tagesweise über alle Pfade parallel (NumPy-Vektoren),
       mit monatlicher Entnahme (alle 21 Handelstage) und Ruin-Erkennung.

    Returns:
        final_capitals: (n_simulations,) — Endkapital je Pfad
        all_capital_histories: (n_simulations, total_days) — vollständige Pfade
    """
    n_source = len(rets_arr)
    n_blocks = int(np.ceil(total_days / block_size))

    # --- Vectorized Paired Block Bootstrap ---
    # Alle Startindizes auf einmal ziehen: (n_simulations, n_blocks)
    start_indices = rng.integers(0, n_source - block_size, size=(n_simulations, n_blocks))

    # Block-Indizes zu vollständigen Zeitreihen-Indizes expandieren
    # offsets: (1, 1, block_size) broadcast mit start_indices: (n_sim, n_blocks, 1)
    offsets = np.arange(block_size)
    # (n_simulations, n_blocks, block_size)
    full_indices = start_indices[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
    # Flatten zu (n_simulations, n_blocks * block_size) und auf total_days trimmen
    full_indices = full_indices.reshape(n_simulations, -1)[:, :total_days]

    sim_rets = rets_arr[full_indices]   # (n_simulations, total_days)
    sim_sigs = sig_arr[full_indices]    # (n_simulations, total_days)

    # --- Vectorized Capital Evolution ---
    capitals = np.full(n_simulations, start_capital, dtype=np.float64)
    all_capital_histories = np.empty((n_simulations, total_days), dtype=np.float64)
    ruined = np.zeros(n_simulations, dtype=bool)

    for i in range(total_days):
        # Rendite anwenden (alle Pfade gleichzeitig)
        capitals *= (1 + sim_rets[:, i])

        # Monatliche Entnahme (alle 21 Handelstage)
        if i % 21 == 0:
            withdrawal_amt = np.full(n_simulations, withdrawal)
            # Liquiditäts-Fee wenn Signal == 0 (Bull-Phase investiert)
            fee_mask = sim_sigs[:, i] == 0
            withdrawal_amt[fee_mask] += withdrawal * fee
            capitals -= withdrawal_amt

        # Ruin-Check: neu ruinierte Pfade auf 0 setzen
        newly_ruined = (capitals <= 0) & ~ruined
        capitals[newly_ruined] = 0.0
        ruined |= newly_ruined

        # Bereits ruinierte Pfade bleiben bei 0
        capitals[ruined] = 0.0

        all_capital_histories[:, i] = capitals

    return capitals.copy(), all_capital_histories


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

    Optimiert für hohe Pfadanzahlen (10.000+):
    - Vektorisierter Block-Bootstrap (NumPy Fancy Indexing)
    - Vektorisierte Capital Evolution (alle Pfade parallel)
    - Parallelisierung über Strategien via concurrent.futures

    Gibt (all_mc_summaries, mcs_paths_collector) zurück:
    - all_mc_summaries: Liste von Dicts mit Ruin-Wahrscheinlichkeit und Median Endkapital
    - mcs_paths_collector: Dict mit allen simulierten Kapitalpfaden
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    total_days = sim_years * trading_days_per_year

    # Prüfen, ob Renditen vorhanden sind
    if daily_rets.empty:
        raise ValueError("daily_rets ist leer. Prüfe die Datenquelle backtesting_results.")

    # Reproduzierbare, unabhängige Seeds pro Job via SeedSequence
    seed_seq = np.random.SeedSequence(random_seed)

    all_mc_summaries = []
    mcs_paths_collector = {}

    # --- Jobs vorbereiten: (Szenario, Strategie) Paare ---
    jobs = []
    job_keys = []
    child_seeds = seed_seq.spawn(len(scenarios) * len(daily_rets.columns))
    seed_idx = 0

    for sc_name, params in scenarios.items():
        for strategy in daily_rets.columns:
            sig_col = find_matching_signal_col(strategy, test_df.columns)
            rets_arr = daily_rets[strategy].values
            sig_arr = (
                test_df[sig_col].values if sig_col
                else np.zeros(len(test_df))
            )

            jobs.append((
                rets_arr,
                sig_arr,
                n_simulations,
                total_days,
                block_size,
                params["start"],
                params["withdrawal"],
                params["fee"],
                child_seeds[seed_idx],
            ))
            job_keys.append((sc_name, strategy))
            seed_idx += 1

    # --- Parallel oder sequentiell ausführen ---
    n_workers = min(len(jobs), max(1, os.cpu_count() - 1))
    results = {}

    if n_workers > 1 and n_simulations >= 1000:
        print(f"MCS: Starte {len(jobs)} Jobs auf {n_workers} Workern "
              f"({n_simulations:,} Pfade je Kombination)...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_key = {}
            for key, job_args in zip(job_keys, jobs):
                # SeedSequence → Generator im Worker erstellen
                future = executor.submit(_run_strategy_job, *job_args)
                future_to_key[future] = key

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                results[key] = future.result()
    else:
        print(f"MCS: Starte {len(jobs)} Jobs sequentiell "
              f"({n_simulations:,} Pfade je Kombination)...")
        for key, job_args in zip(job_keys, jobs):
            results[key] = _run_strategy_job(*job_args)

    # --- Ergebnisse aggregieren ---
    for (sc_name, strategy), (final_capitals, all_histories) in results.items():
        print(f"  ✓ {sc_name} / {strategy}")

        # Pfade in Collector schreiben
        for s in range(n_simulations):
            path_id = f"{sc_name}_{strategy}_path_{s:03d}"
            mcs_paths_collector[path_id] = all_histories[s].tolist()

        # Summary-Statistiken
        ruin_prob = np.mean(final_capitals <= 0)
        median_wealth = np.median(final_capitals)

        all_mc_summaries.append({
            "Szenario": sc_name,
            "Strategie": strategy.replace("_", " "),
            "Ruin-Wahrscheinlichkeit": f"{ruin_prob:.2%}",
            "Median Endkapital": f"{median_wealth:,.2f} €",
        })

    return all_mc_summaries, mcs_paths_collector


def _run_strategy_job(
    rets_arr, sig_arr, n_simulations, total_days, block_size,
    start_capital, withdrawal, fee, child_seed,
):
    """Wrapper für ProcessPoolExecutor — erstellt Generator aus SeedSequence."""
    rng = np.random.default_rng(child_seed)
    return _simulate_strategy(
        rets_arr, sig_arr, n_simulations, total_days, block_size,
        start_capital, withdrawal, fee, rng,
    )