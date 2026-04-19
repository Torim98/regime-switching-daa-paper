"""Performance-Metriken, Strategie-Evaluation und Monte Carlo Simulation.

Enthaelt zusaetzlich die erweiterten Auswertungen aus Issue #13
(Ulcer Index, Classification vs. NBER, ROC/PR, Whipsaw/Churning,
Time-to-Recovery, Switch-Timing, MCS Depletion-CI, H1/H2-Hypothesentests,
Break-Even-Transaktionskosten, Entnahmeraten-Sensitivitaet, Regime-Heatmap).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


# ============================================================
# Issue #13 — Erweiterte Evaluations-Metriken (Kap. 4.1–4.4)
# ============================================================

# ------------------------------------------------------------
# Kap. 4.2/4.4 — Ulcer Index
# ------------------------------------------------------------
def ulcer_index(equity: pd.Series) -> float:
    """
    Martin (1989): RMS-Drawdown. Robusteres Stress-Mass als MaxDD,
    da tiefe UND lange Drawdowns bestraft werden.
    """
    roll_max = equity.cummax()
    dd_pct = (equity / roll_max - 1.0) * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def add_ulcer_to_table(
    backtesting_results: pd.DataFrame,
    evaluation_table: pd.DataFrame,
) -> pd.DataFrame:
    """Hängt eine 'Ulcer Index'-Spalte an die Evaluation-Tabelle an."""
    ui = {
        col.replace("_", " "): round(ulcer_index(backtesting_results[col]), 2)
        for col in backtesting_results.columns
    }
    evaluation_table["Ulcer Index"] = pd.Series(ui)
    return evaluation_table


# ------------------------------------------------------------
# Kap. 4.1 — Classification vs. NBER-Ground-Truth
# ------------------------------------------------------------
def compute_classification_metrics(
    test_df: pd.DataFrame,
    nber_signal: pd.Series,
    models: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Pro Modell: Precision / Recall / F1 / Confusion Matrix gegen NBER.
    `test_df` muss <Model>_Signal-Spalten enthalten.
    """
    from sklearn.metrics import (
        confusion_matrix, f1_score, precision_score, recall_score,
    )

    y_true = nber_signal.reindex(test_df.index).fillna(0).astype(int).values
    rows, cms = [], {}

    for m in models:
        sig_col = f"{m}_Signal"
        if sig_col not in test_df.columns:
            continue
        y_pred = test_df[sig_col].fillna(0).astype(int).values
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cms[m] = cm
        rows.append({
            "Modell": m,
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
            "F1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
            "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
        })
    return pd.DataFrame(rows).set_index("Modell"), cms


def plot_confusion_matrices(
    cms: dict,
    save_path: str,
) -> None:
    """Konfusions-Matrizen als Grid (eine pro Modell)."""
    n = len(cms)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (model, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No-Rec", "Rec"])
        ax.set_yticklabels(["No-Rec", "Rec"])
        ax.set_xlabel("Vorhergesagt"); ax.set_ylabel("NBER (Wahrheit)")
        ax.set_title(model)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_curves(
    test_df: pd.DataFrame,
    nber_signal: pd.Series,
    models: list[str],
    color_map: dict,
    roc_path: str,
    pr_path: str,
) -> pd.DataFrame:
    """
    ROC + PR-Kurven pro Modell (nutzt <Model>_Prob, nicht _Signal,
    um schwellenunabhängig zu vergleichen).
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    y_true = nber_signal.reindex(test_df.index).fillna(0).astype(int).values
    rows = []

    fig_roc, ax_r = plt.subplots(figsize=(7, 6))
    fig_pr,  ax_p = plt.subplots(figsize=(7, 6))
    ax_r.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")

    for m in models:
        prob_col = f"{m}_Prob"
        if prob_col not in test_df.columns:
            continue
        y_score = test_df[prob_col].fillna(0).values
        c = color_map.get(m, None)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax_r.plot(fpr, tpr, color=c, lw=1.6, label=f"{m} (AUC={roc_auc:.2f})")

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec, prec)
        ax_p.plot(rec, prec, color=c, lw=1.6, label=f"{m} (AUC={pr_auc:.2f})")

        rows.append({"Modell": m, "ROC-AUC": round(roc_auc, 3), "PR-AUC": round(pr_auc, 3)})

    ax_r.set_xlabel("False Positive Rate"); ax_r.set_ylabel("True Positive Rate")
    ax_r.set_title("ROC-Kurven (vs. NBER)"); ax_r.legend(loc="lower right")
    ax_r.grid(alpha=0.25)
    fig_roc.savefig(roc_path, dpi=300, bbox_inches="tight"); plt.close(fig_roc)

    ax_p.set_xlabel("Recall"); ax_p.set_ylabel("Precision")
    ax_p.set_title("Precision-Recall-Kurven (vs. NBER)"); ax_p.legend(loc="lower left")
    ax_p.grid(alpha=0.25)
    fig_pr.savefig(pr_path, dpi=300, bbox_inches="tight"); plt.close(fig_pr)

    return pd.DataFrame(rows).set_index("Modell")


# ------------------------------------------------------------
# Kap. 4.1 — Signal-Churning / Whipsaw / Schwellen-Sensitivität
# ------------------------------------------------------------
def churning_stats(
    test_df: pd.DataFrame,
    models: list[str],
    fee_rate: float,
    min_phase_days: int = 5,
) -> pd.DataFrame:
    """
    Pro Modell: Anzahl Signalwechsel, Anteil 'Whipsaws' (Phasen < min_phase_days),
    durchschnittliche Phasenlänge, implizite Kosten bei `fee_rate`.
    """
    rows = []
    for m in models:
        sig = test_df.get(f"{m}_Signal")
        if sig is None:
            continue
        sig = sig.dropna().astype(int)
        n_switches = int((sig.diff().abs() == 1).sum())
        # Phasenlängen über Run-Length-Encoding
        changes = (sig != sig.shift()).cumsum()
        phase_lengths = sig.groupby(changes).size().values
        whipsaws = int((phase_lengths < min_phase_days).sum())
        rows.append({
            "Modell": m,
            "Signalwechsel": n_switches,
            f"Whipsaws (<{min_phase_days}T)": whipsaws,
            "Whipsaw-Anteil": f"{whipsaws / max(len(phase_lengths), 1):.1%}",
            "Ø Phase (Tage)": round(float(np.mean(phase_lengths)), 1),
            "Median Phase (Tage)": int(np.median(phase_lengths)),
            "Kumul. Kosten": f"{n_switches * fee_rate:.2%}",
        })
    return pd.DataFrame(rows).set_index("Modell")


def threshold_sensitivity(
    test_df: pd.DataFrame,
    backtest_fn,              # src.backtest.engine.backtest
    model: str,
    thresholds: list[float],
    fee_rate: float,
    signal_shift: int,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """
    Variiert die Threshold-Schwelle für ein einzelnes Modell und misst,
    wie sich Final-Equity (in €), MaxDD und #Wechsel ändern (Kap. 4.1 Glättung).
    Setzt voraus, dass `<model>_Prob` in test_df vorhanden ist.
    """
    prob_col = f"{model}_Prob"
    assert prob_col in test_df.columns, f"{prob_col} fehlt in test_df"

    rows = []
    for t in thresholds:
        df = test_df.copy()
        sig = (df[prob_col] >= t).astype(int)
        df[f"{model}_Signal_tmp"] = sig
        eq = backtest_fn(df, f"{model}_Signal_tmp", signal_shift, fee_rate)
        dd = (eq / eq.cummax() - 1).min()
        n_switches = int((sig.diff().abs() == 1).sum())
        rows.append({
            "Threshold": t,
            "Final Wealth": f"{float(eq.iloc[-1]) * initial_capital:,.0f} €",
            "Max Drawdown": f"{dd*100:.2f}%",
            "Wechsel": n_switches,
        })
    return pd.DataFrame(rows).set_index("Threshold")


# ------------------------------------------------------------
# Kap. 4.2 — Time-to-Recovery + Switch-Timing
# ------------------------------------------------------------
def time_to_recovery(equity: pd.Series, min_dd: float = -0.05) -> pd.DataFrame:
    """
    Alle Drawdown-Phasen mit DD < min_dd: Peak-Datum, Trough-Datum,
    Recovery-Datum, Dauer in Handelstagen. NaN bei unerholten Phasen.
    """
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    in_dd = dd < -1e-9
    phases = []
    start = None
    for date, flag in in_dd.items():
        if flag and start is None:
            start = date
        elif not flag and start is not None:
            seg = equity.loc[start:date]
            dd_seg = seg / seg.cummax() - 1
            trough = dd_seg.idxmin()
            dd_min = float(dd_seg.min())
            if dd_min <= min_dd:
                phases.append({
                    "Peak":     start.strftime("%Y-%m-%d"),
                    "Trough":   trough.strftime("%Y-%m-%d"),
                    "Recovery": date.strftime("%Y-%m-%d"),
                    "Max DD":   f"{dd_min*100:.2f}%",
                    "Drawdown-Dauer (T)": (trough - start).days,
                    "Recovery-Dauer (T)": (date - trough).days,
                    "Gesamt (T)": (date - start).days,
                })
            start = None
    # Offene Phase am Ende (noch nicht erholt)
    if start is not None:
        seg = equity.loc[start:]
        dd_seg = seg / seg.cummax() - 1
        trough_idx = dd_seg.idxmin()
        dd_min = float(dd_seg.min())
        if dd_min <= min_dd:
            phases.append({
                "Peak":     start.strftime("%Y-%m-%d"),
                "Trough":   trough_idx.strftime("%Y-%m-%d"),
                "Recovery": "—",
                "Max DD":   f"{dd_min*100:.2f}%",
                "Drawdown-Dauer (T)": (trough_idx - start).days,
                "Recovery-Dauer (T)": np.nan,
                "Gesamt (T)": np.nan,
            })
    return pd.DataFrame(phases)


def switch_timing_vs_peak(
    test_df: pd.DataFrame,
    backtesting_results: pd.DataFrame,
    model: str,
    crisis_windows: dict,
) -> pd.DataFrame:
    """
    Pro Krisenfenster: Wie viele Tage VOR dem Drawdown-Peak des Buy-Hold
    hat das Modell bereits ein Bear-Signal gesetzt? Negativ = reagierte
    zu spät.
    """
    sig_col = f"{model}_Signal"
    if sig_col not in test_df.columns:
        return pd.DataFrame()

    rows = []
    bh = backtesting_results["Buy_Hold"]
    for name, (start, end) in crisis_windows.items():
        mask = (bh.index >= start) & (bh.index <= end)
        if mask.sum() < 2:
            continue
        trough_date = (bh[mask] / bh[mask].cummax() - 1).idxmin()
        sig = test_df.loc[mask, sig_col].fillna(0).astype(int)
        first_bear = sig[sig == 1].index.min() if (sig == 1).any() else pd.NaT
        if pd.isna(first_bear):
            lead = np.nan
        else:
            lead = (trough_date - first_bear).days  # pos. = frühzeitig
        rows.append({
            "Krise": name,
            "DD-Trough": trough_date.date(),
            "1. Bear-Signal": first_bear.date() if not pd.isna(first_bear) else None,
            "Lead (Tage)": lead,
        })
    return pd.DataFrame(rows).set_index("Krise")


# ------------------------------------------------------------
# Kap. 4.3 — MCS: Endvermögen, Depletion-CI, H1/H2-Tests
# ------------------------------------------------------------
def mcs_final_capitals(
    mcs_paths_collector: dict,
    scenarios: list[str],
    strategies: list[str],
) -> dict:
    """
    Aus dem Pfad-Collector je (Szenario, Strategie) die Endkapitale
    als 1D-NumPy-Array rekonstruieren.
    """
    finals = {}
    for sc in scenarios:
        for s in strategies:
            prefix = f"{sc}_{s}_path_"
            vals = [p[-1] for k, p in mcs_paths_collector.items() if k.startswith(prefix)]
            if vals:
                finals[(sc, s)] = np.asarray(vals)
    return finals


def depletion_rate_with_ci(
    finals: dict,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Wilson-CI fuer die Depletion Rate p = P(Endkapital <= 0).
    Wilson statt Wald, weil bei p ~ 0 numerisch stabiler.
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)

    rows = []
    for (sc, s), arr in finals.items():
        n = len(arr)
        k = int(np.sum(arr <= 0))
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
        rows.append({
            "Szenario": sc,
            "Strategie": s,
            "Depletion Rate": f"{p:.2%}",
            "95%-CI unten":  f"{max(0, center - half):.2%}",
            "95%-CI oben":   f"{min(1, center + half):.2%}",
            "n_ruin / n_paths": f"{k}/{n}",
        })
    return pd.DataFrame(rows).set_index(["Szenario", "Strategie"])


def mcs_path_maxdd(mcs_paths_collector: dict, prefix: str) -> np.ndarray:
    """MaxDD je Pfad (für Hypothesentests)."""
    dds = []
    for k, path in mcs_paths_collector.items():
        if not k.startswith(prefix):
            continue
        arr = np.asarray(path, dtype=float)
        cummax = np.maximum.accumulate(arr)
        # Schutz vor Division durch 0 in Ruin-Pfaden
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(cummax > 0, arr / cummax - 1, -1.0)
        dds.append(dd.min())
    return np.asarray(dds)


def test_h1_drawdown(
    mcs_paths_collector: dict,
    scenario: str,
    regime_models: list[str],
    benchmark: str = "Buy_Hold",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    H1: Regime-Switching reduziert MaxDD vs. Buy & Hold.
    Gepaarter Wilcoxon-Test (gleiche Bootstrap-Indizes → gepaarte Pfade).
    """
    from scipy.stats import wilcoxon

    dd_bh = mcs_path_maxdd(mcs_paths_collector, f"{scenario}_{benchmark}_path_")
    rows = []
    for m in regime_models:
        dd_m = mcs_path_maxdd(mcs_paths_collector, f"{scenario}_{m}_path_")
        if len(dd_m) != len(dd_bh) or len(dd_m) == 0:
            continue
        # H1: dd_m > dd_bh (weniger negativ) → einseitig "greater"
        try:
            _, p = wilcoxon(dd_m, dd_bh, alternative="greater")
        except ValueError:
            p = np.nan
        rows.append({
            "Modell": m,
            "Median MaxDD (Modell)": f"{np.median(dd_m)*100:.2f}%",
            "Median MaxDD (B&H)":    f"{np.median(dd_bh)*100:.2f}%",
            "Δ Median":              f"{(np.median(dd_m) - np.median(dd_bh))*100:+.2f} pp",
            "Wilcoxon p":            f"{p:.2e}" if not np.isnan(p) else "n/a",
            f"H1 (α={alpha})":       ("bestätigt" if (not np.isnan(p) and p < alpha)
                                      else "abgelehnt"),
        })
    return pd.DataFrame(rows).set_index("Modell")


def test_h2_transformer(
    finals: dict,
    scenario: str,
    challenger: str = "Transformer",
    competitors: tuple = ("MSM", "HMM", "LSTM"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    H2: Transformer schlägt Ökonometrie/LSTM im Endvermögen.
    Gepaarter Wilcoxon-Test auf Endkapital (dieselben Pfade).
    """
    from scipy.stats import wilcoxon
    w_ch = finals.get((scenario, challenger))
    if w_ch is None:
        return pd.DataFrame()

    rows = []
    for c in competitors:
        w_c = finals.get((scenario, c))
        if w_c is None or len(w_c) != len(w_ch) or len(w_c) == 0:
            continue
        try:
            _, p = wilcoxon(w_ch, w_c, alternative="greater")
        except ValueError:
            p = np.nan
        rows.append({
            "Vergleich":        f"{challenger} vs. {c}",
            f"Median {challenger}": f"{np.median(w_ch):,.0f} €",
            f"Median {c}":          f"{np.median(w_c):,.0f} €",
            "Δ Median":         f"{(np.median(w_ch) - np.median(w_c)):+,.0f} €",
            "Wilcoxon p":       f"{p:.2e}" if not np.isnan(p) else "n/a",
            f"H2 (α={alpha})":  ("bestätigt" if (not np.isnan(p) and p < alpha)
                                 else "abgelehnt"),
        })
    return pd.DataFrame(rows).set_index("Vergleich")


def plot_mcs_violins(
    finals: dict,
    scenarios: list[str],
    strategies: list[str],
    color_map: dict,
    save_path_template: str,
) -> None:
    """Violin-Plots je Szenario (zusätzlich zu den Boxplots)."""
    for sc in scenarios:
        data, labels, colors = [], [], []
        for s in strategies:
            arr = finals.get((sc, s))
            if arr is None:
                continue
            data.append(arr)
            labels.append(s.replace("_", " "))
            colors.append(color_map.get(s, "gray"))
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        vp = ax.violinplot(data, showmedians=True)
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c); body.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(f"MCS Endvermögen — Szenario {sc}")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.6)
        ax.grid(alpha=0.25)
        fig.savefig(save_path_template.format(sc.lower()),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------
# Kap. 4.4 — Break-Even-Kosten + Entnahmeraten-Sensitivität
# ------------------------------------------------------------
def break_even_transaction_cost(
    test_df: pd.DataFrame,
    backtest_fn,
    benchmark_equity: pd.Series,
    models: list[str],
    fee_grid_bps: list[int],
    signal_shift: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Pro Modell: Final-Wealth unter variierender Kostenquote.
    Break-Even = kleinste Kostenquote, bei der Final-Wealth <= B&H.
    Gibt (Summary-Tabelle, {Modell: Series(fee_bps -> final_wealth)}).
    """
    bh_final = float(benchmark_equity.iloc[-1])
    summary, curves = [], {}

    for m in models:
        sig_col = f"{m}_Signal"
        if sig_col not in test_df.columns:
            continue
        wealths = {}
        for bps in fee_grid_bps:
            fee = bps / 10_000
            eq = backtest_fn(test_df, sig_col, signal_shift, fee)
            wealths[bps] = float(eq.iloc[-1])
        curves[m] = pd.Series(wealths).sort_index()

        below = curves[m][curves[m] <= bh_final]
        be_bps = int(below.index.min()) if not below.empty else None
        summary.append({
            "Modell": m,
            "Final @10bps": round(wealths.get(10, float("nan")), 3),
            "B&H Final":    round(bh_final, 3),
            "Break-Even (bps)": be_bps if be_bps is not None else ">max",
        })
    return pd.DataFrame(summary).set_index("Modell"), curves


def plot_break_even(
    curves: dict,
    benchmark_final: float,
    color_map: dict,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, s in curves.items():
        ax.plot(s.index, s.values, marker="o",
                color=color_map.get(m, None), label=m)
    ax.axhline(benchmark_final, color="black", linestyle="--",
               label=f"Buy & Hold ({benchmark_final:.2f})")
    ax.set_xlabel("Transaktionskosten (bps)")
    ax.set_ylabel("Final Wealth (kumuliert)")
    ax.set_title("Break-Even-Analyse: Kostenquote vs. Endvermögen")
    ax.legend(); ax.grid(alpha=0.25)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def withdrawal_sensitivity(
    backtesting_results: pd.DataFrame,
    test_df: pd.DataFrame,
    sorr_fn,                  # src.backtest.sorr.run_sorr_simulation
    base_scenario: dict,      # {"start": 500000, "fee": 0.001}
    rates: tuple = (0.035, 0.04, 0.05),
) -> pd.DataFrame:
    """
    Identischer Start, variierende Jahres-Entnahmerate. Ergebnis:
    Endkapital + Jahr-der-Erschöpfung pro Strategie × Rate.
    """
    rows = []
    for r in rates:
        monthly = base_scenario["start"] * r / 12
        sim = sorr_fn(
            backtesting_results, test_df,
            base_scenario["start"], monthly, base_scenario["fee"],
        )
        for col in sim.columns:
            final = float(sim[col].iloc[-1])
            if final > 0:
                status = "Kapitalerhalt"
            else:
                depleted = sim[sim[col] <= 0].index[0]
                status = f"Erschöpft ({depleted.strftime('%Y')})"
            rows.append({
                "Entnahmerate (p.a.)": f"{r:.1%}",
                "Strategie": col,
                "Endkapital": f"{final:,.0f} €",
                "Status": status,
            })
    return (
        pd.DataFrame(rows)
        .pivot(index="Strategie", columns="Entnahmerate (p.a.)",
               values=["Endkapital", "Status"])
    )


# ------------------------------------------------------------
# Kap. 4.1 — Regime-Wahrscheinlichkeits-Heatmap
# ------------------------------------------------------------
def plot_regime_probability_heatmap(
    test_df: pd.DataFrame,
    models: list[str],
    save_path: str,
) -> None:
    """
    Heatmap: y=Modell, x=Zeit, Farbe=Bear-Wahrscheinlichkeit (<Model>_Prob).
    """
    probs = pd.DataFrame({
        m: test_df[f"{m}_Prob"] for m in models if f"{m}_Prob" in test_df.columns
    })
    if probs.empty:
        return
    fig, ax = plt.subplots(figsize=(15, 1.0 * len(probs.columns) + 2))
    im = ax.imshow(probs.T.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1)
    n_ticks = min(10, len(probs))
    tick_idx = np.linspace(0, len(probs) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([probs.index[i].strftime("%Y-%m") for i in tick_idx],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(probs.columns)))
    ax.set_yticklabels(probs.columns)
    ax.set_title("Regime-Bear-Wahrscheinlichkeiten über OOS-Zeitraum")
    fig.colorbar(im, ax=ax, shrink=0.7, label="P(Bear)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)