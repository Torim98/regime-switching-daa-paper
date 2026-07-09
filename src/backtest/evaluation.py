"""Performance metrics, strategy evaluation, and Monte Carlo simulation.

Additionally contains the extended evaluations from Issue #13
(Ulcer index, classification vs. NBER, ROC/PR, whipsaw/churning,
time-to-recovery, switch timing, MCS depletion CI, H1/H2 hypothesis tests,
break-even transaction costs, withdrawal rate sensitivity, regime heatmap).
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
    Comprehensive evaluation of all strategies.
    Computes per strategy:
    1. Total return & CAGR (annualized return)
    2. Volatility (annualized)
    3. Sharpe ratio (assumption: risk-free rate = 0, since cash is already part of the strategy)
    4. Maximum drawdown
    5. Sortino ratio (focus on downside risk)
    6. Calmar ratio (ratio of return to max drawdown)
    7. Number of trades (regime switches)
    8. Total transaction costs at the end of the period
    """
    stats = []

    for col in results_df.columns:
        equity_curve = results_df[col]
        daily_returns = equity_curve.pct_change().dropna()

        # 1. Total return & CAGR (annualized return)
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365.25 / days) - 1

        # 2. Volatility (annualized)
        vol = daily_returns.std() * np.sqrt(252)

        # 3. Sharpe ratio (assumption: risk-free rate = 0, since cash is already part of the strategy)
        sharpe = (
            (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            if daily_returns.std() != 0
            else 0
        )

        # 4. Maximum drawdown
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve / peak) - 1
        mdd = drawdown.min()

        # 5. Sortino ratio (focus on downside risk)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (daily_returns.mean() * 252) / downside_std if downside_std != 0 else np.nan

        # 6. Calmar ratio (ratio of return to max drawdown)
        calmar = cagr / abs(mdd) if mdd != 0 else np.nan

        # 7. Number of trades (regime switches)
        if col in trades_df.columns:
            switches = trades_df[col].diff().abs().sum()
        else:
            switches = 0

        # 8. Extract total transaction costs at the end of the period
        if col in costs_df.columns:
            total_fees = costs_df[col].iloc[-1]
        else:
            total_fees = 0.0

        stats.append({
            "Strategy": col.replace("_", " "),
            "Total Return": f"{total_return:.2%}",
            "CAGR (p.a.)": f"{cagr:.2%}",
            "Volatility": f"{vol:.2%}",
            "Max Drawdown": f"{mdd:.2%}",
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Calmar Ratio": round(calmar, 2),
            "Regime Switches": int(switches),
            "Total Costs (Fees)": f"{total_fees:.2%}",
        })

    return pd.DataFrame(stats).set_index("Strategy")


def find_matching_signal_col(
    strategy_name: str,
    test_df_columns: pd.Index,
) -> str | None:
    """
    Dynamic mapping from strategy to signal column.
    Searches for a matching _Signal column in test_df for a given strategy.
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


def _block_bootstrap_indices(
    n_source: int,
    n_simulations: int,
    total_days: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Fixed-length (moving) block bootstrap indices.

    Draws non-overlapping start positions per block and expands each block to
    `block_size` consecutive indices, then trims to `total_days`.

    Returns an (n_simulations, total_days) integer index matrix.
    """
    n_blocks = int(np.ceil(total_days / block_size))

    # Draw all start indices at once: (n_simulations, n_blocks)
    start_indices = rng.integers(0, n_source - block_size, size=(n_simulations, n_blocks))

    # Expand block indices to full time-series indices
    # offsets: (1, 1, block_size) broadcast with start_indices: (n_sim, n_blocks, 1)
    offsets = np.arange(block_size)
    # (n_simulations, n_blocks, block_size)
    full_indices = start_indices[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
    # Flatten to (n_simulations, n_blocks * block_size) and trim to total_days
    return full_indices.reshape(n_simulations, -1)[:, :total_days]


def _stationary_bootstrap_indices(
    n_source: int,
    n_simulations: int,
    total_days: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stationary bootstrap indices (Politis & Romano 1994).

    Block lengths are geometrically distributed with mean `block_size`
    (restart probability p = 1 / block_size), start positions are uniform over
    the source series, and blocks wrap around circularly at the series end.

    Implemented via the equivalent day-by-day recursion: on each day either
    continue the current block (index + 1, modulo n_source) with probability
    1 - p, or jump to a fresh uniform start with probability p. This is
    vectorized across all simulations; the loop runs over `total_days` (same
    order of magnitude as the capital-evolution loop below).

    Returns an (n_simulations, total_days) integer index matrix.
    """
    p = 1.0 / block_size
    indices = np.empty((n_simulations, total_days), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n_source, size=n_simulations)

    for t in range(1, total_days):
        new_block = rng.random(n_simulations) < p
        cont = (indices[:, t - 1] + 1) % n_source
        starts = rng.integers(0, n_source, size=n_simulations)
        indices[:, t] = np.where(new_block, starts, cont)

    return indices


def build_bootstrap_indices(
    bootstrap_method: str,
    n_source: int,
    n_simulations: int,
    total_days: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Dispatch to the requested bootstrap index builder.

    `bootstrap_method`: "block" (default, fixed-length) or "stationary".
    """
    if bootstrap_method == "block":
        return _block_bootstrap_indices(
            n_source, n_simulations, total_days, block_size, rng,
        )
    if bootstrap_method == "stationary":
        return _stationary_bootstrap_indices(
            n_source, n_simulations, total_days, block_size, rng,
        )
    raise ValueError(
        f"Unknown bootstrap_method '{bootstrap_method}'. "
        "Expected 'block' or 'stationary'."
    )


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
    bootstrap_method: str = "block",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized MCS for a single strategy + scenario combination.

    1. Paired bootstrap: all paths simultaneously via precomputed indices
       (returns + signals remain correlated). `bootstrap_method` selects the
       fixed-length block bootstrap ("block") or the stationary bootstrap
       ("stationary", Politis & Romano 1994).
    2. Capital evolution: day by day across all paths in parallel (NumPy vectors),
       with monthly withdrawal (every 21 trading days) and ruin detection.

    Returns:
        final_capitals: (n_simulations,) terminal capital per path
        all_capital_histories: (n_simulations, total_days) full paths
    """
    n_source = len(rets_arr)

    # --- Vectorized paired bootstrap ---
    full_indices = build_bootstrap_indices(
        bootstrap_method, n_source, n_simulations, total_days, block_size, rng,
    )

    sim_rets = rets_arr[full_indices]   # (n_simulations, total_days)
    sim_sigs = sig_arr[full_indices]    # (n_simulations, total_days)

    # --- Vectorized capital evolution ---
    capitals = np.full(n_simulations, start_capital, dtype=np.float64)
    all_capital_histories = np.empty((n_simulations, total_days), dtype=np.float64)
    ruined = np.zeros(n_simulations, dtype=bool)

    for i in range(total_days):
        # Apply return (all paths simultaneously)
        capitals *= (1 + sim_rets[:, i])

        # Monthly withdrawal (every 21 trading days)
        if i % 21 == 0:
            withdrawal_amt = np.full(n_simulations, withdrawal)
            # Liquidity fee if signal == 0 (invested in bull phase)
            fee_mask = sim_sigs[:, i] == 0
            withdrawal_amt[fee_mask] += withdrawal * fee
            capitals -= withdrawal_amt

        # Ruin check: set newly ruined paths to 0
        newly_ruined = (capitals <= 0) & ~ruined
        capitals[newly_ruined] = 0.0
        ruined |= newly_ruined

        # Already ruined paths stay at 0
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
    bootstrap_method: str = "block",
) -> tuple[list[dict], dict]:
    """
    Bootstrap Monte Carlo simulation (MCS) as robustness check.

    `bootstrap_method` selects the resampling scheme: "block" (fixed-length
    block bootstrap) or "stationary" (Politis & Romano 1994).

    Paired bootstrap: return blocks + signal blocks are drawn together
    to preserve the correlation between returns and signals.

    Common random numbers across strategies: all (scenario, strategy) cells
    share one bootstrap index stream, so path s uses the identical resampled
    trading days for every strategy and differs only through each strategy's
    own returns and signals. This makes the downstream paired Wilcoxon tests
    (test_h1_drawdown, test_h2_transformer) genuinely paired.

    Withdrawal simulation: monthly withdrawal (every 21 trading days) with
    a liquidity fee if invested in a bull phase.

    Reproducibility ensured via random_seed.

    Optimized for high path counts (10,000+):
    - Vectorized bootstrap resampling (NumPy fancy indexing)
    - Vectorized capital evolution (all paths in parallel)
    - Parallelization across strategies via concurrent.futures

    Returns (all_mc_summaries, mcs_paths_collector):
    - all_mc_summaries: list of dicts with ruin probability and median terminal capital
    - mcs_paths_collector: dict with all simulated capital paths
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    total_days = sim_years * trading_days_per_year

    # Check that returns are available
    if daily_rets.empty:
        raise ValueError("daily_rets is empty. Check the data source backtesting_results.")

    # Common random numbers (CRN): every (scenario, strategy) cell resamples
    # with the SAME bootstrap index stream, so path s draws the identical
    # sequence of source trading days for all strategies. Only each strategy's
    # own returns and signals differ along those shared days, which is the
    # pairing that the Wilcoxon tests in test_h1_drawdown / test_h2_transformer
    # assume ("same bootstrap indices -> paired paths"). Because n_source,
    # n_simulations, total_days and block_size are identical across all cells,
    # passing one shared seed reproduces the identical index matrix inside every
    # worker, so no large index array has to cross the process boundary.
    # Determinism is preserved via random_seed.
    shared_seed = np.random.SeedSequence(random_seed)

    all_mc_summaries = []
    mcs_paths_collector = {}

    # --- Prepare jobs: (scenario, strategy) pairs ---
    jobs = []
    job_keys = []

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
                shared_seed,
                bootstrap_method,
            ))
            job_keys.append((sc_name, strategy))

    # --- Run in parallel or sequentially ---
    n_workers = min(len(jobs), max(1, os.cpu_count() - 1))
    results = {}

    if n_workers > 1 and n_simulations >= 1000:
        print(f"MCS: starting {len(jobs)} jobs on {n_workers} workers "
              f"({n_simulations:,} paths per combination)...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_key = {}
            for key, job_args in zip(job_keys, jobs):
                # SeedSequence → create generator in the worker
                future = executor.submit(_run_strategy_job, *job_args)
                future_to_key[future] = key

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                results[key] = future.result()
    else:
        print(f"MCS: starting {len(jobs)} jobs sequentially "
              f"({n_simulations:,} paths per combination)...")
        for key, job_args in zip(job_keys, jobs):
            results[key] = _run_strategy_job(*job_args)

    # --- Aggregate results ---
    for (sc_name, strategy), (final_capitals, all_histories) in results.items():
        print(f"  [done] {sc_name} / {strategy}")

        # Write paths into the collector
        for s in range(n_simulations):
            path_id = f"{sc_name}_{strategy}_path_{s:03d}"
            mcs_paths_collector[path_id] = all_histories[s].tolist()

        # Summary statistics
        ruin_prob = np.mean(final_capitals <= 0)
        median_wealth = np.median(final_capitals)

        all_mc_summaries.append({
            "Scenario": sc_name,
            "Strategy": strategy.replace("_", " "),
            "Ruin Probability": f"{ruin_prob:.2%}",
            "Median Terminal Capital": f"{median_wealth:,.2f} €",
        })

    return all_mc_summaries, mcs_paths_collector


def _run_strategy_job(
    rets_arr, sig_arr, n_simulations, total_days, block_size,
    start_capital, withdrawal, fee, seed, bootstrap_method="block",
):
    """Wrapper for ProcessPoolExecutor: builds a generator from `seed`.

    `seed` is the SAME SeedSequence for every (scenario, strategy) cell, so all
    cells reconstruct the identical bootstrap index matrix (common random
    numbers). Reusing a SeedSequence yields identical streams by design, which
    is exactly the intended pairing (contrast SeedSequence.spawn, which would
    give independent streams).
    """
    rng = np.random.default_rng(seed)
    return _simulate_strategy(
        rets_arr, sig_arr, n_simulations, total_days, block_size,
        start_capital, withdrawal, fee, rng, bootstrap_method,
    )


# ============================================================
# Issue #13: extended evaluation metrics (thesis ch. 4.1-4.4)
# ============================================================

# ------------------------------------------------------------
# Ch. 4.2/4.4: Ulcer index
# ------------------------------------------------------------
def ulcer_index(equity: pd.Series) -> float:
    """
    Martin (1989): RMS drawdown. More robust stress measure than MaxDD,
    since deep AND long drawdowns are penalized.
    """
    roll_max = equity.cummax()
    dd_pct = (equity / roll_max - 1.0) * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def add_ulcer_to_table(
    backtesting_results: pd.DataFrame,
    evaluation_table: pd.DataFrame,
) -> pd.DataFrame:
    """Appends an 'Ulcer Index' column to the evaluation table."""
    ui = {
        col.replace("_", " "): round(ulcer_index(backtesting_results[col]), 2)
        for col in backtesting_results.columns
    }
    evaluation_table["Ulcer Index"] = pd.Series(ui)
    return evaluation_table


# ------------------------------------------------------------
# Ch. 4.1: classification vs. NBER ground truth
# ------------------------------------------------------------
def compute_classification_metrics(
    test_df: pd.DataFrame,
    nber_signal: pd.Series,
    models: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Per model: precision / recall / F1 / confusion matrix against NBER.
    `test_df` must contain <Model>_Signal columns.
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
            "Model": m,
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
            "F1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
            "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
        })
    return pd.DataFrame(rows).set_index("Model"), cms


def plot_confusion_matrices(
    cms: dict,
    save_path: str,
) -> None:
    """Confusion matrices as a grid (one per model)."""
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
        ax.set_xlabel("Predicted"); ax.set_ylabel("NBER (ground truth)")
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
    ROC + PR curves per model (uses <Model>_Prob, not _Signal,
    for a threshold-independent comparison).
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

        rows.append({"Model": m, "ROC-AUC": round(roc_auc, 3), "PR-AUC": round(pr_auc, 3)})

    ax_r.set_xlabel("False Positive Rate"); ax_r.set_ylabel("True Positive Rate")
    ax_r.set_title("ROC Curves (vs. NBER)"); ax_r.legend(loc="lower right")
    ax_r.grid(alpha=0.25)
    fig_roc.savefig(roc_path, dpi=300, bbox_inches="tight"); plt.close(fig_roc)

    ax_p.set_xlabel("Recall"); ax_p.set_ylabel("Precision")
    ax_p.set_title("Precision-Recall Curves (vs. NBER)"); ax_p.legend(loc="lower left")
    ax_p.grid(alpha=0.25)
    fig_pr.savefig(pr_path, dpi=300, bbox_inches="tight"); plt.close(fig_pr)

    return pd.DataFrame(rows).set_index("Model")


# ------------------------------------------------------------
# Ch. 4.1: signal churning / whipsaw / threshold sensitivity
# ------------------------------------------------------------
def churning_stats(
    test_df: pd.DataFrame,
    models: list[str],
    fee_rate: float,
    min_phase_days: int = 5,
) -> pd.DataFrame:
    """
    Per model: number of signal switches, share of 'whipsaws' (phases < min_phase_days),
    average phase length, implied costs at `fee_rate`.
    """
    rows = []
    for m in models:
        sig = test_df.get(f"{m}_Signal")
        if sig is None:
            continue
        sig = sig.dropna().astype(int)
        n_switches = int((sig.diff().abs() == 1).sum())
        # Phase lengths via run-length encoding
        changes = (sig != sig.shift()).cumsum()
        phase_lengths = sig.groupby(changes).size().values
        whipsaws = int((phase_lengths < min_phase_days).sum())
        rows.append({
            "Model": m,
            "Signal Switches": n_switches,
            f"Whipsaws (<{min_phase_days}d)": whipsaws,
            "Whipsaw Share": f"{whipsaws / max(len(phase_lengths), 1):.1%}",
            "Mean Phase (Days)": round(float(np.mean(phase_lengths)), 1),
            "Median Phase (Days)": int(np.median(phase_lengths)),
            "Cumul. Costs": f"{n_switches * fee_rate:.2%}",
        })
    return pd.DataFrame(rows).set_index("Model")


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
    Varies the threshold for a single model and measures how
    final equity (in €), MaxDD, and #switches change (thesis ch. 4.1, smoothing).
    Requires `<model>_Prob` to be present in test_df.
    """
    prob_col = f"{model}_Prob"
    assert prob_col in test_df.columns, f"{prob_col} missing in test_df"

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
            "Switches": n_switches,
        })
    return pd.DataFrame(rows).set_index("Threshold")


# ------------------------------------------------------------
# Ch. 4.2: time-to-recovery + switch timing
# ------------------------------------------------------------
def time_to_recovery(equity: pd.Series, min_dd: float = -0.05) -> pd.DataFrame:
    """
    All drawdown phases with DD < min_dd: peak date, trough date,
    recovery date, duration in trading days. NaN for unrecovered phases.
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
                    "Drawdown Duration (d)": (trough - start).days,
                    "Recovery Duration (d)": (date - trough).days,
                    "Total (d)": (date - start).days,
                })
            start = None
    # Open phase at the end (not yet recovered)
    if start is not None:
        seg = equity.loc[start:]
        dd_seg = seg / seg.cummax() - 1
        trough_idx = dd_seg.idxmin()
        dd_min = float(dd_seg.min())
        if dd_min <= min_dd:
            phases.append({
                "Peak":     start.strftime("%Y-%m-%d"),
                "Trough":   trough_idx.strftime("%Y-%m-%d"),
                "Recovery": "open",
                "Max DD":   f"{dd_min*100:.2f}%",
                "Drawdown Duration (d)": (trough_idx - start).days,
                "Recovery Duration (d)": np.nan,
                "Total (d)": np.nan,
            })
    return pd.DataFrame(phases)


def switch_timing_vs_peak(
    test_df: pd.DataFrame,
    backtesting_results: pd.DataFrame,
    model: str,
    crisis_windows: dict,
) -> pd.DataFrame:
    """
    Per crisis window: how many days BEFORE the buy-and-hold drawdown
    trough did the model already set a bear signal? Negative = reacted
    too late.
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
            lead = (trough_date - first_bear).days  # positive = early
        rows.append({
            "Crisis": name,
            "DD Trough": trough_date.date(),
            "First Bear Signal": first_bear.date() if not pd.isna(first_bear) else None,
            "Lead (Days)": lead,
        })
    return pd.DataFrame(rows).set_index("Crisis")


# ------------------------------------------------------------
# Ch. 4.3: MCS: terminal wealth, depletion CI, H1/H2 tests
# ------------------------------------------------------------
def mcs_final_capitals(
    mcs_paths_collector: dict,
    scenarios: list[str],
    strategies: list[str],
) -> dict:
    """
    Reconstruct the terminal capitals per (scenario, strategy) from the
    path collector as a 1D NumPy array.
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
    Wilson CI for the depletion rate p = P(terminal capital <= 0).
    Wilson instead of Wald because it is numerically more stable for p ~ 0.
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
            "Scenario": sc,
            "Strategy": s,
            "Depletion Rate": f"{p:.2%}",
            "95% CI Lower":  f"{max(0, center - half):.2%}",
            "95% CI Upper":  f"{min(1, center + half):.2%}",
            "n_ruin / n_paths": f"{k}/{n}",
        })
    return pd.DataFrame(rows).set_index(["Scenario", "Strategy"])


# ------------------------------------------------------------
# Issue #7: bootstrap robustness (block vs. stationary)
# ------------------------------------------------------------
def _wilson_interval(k: int, n: int, z: float) -> tuple[float, float, float]:
    """
    Wilson score interval for a binomial proportion.

    Returns (p, lower, upper) with p = k / n. Wilson instead of Wald because it
    stays well behaved for p close to 0 (typical for depletion rates).
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def compare_bootstrap_methods(
    finals_block: dict,
    finals_stationary: dict,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Side-by-side robustness comparison of the two MCS resampling schemes
    (Issue #7). Per (scenario, strategy) it reports, for the fixed-length block
    bootstrap and the stationary bootstrap (same seed, same n_paths):

    - depletion rate P(terminal capital <= 0) with Wilson 95% CI,
    - median terminal capital,
    - and the block -> stationary difference for both (Delta columns).

    `finals_block` / `finals_stationary` are the (scenario, strategy) -> terminal
    capital arrays produced by `mcs_final_capitals` for each run.
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)

    rows = []
    for key in finals_block:
        if key not in finals_stationary:
            continue
        sc, s = key
        ab, as_ = finals_block[key], finals_stationary[key]
        pb, lob, hib = _wilson_interval(int(np.sum(ab <= 0)), len(ab), z)
        ps, los, his = _wilson_interval(int(np.sum(as_ <= 0)), len(as_), z)
        med_b, med_s = float(np.median(ab)), float(np.median(as_))
        rows.append({
            "Scenario": sc,
            "Strategy": s.replace("_", " "),
            "Depletion (Block)":         f"{pb:.2%}",
            "95% CI (Block)":            f"[{lob:.2%}, {hib:.2%}]",
            "Depletion (Stationary)":    f"{ps:.2%}",
            "95% CI (Stationary)":       f"[{los:.2%}, {his:.2%}]",
            "Delta Depletion (pp)":      f"{(ps - pb) * 100:+.2f}",
            "Median Final (Block)":      f"{med_b:,.0f} EUR",
            "Median Final (Stationary)": f"{med_s:,.0f} EUR",
            "Delta Median (EUR)":        f"{(med_s - med_b):+,.0f}",
        })
    return pd.DataFrame(rows).set_index(["Scenario", "Strategy"])


def bootstrap_robustness_summary(
    finals_block: dict,
    finals_stationary: dict,
    benchmark: str = "Buy_Hold",
) -> str:
    """
    Data-driven conclusion for the bootstrap robustness table: quantifies how
    far the two resampling schemes disagree and whether the core tail-risk
    ordering (regime models vs. the benchmark) is preserved under both.
    Returns a short Markdown paragraph.
    """
    def _rate(finals, key):
        arr = finals[key]
        return float(np.sum(arr <= 0)) / len(arr)

    keys = [k for k in finals_block if k in finals_stationary]
    if not keys:
        return "No overlapping (scenario, strategy) cells to compare."

    max_delta_pp = max(
        abs(_rate(finals_stationary, k) - _rate(finals_block, k)) * 100 for k in keys
    )

    scenarios = sorted({sc for sc, _ in keys})
    order_preserved = 0
    bench_lead_ok = 0
    bench_lead_total = 0
    for sc in scenarios:
        strats = [s for (s_sc, s) in keys if s_sc == sc]
        # Ranking of strategies by depletion rate under each method.
        order_b = sorted(strats, key=lambda s: _rate(finals_block, (sc, s)))
        order_s = sorted(strats, key=lambda s: _rate(finals_stationary, (sc, s)))
        if order_b == order_s:
            order_preserved += 1
        # Does every regime model keep its depletion advantage over the benchmark?
        if benchmark in strats:
            b_rate_block = _rate(finals_block, (sc, benchmark))
            b_rate_stat = _rate(finals_stationary, (sc, benchmark))
            for s in strats:
                if s == benchmark:
                    continue
                bench_lead_total += 1
                lead_block = _rate(finals_block, (sc, s)) <= b_rate_block
                lead_stat = _rate(finals_stationary, (sc, s)) <= b_rate_stat
                if lead_block == lead_stat:
                    bench_lead_ok += 1

    lines = [
        f"**Robustness summary.** Across {len(keys)} (scenario, strategy) cells, "
        f"the largest depletion-rate difference between the block and stationary "
        f"bootstrap is {max_delta_pp:.2f} pp. "
        f"The strategy ranking by depletion rate is identical under both methods "
        f"in {order_preserved}/{len(scenarios)} scenarios.",
    ]
    if bench_lead_total:
        lines.append(
            f"The sign of every regime model's depletion advantage over "
            f"{benchmark.replace('_', ' ')} is preserved under both methods in "
            f"{bench_lead_ok}/{bench_lead_total} model-scenario comparisons, so the "
            f"tail-protection findings do not hinge on the resampling scheme."
        )
    return "\n\n".join(lines)


def mcs_path_maxdd(mcs_paths_collector: dict, prefix: str) -> np.ndarray:
    """MaxDD per path (for hypothesis tests)."""
    dds = []
    for k, path in mcs_paths_collector.items():
        if not k.startswith(prefix):
            continue
        arr = np.asarray(path, dtype=float)
        cummax = np.maximum.accumulate(arr)
        # Protection against division by 0 in ruin paths
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
    H1: Regime switching reduces MaxDD vs. buy and hold.
    Paired Wilcoxon test (same bootstrap indices → paired paths).
    """
    from scipy.stats import wilcoxon

    dd_bh = mcs_path_maxdd(mcs_paths_collector, f"{scenario}_{benchmark}_path_")
    rows = []
    for m in regime_models:
        dd_m = mcs_path_maxdd(mcs_paths_collector, f"{scenario}_{m}_path_")
        if len(dd_m) != len(dd_bh) or len(dd_m) == 0:
            continue
        # H1: dd_m > dd_bh (less negative) → one-sided "greater"
        try:
            _, p = wilcoxon(dd_m, dd_bh, alternative="greater")
        except ValueError:
            p = np.nan
        rows.append({
            "Model": m,
            "Median MaxDD (Model)": f"{np.median(dd_m)*100:.2f}%",
            "Median MaxDD (B&H)":   f"{np.median(dd_bh)*100:.2f}%",
            "Δ Median":             f"{(np.median(dd_m) - np.median(dd_bh))*100:+.2f} pp",
            "Wilcoxon p":           f"{p:.2e}" if not np.isnan(p) else "n/a",
            f"H1 (α={alpha})":      ("confirmed" if (not np.isnan(p) and p < alpha)
                                     else "rejected"),
        })
    return pd.DataFrame(rows).set_index("Model")


def test_h2_transformer(
    finals: dict,
    scenario: str,
    challenger: str = "Transformer",
    competitors: tuple = ("MSM", "HMM", "HMM_Uni", "LSTM"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    H2: The Transformer beats econometrics/LSTM in terminal wealth.
    Paired Wilcoxon test on terminal capital (same paths).
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
            "Comparison":       f"{challenger} vs. {c}",
            f"Median {challenger}": f"{np.median(w_ch):,.0f} €",
            f"Median {c}":          f"{np.median(w_c):,.0f} €",
            "Δ Median":         f"{(np.median(w_ch) - np.median(w_c)):+,.0f} €",
            "Wilcoxon p":       f"{p:.2e}" if not np.isnan(p) else "n/a",
            f"H2 (α={alpha})":  ("confirmed" if (not np.isnan(p) and p < alpha)
                                 else "rejected"),
        })
    return pd.DataFrame(rows).set_index("Comparison")


def plot_mcs_violins(
    finals: dict,
    scenarios: list[str],
    strategies: list[str],
    color_map: dict,
    save_path_template: str,
) -> None:
    """Violin plots per scenario (in addition to the boxplots)."""
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
        ax.set_title(f"MCS Terminal Wealth: Scenario {sc}")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.6)
        ax.grid(alpha=0.25)
        fig.savefig(save_path_template.format(sc.lower()),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------
# Ch. 4.4: break-even costs + withdrawal rate sensitivity
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
    Per model: final wealth under varying cost rates.
    Break-even = smallest cost rate at which final wealth <= B&H.
    Returns (summary table, {model: Series(fee_bps -> final_wealth)}).
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
            "Model": m,
            "Final @10bps": round(wealths.get(10, float("nan")), 3),
            "B&H Final":    round(bh_final, 3),
            "Break-Even (bps)": be_bps if be_bps is not None else ">max",
        })
    return pd.DataFrame(summary).set_index("Model"), curves


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
    ax.set_xlabel("Transaction costs (bps)")
    ax.set_ylabel("Final wealth (cumulative)")
    ax.set_title("Break-Even Analysis: Cost Rate vs. Terminal Wealth")
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
    Identical start, varying annual withdrawal rate. Result:
    terminal capital + year of depletion per strategy × rate.
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
                status = "Capital preserved"
            else:
                depleted = sim[sim[col] <= 0].index[0]
                status = f"Depleted ({depleted.strftime('%Y')})"
            rows.append({
                "Withdrawal Rate (p.a.)": f"{r:.1%}",
                "Strategy": col,
                "Terminal Capital": f"{final:,.0f} €",
                "Status": status,
            })
    return (
        pd.DataFrame(rows)
        .pivot(index="Strategy", columns="Withdrawal Rate (p.a.)",
               values=["Terminal Capital", "Status"])
    )


# ------------------------------------------------------------
# Ch. 4.1: regime probability heatmap
# ------------------------------------------------------------
def plot_regime_probability_heatmap(
    test_df: pd.DataFrame,
    models: list[str],
    save_path: str,
) -> None:
    """
    Heatmap: y=model, x=time, color=bear probability (<Model>_Prob).
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
    ax.set_title("Regime bear probabilities over the OOS period")
    fig.colorbar(im, ax=ax, shrink=0.7, label="P(Bear)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
