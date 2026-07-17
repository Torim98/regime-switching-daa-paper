import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pandas as pd


def plot_walk_forward_schema(
    splits_summary: pd.DataFrame,
    save_path: str,
    mode: str = "rolling",
    train_window_years: int | None = None,
    test_window_months: int | None = None,
    train_color: str = "#4C72B0",
    test_color: str = "#DD8452",
) -> None:
    """
    Visualization of the walk-forward schema as a horizontal Gantt-style plot.

    For each fold, a blue bar is drawn for the training window and an orange
    bar for the OOS test window. Fold IDs increase downwards (fold 1 at the top),
    which makes the rolling shift of the train/test window over time
    immediately visible.

    Parameters
    ----------
    splits_summary : pd.DataFrame
        Output of src.backtest.walk_forward.summarize_splits:
        index = fold ID, columns = train_start, train_end, test_start, test_end,
        n_train, n_test.
    save_path : str
        Target path for the PNG file (DPI=300).
    mode : str
        "rolling" or "expanding"; only used in the title.
    train_window_years : int | None
        Length of the training window (years) for the title.
    test_window_months : int | None
        Length of the test window (months) for the title.
    train_color, test_color : str
        Colors of the train/test bars. Defaults consistent with the other
        pipeline plots.
    """
    n_folds = len(splits_summary)
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.25 * n_folds)))

    for fold_id, row in splits_summary.iterrows():
        train_width = row["train_end"] - row["train_start"]
        test_width = row["test_end"] - row["test_start"]
        ax.barh(fold_id, train_width, left=row["train_start"],
                height=0.7, color=train_color, edgecolor="none")
        ax.barh(fold_id, test_width, left=row["test_start"],
                height=0.7, color=test_color, edgecolor="none")

    ax.invert_yaxis()  # Fold 1 at the top
    ax.set_xlabel("Date")
    ax.set_ylabel("Fold")

    subtitle_parts = [f"Mode: {mode}"]
    if train_window_years is not None:
        subtitle_parts.append(f"Train: {train_window_years}y")
    if test_window_months is not None:
        subtitle_parts.append(f"Test: {test_window_months}m")
    subtitle_parts.append(f"{n_folds} folds")
    ax.set_title(
        "Walk-Forward Schema: Train/Test Windows Over Time\n"
        + " | ".join(subtitle_parts),
        fontsize=12,
    )

    handles = [
        mpatches.Patch(color=train_color, label="Train"),
        mpatches.Patch(color=test_color, label="Test (OOS)"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curves(backtesting_results, color_map: dict, save_path: str,
                       initial_capital: float = 1.0):
    """Equity curves of all strategies in € (unnormalized)."""
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(backtesting_results['Buy_Hold'] * initial_capital,
            label='Static 60/40 portfolio (benchmark)',
            color=color_map.get('Buy_Hold', 'gray'), alpha=0.5, linestyle='--')

    for col in backtesting_results.columns:
        if col == 'Buy_Hold':
            continue
        color = color_map.get(col, None)
        ax.plot(backtesting_results[col] * initial_capital,
                label=f'Strategy: {col.replace("_", " ")}',
                color=color, linewidth=1.5, alpha=0.8)

    ax.set_title(
        f"Equity Curves: Dynamic Comparison of the Regime-Switching Models "
        f"(Initial Capital {initial_capital:,.0f} €)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital (€)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.2)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_transaction_costs(backtesting_costs, fee_rate: float, color_map: dict, save_path: str):
    """Cumulative transaction costs."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for col in backtesting_costs.columns:
        if col == 'Buy_Hold':
            continue
        color = color_map.get(col, None)
        ax.plot(backtesting_costs[col] * 100,
                label=f'Costs: {col.replace("_", " ")}', color=color)

    ax.set_title(f"Cumulative Transaction Costs Over Time (Fee: {fee_rate*100}%)")
    ax.set_ylabel("Costs in %")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_sorr_scenario(sim_results, scenario_name: str, params: dict,
                       color_map: dict, save_path: str):
    """SORR simulation for a single scenario."""
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, col in enumerate(sim_results.columns):
        color = color_map.get(col, default_colors[i % len(default_colors)])
        ax.plot(sim_results[col], label=col.replace('_', ' '), color=color)

    ax.set_title(f"SORR Scenario {scenario_name}: Start {params['start']:,.0f}€, "
                 f"Withdrawal {params['withdrawal']:,.0f}€")
    ax.axhline(y=0, color='black', linestyle='-')
    ax.legend(loc='upper left', fontsize='small')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_mcs_boxplots(finals, daily_rets_columns, scenarios,
                      sim_years: int, save_path_template: str):
    """MCS violin + boxplots per scenario (optimized for 10,000+ paths).

    `finals`: MCSResult.finals, keyed (scenario, strategy), with the terminal
    capitals of ALL simulated paths per cell.
    """
    for sc_name, params in scenarios.items():
        mc_results_scenario = {}
        for strategy in daily_rets_columns:
            arr = finals.get((sc_name, strategy))
            if arr is not None and len(arr):
                mc_results_scenario[strategy] = arr

        if mc_results_scenario:
            labels = [s.replace('_', ' ') for s in mc_results_scenario.keys()]
            data = list(mc_results_scenario.values())

            fig, ax = plt.subplots(figsize=(12, 6))

            # Violin plot for the distribution shape
            vp = ax.violinplot(data, showmedians=False, showextrema=False)
            for i, body in enumerate(vp['bodies']):
                body.set_alpha(0.3)

            # Boxplot on top for quartiles + outliers
            bp = ax.boxplot(
                data, tick_labels=labels, widths=0.15,
                showfliers=False,  # no outlier points with 10k values
                medianprops=dict(color='red', linewidth=2),
            )

            # Annotate median values
            for i, vals in enumerate(data, start=1):
                import numpy as _np
                med = _np.median(vals)
                ax.annotate(
                    f"{med:,.0f}€", xy=(i, med),
                    xytext=(18, 5), textcoords="offset points",
                    fontsize=8, color="red",
                )

            ax.set_title(
                f"MCS {sc_name}: Distribution of Terminal Capital "
                f"(n={len(data[0]):,}, start: {params['start']:,.0f}€)"
            )
            ax.set_ylabel(f"Terminal capital after {sim_years} years in €")
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.grid(axis='y', alpha=0.3)

            save_path = save_path_template.format(sc_name.lower())
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

def plot_mcs_paths(mcs_results_df, scenarios_list: list, strategies,
                   color_map: dict, save_path: str,
                   trading_days_per_year: int = 252,
                   start_year: int | None = None):
    """
    MCS path trajectories for all scenarios (optimized for 10,000+ paths).

    Instead of drawing 10,000 individual lines (extremely slow), a quantile
    band (5%-25%-50%-75%-95%) is plotted per strategy. In addition, at most
    50 random paths are drawn as a spaghetti overlay to visually preserve
    the dispersion of the individual paths.
    """
    import numpy as np

    MAX_SAMPLE_PATHS = 50

    fig, axes = plt.subplots(len(scenarios_list), 1,
                             figsize=(15, 6 * len(scenarios_list)), sharex=True)
    if len(scenarios_list) == 1:
        axes = [axes]

    for ax, sc_name in zip(axes, scenarios_list):
        for strat in strategies:
            prefix = f"{sc_name}_{strat}_path_"
            strat_paths = mcs_results_df.filter(like=prefix)

            if strat_paths.empty:
                continue

            color = color_map.get(strat, 'black')
            values = strat_paths.values  # (total_days, n_paths)

            q05 = np.quantile(values, 0.05, axis=1)
            q25 = np.quantile(values, 0.25, axis=1)
            q50 = np.quantile(values, 0.50, axis=1)
            q75 = np.quantile(values, 0.75, axis=1)
            q95 = np.quantile(values, 0.95, axis=1)

            x = np.arange(values.shape[0])

            ax.fill_between(x, q05, q95, color=color, alpha=0.08)
            ax.fill_between(x, q25, q75, color=color, alpha=0.15)

            n_paths = values.shape[1]
            sample_idx = np.random.choice(
                n_paths, size=min(MAX_SAMPLE_PATHS, n_paths), replace=False
            )
            ax.plot(values[:, sample_idx], color=color, alpha=0.06, linewidth=0.5)

            ax.plot(x, q50, color=color, linewidth=2,
                    label=strat.replace('_', ' '))

        # Year ticks: calendar years from start_year (default: current year).
        # For long horizons (> 15 years) label only every 5th year to keep
        # the axis legible (31 labels at the 30-year horizon otherwise).
        if start_year is None:
            from datetime import datetime
            start_year = datetime.now().year
        n_years = values.shape[0] // trading_days_per_year
        tick_step = 1 if n_years <= 15 else 5
        year_ticks = [y * trading_days_per_year for y in range(0, n_years + 1, tick_step)]
        year_labels = [str(start_year + y) for y in range(0, n_years + 1, tick_step)]
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels)

        ax.set_title(f"MCS Path Trajectories: Scenario {sc_name} "
                     f"(bands: 25-75% / 5-95%, n={n_paths:,})")
        ax.set_ylabel("Capital in €")
        ax.axhline(y=0, color='black', linewidth=1.5)
        ax.grid(alpha=0.2)
        ax.legend(loc='upper left', ncol=2)

    plt.xlabel("Simulation time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_mcs_quantiles(mcs_results_df, scenarios_list: list, strategies,
                       total_days: int, color_map: dict, save_path: str,
                       trading_days_per_year: int = 252,
                       start_year: int | None = None):
    """MCS confidence intervals (5%-95%) for all scenarios (optimized via NumPy)."""
    import numpy as np

    fig, axes = plt.subplots(len(scenarios_list), 1,
                             figsize=(15, 6 * len(scenarios_list)), sharex=True)
    if len(scenarios_list) == 1:
        axes = [axes]

    for ax, sc_name in zip(axes, scenarios_list):
        n_paths_display = "?"
        for strat in strategies:
            prefix = f"{sc_name}_{strat}_path_"
            strat_paths = mcs_results_df.filter(like=prefix)

            if strat_paths.empty:
                continue

            values = strat_paths.values
            color = color_map.get(strat, 'black')

            q05 = np.quantile(values, 0.05, axis=1)
            q50 = np.quantile(values, 0.50, axis=1)
            q95 = np.quantile(values, 0.95, axis=1)

            x = np.arange(total_days)
            ax.fill_between(x, q05, q95, color=color, alpha=0.15)
            ax.plot(x, q50, color=color, linewidth=1.5,
                    label=f"{strat.replace('_', ' ')} (median)")

            n_paths_display = values.shape[1]

        # Year ticks: calendar years from start_year (default: current year).
        # For long horizons (> 15 years) label only every 5th year to keep
        # the axis legible (31 labels at the 30-year horizon otherwise).
        if start_year is None:
            from datetime import datetime
            start_year = datetime.now().year
        n_years = total_days // trading_days_per_year
        tick_step = 1 if n_years <= 15 else 5
        year_ticks = [y * trading_days_per_year for y in range(0, n_years + 1, tick_step)]
        year_labels = [str(start_year + y) for y in range(0, n_years + 1, tick_step)]
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels)

        ax.set_title(f"MCS Confidence Intervals (5% - 95%): "
                     f"Scenario {sc_name} (n={n_paths_display:,})")
        ax.set_ylabel("Capital in €")
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1,
                   label="Depletion threshold")
        ax.grid(alpha=0.2)
        ax.legend(loc='upper left', ncol=2)

    plt.xlabel("Simulation time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_rolling_sharpe(rolling_sharpe: pd.DataFrame, color_map: dict, save_path: str):
    """Rolling 1-year Sharpe ratio of all strategies."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # .dropna() + preserving the original index: matplotlib draws a continuous
    # line over the available points and visually skips NaN indices.
    bh = rolling_sharpe["Buy_Hold"].dropna()
    ax.plot(bh.index, bh.values,
            label="Buy & Hold (benchmark)",
            color=color_map.get("Buy_Hold", "gray"), alpha=0.5, linestyle="--")

    for col in rolling_sharpe.columns:
        if col == "Buy_Hold":
            continue
        series = rolling_sharpe[col].dropna()
        color = color_map.get(col, None)
        ax.plot(series.index, series.values,
                label=f"Strategy: {col}", color=color, linewidth=1.5, alpha=0.8)

    # Reference lines for interpretation
    ax.axhline(y=1.0, color="green", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axhline(y=2.0, color="darkgreen", linewidth=0.5, linestyle=":", alpha=0.5)
    # Fix the y-axis to a plausible range
    ax.set_ylim(-3, 5)

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax.set_title("Rolling Sharpe Ratio (252-day window, cap ±10, low-vol NaN)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe ratio (annualized)")
    ax.legend(loc="upper left", ncol=2, fontsize="small")
    ax.grid(True, alpha=0.2)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_drawdown(backtesting_results: pd.DataFrame, color_map: dict, save_path: str):
    """Drawdown paths of all strategies."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for col in backtesting_results.columns:
        equity = backtesting_results[col]
        dd = (equity / equity.cummax() - 1) * 100
        color = color_map.get(col, None)
        style = "--" if col == "Buy_Hold" else "-"
        alpha = 0.5 if col == "Buy_Hold" else 0.8
        ax.fill_between(dd.index, dd, 0, alpha=0.15, color=color)
        ax.plot(dd, label=col, color=color, linewidth=1.2,
                linestyle=style, alpha=alpha)

    ax.set_title("Drawdown Paths of All Strategies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", ncol=2, fontsize="small")
    ax.grid(True, alpha=0.2)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_optuna_plots(study, model_name: str, cfg) -> dict[str, str]:
    """Save Optuna visualizations (history, importance, contour, slice) as PNG.

    The contour plot is only created for models with >= 2 hyperparameters.
    With a single parameter (e.g. MSM with only `threshold`), the matrix
    degenerates and the plot carries no information.
    """
    from pathlib import Path
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_contour,
        plot_slice,
    )
    from optuna.importance import FanovaImportanceEvaluator

    # Number of hyperparameters in the search space (for the contour decision + size)
    n_params = max(
        (len(t.params) for t in study.trials if t.params), default=1
    )

    # fANOVA is stochastic (random-forest sampling) → fixed seed so that the
    # pipeline PNG and the dashboard live chart show exactly the same numbers.
    importance_evaluator = FanovaImportanceEvaluator(seed=42)

    plots = {
        "optuna_history": plot_optimization_history(study),
        "optuna_importance": plot_param_importances(
            study, evaluator=importance_evaluator,
        ),
        "optuna_slice": plot_slice(study),
    }
    if n_params >= 2:
        plots["optuna_contour"] = plot_contour(study)
    else:
        print(
            f"  ℹ {model_name}: contour plot skipped "
            f"(only {n_params} hyperparameter(s) in the search space)."
        )

    # Source pixel sizes per plot type (before scale=2)
    # contour is an n×n matrix → size must scale with n_params,
    # otherwise tick values overlap with the axis titles of the neighboring cell.
    sizes = {
        "optuna_history":    (1200, 700),
        "optuna_importance": (1200, max(500, 60 * n_params)),
        "optuna_contour":    (max(1400, 260 * n_params), max(1200, 220 * n_params)),
        "optuna_slice":      (max(1400, 300 * n_params), 700),
    }

    saved = {}
    for key, fig in plots.items():
        w, h = sizes[key]
        fig.update_layout(
            width=w,
            height=h,
            font=dict(size=11),
            margin=dict(l=110, r=80, t=80, b=110),
        )
        # Smaller tick fonts + more axis standoff prevent overlap in matrix plots
        fig.update_xaxes(tickfont=dict(size=9), title_standoff=20, automargin=True)
        fig.update_yaxes(tickfont=dict(size=9), title_standoff=20, automargin=True)

        raw_template = getattr(cfg.paths.assets, key)
        filename = raw_template.replace("{model}", model_name)
        path = Path(cfg.asset_path(key).replace(raw_template, filename))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path), scale=2)
        saved[key] = str(path)
        print(f"  ✓ {path}")

    # Persist the importance values in a JSON cache so that the dashboard
    # shows exactly the same numbers as the PNG file (fANOVA is stochastic).
    # The cache is updated incrementally per model.
    try:
        import json
        import optuna
        importance_dict = optuna.importance.get_param_importances(
            study, evaluator=importance_evaluator,
        )
        cache_path = Path(cfg._base_dir) / "assets" / "optuna_importance_values.json"
        cache_payload = {"studies": {}}
        if cache_path.exists():
            try:
                cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
                cache_payload.setdefault("studies", {})
            except Exception:
                pass
        cache_payload["studies"][model_name] = {
            k: float(v) for k, v in importance_dict.items()
        }
        cache_payload["source"] = "auto-written by save_optuna_plots"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
        saved["optuna_importance_json"] = str(cache_path)
    except Exception as e:
        print(f"  ⚠ Importance JSON cache not written ({model_name}): {e}")

    return saved
