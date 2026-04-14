import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curves(backtesting_results, color_map: dict, save_path: str):
    """Equity Curves aller Strategien."""
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(backtesting_results['Buy_Hold'],
            label='Statisches 60/40 Portfolio (Benchmark)',
            color=color_map.get('Buy_Hold', 'gray'), alpha=0.5, linestyle='--')

    for col in backtesting_results.columns:
        if col == 'Buy_Hold':
            continue
        color = color_map.get(col, None)
        ax.plot(backtesting_results[col],
                label=f'Strategie: {col.replace("_", " ")}',
                color=color, linewidth=1.5, alpha=0.8)

    ax.set_title("Equity Curves: Dynamischer Vergleich der Regime-Switching-Modelle")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Kumuliertes Vermögen (Start = 1.0)")
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.2)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_transaction_costs(backtesting_costs, fee_rate: float, color_map: dict, save_path: str):
    """Kumulierte Transaktionskosten."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for col in backtesting_costs.columns:
        if col == 'Buy_Hold':
            continue
        color = color_map.get(col, None)
        ax.plot(backtesting_costs[col] * 100,
                label=f'Kosten: {col.replace("_", " ")}', color=color)

    ax.set_title(f"Kumulierte Transaktionskosten im Zeitverlauf (Gebühr: {fee_rate*100}%)")
    ax.set_ylabel("Kosten in %")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_sorr_scenario(sim_results, scenario_name: str, params: dict,
                       color_map: dict, save_path: str):
    """SORR-Simulation für ein einzelnes Szenario."""
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, col in enumerate(sim_results.columns):
        color = color_map.get(col, default_colors[i % len(default_colors)])
        ax.plot(sim_results[col], label=col.replace('_', ' '), color=color)

    ax.set_title(f"SORR Szenario {scenario_name}: Start {params['start']:,.0f}€, "
                 f"Entnahme {params['withdrawal']:,.0f}€")
    ax.axhline(y=0, color='black', linestyle='-')
    ax.legend(loc='upper left', fontsize='small')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_mcs_boxplots(mcs_paths_collector, daily_rets_columns, scenarios,
                      sim_years: int, save_path_template: str):
    """MCS Violin + Boxplots pro Szenario (optimiert für 10.000+ Pfade)."""
    for sc_name, params in scenarios.items():
        mc_results_scenario = {}
        for strategy in daily_rets_columns:
            prefix = f"{sc_name}_{strategy}_path_"
            # Direkt nur Endkapitalwerte sammeln (letzter Wert je Pfad)
            finals = [
                path[-1] for k, path in mcs_paths_collector.items()
                if k.startswith(prefix)
            ]
            if finals:
                mc_results_scenario[strategy] = finals

        if mc_results_scenario:
            labels = [s.replace('_', ' ') for s in mc_results_scenario.keys()]
            data = list(mc_results_scenario.values())

            fig, ax = plt.subplots(figsize=(12, 6))

            # Violin-Plot für Verteilungsform
            vp = ax.violinplot(data, showmedians=False, showextrema=False)
            for i, body in enumerate(vp['bodies']):
                body.set_alpha(0.3)

            # Boxplot darüber für Quartile + Ausreißer
            bp = ax.boxplot(
                data, tick_labels=labels, widths=0.15,
                showfliers=False,  # Keine Outlier-Punkte bei 10k Werten
                medianprops=dict(color='red', linewidth=2),
            )

            # Median-Werte annotieren
            for i, vals in enumerate(data, start=1):
                import numpy as _np
                med = _np.median(vals)
                ax.annotate(
                    f"{med:,.0f}€", xy=(i, med),
                    xytext=(18, 5), textcoords="offset points",
                    fontsize=8, color="red",
                )

            ax.set_title(
                f"MCS {sc_name}: Verteilung des Endkapitals "
                f"(n={len(data[0]):,}, Start: {params['start']:,.0f}€)"
            )
            ax.set_ylabel(f"Endkapital nach {sim_years} Jahren in €")
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.grid(axis='y', alpha=0.3)

            save_path = save_path_template.format(sc_name.lower())
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

def plot_mcs_paths(mcs_results_df, scenarios_list: list, strategies,
                   color_map: dict, save_path: str,
                   trading_days_per_year: int = 252):
    """
    MCS Pfad-Verläufe für alle Szenarien (optimiert für 10.000+ Pfade).

    Statt 10.000 individuelle Linien zu zeichnen (extrem langsam), wird
    ein Quantil-Band (5%-25%-50%-75%-95%) pro Strategie geplottet. Zusätzlich
    werden max. 50 zufällige Pfade als Spaghetti-Overlay gezeichnet, um die
    Streuung der Einzelpfade visuell zu erhalten.
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

        # Jahres-Ticks: Kalenderjahre ab heute
        from datetime import datetime
        start_year = datetime.now().year
        n_years = values.shape[0] // trading_days_per_year
        year_ticks = [y * trading_days_per_year for y in range(n_years + 1)]
        year_labels = [str(start_year + y) for y in range(n_years + 1)]
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels)

        ax.set_title(f"MCS Pfad-Verläufe: Szenario {sc_name} "
                     f"(Bänder: 25-75% / 5-95%, n={n_paths:,})")
        ax.set_ylabel("Kapital in €")
        ax.axhline(y=0, color='black', linewidth=1.5)
        ax.grid(alpha=0.2)
        ax.legend(loc='upper left', ncol=2)

    plt.xlabel("Simulationszeit")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_mcs_quantiles(mcs_results_df, scenarios_list: list, strategies,
                       total_days: int, color_map: dict, save_path: str,
                       trading_days_per_year: int = 252):
    """MCS Konfidenz-Intervalle (5%-95%) für alle Szenarien (optimiert via NumPy)."""
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
                    label=f"{strat.replace('_', ' ')} (Median)")

            n_paths_display = values.shape[1]

        # Jahres-Ticks: Kalenderjahre ab heute
        from datetime import datetime
        start_year = datetime.now().year
        n_years = total_days // trading_days_per_year
        year_ticks = [y * trading_days_per_year for y in range(n_years + 1)]
        year_labels = [str(start_year + y) for y in range(n_years + 1)]
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels)

        ax.set_title(f"MCS Konfidenz-Intervalle (5% - 95%): "
                     f"Szenario {sc_name} (n={n_paths_display:,})")
        ax.set_ylabel("Kapital in €")
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1,
                   label="Erschöpfungsgrenze")
        ax.grid(alpha=0.2)
        ax.legend(loc='upper left', ncol=2)

    plt.xlabel("Simulationszeit")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def plot_rolling_sharpe(rolling_sharpe: pd.DataFrame, color_map: dict, save_path: str):
    """Rollierender 1-Jahres Sharpe Ratio aller Strategien."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # .dropna() + Original-Index bewahren: matplotlib zeichnet durchgehende
    # Linie über vorhandene Punkte, überspringt NaN-Indizes visuell.
    bh = rolling_sharpe["Buy_Hold"].dropna()
    ax.plot(bh.index, bh.values,
            label="Buy & Hold (Benchmark)",
            color=color_map.get("Buy_Hold", "gray"), alpha=0.5, linestyle="--")

    for col in rolling_sharpe.columns:
        if col == "Buy_Hold":
            continue
        series = rolling_sharpe[col].dropna()
        color = color_map.get(col, None)
        ax.plot(series.index, series.values,
                label=f"Strategie: {col}", color=color, linewidth=1.5, alpha=0.8)

    # Referenzlinien für Interpretation
    ax.axhline(y=1.0, color="green", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axhline(y=2.0, color="darkgreen", linewidth=0.5, linestyle=":", alpha=0.5)
    # Y-Achse auf plausiblen Bereich fixieren
    ax.set_ylim(-3, 5)

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax.set_title("Rollierender Sharpe Ratio (252-Tage-Fenster, Cap ±10, Low-Vol NaN)")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Sharpe Ratio (annualisiert)")
    ax.legend(loc="upper left", ncol=2, fontsize="small")
    ax.grid(True, alpha=0.2)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
 
def plot_drawdown(backtesting_results: pd.DataFrame, color_map: dict, save_path: str):
    """Drawdown-Verläufe aller Strategien."""
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

    ax.set_title("Drawdown-Verlauf aller Strategien")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", ncol=2, fontsize="small")
    ax.grid(True, alpha=0.2)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
 
def save_optuna_plots(study, model_name: str, cfg) -> dict[str, str]:
    """Optuna-Visualisierungen (History, Importance, Contour, Slice) als PNG speichern."""
    from pathlib import Path
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_contour,
        plot_slice,
    )

    plots = {
        "optuna_history": plot_optimization_history(study),
        "optuna_importance": plot_param_importances(study),
        "optuna_contour": plot_contour(study),
        "optuna_slice": plot_slice(study),
    }

    saved = {}
    for key, fig in plots.items():
        # Template enthält {model} → ersetzen, dann über cfg.asset_path auflösen
        raw_template = getattr(cfg.paths.assets, key)       # "optuna_{model}_history.png"
        filename = raw_template.replace("{model}", model_name)
        # asset_path nutzt _base_dir → funktioniert in Jupyter UND Docker
        path = Path(cfg.asset_path(key).replace(raw_template, filename))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path), scale=2)
        saved[key] = str(path)
        print(f"  ✓ {path}")

    return saved