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
    """MCS Boxplots pro Szenario. save_path_template: z.B. 'assets/mcs_boxplot_{}.png'"""
    for sc_name, params in scenarios.items():
        mc_results_scenario = {}
        for strategy in daily_rets_columns:
            prefix = f"{sc_name}_{strategy}_path_"
            strat_paths = {k: v for k, v in mcs_paths_collector.items() if k.startswith(prefix)}
            if strat_paths:
                mc_results_scenario[strategy] = [path[-1] for path in strat_paths.values()]

        if mc_results_scenario:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.boxplot(mc_results_scenario.values(),
                       tick_labels=[s.replace('_', ' ') for s in mc_results_scenario.keys()])
            ax.set_title(f"MCS {sc_name}: Verteilung des Endkapitals (Start: {params['start']:,.0f}€)")
            ax.set_ylabel(f"Endkapital nach {sim_years} Jahren in €")
            ax.axhline(y=0, color='red', linestyle='--')
            ax.grid(axis='y', alpha=0.3)

            save_path = save_path_template.format(sc_name.lower())
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)


def plot_mcs_paths(mcs_results_df, scenarios_list: list, strategies,
                   color_map: dict, save_path: str):
    """MCS Pfad-Verläufe für alle Szenarien."""
    fig, axes = plt.subplots(len(scenarios_list), 1,
                             figsize=(15, 6 * len(scenarios_list)), sharex=True)
    if len(scenarios_list) == 1:
        axes = [axes]

    for ax, sc_name in zip(axes, scenarios_list):
        for strat in strategies:
            prefix = f"{sc_name}_{strat}_path_"
            strat_paths = mcs_results_df.filter(like=prefix)
            color = color_map.get(strat, 'black')
            ax.plot(strat_paths, color=color, alpha=0.05, linewidth=1)
            ax.plot([], [], color=color, label=strat.replace('_', ' '))

        ax.set_title(f"MCS Pfad-Verläufe: Szenario {sc_name}")
        ax.set_ylabel("Kapital in €")
        ax.axhline(y=0, color='black', linewidth=1.5)
        ax.grid(alpha=0.2)
        ax.legend(loc='upper left', ncol=2)

    plt.xlabel("Handelstage (T+N)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_mcs_quantiles(mcs_results_df, scenarios_list: list, strategies,
                       total_days: int, color_map: dict, save_path: str):
    """MCS Konfidenz-Intervalle (5%-95%) für alle Szenarien."""
    fig, axes = plt.subplots(len(scenarios_list), 1,
                             figsize=(15, 6 * len(scenarios_list)), sharex=True)
    if len(scenarios_list) == 1:
        axes = [axes]

    for ax, sc_name in zip(axes, scenarios_list):
        for strat in strategies:
            prefix = f"{sc_name}_{strat}_path_"
            strat_paths = mcs_results_df.filter(like=prefix)

            upper_95 = strat_paths.quantile(0.95, axis=1)
            lower_05 = strat_paths.quantile(0.05, axis=1)
            median_50 = strat_paths.median(axis=1)

            color = color_map.get(strat, 'black')
            ax.fill_between(range(total_days), lower_05, upper_95, color=color, alpha=0.15)
            ax.plot(median_50, color=color, linewidth=1.5,
                    label=f"{strat.replace('_', ' ')} (Median)")

        ax.set_title(f"MCS Konfidenz-Intervalle (5% - 95%): Szenario {sc_name}")
        ax.set_ylabel("Kapital in €")
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label="Erschöpfungsgrenze")
        ax.grid(alpha=0.2)
        ax.legend(loc='upper left', ncol=2)

    plt.xlabel("Handelstage (T+N)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)