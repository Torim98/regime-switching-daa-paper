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
    Visualisierung des Walk-Forward-Schemas als horizontaler Gantt-artiger Plot.

    Pro Fold wird ein blauer Balken für das Trainingsfenster und ein orangefarbener
    Balken für das OOS-Testfenster gezeichnet. Fold-IDs wachsen nach unten
    (Fold 1 oben), wodurch die rollierende Verschiebung des Train/Test-Fensters
    über die Zeit unmittelbar ablesbar wird.

    Parameter
    ---------
    splits_summary : pd.DataFrame
        Ausgabe von src.backtest.walk_forward.summarize_splits:
        Index = fold-ID, Spalten = train_start, train_end, test_start, test_end,
        n_train, n_test.
    save_path : str
        Zielpfad für die PNG-Datei (DPI=300).
    mode : str
        "rolling" oder "expanding" — fließt nur in den Titel ein.
    train_window_years : int | None
        Länge des Trainingsfensters (Jahre) für den Titel.
    test_window_months : int | None
        Länge des Testfensters (Monate) für den Titel.
    train_color, test_color : str
        Farben der Train/Test-Balken. Defaults konsistent mit den übrigen
        Pipeline-Plots.
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

    ax.invert_yaxis()  # Fold 1 oben
    ax.set_xlabel("Datum")
    ax.set_ylabel("Fold")

    subtitle_parts = [f"Modus: {mode}"]
    if train_window_years is not None:
        subtitle_parts.append(f"Train: {train_window_years}J")
    if test_window_months is not None:
        subtitle_parts.append(f"Test: {test_window_months}M")
    subtitle_parts.append(f"{n_folds} Folds")
    ax.set_title(
        "Walk-Forward-Schema — Train/Test-Fenster über die Zeit\n"
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
    """Equity Curves aller Strategien in €-Darstellung (unnormiert)."""
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(backtesting_results['Buy_Hold'] * initial_capital,
            label='Statisches 60/40 Portfolio (Benchmark)',
            color=color_map.get('Buy_Hold', 'gray'), alpha=0.5, linestyle='--')

    for col in backtesting_results.columns:
        if col == 'Buy_Hold':
            continue
        color = color_map.get(col, None)
        ax.plot(backtesting_results[col] * initial_capital,
                label=f'Strategie: {col.replace("_", " ")}',
                color=color, linewidth=1.5, alpha=0.8)

    ax.set_title(
        f"Equity Curves: Dynamischer Vergleich der Regime-Switching-Modelle "
        f"(Startkapital {initial_capital:,.0f} €)"
    )
    ax.set_xlabel("Datum")
    ax.set_ylabel("Kapital (€)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
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
                   trading_days_per_year: int = 252,
                   start_year: int | None = None):
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

        # Jahres-Ticks: Kalenderjahre ab start_year (Default: aktuelles Jahr)
        if start_year is None:
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
                       trading_days_per_year: int = 252,
                       start_year: int | None = None):
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

        # Jahres-Ticks: Kalenderjahre ab start_year (Default: aktuelles Jahr)
        if start_year is None:
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
    """Optuna-Visualisierungen (History, Importance, Contour, Slice) als PNG speichern.

    Der Contour-Plot wird ausschließlich für Modelle mit ≥ 2 Hyperparametern
    erzeugt — bei einem einzigen Parameter (z.B. MSM mit nur `threshold`)
    degeneriert die Matrix und der Plot ist inhaltslos.
    """
    from pathlib import Path
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_contour,
        plot_slice,
    )

    # Anzahl Hyperparameter im Search-Space (für Contour-Entscheidung + Größe)
    n_params = max(
        (len(t.params) for t in study.trials if t.params), default=1
    )

    plots = {
        "optuna_history": plot_optimization_history(study),
        "optuna_importance": plot_param_importances(study),
        "optuna_slice": plot_slice(study),
    }
    if n_params >= 2:
        plots["optuna_contour"] = plot_contour(study)
    else:
        print(
            f"  ℹ {model_name}: Contour-Plot übersprungen "
            f"(nur {n_params} Hyperparameter im Search-Space)."
        )

    # Quell-Pixelmaße je Plot-Typ (vor scale=2)
    # contour ist eine n×n-Matrix → Größe muss mit n_params skalieren,
    # sonst überlappen Tick-Werte mit den Achsen-Titeln der Nachbarzelle.
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
        # Kleinere Tick-Fonts + mehr Achsen-Abstand verhindern Overlap in Matrix-Plots
        fig.update_xaxes(tickfont=dict(size=9), title_standoff=20, automargin=True)
        fig.update_yaxes(tickfont=dict(size=9), title_standoff=20, automargin=True)

        raw_template = getattr(cfg.paths.assets, key)
        filename = raw_template.replace("{model}", model_name)
        path = Path(cfg.asset_path(key).replace(raw_template, filename))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path), scale=2)
        saved[key] = str(path)
        print(f"  ✓ {path}")

    return saved