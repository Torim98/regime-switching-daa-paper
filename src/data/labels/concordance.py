"""Concordance-Analyse und Timeline-Visualisierung fuer Label-Schemata."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def compute_concordance_matrix(labels: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Paarweise Konkordanz (Anteil uebereinstimmender Labels) aller Methoden.

    Parameter
    ---------
    labels : dict[str, pd.Series]
        Mapping Methoden-Name -> 0/1-Serie. Alle Serien muessen identischen Index haben.
    """
    # Auf gemeinsamen Index zuschneiden (innere Schnittmenge)
    common_index = None
    for s in labels.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)

    aligned = {k: v.reindex(common_index).dropna() for k, v in labels.items()}
    # Nach Alignment ggf. erneut schneiden
    common_index = sorted(set.intersection(*[set(s.index) for s in aligned.values()]))
    aligned = {k: v.loc[common_index] for k, v in aligned.items()}

    names = list(aligned.keys())
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            matrix.loc[a, b] = (aligned[a].values == aligned[b].values).mean()

    return matrix.astype(float)


def compute_kappa_matrix(labels: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Paarweise Cohen's-κ-Matrix. κ ∈ [-1, 1], 1 = perfekte Übereinstimmung,
    0 = Zufallsniveau. Chance-korrigiert → robust gegen ungleiche Klassenverteilungen.
    """
    # Auf gemeinsamen Index zuschneiden (wie compute_concordance_matrix)
    common_index = None
    for s in labels.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)

    aligned = {k: v.reindex(common_index).dropna() for k, v in labels.items()}
    common_index = sorted(set.intersection(*[set(s.index) for s in aligned.values()]))
    aligned = {k: v.loc[common_index].astype(int) for k, v in aligned.items()}

    names = list(aligned.keys())
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            matrix.loc[a, b] = cohen_kappa_score(aligned[a].values, aligned[b].values)
    return matrix.astype(float)


def plot_kappa_heatmap(matrix: pd.DataFrame, save_path: str) -> None:
    """Heatmap-Darstellung der Cohen's-κ-Matrix (-0.2 … 1.0)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if matrix.iloc[i, j] > 0.5 else "white")
    ax.set_title("Cohen's κ: Label-Konkordanz (chance-korrigiert)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_concordance_heatmap(matrix: pd.DataFrame, save_path: str) -> None:
    """Heatmap-Darstellung der Konkordanz-Matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if matrix.iloc[i, j] > 0.7 else "white")
    ax.set_title("Label-Konkordanz (Anteil uebereinstimmender Tage)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_label_timeline(
    labels: dict[str, pd.Series],
    prices: pd.Series,
    save_path: str,
) -> None:
    """
    Horizontale Baender pro Labeling-Methode; Bear-Phasen rot eingefaerbt.
    Gemeinsame S&P-500-Kurslinie als Referenz oberhalb.
    """
    n_methods = len(labels)
    fig, axes = plt.subplots(
        n_methods + 1, 1,
        figsize=(14, 1.0 * (n_methods + 1) + 2),
        sharex=True,
        gridspec_kw={"height_ratios": [3] + [1] * n_methods},
    )

    # Preis-Panel
    axes[0].plot(prices.index, prices.values, color="black", linewidth=0.8)
    axes[0].set_title("S&P 500 mit Regime-Labels (rot = Bear)")
    axes[0].set_ylabel("Preis")
    axes[0].grid(alpha=0.2)

    # Label-Baender
    for ax, (name, series) in zip(axes[1:], labels.items()):
        ax.fill_between(series.index, 0, 1,
                        where=(series.values == 1),
                        color="red", alpha=0.5, step="post")
        ax.set_ylabel(name, rotation=0, labelpad=40, va="center")
        ax.set_yticks([])
        ax.set_ylim(0, 1)

    axes[-1].set_xlabel("Datum")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
def run_label_analysis(
    test_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    concordance_path: str,
    timeline_path: str,
    kappa_path: str | None = None,
) -> dict:
    """
    Vergleicht MSM/HMM-Labels mit preisbasierten + makro Alternativen.

    Schreibt Heatmap (% Uebereinstimmung), Cohen's-kappa-Heatmap und
    Timeline-PNG und liefert eine kompakte Statistik pro Methode
    (bear_share, n_switches, avg_phase_days) sowie Konkordanz- UND
    Kappa-Matrix zurueck.
    """
    from src.data.labels import (
        label_pagan_sossounov,
        label_peak_to_trough,
        label_lunde_timmermann,
        load_nber_recession,
    )

    prices = test_df["Cumulative_Returns"]

    labels = {
        "MSM":     test_df["MSM_Signal"].astype("int8"),
        "HMM":     test_df["HMM_Signal"].astype("int8"),
        "PagSoss": label_pagan_sossounov(prices),
        "P2T":     label_peak_to_trough(prices, threshold=0.20),
        "LundeT":  label_lunde_timmermann(prices),
        "NBER":    load_nber_recession(test_df.index),
    }

    # Heatmap (Anteil uebereinstimmender Tage)
    concordance = compute_concordance_matrix(labels)
    plot_concordance_heatmap(concordance, concordance_path)

    # Cohen's kappa (chance-korrigiert)
    kappa = compute_kappa_matrix(labels)
    if kappa_path is not None:
        plot_kappa_heatmap(kappa, kappa_path)

    # Timeline (S&P-500-Kurslinie aus raw)
    plot_prices = raw_df["^GSPC"].reindex(test_df.index).ffill()
    plot_label_timeline(labels, plot_prices, timeline_path)

    # Switch-Statistik
    switch_stats = pd.DataFrame({
        name: {
            "bear_share_pct": float(s.mean() * 100),
            "n_switches": int((s.diff().abs() == 1).sum()),
            "avg_phase_days": float(
                len(s) / max((s.diff().abs() == 1).sum(), 1)
            ),
        }
        for name, s in labels.items()
    }).T

    return {
        "concordance": concordance.round(4).to_dict(),
        "kappa":       kappa.round(4).to_dict(),
        "switch_stats": switch_stats.round(2).to_dict(orient="index"),
    }