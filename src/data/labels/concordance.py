"""Concordance-Analyse und Timeline-Visualisierung fuer Label-Schemata."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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