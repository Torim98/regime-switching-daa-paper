"""Concordance analysis and timeline visualization for label schemes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def compute_concordance_matrix(labels: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Pairwise concordance (share of matching labels) of all methods.

    Parameters
    ----------
    labels : dict[str, pd.Series]
        Mapping method name -> 0/1 series. All series must have an identical index.
    """
    # Trim to the common index (inner intersection)
    common_index = None
    for s in labels.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)

    aligned = {k: v.reindex(common_index).dropna() for k, v in labels.items()}
    # After alignment, intersect again if necessary
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
    Pairwise Cohen's kappa matrix. κ ∈ [-1, 1], 1 = perfect agreement,
    0 = chance level. Chance-corrected → robust against unequal class distributions.
    """
    # Trim to the common index (as in compute_concordance_matrix)
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
    """Heatmap of the Cohen's kappa matrix (-0.2 to 1.0)."""
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
    ax.set_title("Cohen's κ: Label Concordance (Chance-Corrected)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_concordance_heatmap(matrix: pd.DataFrame, save_path: str) -> None:
    """Heatmap of the concordance matrix."""
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
    ax.set_title("Label Concordance (Share of Matching Days)")
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
    Horizontal bands per labeling method; bear phases shaded in red.
    Shared S&P 500 price line as reference on top.
    """
    n_methods = len(labels)
    fig, axes = plt.subplots(
        n_methods + 1, 1,
        figsize=(14, 1.0 * (n_methods + 1) + 2),
        sharex=True,
        gridspec_kw={"height_ratios": [3] + [1] * n_methods},
    )

    # Price panel
    axes[0].plot(prices.index, prices.values, color="black", linewidth=0.8)
    axes[0].set_title("S&P 500 with Regime Labels (red = Bear)")
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.2)

    # Label bands
    for ax, (name, series) in zip(axes[1:], labels.items()):
        ax.fill_between(series.index, 0, 1,
                        where=(series.values == 1),
                        color="red", alpha=0.5, step="post")
        ax.set_ylabel(name, rotation=0, labelpad=40, va="center")
        ax.set_yticks([])
        ax.set_ylim(0, 1)

    axes[-1].set_xlabel("Date")
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
    Compares MSM/HMM labels with price-based + macro alternatives.

    Writes the heatmap (% agreement), Cohen's kappa heatmap, and
    timeline PNG, and returns compact statistics per method
    (bear_share, n_switches, avg_phase_days) as well as the concordance
    AND kappa matrix.
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

    # Heatmap (share of matching days)
    concordance = compute_concordance_matrix(labels)
    plot_concordance_heatmap(concordance, concordance_path)

    # Cohen's kappa (chance-corrected)
    kappa = compute_kappa_matrix(labels)
    if kappa_path is not None:
        plot_kappa_heatmap(kappa, kappa_path)

    # Timeline (S&P 500 price line from raw)
    plot_prices = raw_df["^GSPC"].reindex(test_df.index).ffill()
    plot_label_timeline(labels, plot_prices, timeline_path)

    # Switch statistics
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
