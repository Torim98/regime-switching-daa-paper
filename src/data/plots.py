import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf


def plot_volatility_clusters(df, save_path: str):
    """Rendite-Verlauf + ACF der quadrierten Renditen."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(df.index, df['Returns_GSPC'], color='blue', alpha=0.6)
    axes[0].set_title('S&P 500 Tägliche Renditen (Visualisierung von Volatilitätsclustern)')
    axes[0].set_ylabel('Rendite')
    axes[0].grid(True, alpha=0.3)

    squared_returns = df['Returns_GSPC'] ** 2
    plot_acf(squared_returns.dropna(), lags=40, ax=axes[1], alpha=0.05, color='red')
    axes[1].set_title('Autokorrelation der quadrierten Renditen (S&P 500) - Nachweis von GARCH-Effekten')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_historical_drawdowns(df, save_path: str):
    """Historische Drawdowns des 60/40 Portfolios."""
    cum_returns = np.exp(df['Returns'].cumsum())
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(drawdowns.index, drawdowns, color='darkred')
    ax.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
    ax.set_title('Historische Drawdowns des 60/40 Portfolios (Benchmark)')
    ax.set_ylabel('Drawdown')
    ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_capital_curve(df, save_path: str):
    """Kapitalkurve des 60/40 Portfolios."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Cumulative_Returns'])
    ax.set_title("Kapitalkurve des 60/40 Portfolios (Start = 1.0)")
    ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_feature_correlation(df, feature_cols: list, save_path: str, md_path: str = None):
    """Pearson-Korrelationsmatrix als Heatmap."""
    import seaborn as sns

    corr_matrix = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, ax=ax,
    )
    ax.set_title("Pearson-Korrelationsmatrix der Modell-Features")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if md_path:
        corr_matrix.to_markdown(md_path)

    return corr_matrix