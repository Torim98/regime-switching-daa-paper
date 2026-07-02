"""Shared model helper functions and constants."""

import numpy as np
import pandas as pd


# --- Regime constants ---
BEAR_REGIME = 1
BULL_REGIME = 0


def validate_regime_signal(
    data: pd.DataFrame,
    model_name: str,
    auto_invert: bool = True,
) -> None:
    """
    Standardized sanity check for regime signals.
    Expects {model_name}_Prob and {model_name}_Signal in data.

    Checks:
    - Regime statistics (means per regime)
    - Signal distribution (bull/bear counts)
    - Plausibility: the bear regime must NOT have higher returns than bull
      (→ automatic label inversion if auto_invert=True)
    - Formal validation: signal ∈ {0,1}, no NaNs, prob ∈ [0,1]
    """
    prob_col = f"{model_name}_Prob"
    signal_col = f"{model_name}_Signal"
    stats_cols = ["Returns", "VIX", "Yield_Spread", prob_col]

    # Regime statistics
    print(f"\n{'='*60}")
    print(f"   {model_name}: regime statistics")
    print(f"{'='*60}")
    available = [c for c in stats_cols if c in data.columns]
    print(data.groupby(signal_col)[available].mean())
    print(f"\nSignal distribution:\n{data[signal_col].value_counts()}")

    # Plausibility check
    mean_returns = data.groupby(signal_col)["Returns"].mean()
    if mean_returns.get(BEAR_REGIME, 0) > mean_returns.get(BULL_REGIME, 0):
        print(f"\n WARNING: {model_name} bear regime ({BEAR_REGIME}) "
              f"has higher returns than bull ({BULL_REGIME})!")
        if auto_invert:
            print("    → Labels may be swapped. Inverting:")
            data[signal_col] = 1 - data[signal_col]
            data[prob_col] = 1 - data[prob_col]
            print("    → Labels inverted automatically.")
    else:
        print(f"\n{model_name} plausibility check passed.")

    # Validation
    assert prob_col in data.columns, f"{prob_col} missing!"
    assert signal_col in data.columns, f"{signal_col} missing!"
    assert data[signal_col].isin([BULL_REGIME, BEAR_REGIME]).all(), "Signal contains invalid values!"
    assert data[signal_col].isna().sum() == 0, "NaN in signal!"
    assert data[prob_col].between(0, 1).all(), "Prob outside [0,1]!"
    print("All formal checks passed.")


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling window for time-series-based models (LSTM/Transformer).
    Produces sequences of length `window` as input (X)
    and the corresponding label (y) at the end of each sequence.
    """
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(target[i])
    return np.array(X), np.array(y)
