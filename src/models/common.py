"""Gemeinsame Modell-Hilfsfunktionen und Konstanten."""

import numpy as np
import pandas as pd


# --- Regime-Konstanten ---
BEAR_REGIME = 1
BULL_REGIME = 0


def validate_regime_signal(
    data: pd.DataFrame,
    model_name: str,
    auto_invert: bool = True,
) -> None:
    """
    Standardized Sanity Check für Regime-Signale.
    Erwartet {model_name}_Prob und {model_name}_Signal in data.

    Prüft:
    - Regime-Statistik (Mittelwerte pro Regime)
    - Signal-Verteilung (Anzahl Bull/Bear)
    - Plausibilität: Bear-Regime darf NICHT höhere Returns haben als Bull
      (→ automatische Label-Invertierung wenn auto_invert=True)
    - Formale Validierung: Signal ∈ {0,1}, keine NaNs, Prob ∈ [0,1]
    """
    prob_col = f"{model_name}_Prob"
    signal_col = f"{model_name}_Signal"
    stats_cols = ["Returns", "VIX", "Yield_Spread", prob_col]

    # Regime Statistics
    print(f"\n{'='*60}")
    print(f"   {model_name} — Regime-Statistik")
    print(f"{'='*60}")
    available = [c for c in stats_cols if c in data.columns]
    print(data.groupby(signal_col)[available].mean())
    print(f"\nSignal-Verteilung:\n{data[signal_col].value_counts()}")

    # Plausibility Check
    mean_returns = data.groupby(signal_col)["Returns"].mean()
    if mean_returns.get(BEAR_REGIME, 0) > mean_returns.get(BULL_REGIME, 0):
        print(f"\n WARNUNG: {model_name} Bear-Regime ({BEAR_REGIME}) "
              f"hat höhere Returns als Bull ({BULL_REGIME})!")
        if auto_invert:
            print("    → Labels könnten vertauscht sein. Invertiere:")
            data[signal_col] = 1 - data[signal_col]
            data[prob_col] = 1 - data[prob_col]
            print("    → Labels automatisch invertiert.")
    else:
        print(f"\n{model_name} Plausibilitäts-Check bestanden.")

    # Validierung
    assert prob_col in data.columns, f"{prob_col} fehlt!"
    assert signal_col in data.columns, f"{signal_col} fehlt!"
    assert data[signal_col].isin([BULL_REGIME, BEAR_REGIME]).all(), "Signal enthält ungültige Werte!"
    assert data[signal_col].isna().sum() == 0, "NaN im Signal!"
    assert data[prob_col].between(0, 1).all(), "Prob außerhalb [0,1]!"
    print("Alle formalen Prüfungen bestanden.")


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rollendes Fenster für zeitreihenbasierte Modelle (LSTM/Transformer).
    Erzeugt Sequenzen der Länge `window` als Input (X)
    und das zugehörige Label (y) am Ende jeder Sequenz.
    """
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(target[i])
    return np.array(X), np.array(y)