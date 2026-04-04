"""Markov-Switching-Modell (MS Univariate) — statsmodels."""

import statsmodels.api as sm
import pandas as pd
import numpy as np
from pathlib import Path


def train_msm(
    returns: pd.Series,
    k_regimes: int,
    switching_variance: bool,
    model_file: str,
) -> object:
    """
    Markov-Switching-Regression trainieren.
    Typ: Ökonometrie (Regression).
    Zustandsabhängiges Regressionsmodell mit switching variance.
    MS Univariate: Nur Returns.

    Modell wird nach dem Training unter model_file persistiert.
    Gibt das gefittete Results-Objekt zurück.
    """
    # Nur Returns zur Bestimmung von Mittelwert und Varianz
    ms_model = sm.tsa.MarkovRegression(
        returns,
        k_regimes=k_regimes,
        switching_variance=switching_variance,
    )
    ms_results = ms_model.fit()

    # Modell persistieren
    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    ms_results.save(model_file)
    print(f"MSM: Modell gespeichert unter {model_file}")

    return ms_results


def load_msm(
    model_file: str,
    returns: pd.Series = None,
    k_regimes: int = None,
    switching_variance: bool = None,
) -> object:
    """
    Persistiertes MSM-Modell laden (Training überspringen).
    Prüft ob die Daten noch zum Modell passen.
    Falls der Index abweicht, wird das Modell auf die neuen Daten angewendet (smooth).
    """
    ms_results = sm.load(model_file)

    # Prüfe ob die Daten noch zum Modell passen
    if returns is not None:
        stored_probs = ms_results.smoothed_marginal_probabilities[1]
        if not returns.index.equals(stored_probs.index):
            print("MSM: Daten haben sich geändert, wende Modell auf neue Daten an...")
            ms_model = sm.tsa.MarkovRegression(
                returns,
                k_regimes=k_regimes,
                switching_variance=switching_variance,
            )
            ms_results = ms_model.smooth(ms_results.params)

    return ms_results


def predict_msm(
    ms_results: object,
    threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Regime-Wahrscheinlichkeiten und binäres Signal ableiten.

    Label-Alignment: Bear (1) = Regime mit höherer Varianz.
    Identifikation des Bärenmarktes über den Vergleich der geschätzten Varianzen.
    Signal aus Prob via Threshold ableiten.

    Gibt (probabilities, signal) zurück.
    """
    prob_regime_1 = ms_results.smoothed_marginal_probabilities[1]

    # Bear (1) = Regime mit höherer Varianz
    if ms_results.params["sigma2[1]"] > ms_results.params["sigma2[0]"]:
        probs = prob_regime_1.clip(0, 1)
    else:
        probs = (1 - prob_regime_1).clip(0, 1)

    # Signal aus Prob via Threshold ableiten
    signal = (probs >= threshold).astype(int)

    return probs, signal