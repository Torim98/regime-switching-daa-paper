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

def train_msm_fold(
    returns_train: pd.Series,
    returns_test: pd.Series,
    k_regimes: int,
    switching_variance: bool,
    threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Markov-Switching-Modell auf einem Walk-Forward-Fold trainieren.

    Logik:
    1. MarkovRegression auf returns_train fitten (Parameter-Schätzung).
    2. Neues Modell-Objekt auf der KOMBINIERTEN Serie (train + test) instanziieren.
    3. smooth(params) anwenden; wendet die TRAIN-Parameter auf den
       Gesamtbereich an, OHNE neuen Fit. Damit hat der Test-Bereich keinen
       Einfluss auf die Modellschätzung (kein Leakage).
    4. FILTERED marginal probabilities verwenden; diese nutzen pro Zeitpunkt t
       nur Information bis t (Forward-Pass), im Gegensatz zu smoothed, das
       auch zukünftige Beobachtungen einbezieht. Methodisch konsistent mit dem
       Walk-Forward-Argument der look-ahead-Vermeidung.
    5. Bear-Regime-Identifikation auf Basis der TRAIN-Parameter (sigma2-Vergleich).
    6. Test-Slice extrahieren, Threshold anwenden.

    Parameter
    ---------
    returns_train : pd.Series
        Returns-Serie für das Trainingsfenster (DatetimeIndex).
    returns_test : pd.Series
        Returns-Serie für das Testfenster (DatetimeIndex), zeitlich strikt
        nach returns_train.
    k_regimes, switching_variance, threshold :
        Identische Bedeutung wie in train_msm / predict_msm.

    Rückgabe
    --------
    tuple[pd.Series, pd.Series]
        (probs, signal), beide indexiert auf returns_test.index.
        probs : Bear-Wahrscheinlichkeit (filtered) für jeden Test-Tag.
        signal : Binäres Signal (0=Bull, 1=Bear) via Threshold.
    """
    # --- 1. Sanity-Checks ---
    if len(returns_train) == 0 or len(returns_test) == 0:
        raise ValueError("returns_train und returns_test dürfen nicht leer sein.")
    if returns_train.index.max() >= returns_test.index.min():
        raise ValueError(
            f"returns_train endet ({returns_train.index.max()}) nicht strikt vor "
            f"returns_test ({returns_test.index.min()}) — Look-Ahead-Verdacht!"
        )

    # --- 2. Auf Train-Bereich fitten ---
    ms_train = sm.tsa.MarkovRegression(
        returns_train,
        k_regimes=k_regimes,
        switching_variance=switching_variance,
    )
    ms_train_results = ms_train.fit()

    # --- 3. Bear-Regime-Identifikation aus TRAIN-Parametern ---
    # (NICHT aus der kombinierten Serie, sonst hätte der Test-Bereich
    # indirekten Einfluss auf das Label-Mapping.)
    if ms_train_results.params["sigma2[1]"] > ms_train_results.params["sigma2[0]"]:
        bear_state = 1
    else:
        bear_state = 0
    
    # --- 3b. Train-Signal für DL-Labels (LSTM/Transformer) ---
    # Filtered probs auf den TRAIN-Daten; nutzt nur Train-Information.
    # Wird vom Orchestrator als labels_col in df_train injiziert.
    filtered_bear_train = ms_train_results.filtered_marginal_probabilities[bear_state]
    signal_train = (filtered_bear_train.clip(0, 1) >= threshold).astype(int)
    signal_train.index = returns_train.index

    # --- 4. Train-Parameter auf kombinierten Bereich anwenden (ohne Re-Fit) ---
    returns_combined = pd.concat([returns_train, returns_test])
    ms_combined = sm.tsa.MarkovRegression(
        returns_combined,
        k_regimes=k_regimes,
        switching_variance=switching_variance,
    )
    combined_results = ms_combined.smooth(ms_train_results.params)

    # --- 5. FILTERED Probabilities verwenden (kein Look-Ahead innerhalb des Folds) ---
    filtered_bear = combined_results.filtered_marginal_probabilities[bear_state]

    # --- 6. Test-Slice extrahieren ---
    probs = filtered_bear.loc[returns_test.index].clip(0, 1)
    signal = (probs >= threshold).astype(int)

    return probs, signal, signal_train

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