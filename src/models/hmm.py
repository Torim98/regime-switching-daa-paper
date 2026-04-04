"""Hidden Markov Model (HMM) — hmmlearn mit Gaussian-Emissions."""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler
import joblib
from pathlib import Path


def train_hmm(
    features_df: pd.DataFrame,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    random_state: int,
    model_file: str,
    scaler_file: str,
) -> tuple[GaussianHMM, RobustScaler, np.ndarray]:
    """
    Hidden Markov Model trainieren.
    Typ: Unsupervised (Clustering).
    Identifiziert Regime-Cluster über Gaussian-Emissions in
    Returns, VIX und Yield_Spread ohne gelabelte Daten.

    Skalierung (Standardisierung auf Mittelwert 0 und Varianz 1) via RobustScaler.
    Modell + Scaler werden unter model_file / scaler_file persistiert.

    Gibt (model, scaler, skalierte_daten) zurück.
    """
    X = features_df.values

    # Skalierung (Standardisierung auf Mittelwert 0 und Varianz 1)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # HMM Modellierung
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X_scaled)

    # Modell + Scaler persistieren
    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"HMM: Modell gespeichert unter {model_file}")

    return model, scaler, X_scaled


def load_hmm(
    features_df: pd.DataFrame,
    model_file: str,
    scaler_file: str,
) -> tuple[GaussianHMM, RobustScaler, np.ndarray]:
    """
    Persistiertes HMM + Scaler laden (Training überspringen).
    Skalierung mit geladenem Scaler (transform, NICHT fit_transform!).

    Gibt (model, scaler, skalierte_daten) zurück.
    """
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    # Skalierung mit geladenem Scaler (transform, NICHT fit_transform!)
    X_scaled = scaler.transform(features_df.values)

    return model, scaler, X_scaled


def predict_hmm(
    model: GaussianHMM,
    X_scaled: np.ndarray,
    returns: pd.Series,
    threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Regimes und Wahrscheinlichkeiten vorhersagen.

    predict() liefert 0 oder 1.
    predict_proba() liefert die Wahrscheinlichkeit für beide Zustände [Prob_0, Prob_1].

    Label-Alignment: Bear (1) = Regime mit höherer Volatilität.
    Wir definieren Bear (1) als das Regime mit der höheren Volatilität der Renditen.
    Signal aus Prob via Threshold ableiten.

    Gibt (probabilities, signal) zurück.
    """
    # predict() liefert 0 oder 1
    # predict_proba() liefert die Wahrscheinlichkeit für beide Zustände [Prob_0, Prob_1]
    hmm_regimes_raw = model.predict(X_scaled)
    hmm_probs_raw = model.predict_proba(X_scaled)

    # Label-Alignment: Bear (1) = Regime mit höherer Volatilität
    # Wir definieren Bear (1) als das Regime mit der höheren Volatilität der Renditen
    state_0_vol = returns[hmm_regimes_raw == 0].std()
    state_1_vol = returns[hmm_regimes_raw == 1].std()

    # Wir wollen, dass Regime 1 immer "Bear" ist (höhere Vola)
    if state_1_vol > state_0_vol:
        # Fall: Modell-Zustand 1 ist bereits der Bear-Markt
        probs = pd.Series(hmm_probs_raw[:, 1], index=returns.index)
    else:
        # Fall: Modell-Zustand 0 war eigentlich der Bear-Markt → wir flippen alles
        probs = pd.Series(hmm_probs_raw[:, 0], index=returns.index)

    # Signal aus Prob via Threshold ableiten
    signal = (probs >= threshold).astype(int)

    return probs, signal