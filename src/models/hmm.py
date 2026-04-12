"""Hidden Markov Model (HMM) — hmmlearn mit Gaussian-Emissions."""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler
import joblib
from pathlib import Path


def train_hmm(
    features_df_train: pd.DataFrame,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    random_state: int,
    model_file: str,
    scaler_file: str,
) -> tuple[GaussianHMM, RobustScaler]:
    X_train = features_df_train.values

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X_train_scaled)

    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"HMM: Modell gespeichert unter {model_file}")

    return model, scaler

    return model, scaler, X_scaled

def train_hmm_fold(
    features_df_train: pd.DataFrame,
    features_df_test: pd.DataFrame,
    returns_train: pd.Series,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    random_state: int,
    threshold: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Hidden Markov Model auf einem Walk-Forward-Fold trainieren.

    Logik:
    1. Scaler NUR auf Train-Features fitten.
    2. HMM NUR auf skalierten Train-Features fitten.
    3. Bear-State-Identifikation auf TRAIN-Predictions:
       Welches Regime hat im Trainingszeitraum die höhere Returns-Volatilität?
       Diese Zuordnung wird gespeichert und auf Test angewendet.
       (KRITISCH: NICHT auf Test-Daten basierend bestimmen, sonst Leakage!)
    4. predict_proba auf skalierten Test-Features.
    5. Bear-Wahrscheinlichkeit gemäß Train-Mapping extrahieren, Threshold anwenden.

    Parameter
    ---------
    features_df_train : pd.DataFrame
        Feature-Slice für Training (z.B. df.loc[train_idx, hmm_features]).
    features_df_test : pd.DataFrame
        Feature-Slice für Test, zeitlich strikt nach Train.
    returns_train : pd.Series
        Returns für den Trainingsbereich, wird AUSSCHLIESSLICH zur
        Bear-State-Identifikation auf Train-Predictions benötigt.
    n_components, covariance_type, n_iter, random_state, threshold :
        Identische Bedeutung wie in train_hmm / predict_hmm.

    Rückgabe
    --------
    tuple[pd.Series, pd.Series]
        (probs, signal), beide indexiert auf features_df_test.index.

    Hinweise
    --------
    - Cold-Start im Test: hmmlearn's predict_proba startet im Test ohne
      Kenntnis der Train-Endzustände. Bei kurzen Test-Fenstern (z.B. 6 Monate)
      kann das die ersten ~10 Tage leicht verzerren. Methodisch akzeptabel,
      sollte aber in 3.5.5 als Limitation erwähnt werden.
    - Falls sich bear_state zwischen Folds wild ändert, deutet das auf
      instabile HMM-Konvergenz oder zu kurze Train-Fenster hin; im
      Orchestrator entsprechend loggen.
    """
    # --- 1. Sanity-Checks ---
    if features_df_train.index.max() >= features_df_test.index.min():
        raise ValueError(
            f"features_df_train endet ({features_df_train.index.max()}) nicht strikt vor "
            f"features_df_test ({features_df_test.index.min()}) — Look-Ahead-Verdacht!"
        )
    if not features_df_train.index.equals(returns_train.index):
        raise ValueError(
            "features_df_train.index und returns_train.index müssen identisch sein."
        )

    # --- 2. Skalierung — fit NUR auf Train ---
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(features_df_train.values)
    X_test_scaled = scaler.transform(features_df_test.values)

    # --- 3. HMM auf Train fitten ---
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X_train_scaled)

    # --- 4. Bear-State-Identifikation aus TRAIN-Predictions ---
    # Welches Regime hat im Training die höhere Volatilität der Returns?
    # Diese Zuordnung wird gemerkt und auf Test angewendet.
    train_states = model.predict(X_train_scaled)
    state_0_vol = returns_train[train_states == 0].std()
    state_1_vol = returns_train[train_states == 1].std()

    # NaN-Schutz: falls ein State im Train gar nicht auftritt
    if pd.isna(state_0_vol) or pd.isna(state_1_vol):
        raise ValueError(
            f"HMM-Fold: Ein Regime wurde im Train nie besucht "
            f"(state_0_vol={state_0_vol}, state_1_vol={state_1_vol}). "
            f"Train-Fenster zu kurz oder zu homogen?"
        )

    bear_state = 1 if state_1_vol > state_0_vol else 0

    # --- 5. predict_proba auf Test ---
    test_probs_raw = model.predict_proba(X_test_scaled)
    bear_probs = test_probs_raw[:, bear_state]

    probs = pd.Series(bear_probs, index=features_df_test.index)
    signal = (probs >= threshold).astype(int)

    return probs, signal

def load_hmm(
    model_file: str,
    scaler_file: str,
) -> tuple[GaussianHMM, RobustScaler]:
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return model, scaler

def predict_hmm(
    model: GaussianHMM,
    scaler: RobustScaler,
    features_df_train: pd.DataFrame,
    features_df_test: pd.DataFrame,
    returns_train: pd.Series,
    threshold: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Regimes und Wahrscheinlichkeiten bias-frei vorhersagen.

    1. Bear-State aus TRAIN-Predictions bestimmen (Volatilitätsvergleich).
    2. predict_proba auf Train und Test separat.

    Gibt (probs_train, signal_train, probs_test, signal_test) zurück.
    """
    X_train_scaled = scaler.transform(features_df_train.values)
    X_test_scaled = scaler.transform(features_df_test.values)

    # --- Bear-State aus Train-Predictions ---
    train_states = model.predict(X_train_scaled)
    state_0_vol = returns_train[train_states == 0].std()
    state_1_vol = returns_train[train_states == 1].std()
    bear_state = 1 if state_1_vol > state_0_vol else 0

    # --- Train-Probs ---
    train_probs_raw = model.predict_proba(X_train_scaled)
    probs_train = pd.Series(train_probs_raw[:, bear_state], index=features_df_train.index)
    signal_train = (probs_train >= threshold).astype(int)

    # --- Test-Probs ---
    test_probs_raw = model.predict_proba(X_test_scaled)
    probs_test = pd.Series(test_probs_raw[:, bear_state], index=features_df_test.index)
    signal_test = (probs_test >= threshold).astype(int)

    return probs_train, signal_train, probs_test, signal_test