"""Hidden Markov model (HMM), hmmlearn with Gaussian emissions."""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler
from scipy.special import logsumexp
import joblib
from pathlib import Path

def _filtered_probs(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """P(state_t | x_1..t) via a pure forward pass; no backward pass,
    hence no look-ahead (counterpart to predict_proba = smoothed)."""
    framelogprob = model._compute_log_likelihood(X)          # (T, n_states)
    log_alpha = np.empty_like(framelogprob)
    log_alpha[0] = np.log(model.startprob_ + 1e-300) + framelogprob[0]
    log_trans = np.log(model.transmat_ + 1e-300)
    for t in range(1, len(X)):
        log_alpha[t] = framelogprob[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_trans, axis=0
        )
    return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))

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
    print(f"HMM: model saved at {model_file}")

    return model, scaler

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
    Train the hidden Markov model on a walk-forward fold.

    Logic:
    1. Fit the scaler ONLY on the train features.
    2. Fit the HMM ONLY on the scaled train features.
    3. Bear-state identification on the TRAIN predictions:
       which regime has the higher return volatility in the training period?
       This mapping is stored and applied to the test range.
       (CRITICAL: do NOT determine it based on test data, otherwise leakage!)
    4. Filtered probabilities P(state_t | x_1..t) via a forward pass on train+test;
       the train window serves as burn-in context. NO predict_proba (= smoothed,
       look-ahead within the fold).
    5. Extract the bear probability per the train mapping, apply the threshold.

    Parameters
    ----------
    features_df_train : pd.DataFrame
        Feature slice for training (e.g. df.loc[train_idx, hmm_features]).
    features_df_test : pd.DataFrame
        Feature slice for the test range, strictly after train in time.
    returns_train : pd.Series
        Returns for the training range, needed EXCLUSIVELY for
        bear-state identification on the train predictions.
    n_components, covariance_type, n_iter, random_state, threshold :
        Same meaning as in train_hmm / predict_hmm.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (probs, signal), both indexed on features_df_test.index.

    Notes
    -----
    - No cold start: the forward pass runs over train+test, so the test
      starts with the correct state prior.
    - If bear_state changes erratically between folds, this indicates
      unstable HMM convergence or training windows that are too short;
      log accordingly in the orchestrator.
    """
    # --- 1. Sanity checks ---
    if features_df_train.index.max() >= features_df_test.index.min():
        raise ValueError(
            f"features_df_train does not end ({features_df_train.index.max()}) strictly before "
            f"features_df_test ({features_df_test.index.min()}): possible look-ahead!"
        )
    if not features_df_train.index.equals(returns_train.index):
        raise ValueError(
            "features_df_train.index and returns_train.index must be identical."
        )

    # --- 2. Scaling: fit ONLY on train ---
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(features_df_train.values)
    X_test_scaled = scaler.transform(features_df_test.values)

    # --- 3. Fit the HMM on train ---
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X_train_scaled)

    # --- 4. Bear-state identification from the TRAIN predictions ---
    # Which regime has the higher return volatility in training?
    # This mapping is stored and applied to the test range.
    train_states = model.predict(X_train_scaled)
    state_0_vol = returns_train[train_states == 0].std()
    state_1_vol = returns_train[train_states == 1].std()

    # NaN protection: in case one state is never visited in training
    if pd.isna(state_0_vol) or pd.isna(state_1_vol):
        raise ValueError(
            f"HMM fold: one regime was never visited in training "
            f"(state_0_vol={state_0_vol}, state_1_vol={state_1_vol}). "
            f"Training window too short or too homogeneous?"
        )

    bear_state = 1 if state_1_vol > state_0_vol else 0

    # --- 5. FILTERED probs on test (train as context, no look-ahead) ---
    X_all = np.vstack([X_train_scaled, X_test_scaled])
    filtered = _filtered_probs(model, X_all)
    bear_probs = filtered[len(X_train_scaled):, bear_state]

    probs = pd.Series(bear_probs, index=features_df_test.index)
    signal = (probs >= threshold).astype(int)

    # Train signal for DL label injection (analogous to train_msm_fold)
    train_probs_raw = model.predict_proba(X_train_scaled)[:, bear_state]
    signal_train = pd.Series(
        (train_probs_raw >= threshold).astype(int),
        index=features_df_train.index,
    )

    return probs, signal, signal_train

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
    Predict regimes and probabilities bias-free.

    1. Determine the bear state from the TRAIN predictions (volatility comparison).
    2. Filtered probs (forward pass), test with train as context

    Returns (probs_train, signal_train, probs_test, signal_test).
    """
    X_train_scaled = scaler.transform(features_df_train.values)
    X_test_scaled = scaler.transform(features_df_test.values)

    # --- Bear state from the train predictions ---
    train_states = model.predict(X_train_scaled)
    state_0_vol = returns_train[train_states == 0].std()
    state_1_vol = returns_train[train_states == 1].std()
    bear_state = 1 if state_1_vol > state_0_vol else 0

    # --- Train probs (filtered, train information only) ---
    filtered_train = _filtered_probs(model, X_train_scaled)
    probs_train = pd.Series(filtered_train[:, bear_state], index=features_df_train.index)
    signal_train = (probs_train >= threshold).astype(int)

    # --- Test probs (filtered, train as burn-in context, no look-ahead) ---
    X_all = np.vstack([X_train_scaled, X_test_scaled])
    filtered = _filtered_probs(model, X_all)
    probs_test = pd.Series(filtered[len(X_train_scaled):, bear_state], index=features_df_test.index)
    signal_test = (probs_test >= threshold).astype(int)

    return probs_train, signal_train, probs_test, signal_test
