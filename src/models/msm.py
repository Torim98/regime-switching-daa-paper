"""Markov-switching model (MS univariate), statsmodels."""

import statsmodels.api as sm
import pandas as pd
import numpy as np
from pathlib import Path


def train_msm(
    returns_train: pd.Series,
    k_regimes: int,
    switching_variance: bool,
    model_file: str,
) -> object:
    ms_model = sm.tsa.MarkovRegression(
        returns_train,
        k_regimes=k_regimes,
        switching_variance=switching_variance,
    )
    ms_results = ms_model.fit()

    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    ms_results.save(model_file)
    print(f"MSM: model saved at {model_file}")

    return ms_results

def train_msm_fold(
    returns_train: pd.Series,
    returns_test: pd.Series,
    k_regimes: int,
    switching_variance: bool,
    threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Train the Markov-switching model on a walk-forward fold.

    Logic:
    1. Fit MarkovRegression on returns_train (parameter estimation).
    2. Instantiate a new model object on the COMBINED series (train + test).
    3. Apply smooth(params); applies the TRAIN parameters to the full
       range WITHOUT refitting. The test range therefore has no influence
       on the model estimation (no leakage).
    4. Use FILTERED marginal probabilities; at each time t these use only
       information up to t (forward pass), in contrast to smoothed, which
       also incorporates future observations. Methodologically consistent
       with the walk-forward argument of look-ahead prevention.
    5. Bear-regime identification based on the TRAIN parameters (sigma2 comparison).
    6. Extract the test slice, apply the threshold.

    Parameters
    ----------
    returns_train : pd.Series
        Return series for the training window (DatetimeIndex).
    returns_test : pd.Series
        Return series for the test window (DatetimeIndex), strictly after
        returns_train in time.
    k_regimes, switching_variance, threshold :
        Same meaning as in train_msm / predict_msm.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (probs, signal), both indexed on returns_test.index.
        probs : bear probability (filtered) for each test day.
        signal : binary signal (0=bull, 1=bear) via threshold.
    """
    # --- 1. Sanity checks ---
    if len(returns_train) == 0 or len(returns_test) == 0:
        raise ValueError("returns_train and returns_test must not be empty.")
    if returns_train.index.max() >= returns_test.index.min():
        raise ValueError(
            f"returns_train does not end ({returns_train.index.max()}) strictly before "
            f"returns_test ({returns_test.index.min()}): possible look-ahead!"
        )

    # --- 2. Fit on the train range ---
    ms_train = sm.tsa.MarkovRegression(
        returns_train,
        k_regimes=k_regimes,
        switching_variance=switching_variance,
    )
    ms_train_results = ms_train.fit()

    # --- 3. Bear-regime identification from the TRAIN parameters ---
    # (NOT from the combined series, otherwise the test range would have
    # an indirect influence on the label mapping.)
    if ms_train_results.params["sigma2[1]"] > ms_train_results.params["sigma2[0]"]:
        bear_state = 1
    else:
        bear_state = 0

    # --- 3b. Train signal for DL labels (LSTM/Transformer) ---
    # Filtered probs on the TRAIN data; uses train information only.
    # Injected by the orchestrator as labels_col into df_train.
    filtered_bear_train = ms_train_results.filtered_marginal_probabilities[bear_state]
    signal_train = (filtered_bear_train.clip(0, 1) >= threshold).astype(int)
    signal_train.index = returns_train.index

    # --- 4. Apply the train parameters to the combined range (no refit) ---
    returns_combined = pd.concat([returns_train, returns_test])
    ms_combined = sm.tsa.MarkovRegression(
        returns_combined,
        k_regimes=k_regimes,
        switching_variance=switching_variance,
    )
    combined_results = ms_combined.smooth(ms_train_results.params)

    # --- 5. Use FILTERED probabilities (no look-ahead within the fold) ---
    filtered_bear = combined_results.filtered_marginal_probabilities[bear_state]

    # --- 6. Extract the test slice ---
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
    Load a persisted MSM model (skip training).
    Checks whether the data still matches the model.
    If the index differs, the model is applied to the new data (smooth).
    """
    ms_results = sm.load(model_file)

    # Check whether the data still matches the model
    if returns is not None:
        stored_probs = ms_results.smoothed_marginal_probabilities[1]
        if not returns.index.equals(stored_probs.index):
            print("MSM: data has changed, applying the model to the new data...")
            ms_model = sm.tsa.MarkovRegression(
                returns,
                k_regimes=k_regimes,
                switching_variance=switching_variance,
            )
            ms_results = ms_model.smooth(ms_results.params)

    return ms_results


def predict_msm(
    ms_results: object,
    returns_train: pd.Series,
    returns_test: pd.Series,
    k_regimes: int,
    switching_variance: bool,
    threshold: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Derive regime probabilities and the binary signal bias-free.

    1. Identify the bear regime from the TRAIN parameters (sigma2 comparison).
    2. Train signal via filtered probs on the train data.
    3. smooth(train_params) on the combined series (no refit).
    4. Extract the filtered marginal probabilities for the test range.

    Returns (probs_train, signal_train, probs_test, signal_test).
    """
    # --- Bear regime from the train parameters ---
    if ms_results.params["sigma2[1]"] > ms_results.params["sigma2[0]"]:
        bear_state = 1
    else:
        bear_state = 0

    # --- Train probs (filtered, train information only) ---
    filtered_bear_train = ms_results.filtered_marginal_probabilities[bear_state]
    probs_train = filtered_bear_train.clip(0, 1)
    probs_train.index = returns_train.index
    signal_train = (probs_train >= threshold).astype(int)

    # --- Apply the train parameters to the combined range (no refit) ---
    returns_combined = pd.concat([returns_train, returns_test])
    ms_combined = sm.tsa.MarkovRegression(
        returns_combined,
        k_regimes=k_regimes,
        switching_variance=switching_variance,
    )
    combined_results = ms_combined.smooth(ms_results.params)

    # --- Filtered probabilities for the test range (no look-ahead) ---
    filtered_bear = combined_results.filtered_marginal_probabilities[bear_state]
    probs_test = filtered_bear.loc[returns_test.index].clip(0, 1)
    signal_test = (probs_test >= threshold).astype(int)

    return probs_train, signal_train, probs_test, signal_test
