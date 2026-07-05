"""LSTM network: supervised regime classification (TensorFlow/Keras)."""

import numpy as np
import pandas as pd
import math
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import joblib
from tensorflow.keras.callbacks import EarlyStopping

from .common import create_sequences

def weighted_bce(pos_weight: float):
    """
    Binary cross-entropy with positive class weighting.
    Corresponds to the pos_weight mechanism of torch.nn.BCEWithLogitsLoss
    and ensures that LSTM and Transformer use the same loss function
    (incl. the identical weighting formula sqrt(n_neg/n_pos)).
    """
    pw = tf.constant(pos_weight, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = -(pw * y_true * tf.math.log(y_pred)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce)

    return loss

def build_lstm(
    window_size: int,
    n_features: int,
    units_l1: int,
    units_l2: int,
    return_sequences: bool,
    dropout: float,
    dense: int,
    activation: str,
    optimizer: str,
    loss,
    metrics: str,
) -> Sequential:
    """
    Build the LSTM architecture per the config.
    input_shape adapts automatically to the number of features.
    Binary classification (Dense 1, sigmoid).
    """
    model = Sequential([
        LSTM(units_l1,
             return_sequences=return_sequences,
             input_shape=(window_size, n_features)),
        Dropout(dropout),
        LSTM(units_l2),
        Dropout(dropout),
        # dtype="float32" forces the output layer into FP32, even if
        # Keras is globally set to mixed_float16; numerically stable for BCE.
        Dense(dense, activation=activation, dtype="float32"),
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model


def train_lstm(
    df: pd.DataFrame,
    features: list[str],
    labels_col: str,
    window_size: int,
    train_test_split: float,
    units_l1: int,
    units_l2: int,
    return_sequences: bool,
    dropout: float,
    dense: int,
    activation: str,
    optimizer: str,
    metrics: str,
    epochs: int,
    batch_size: int,
    validation_split: float,
    verbose: int,
    model_file: str,
    scaler_file: str,
) -> tuple[Sequential, RobustScaler, np.ndarray, int]:
    """
    Train the LSTM network.
    Type: supervised (labels from MS_Univariate).
    LSTM network with a rolling window for time-series-based regime classification.

    Scaling: fit ONLY on the training data (avoid data leakage).
    Bear/bull weighting via pos_weight = sqrt(n_neg/n_pos) in the weighted BCE.
    Model + scaler are persisted.

    Returns (model, scaler, test_probs, split_index).
    """
    n_features = len(features)

    # Scaling: fit ONLY on the training data (avoid data leakage)
    split_point = int(len(df) * train_test_split)
    scaler = RobustScaler()
    scaler.fit(df[features].iloc[:split_point])          # fit on train only
    scaled_data = scaler.transform(df[features])          # transform on everything

    # Labels and sequences
    # Choice of the appropriate labels (in the config file)
    X, y = create_sequences(scaled_data, df[labels_col].values, window_size)

    # Split (train/test): 80% training, 20% test
    split = int(len(X) * train_test_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Determine pos_weight identically to the Transformer
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    raw_weight = n_neg / n_pos
    pos_weight = math.sqrt(raw_weight)
    print(f"Class balance: bull: {n_neg}, bear: {n_pos}, "
          f"raw_weight: {raw_weight:.2f}, pos_weight (sqrt): {pos_weight:.2f}")
    # Expected: raw_weight: 3.31, pos_weight (sqrt): ~1.82

    # LSTM architecture
    model = build_lstm(
        window_size=window_size,
        n_features=n_features,
        units_l1=units_l1,
        units_l2=units_l2,
        return_sequences=return_sequences,
        dropout=dropout,
        dense=dense,
        activation=activation,
        optimizer=optimizer,
        loss=weighted_bce(pos_weight),
        metrics=metrics,
    )

    # Training
    print("Starting LSTM training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )

    # Generate predictions
    lstm_probs_raw = model.predict(X_test)

    # Persist model + scaler
    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_file)
    joblib.dump(scaler, scaler_file)
    print(f"LSTM: model saved at {model_file}")
    print(f"Final test accuracy: {history.history['val_accuracy'][-1]:.2%}")

    return model, scaler, lstm_probs_raw, split

def train_lstm_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    labels_col: str,
    window_size: int,
    units_l1: int,
    units_l2: int,
    return_sequences: bool,
    dropout: float,
    dense: int,
    activation: str,
    optimizer: str,
    metrics: str,
    epochs: int,
    batch_size: int,
    validation_split: float,
    verbose: int,
    init_weights: list | None = None,
    epochs_warm: int | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex, list | None]:
    """
    Train the LSTM network on a walk-forward fold.

    In contrast to train_lstm:
    - Receives train and test slices explicitly (no internal ratio split).
    - Fits the scaler exclusively on df_train (no leakage).
    - Persists NOTHING (each fold produces its own model).
    - Returns the OOS probabilities together with the corresponding
      DatetimeIndex, so that the walk-forward orchestrator (run_walk_forward)
      can insert the predictions into a continuous series via index alignment.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training slice (DatetimeIndex). Must contain at least window_size+1
        rows, otherwise not a single training sequence can be formed.
    df_test : pd.DataFrame
        Test slice (DatetimeIndex). Must lie strictly after df_train in time.
        A warm-up buffer of the last window_size train rows is prepended, so
        every test day gets a prediction; the prediction index spans the full
        df_test.index. df_test may therefore be shorter than window_size (the
        buffer supplies the missing history), it only has to be non-empty.
    features, labels_col, window_size, ... :
        Same meaning as in train_lstm. Passed through 1:1 from cfg.models.lstm.

    Returns
    -------
    tuple[np.ndarray, pd.DatetimeIndex, list | None]
        (probs_raw, prediction_index, final_weights)
        probs_raw : 1D array of the raw probabilities (sigmoid output) on the
                    OOS test range.
        prediction_index : DatetimeIndex, exactly len(probs_raw) entries,
                           aligned to df_test.index[window_size:].
        final_weights : list of the Keras weights (model.get_weights()) at
                        the end of training; serves as the warm-start basis
                        for the following fold. None in the single-class fallback.

    Notes
    -----
    - Sequence boundary: the test sequences use a warm-up buffer of the last
      window_size rows of df_train as history, so the first test day can already
      be predicted (predictions cover the full df_test range). Those buffer rows
      only supply input features; they are never trained on and carry train
      labels only, so no OOS label leaks into training.
    - pos_weight is computed exclusively from the train labels.
    - validation_split acts as in train_lstm: the last X% of the generated
      X_train tensor serve as Keras-internal validation. Since X_train is
      chronologically ordered, this internal split is also look-ahead-free.
    """
    n_features = len(features)

    # --- 1. Sanity checks ---
    if len(df_train) <= window_size:
        raise ValueError(
            f"df_train has only {len(df_train)} rows, requires > window_size={window_size}."
        )
    if len(df_test) < 1:
        raise ValueError(
            f"df_test is empty ({len(df_test)} rows); at least 1 test row is required."
        )
    if df_train.index.max() >= df_test.index.min():
        raise ValueError(
            f"df_train does not end ({df_train.index.max()}) strictly before df_test "
            f"({df_test.index.min()}): possible look-ahead!"
        )

    # --- 2. Scaling: fit ONLY on the training data ---
    scaler = RobustScaler()
    scaler.fit(df_train[features])
    train_scaled = scaler.transform(df_train[features])
    test_scaled = scaler.transform(df_test[features])

    # --- 3. Create sequences, with a warm-up buffer for the test range ---
    # Train sequences from df_train only.
    X_train, y_train = create_sequences(
        train_scaled, df_train[labels_col].values, window_size,
    )

    # Test sequences WITH warm-up: prepend the last window_size rows of df_train
    # as "history", so that the first test sequence can predict on the first
    # test day (instead of only window_size days later).
    # IMPORTANT: these buffer rows are NOT used for training;
    # they only provide the input features for the test sequences.
    buffer_scaled = train_scaled[-window_size:]
    test_scaled_with_buffer = np.concatenate([buffer_scaled, test_scaled], axis=0)

    buffer_labels = df_train[labels_col].values[-window_size:]
    test_labels_with_buffer = np.concatenate(
        [buffer_labels, df_test[labels_col].values], axis=0,
    )

    X_test, _ = create_sequences(
        test_scaled_with_buffer, test_labels_with_buffer, window_size,
    )

    # prediction_index: now the ENTIRE df_test.index (no longer [window_size:]),
    # because the first test sequence can already predict on the first test day
    # thanks to the buffer.
    prediction_index = df_test.index
    assert len(prediction_index) == len(X_test), (
        f"Index mismatch after warm-up buffer: prediction_index={len(prediction_index)}, "
        f"X_test={len(X_test)}"
    )

    # --- 4. Class weighting (train labels only!) ---
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    if n_pos == 0 or n_neg == 0:
        import warnings
        majority_prob = 1.0 if n_pos > n_neg else 0.0
        warnings.warn(
            f"  [LSTM] Train fold is single-class (n_neg={n_neg}, n_pos={n_pos}). "
            f"Fallback: constant prediction P(Bear)={majority_prob}."
        )
        # With the warm-up buffer we predict the ENTIRE test range,
        # so the fallback must return the same pred_idx.
        pred_idx = df_test.index
        probs = np.full(len(pred_idx), majority_prob, dtype=np.float32)
        # Warm start: None; in the single-class case no real model was trained.
        return probs, pred_idx, None

    raw_weight = n_neg / n_pos
    pos_weight = math.sqrt(raw_weight)
    if verbose:
        print(
            f"  [LSTM Fold] Train: {len(df_train)} rows, Test: {len(df_test)} rows | "
            f"Bull: {n_neg}, Bear: {n_pos}, pos_weight (sqrt): {pos_weight:.2f}"
        )

    # --- 5. Build the architecture ---
    model = build_lstm(
        window_size=window_size,
        n_features=n_features,
        units_l1=units_l1,
        units_l2=units_l2,
        return_sequences=return_sequences,
        dropout=dropout,
        dense=dense,
        activation=activation,
        optimizer=optimizer,
        loss=weighted_bce(pos_weight),
        metrics=metrics,
    )

    # --- 5a. Warm start from the previous fold (if available) ---
    # 90% train overlap between rolling folds -> the initial weights are already
    # "preconditioned". We then only fine-tune for a few epochs.
    # An architecture mismatch (e.g. changed feature count) raises a
    # ValueError -> fall back to a cold start (discard init_weights).
    effective_epochs = epochs
    if init_weights is not None:
        try:
            model.set_weights(init_weights)
            if epochs_warm is not None and epochs_warm > 0:
                effective_epochs = epochs_warm
            if verbose:
                print(f"  [LSTM Fold] Warm start active, epochs={effective_epochs}")
        except ValueError as e:
            if verbose:
                print(f"  [LSTM Fold] Warm start discarded (architecture mismatch): {e}")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=0,
    )

    # --- 6. Training ---
    model.fit(
        X_train, y_train,
        epochs=effective_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=verbose,
    )

    # --- 7. OOS predictions on the test sequences ---
    probs_raw = model.predict(X_test, verbose=0).flatten()

    # --- 8. Weights for the warm start of the next fold ---
    final_weights = model.get_weights()

    return probs_raw, prediction_index, final_weights

def load_lstm_model(
    df: pd.DataFrame,
    features: list[str],
    labels_col: str,
    window_size: int,
    train_test_split: float,
    model_file: str,
    scaler_file: str,
) -> tuple[Sequential, RobustScaler, np.ndarray, int]:
    """
    Load the persisted LSTM model + scaler (skip training).
    Scaling with the loaded scaler (transform, NOT fit_transform!).

    Returns (model, scaler, test_probs, split_index).
    """
    print(f"LSTM: loading persisted model from {model_file}")
    # compile=False: the loaded model is used for inference only,
    # so Keras does not need to deserialize the custom weighted_bce loss.
    # If incremental training is desired later, pass
    # custom_objects={"loss": weighted_bce(pos_weight)} instead or
    # recompile the model after loading.
    model = load_model(model_file, compile=False)
    scaler = joblib.load(scaler_file)

    # Scaling with the loaded scaler (transform, NOT fit_transform!)
    scaled_data = scaler.transform(df[features])

    # Labels and sequences
    X, y = create_sequences(scaled_data, df[labels_col].values, window_size)

    # Split (train/test): 80% training, 20% test
    split = int(len(X) * train_test_split)
    X_test = X[split:]

    # Generate predictions
    lstm_probs_raw = model.predict(X_test)

    return model, scaler, lstm_probs_raw, split


def predict_lstm(
    probs_raw: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive probabilities and the binary signal.
    Generate signals via threshold.

    Returns (probabilities, signal).
    """
    probs = probs_raw.flatten()
    signal = (probs >= threshold).astype(int)
    return probs, signal
