"""LSTM-Netzwerk — Supervised Regime Classification (TensorFlow/Keras)."""

import numpy as np
import pandas as pd
import math
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import RobustScaler
import joblib

from .common import create_sequences


def build_lstm(
    window_size: int,
    n_features: int,
    units: int,
    return_sequences: bool,
    dropout: float,
    dense: int,
    activation: str,
    optimizer: str,
    loss: str,
    metrics: str,
) -> Sequential:
    """
    LSTM-Architektur gemäß Config aufbauen.
    input_shape passt sich automatisch an die Anzahl der Features an.
    Binäre Klassifikation (Dense 1, Sigmoid).
    """
    model = Sequential([
        LSTM(window_size, return_sequences=return_sequences, input_shape=(window_size, n_features)),
        Dropout(dropout),
        LSTM(units),
        Dropout(dropout),
        Dense(dense, activation=activation),
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model


def train_lstm(
    df: pd.DataFrame,
    features: list[str],
    labels_col: str,
    window_size: int,
    train_test_split: float,
    units: int,
    return_sequences: bool,
    dropout: float,
    dense: int,
    activation: str,
    optimizer: str,
    loss: str,
    metrics: str,
    epochs: int,
    batch_size: int,
    validation_split: float,
    verbose: int,
    model_file: str,
    scaler_file: str,
) -> tuple[Sequential, RobustScaler, np.ndarray, int]:
    """
    LSTM-Netzwerk trainieren.
    Typ: Supervised (Labels von MS_Univariate).
    LSTM-Netzwerk mit rollendem Fenster für zeitreihenbasierte Regime-Klassifikation.

    Skalierung — fit NUR auf Trainingsdaten (Data Leakage vermeiden).
    Gewichtung Bear/Bull via sqrt(raw_weight).
    Modell + Scaler werden persistiert.

    Gibt (model, scaler, test_probs, split_index) zurück.
    """
    n_features = len(features)

    # Skalierung — fit NUR auf Trainingsdaten (Data Leakage vermeiden)
    split_point = int(len(df) * train_test_split)
    scaler = RobustScaler()
    scaler.fit(df[features].iloc[:split_point])          # fit nur auf Train
    scaled_data = scaler.transform(df[features])          # transform auf alles

    # Labels und Sequenzen
    # Wahl der passenden Labels (in config-file)
    X, y = create_sequences(scaled_data, df[labels_col].values, window_size)

    # Split (Train/Test) - 80% Training, 20% Test
    split = int(len(X) * train_test_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM Architektur
    model = build_lstm(
        window_size=window_size,
        n_features=n_features,
        units=units,
        return_sequences=return_sequences,
        dropout=dropout,
        dense=dense,
        activation=activation,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    # Gewichtung Bear/Bull
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    raw_weight = n_neg / n_pos
    sqrt_weight = math.sqrt(raw_weight)
    class_weight = {0: 1.0, 1: sqrt_weight}
    print(f"Class Balance — Bull: {int(n_neg)}, Bear: {int(n_pos)}, "
          f"raw_weight: {raw_weight:.2f}, pos_weight (sqrt): {sqrt_weight:.2f}")
    # Erwartet: raw_weight: 3.31, pos_weight (sqrt): ~1.82

    # Training
    print("Starte LSTM Training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
        class_weight=class_weight,
    )

    # Vorhersagen generieren
    lstm_probs_raw = model.predict(X_test)

    # Modell + Scaler persistieren
    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_file)
    joblib.dump(scaler, scaler_file)
    print(f"LSTM: Modell gespeichert unter {model_file}")
    print(f"Finale Test-Genauigkeit: {history.history['val_accuracy'][-1]:.2%}")

    return model, scaler, lstm_probs_raw, split


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
    Persistiertes LSTM-Modell + Scaler laden (Training überspringen).
    Skalierung mit geladenem Scaler (transform, NICHT fit_transform!).

    Gibt (model, scaler, test_probs, split_index) zurück.
    """
    print(f"LSTM: Lade persistiertes Modell aus {model_file}")
    model = load_model(model_file)
    scaler = joblib.load(scaler_file)

    # Skalierung mit geladenem Scaler (transform, NICHT fit_transform!)
    scaled_data = scaler.transform(df[features])

    # Labels und Sequenzen
    X, y = create_sequences(scaled_data, df[labels_col].values, window_size)

    # Split (Train/Test) - 80% Training, 20% Test
    split = int(len(X) * train_test_split)
    X_test = X[split:]

    # Vorhersagen generieren
    lstm_probs_raw = model.predict(X_test)

    return model, scaler, lstm_probs_raw, split


def predict_lstm(
    probs_raw: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wahrscheinlichkeiten und binäres Signal ableiten.
    Signale generieren via Threshold.

    Gibt (probabilities, signal) zurück.
    """
    probs = probs_raw.flatten()
    signal = (probs >= threshold).astype(int)
    return probs, signal