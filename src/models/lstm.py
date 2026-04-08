"""LSTM-Netzwerk — Supervised Regime Classification (TensorFlow/Keras)."""

import numpy as np
import pandas as pd
import math
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import joblib

from .common import create_sequences

def weighted_bce(pos_weight: float):
    """
    Binary Cross-Entropy mit positiver Klassengewichtung.
    Entspricht dem pos_weight-Mechanismus von torch.nn.BCEWithLogitsLoss
    und stellt sicher, dass LSTM und Transformer dieselbe Verlustfunktion
    (inkl. identischer Gewichtungsformel sqrt(n_neg/n_pos)) verwenden.
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
    LSTM-Architektur gemäß Config aufbauen.
    input_shape passt sich automatisch an die Anzahl der Features an.
    Binäre Klassifikation (Dense 1, Sigmoid).
    """
    model = Sequential([
        LSTM(units_l1,
             return_sequences=return_sequences,
             input_shape=(window_size, n_features)),
        Dropout(dropout),
        LSTM(units_l2),
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
    LSTM-Netzwerk trainieren.
    Typ: Supervised (Labels von MS_Univariate).
    LSTM-Netzwerk mit rollendem Fenster für zeitreihenbasierte Regime-Klassifikation.

    Skalierung — fit NUR auf Trainingsdaten (Data Leakage vermeiden).
    Gewichtung Bear/Bull via pos_weight = sqrt(n_neg/n_pos) in weighted BCE.
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
    
    # pos_weight identisch zum Transformer bestimmen
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    raw_weight = n_neg / n_pos
    pos_weight = math.sqrt(raw_weight)
    print(f"Class Balance — Bull: {n_neg}, Bear: {n_pos}, "
          f"raw_weight: {raw_weight:.2f}, pos_weight (sqrt): {pos_weight:.2f}")
    # Erwartet: raw_weight: 3.31, pos_weight (sqrt): ~1.82

    # LSTM Architektur
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
    print("Starte LSTM Training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
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
    # compile=False: das geladene Modell wird nur für Inference genutzt,
    # daher muss Keras die custom weighted_bce-Loss nicht deserialisieren.
    # Falls später inkrementell weitertrainiert werden soll, hier stattdessen
    # custom_objects={"loss": weighted_bce(pos_weight)} übergeben oder das
    # Modell nach dem Laden neu compilen.
    model = load_model(model_file, compile=False)
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