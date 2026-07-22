"""The production LSTM builder must honor the configured learning rate."""

import pytest

tf = pytest.importorskip("tensorflow")

from src.models.lstm import build_lstm, weighted_bce


def test_build_lstm_applies_explicit_learning_rate():
    model = build_lstm(
        window_size=4,
        n_features=2,
        units_l1=2,
        units_l2=2,
        return_sequences=True,
        dropout=0.0,
        dense=1,
        activation="sigmoid",
        optimizer="adam",
        learning_rate=1.25e-5,
        loss=weighted_bce(1.0),
        metrics="accuracy",
    )
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(1.25e-5)

