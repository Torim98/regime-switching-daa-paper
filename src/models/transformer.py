"""Transformer network: regime detection via multi-head self-attention (PyTorch)."""

import numpy as np
import pandas as pd
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
import joblib

from .common import create_sequences


# --- Positional encoding ---
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the Transformer architecture."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# --- Transformer classifier ---
class TransformerRegimeClassifier(nn.Module):
    """
    Linear(n_features → d_model) → PositionalEncoding
    → TransformerEncoder × n_layers → Linear(d_model → 1)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # last time step
        x = self.classifier(x)
        return x


def _build_model(
    n_features: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_feedforward: int,
    dropout: float,
    device: torch.device,
) -> TransformerRegimeClassifier:
    """Instantiate the Transformer model and move it to the device."""
    return TransformerRegimeClassifier(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)


def train_transformer(
    df: pd.DataFrame,
    features: list[str],
    labels_col: str,
    window_size: int,
    train_test_split: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_feedforward: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    validation_split: float,
    verbose: int,
    model_file: str,
    scaler_file: str,
) -> tuple[TransformerRegimeClassifier, RobustScaler, np.ndarray, int]:
    """
    Train the Transformer network.
    Type: supervised (labels from MS_Univariate).
    Transformer encoder with positional encoding for time-series-based
    regime classification. Tests hypothesis H2 (attention > econometric MSM).

    Scaling: fit ONLY on the training data (avoid data leakage).
    BCEWithLogitsLoss expects RAW logits (no sigmoid in the model output!).
    Bear/bull weighting via sqrt(raw_weight).
    Model + scaler are persisted.

    Returns (model, scaler, test_probs, split_index).
    """
    n_features = len(features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scaling: fit ONLY on the training data (avoid data leakage)
    split_point = int(len(df) * train_test_split)
    scaler = RobustScaler()
    scaler.fit(df[features].iloc[:split_point])               # fit on train only
    scaled_data = scaler.transform(df[features])               # transform on everything

    # Create sequences
    X, y = create_sequences(scaled_data, df[labels_col].values, window_size)

    # Train/test split (identical to the LSTM)
    split = int(len(X) * train_test_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Validation split (last 10% of the training data) ---
    val_split = int(len(X_train) * (1 - validation_split))
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]

    # Instantiate the model
    model = _build_model(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        device=device,
    )

    # Bear/bull weighting
    n_pos = y_train.sum()           # number of bear samples in training
    n_neg = len(y_train) - n_pos    # number of bull samples in training
    raw_weight = n_neg / n_pos
    pos_weight = torch.tensor([math.sqrt(raw_weight)], dtype=torch.float32).to(device)
    print(f"Class balance: bull: {int(n_neg)}, bear: {int(n_pos)}, "
          f"raw_weight: {raw_weight:.2f}, pos_weight (sqrt): {pos_weight.item():.2f}")
    # Expected: raw_weight: 3.31, pos_weight (sqrt): ~1.82

    # BCEWithLogitsLoss expects RAW logits (no sigmoid in the model output!)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.classifier = nn.Linear(d_model, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training
    print("Starting Transformer training...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)          # raw logits
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Accuracy tracking
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total

            # Compute the validation loss
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
            model.train()

            print(f"  Epoch {epoch+1}/{epochs}: loss: {avg_loss:.4f}, "
                  f"accuracy: {accuracy:.2%}, val loss: {val_loss:.4f}")

    # Predictions on the test set
    # Apply the sigmoid only at inference (logits → probabilities)
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_tensor)
        transformer_probs_raw = torch.sigmoid(logits_test).cpu().numpy().flatten()

    # Persist model + scaler
    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_file)
    joblib.dump(scaler, scaler_file)
    print(f"Transformer: model saved at {model_file}")

    return model, scaler, transformer_probs_raw, split

def train_transformer_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    labels_col: str,
    window_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_feedforward: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    validation_split: float,
    verbose: int,
    init_state_dict: dict | None = None,
    epochs_warm: int | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex, dict | None]:
    """
    Train the Transformer network on a walk-forward fold.

    Analogous to train_lstm_fold:
    - Receives train and test slices explicitly.
    - Fits the scaler exclusively on df_train.
    - Persists NOTHING.
    - Returns the OOS probabilities together with the corresponding DatetimeIndex.

    Parameters
    ----------
    df_train, df_test : pd.DataFrame
        Train and test slice (DatetimeIndex), temporally disjoint.
    init_state_dict : dict | None
        State dict from the previous fold for a warm start (rolling WF has 90%
        train overlap between folds → warm start is look-ahead-free).
        None = from scratch.
    epochs_warm : int | None
        Reduced epoch count for the warm start. None → max(5, epochs // 4).
    Remaining parameters :
        Same meaning as in train_transformer.

    Returns
    -------
    tuple[np.ndarray, pd.DatetimeIndex, dict | None]
        (probs_raw, prediction_index, final_state_dict)
        final_state_dict is None in the single-class fallback.

    Notes
    -----
    - BCEWithLogitsLoss expects RAW logits; the classifier is therefore a
      plain nn.Linear without sigmoid (identical to train_transformer).
      The sigmoid is applied only at inference.
    - pos_weight is computed exclusively from the train labels.
    - validation_split: the last X% of X_train are used as internal
      validation (chronologically ordered, leakage-free).
    - The single-class fallback is checked on the FULL y_train (before the
      val split), so that the val split cannot artificially destroy the
      class balance.
    """
    n_features = len(features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Targets are irrelevant for inference sequences. Use a dummy vector so
    # test-period ex-post labels are never required or accidentally consumed.
    test_targets = np.zeros(len(test_scaled_with_buffer), dtype=np.int8)
    X_test, _ = create_sequences(
        test_scaled_with_buffer, test_targets, window_size,
    )

    # prediction_index: now the ENTIRE df_test.index (no longer [window_size:]),
    # because the first test sequence can already predict on the first test day
    # thanks to the buffer.
    prediction_index = df_test.index
    assert len(prediction_index) == len(X_test), (
        f"Index mismatch after warm-up buffer: prediction_index={len(prediction_index)}, "
        f"X_test={len(X_test)}"
    )

    # --- 4. Single-class check on the FULL train set (BEFORE the val split) ---
    # Must happen here, not after the val split; otherwise the fallback would
    # also trigger in folds where all positives just happen to land in the
    # val window.
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos == 0 or n_neg == 0:
        import warnings
        majority_prob = 1.0 if n_pos > n_neg else 0.0
        warnings.warn(
            f"  [Transformer] Train fold is single-class (n_neg={n_neg}, n_pos={n_pos}). "
            f"Fallback: constant prediction P(Bear)={majority_prob}."
        )
        # Full test range (consistent with the warm-up buffer logic)
        pred_idx = df_test.index
        probs = np.full(len(pred_idx), majority_prob, dtype=np.float32)
        return probs, pred_idx, None

    # --- 5. Validation split within the train sequences (at the end in time) ---
    val_split = int(len(X_train) * (1 - validation_split))
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]

    # --- 6. Instantiate the model ---
    model = _build_model(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        device=device,
    )

    # --- 7. Class weighting (train labels only!) ---
    # Recompute on the reduced y_train after the val split, so that
    # pos_weight matches the subset that is actually trained on.
    n_pos_tr = int(y_train.sum())
    n_neg_tr = int(len(y_train) - n_pos_tr)
    # Edge case: the val split removed all samples of one class; then fall back
    # to a weight of 1.0. Training still runs (both classes exist in the full y_train).
    raw_weight = (n_neg_tr / n_pos_tr) if (n_pos_tr > 0 and n_neg_tr > 0) else 1.0
    pos_weight = torch.tensor([math.sqrt(raw_weight)], dtype=torch.float32).to(device)
    if verbose:
        print(
            f"  [Transformer Fold] Train: {len(df_train)} rows, Test: {len(df_test)} rows | "
            f"Bull: {n_neg_tr}, Bear: {n_pos_tr}, pos_weight (sqrt): {pos_weight.item():.2f}"
        )

    # --- 8. Loss + classifier override (raw logits) ---
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.classifier = nn.Linear(d_model, 1).to(device)

    # --- 8b. Warm start from the previous fold (optional) ---
    if init_state_dict is not None:
        try:
            # The state dict is stored on CPU → move to device
            moved = {k: v.to(device) for k, v in init_state_dict.items()}
            model.load_state_dict(moved)
            epochs_used = epochs_warm if epochs_warm is not None else max(5, epochs // 4)
            if verbose:
                print(f"    [Transformer] Warm start active, epochs_used={epochs_used}")
        except RuntimeError as e:
            # Shape mismatch (e.g. changed d_model / window_size) → from scratch
            if verbose:
                print(f"    [Transformer] Warm start discarded ({e.__class__.__name__}), from scratch")
            epochs_used = epochs
    else:
        epochs_used = epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- 8c. AMP setup ---
    # torch.amp.* replaces torch.cuda.amp.* (deprecated since PyTorch 2.3).
    # device_type is passed to GradScaler + autocast.
    use_amp = torch.cuda.is_available()
    device_type = "cuda" if use_amp else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    # --- 9. Tensors + DataLoader ---
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- 10. Training: AMP + early stopping ---
    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = 5, 0

    for epoch in range(epochs_used):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type, enabled=use_amp):
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Val loss per epoch (for early stopping)
        model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type, enabled=use_amp):
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits.float(), y_val_tensor).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    [Transformer] Early stop at epoch {epoch+1}/{epochs_used}")
                break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs_used}: val loss: {val_loss:.4f}")

    # Restore the best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- 11. OOS predictions ---
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type, enabled=use_amp):
        logits_test = model(X_test_tensor)
        probs_raw = torch.sigmoid(logits_test.float()).cpu().numpy().flatten()

    # --- 12. State dict for the warm start of the next fold ---
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return probs_raw, prediction_index, final_state

def load_transformer_model(
    df: pd.DataFrame,
    features: list[str],
    labels_col: str,
    window_size: int,
    train_test_split: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_feedforward: int,
    dropout: float,
    model_file: str,
    scaler_file: str,
) -> tuple[TransformerRegimeClassifier, RobustScaler, np.ndarray, int]:
    """
    Load the persisted Transformer model + scaler (skip training).

    Returns (model, scaler, test_probs, split_index).
    """
    n_features = len(features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model and load the weights
    model = _build_model(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        device=device,
    )
    # BCEWithLogitsLoss → classifier without sigmoid during training
    model.classifier = nn.Linear(d_model, 1).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    print(f"Transformer: loading persisted model from {model_file}")

    # Scaling with the loaded scaler
    scaler = joblib.load(scaler_file)
    scaled_data = scaler.transform(df[features])

    # Create sequences
    X, y = create_sequences(scaled_data, df[labels_col].values, window_size)

    # Split
    split = int(len(X) * train_test_split)
    X_test = X[split:]

    # Predictions on the test set
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        logits_test = model(X_test_tensor)
        transformer_probs_raw = torch.sigmoid(logits_test).cpu().numpy().flatten()

    return model, scaler, transformer_probs_raw, split


def predict_transformer(
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
