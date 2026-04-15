"""Transformer-Netzwerk — Regime Detection via Multi-Head Self-Attention (PyTorch)."""

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


# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    """Sinusoidale Positionsencoding für Transformer-Architektur."""

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


# --- Transformer-Klassifikator ---
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
        x = x[:, -1, :]  # Letzter Zeitschritt
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
    """Transformer-Modell instanziieren und auf Device verschieben."""
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
    Transformer-Netzwerk trainieren.
    Typ: Supervised (Labels von MS_Univariate).
    Transformer-Encoder mit Positional Encoding für zeitreihenbasierte
    Regime-Klassifikation. Testet Hypothese H2 (Attention > Econometric MSM).

    Skalierung — fit NUR auf Trainingsdaten (Data Leakage vermeiden).
    BCEWithLogitsLoss erwartet RAW Logits (kein Sigmoid im Modell-Output!).
    Gewichtung Bear/Bull via sqrt(raw_weight).
    Modell + Scaler werden persistiert.

    Gibt (model, scaler, test_probs, split_index) zurück.
    """
    n_features = len(features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Skalierung — fit NUR auf Trainingsdaten (Data Leakage vermeiden)
    split_point = int(len(df) * train_test_split)
    scaler = RobustScaler()
    scaler.fit(df[features].iloc[:split_point])               # fit nur auf Train
    scaled_data = scaler.transform(df[features])               # transform auf alles

    # Sequenzen erstellen
    X, y = create_sequences(scaled_data, df[labels_col].values, window_size)

    # Train/Test Split (identisch zum LSTM)
    split = int(len(X) * train_test_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Validation-Split (letzte 10% der Trainingsdaten) ---
    val_split = int(len(X_train) * (1 - validation_split))
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]

    # Modell instanziieren
    model = _build_model(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        device=device,
    )

    # Gewichtung Bear/Bull
    n_pos = y_train.sum()           # Anzahl Bear-Samples im Training
    n_neg = len(y_train) - n_pos    # Anzahl Bull-Samples im Training
    raw_weight = n_neg / n_pos
    pos_weight = torch.tensor([math.sqrt(raw_weight)], dtype=torch.float32).to(device)
    print(f"Class Balance — Bull: {int(n_neg)}, Bear: {int(n_pos)}, "
          f"raw_weight: {raw_weight:.2f}, pos_weight (sqrt): {pos_weight.item():.2f}")
    # Erwartet: raw_weight: 3.31, pos_weight (sqrt): ~1.82

    # BCEWithLogitsLoss erwartet RAW Logits (kein Sigmoid im Modell-Output!)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.classifier = nn.Linear(d_model, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoader erstellen
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training
    print("Starte Transformer Training...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)          # Raw Logits
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

            # Validation-Loss berechnen
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
            model.train()

            print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, "
                  f"Accuracy: {accuracy:.2%}, Val-Loss: {val_loss:.4f}")

    # Vorhersagen auf Test-Set
    # Sigmoid erst bei der Inference anwenden (Logits → Probabilities)
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_tensor)
        transformer_probs_raw = torch.sigmoid(logits_test).cpu().numpy().flatten()

    # Modell + Scaler persistieren
    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_file)
    joblib.dump(scaler, scaler_file)
    print(f"Transformer: Modell gespeichert unter {model_file}")

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
    Transformer-Netzwerk auf einem Walk-Forward-Fold trainieren.

    Analog zu train_lstm_fold:
    - Erhält Train- und Test-Slices explizit.
    - Fittet den Scaler ausschließlich auf df_train.
    - Persistiert NICHTS.
    - Gibt OOS-Probabilities samt zugehörigem DatetimeIndex zurück.

    Parameter
    ---------
    df_train, df_test : pd.DataFrame
        Train- bzw. Test-Slice (DatetimeIndex), zeitlich disjunkt.
    init_state_dict : dict | None
    State-Dict vom vorigen Fold fuer Warm-Start (rolling WF hat 90%
    Train-Ueberlappung zwischen Folds → Warm-Start ist look-ahead-frei).
    None = from scratch.
    epochs_warm : int | None
    Reduzierte Epochen-Zahl beim Warm-Start. None → max(5, epochs // 4).
    Übrige Parameter :
        Identische Bedeutung wie in train_transformer.

    Rückgabe
    --------
    tuple[np.ndarray, pd.DatetimeIndex, dict | None]
        (probs_raw, prediction_index, final_state_dict)
        final_state_dict ist None im Einklassen-Fallback.

    Hinweise
    --------
    - BCEWithLogitsLoss erwartet RAW Logits; der Classifier ist daher ein
      reines nn.Linear ohne Sigmoid (identisch zu train_transformer).
      Sigmoid wird erst bei der Inference angewendet.
    - pos_weight wird ausschließlich aus den Train-Labels berechnet.
    - validation_split: die letzten X% von X_train werden als interne
      Validation verwendet (zeitlich geordnet, leakage-frei).
    - Einklassen-Fallback wird auf dem VOLLEN y_train (vor Val-Split) geprüft,
      damit der Val-Split die Klassenbalance nicht künstlich zerstören kann.
    """
    n_features = len(features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Sanity-Checks ---
    if len(df_train) <= window_size:
        raise ValueError(
            f"df_train hat nur {len(df_train)} Zeilen, benötigt > window_size={window_size}."
        )
    if len(df_test) < window_size:
        raise ValueError(
            f"df_test hat nur {len(df_test)} Zeilen, benötigt >= window_size={window_size}."
        )
    if df_train.index.max() >= df_test.index.min():
        raise ValueError(
            f"df_train endet ({df_train.index.max()}) nicht strikt vor df_test "
            f"({df_test.index.min()}) — Look-Ahead-Verdacht!"
        )

    # --- 2. Skalierung — fit NUR auf Trainingsdaten ---
    scaler = RobustScaler()
    scaler.fit(df_train[features])
    train_scaled = scaler.transform(df_train[features])
    test_scaled = scaler.transform(df_test[features])

    # --- 3. Sequenzen erzeugen — mit Warm-up-Buffer für Test ---
    # Train-Sequenzen nur aus df_train.
    X_train, y_train = create_sequences(
        train_scaled, df_train[labels_col].values, window_size,
    )

    # Test-Sequenzen MIT Warm-up: die letzten window_size Zeilen aus df_train
    # als "Geschichte" voranstellen, damit die erste Test-Sequenz am ersten
    # Test-Tag prädizieren kann (statt erst window_size Tage später).
    # WICHTIG: Diese Buffer-Zeilen werden NICHT zum Trainieren verwendet;
    # sie liefern nur die Input-Features für die Test-Sequenzen.
    buffer_scaled = train_scaled[-window_size:]
    test_scaled_with_buffer = np.concatenate([buffer_scaled, test_scaled], axis=0)

    buffer_labels = df_train[labels_col].values[-window_size:]
    test_labels_with_buffer = np.concatenate(
        [buffer_labels, df_test[labels_col].values], axis=0,
    )

    X_test, _ = create_sequences(
        test_scaled_with_buffer, test_labels_with_buffer, window_size,
    )

    # prediction_index: jetzt der GESAMTE df_test.index (nicht mehr [window_size:]),
    # weil die erste Test-Sequenz dank Buffer schon am ersten Test-Tag prädizieren kann.
    prediction_index = df_test.index
    assert len(prediction_index) == len(X_test), (
        f"Index-Mismatch nach Warm-up-Buffer: prediction_index={len(prediction_index)}, "
        f"X_test={len(X_test)}"
    )

    # --- 4. Einklassen-Check auf dem VOLLEN Train-Set (VOR Val-Split) ---
    # Muss hier passieren, nicht nach dem Val-Split — sonst triggert der
    # Fallback auch in Folds, in denen nur zufällig alle Positives im
    # Val-Fenster landen würden.
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos == 0 or n_neg == 0:
        import warnings
        majority_prob = 1.0 if n_pos > n_neg else 0.0
        warnings.warn(
            f"  [Transformer] Train-Fold einklassig (n_neg={n_neg}, n_pos={n_pos}). "
            f"Fallback: konstante Vorhersage P(Bear)={majority_prob}."
        )
        # Voller Test-Bereich (konsistent mit Warm-up-Buffer-Logik)
        pred_idx = df_test.index
        probs = np.full(len(pred_idx), majority_prob, dtype=np.float32)
        return probs, pred_idx, None

    # --- 5. Validation-Split innerhalb der Train-Sequenzen (zeitlich am Ende) ---
    val_split = int(len(X_train) * (1 - validation_split))
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]

    # --- 6. Modell instanziieren ---
    model = _build_model(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        device=device,
    )

    # --- 7. Klassengewichtung (nur auf Train-Labels!) ---
    # Nach dem Val-Split auf dem reduzierten y_train neu berechnen,
    # damit pos_weight zum tatsächlich trainierten Subset passt.
    n_pos_tr = int(y_train.sum())
    n_neg_tr = int(len(y_train) - n_pos_tr)
    # Edge-Case: Val-Split hat alle einer Klasse entzogen — dann mit 1.0 fallback-gewichten,
    # Training läuft trotzdem durch (Klassen existieren im vollen y_train).
    raw_weight = (n_neg_tr / n_pos_tr) if (n_pos_tr > 0 and n_neg_tr > 0) else 1.0
    pos_weight = torch.tensor([math.sqrt(raw_weight)], dtype=torch.float32).to(device)
    if verbose:
        print(
            f"  [Transformer Fold] Train: {len(df_train)} rows, Test: {len(df_test)} rows | "
            f"Bull: {n_neg_tr}, Bear: {n_pos_tr}, pos_weight (sqrt): {pos_weight.item():.2f}"
        )

    # --- 8. Loss + Classifier-Override (Raw Logits) ---
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.classifier = nn.Linear(d_model, 1).to(device)

    # --- 8b. Warm-Start vom vorherigen Fold (optional) ---
    if init_state_dict is not None:
        try:
            # State-Dict ist auf CPU gespeichert → auf device verschieben
            moved = {k: v.to(device) for k, v in init_state_dict.items()}
            model.load_state_dict(moved)
            epochs_used = epochs_warm if epochs_warm is not None else max(5, epochs // 4)
            if verbose:
                print(f"    [Transformer] Warm-Start aktiv, epochs_used={epochs_used}")
        except RuntimeError as e:
            # Shape-Mismatch (z.B. d_model / window_size geaendert) → from scratch
            if verbose:
                print(f"    [Transformer] Warm-Start verworfen ({e.__class__.__name__}), from-scratch")
            epochs_used = epochs
    else:
        epochs_used = epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- 8c. AMP-Setup ---
    # torch.amp.* ersetzt torch.cuda.amp.* (deprecated ab PyTorch 2.3).
    # device_type wird an GradScaler + autocast weitergereicht.
    use_amp = torch.cuda.is_available()
    device_type = "cuda" if use_amp else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    # --- 9. Tensoren + DataLoader ---
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- 10. Training: AMP + Early Stopping ---
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

        # Val-Loss pro Epoche (fuer Early Stopping)
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
            print(f"    Epoch {epoch+1}/{epochs_used} — Val-Loss: {val_loss:.4f}")

    # Beste Gewichte wiederherstellen
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- 11. OOS-Vorhersagen ---
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type, enabled=use_amp):
        logits_test = model(X_test_tensor)
        probs_raw = torch.sigmoid(logits_test.float()).cpu().numpy().flatten()

    # --- 12. State-Dict fuer Warm-Start des naechsten Folds ---
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
    Persistiertes Transformer-Modell + Scaler laden (Training überspringen).

    Gibt (model, scaler, test_probs, split_index) zurück.
    """
    n_features = len(features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modell instanziieren und Gewichte laden
    model = _build_model(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        device=device,
    )
    # BCEWithLogitsLoss → Classifier ohne Sigmoid beim Training
    model.classifier = nn.Linear(d_model, 1).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    print(f"Transformer: Lade persistiertes Modell aus {model_file}")

    # Skalierung mit geladenem Scaler
    scaler = joblib.load(scaler_file)
    scaled_data = scaler.transform(df[features])

    # Sequenzen erstellen
    X, y = create_sequences(scaled_data, df[labels_col].values, window_size)

    # Split
    split = int(len(X) * train_test_split)
    X_test = X[split:]

    # Vorhersagen auf Test-Set
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
    Wahrscheinlichkeiten und binäres Signal ableiten.
    Signale generieren via Threshold.

    Gibt (probabilities, signal) zurück.
    """
    probs = probs_raw.flatten()
    signal = (probs >= threshold).astype(int)
    return probs, signal