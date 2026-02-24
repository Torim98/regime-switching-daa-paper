Input (batch, 30, 6)                     ← Window von 30 Tagen, 6 Features

&nbsp;   ↓

Linear Projection → (batch, 30, 64)      ← Projiziert Features auf d\_model=64

&nbsp;   ↓

Positional Encoding                       ← Sinusoidal PE für Zeitreihen-Position

&nbsp;   ↓

TransformerEncoder × 2                    ← 2 Encoder-Layer mit 4-Head Self-Attention

&nbsp; ├─ Multi-Head Self-Attention (4 Heads)

&nbsp; ├─ Feed-Forward (128 Hidden)

&nbsp; └─ Dropout (0.1) + LayerNorm

&nbsp;   ↓

Letzter Zeitschritt → (batch, 64)        ← Nur den finalen Hidden State verwenden

&nbsp;   ↓

Dense(64 → 1) + Sigmoid                  ← Bear-Wahrscheinlichkeit \[0, 1]

