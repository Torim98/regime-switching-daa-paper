regime-switching-daa/

├── src/                              # Shared Business Logic

│   ├── \_\_init\_\_.py

│   ├── data/

│   │   ├── \_\_init\_\_.py

│   │   ├── ingestion.py              # aus 01: yfinance-Download

│   │   ├── preprocessing.py          # aus 01: Portfolio-Konstruktion, Returns

│   │   ├── feature\_engineering.py    # aus 02: Rolling Features

│   │   └── eda.py                    # aus 01: Deskriptive Stats, ADF-Tests

│   ├── models/

│   │   ├── \_\_init\_\_.py

│   │   ├── common.py                 # Konstanten, validate\_regime\_signal(), create\_sequences()

│   │   ├── msm.py                    # Markov-Switching

│   │   ├── hmm.py                    # Hidden Markov Model

│   │   ├── lstm.py                   # LSTM

│   │   └── transformer.py            # Transformer (PositionalEncoding + Classifier)

│   └── backtest/

│       ├── \_\_init\_\_.py

│       ├── engine.py                 # aus 04: backtest()

│       ├── sorr.py                   # aus 04: run\_sorr\_simulation()

│       ├── evaluation.py             # aus 05: evaluate\_strategies(), MCS

│       └── reporting.py              # aus 99: statistics.md Generierung

├── services/                         # FastAPI-Services

│   ├── data\_service/

│   │   ├── Dockerfile

│   │   ├── main.py

│   │   └── routes.py

│   ├── model\_service/

│   │   ├── Dockerfile

│   │   ├── main.py

│   │   └── routes.py

│   └── backtest\_service/

│       ├── Dockerfile

│       ├── main.py

│       └── routes.py

├── docker-compose.yml

├── pyproject.toml

├── config/

├── jupyter/

├── data/

├── models/

└── assets/

