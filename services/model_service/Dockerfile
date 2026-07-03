FROM python:3.11-slim

WORKDIR /app

# Chromium for Kaleido/Plotly PNG export
RUN apt-get update && \
    apt-get install -y --no-install-recommends chromium && \
    rm -rf /var/lib/apt/lists/*
ENV CHROME_PATH=/usr/bin/chromium

# harden pip against slow/flaky downloads
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

COPY pyproject.toml .

# Install CPU-only Torch separately first, saves ~3.5 GB of CUDA wheels.
# Afterwards ".[services,ml]" only pulls tensorflow/keras/fastapi,
# torch is already satisfied.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
RUN pip install --no-cache-dir ".[services,ml]"

COPY config/ config/
COPY src/ src/
COPY services/__init__.py services/__init__.py
COPY services/logging_config.py services/logging_config.py
COPY services/__init__.py services/__init__.py
COPY services/logging_config.py services/logging_config.py
COPY services/warnings_config.py services/warnings_config.py
COPY services/data_service/ services/data_service/
COPY services/model_service/ services/model_service/

ENV PYTHONPATH=/app

EXPOSE 8002
CMD ["uvicorn", "services.model_service.main:app", "--host", "0.0.0.0", "--port", "8002"]