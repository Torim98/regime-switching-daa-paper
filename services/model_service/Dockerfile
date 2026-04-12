FROM python:3.11-slim

WORKDIR /app

# Chromium für Kaleido/Plotly PNG-Export
RUN apt-get update && \
    apt-get install -y --no-install-recommends chromium && \
    rm -rf /var/lib/apt/lists/*
ENV CHROME_PATH=/usr/bin/chromium

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[services]"

COPY config/ config/
COPY src/ src/
COPY services/__init__.py services/__init__.py
COPY services/logging_config.py services/logging_config.py
COPY services/model_service/ services/model_service/

ENV PYTHONPATH=/app

EXPOSE 8002
CMD ["uvicorn", "services.model_service.main:app", "--host", "0.0.0.0", "--port", "8002"]