FROM python:3.11-slim

WORKDIR /app

# pip gegen langsame/flaky Downloads härten
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

# Dependencies zuerst (Docker Layer Caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[services,dashboard]"

# Projektcode kopieren
COPY config/ config/
COPY src/ src/
COPY services/__init__.py services/__init__.py
COPY services/logging_config.py services/logging_config.py
COPY services/warnings_config.py services/warnings_config.py
COPY services/dashboard_service/ services/dashboard_service/

ENV PYTHONPATH=/app

EXPOSE 8004
CMD ["uvicorn", "services.dashboard_service.main:app", "--host", "0.0.0.0", "--port", "8004"]
