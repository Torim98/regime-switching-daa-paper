FROM python:3.11-slim

WORKDIR /app

# harden pip against slow/flaky downloads
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

# dependencies first (Docker layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[services,dashboard]"

# copy project code
COPY config/ config/
COPY src/ src/
COPY services/__init__.py services/__init__.py
COPY services/logging_config.py services/logging_config.py
COPY services/warnings_config.py services/warnings_config.py
COPY services/dashboard_service/ services/dashboard_service/

ENV PYTHONPATH=/app

EXPOSE 8004
CMD ["uvicorn", "services.dashboard_service.main:app", "--host", "0.0.0.0", "--port", "8004"]
