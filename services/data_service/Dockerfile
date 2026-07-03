FROM python:3.11-slim

WORKDIR /app

# harden pip against slow/flaky downloads
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

# dependencies first (Docker layer caching), without ML frameworks
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[services]"

# copy project code
COPY config/ config/
COPY src/ src/
COPY services/__init__.py services/__init__.py
COPY services/logging_config.py services/logging_config.py
COPY services/warnings_config.py services/warnings_config.py
COPY services/data_service/ services/data_service/

ENV PYTHONPATH=/app

EXPOSE 8001
CMD ["uvicorn", "services.data_service.main:app", "--host", "0.0.0.0", "--port", "8001"]