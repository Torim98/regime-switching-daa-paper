FROM python:3.11-slim

WORKDIR /app

# Dependencies zuerst (Docker Layer Caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[services]"

# Projektcode kopieren
COPY config/ config/
COPY src/ src/
COPY services/__init__.py services/__init__.py
COPY services/logging_config.py services/logging_config.py
COPY services/data_service/ services/data_service/

ENV PYTHONPATH=/app

EXPOSE 8001
CMD ["uvicorn", "services.data_service.main:app", "--host", "0.0.0.0", "--port", "8001"]