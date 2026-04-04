FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[services]"

COPY config/ config/
COPY src/ src/
COPY services/__init__.py services/__init__.py
COPY services/backtest_service/ services/backtest_service/

ENV PYTHONPATH=/app

EXPOSE 8003
CMD ["uvicorn", "services.backtest_service.main:app", "--host", "0.0.0.0", "--port", "8003"]