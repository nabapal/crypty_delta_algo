# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_DIR=/app/logs \
    STORAGE_DIR=/app/storage \
    TRADE_LEDGER_PATH=/app/storage/delta_trader_trades.jsonl

WORKDIR /app

# Install build tools needed for some python packages (kept minimal)
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends build-essential \ 
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/logs /app/storage

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh

ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Default command runs the live trading engine; override in docker compose for other services
CMD ["python", "production_delta_trader.py"]
