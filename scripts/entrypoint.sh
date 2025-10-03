#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-/app/logs}"
STORAGE_DIR="${STORAGE_DIR:-/app/storage}"
LEDGER_PATH="${TRADE_LEDGER_PATH:-/app/storage/delta_trader_trades.jsonl}"

mkdir -p "${LOG_DIR}" "${STORAGE_DIR}"

if [ ! -f "${LEDGER_PATH}" ]; then
  touch "${LEDGER_PATH}"
fi

"$(dirname "$0")/rotate_logs.sh" || true

exec "$@"
