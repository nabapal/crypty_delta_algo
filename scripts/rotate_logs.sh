#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-/app/logs}"

mkdir -p "${LOG_DIR}"

# Remove log files older than 7 days while keeping the trade ledger intact
find "${LOG_DIR}" -type f -name "*.log" -mtime +7 -print -delete || true
