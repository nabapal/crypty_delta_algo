# Crypty Delta Algo

Automated delta-hedged options trading system for Delta Exchange, paired with a Streamlit control panel for live monitoring. The trading engine orchestrates short strangle entries, manages risk through configurable rules, and exposes rich telemetry for operators.

## Features

- **Contract-aware P&L** — scales performance metrics by the BTC contract size used on Delta Exchange.
- **Concurrent leg execution** — enters and exits both call and put legs simultaneously for tighter fills.
- **Resilient scheduling** — supports next-day exit rollovers when exit times fall before the trade window ends.
- **Live dashboard** — Streamlit UI surfaces current positions, premium capture, and UTC-stamped logs.
- **Trade history ledger** — every completed strangle is persisted to JSONL for retrospective analytics and UI summaries.
- **Structured logging** — summary, detailed, and ledger files power analytics and troubleshooting.
- **Containerized deployment** — Dockerfile and Compose stack run the trader and dashboard together for production use.

## Project structure

```text
production_delta_trader.py  # Main trading engine loop
streamlit_app.py            # Streamlit dashboard for monitoring & control
Dockerfile                  # Container image definition (Python 3.11 slim)
docker-compose.yml          # Orchestrates trader and dashboard services
.dockerignore               # Slims Docker build context
config_loader.py            # Shared configuration helpers
trading_config.py           # Baseline trading parameters
ui_config_snapshot.json     # Snapshot of UI selections (read by Streamlit)
ui_overrides.json           # Operator overrides persisted from the UI
requirements.txt            # Python dependencies
.env                        # Environment variables (API keys and tuning)
scripts/                    # Utility scripts (entrypoint, log rotation, redeploy)
storage/
	├── logs/                 # Host-mounted logs (rotated automatically)
	├── trades/               # Host-mounted trade ledger file(s)
	└── backups/              # Auto-generated config backups (redeploy script)
```

## Getting started

### Prerequisites

- Python 3.10+
- Delta Exchange account with API credentials
- Optional: virtual environment manager (`python -m venv`, `conda`, etc.)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration

1. Review `.env` for API keys and runtime parameters. Replace placeholder values with live credentials if needed.
2. Adjust `trading_config.py` for global defaults (delta range, quantity, timings).
3. Update `ui_overrides.json` if you want to seed the dashboard with predefined overrides.

> **Note:** The repository currently commits `.env` for operational convenience. Rotate credentials if this repo becomes public and consider encrypting secrets (e.g., with git-crypt or a secrets manager) before wider distribution.

## Running the trading engine

```bash
source .venv/bin/activate
python production_delta_trader.py
```

The engine:
- Loads configuration from `.env`, `trading_config.py`, and override files.
- Places short strangle positions when the configured trade window opens.
- Monitors mark prices, trailing rules, and exit criteria on a recurring loop.
- Logs events to `${LOG_DIR:-logs}/delta_trader_summary.log` (high-level) and `${LOG_DIR:-logs}/delta_trader_detailed.log` (diagnostics).

## Streamlit dashboard

```bash
source .venv/bin/activate
streamlit run streamlit_app.py --server.port 8502
```

The dashboard is organized into tabs:
- **Live Dashboard** — strategy status, controls, live metrics, and open-leg breakdowns.
- **Trade History** — aggregated performance, sortable ledger table, and leg-level drilldowns backed by `delta_trader_trades.jsonl`.
- **Logs** — summary and detailed log tails with UTC/IST context for troubleshooting.

## Logging & monitoring

- `${LOG_DIR:-logs}/delta_trader_summary.log` — snapshots of key metrics and lifecycle events.
- `${LOG_DIR:-logs}/delta_trader_detailed.log` — verbose debugging output (API calls, reconciliation steps).
- `${TRADE_LEDGER_PATH:-storage/delta_trader_trades.jsonl}` — append-only ledger of completed trades consumed by the dashboard's Trade History tab.
- Streamlit surfaces log tails with UTC context and the Trade History tab for rich retrospectives.
- Containers automatically prune `.log` files older than 7 days on startup while preserving the trade ledger for long-term analytics.

## Production deployment with Docker Compose

The repository ships with a Dockerfile and a Compose stack that run the live trader and the Streamlit dashboard side by side.

### 1. Prepare persistent storage

Logs and trade history now live outside the container so they survive rebuilds. Create the host directories once:

```bash
mkdir -p storage/logs storage/trades storage/backups
touch storage/trades/delta_trader_trades.jsonl
```

### 2. Fetch the latest code from GitHub

```bash
git fetch origin
git checkout main
git pull origin main
```

> Replace `origin` if your remote has a different name.

### 3. Build the containers

```bash
docker compose build
```

This step installs Python dependencies into the image using the pinned `requirements.txt`.

### 4. Launch the stack

```bash
docker compose up -d
```

- `trader` runs `python production_delta_trader.py` with environment variables sourced from `.env`.
- `dashboard` launches Streamlit on port `8502` (exposed to the host as `8502`).

Monitor logs with:

```bash
docker compose logs -f trader
docker compose logs -f dashboard
```

Stop everything with:

```bash
docker compose down
```

> Ensure `.env` contains valid API credentials before starting the containers. Update the file and re-run `docker compose up -d --build` whenever configuration changes. Log files are persisted under `storage/logs/` on the host, and the trade ledger remains at `storage/trades/delta_trader_trades.jsonl`.

### 5. One-command redeploy (pull, rebuild, restart)

To fetch the latest Git changes, rebuild, and restart the stack safely (even with uncommitted UI overrides), run:

```bash
./scripts/redeploy.sh
```

What it does:
- Backs up `ui_config_snapshot.json` and `ui_overrides.json` to `storage/backups/<timestamp>/` if the working tree is dirty.
- Stashes local changes (including untracked files) so `git pull` always succeeds.
- Rebuilds the containers and restarts the stack via Docker Compose.
- Attempts to reapply the stash; if conflicts arise, restores the backed-up config files.
- Finishes with `docker compose ps` so you can confirm both services are running.

## Deployment tips

- Run the engine under a process supervisor (systemd, pm2, Supervisor) on a reliable VPS.
- Enable OS-level log rotation for `*.log` files if running continuously.
- Keep API credentials restricted to minimal permissions set required for trading.

## Contributing

1. Fork or create a feature branch.
2. Run `pip install -r requirements.txt` to sync dependencies.
3. Validate with `python -m compileall production_delta_trader.py streamlit_app.py` before opening a PR.
4. Submit changes with descriptive commit messages and include updates to docs/tests where relevant.

## License

No explicit license provided. Treat the repository as proprietary unless stated otherwise.
