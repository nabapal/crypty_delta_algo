# Production Delta Trader — System Notes

_Last reviewed: October 3, 2025_

## Executive Summary

`production_delta_trader.py` implements a production-oriented automated short strangle trading system for Delta Exchange BTC/ETH options. The script covers the entire lifecycle of an options trade—from strike discovery through live monitoring and exit—with special attention to risk controls, duplicate-trade prevention, and operational resilience suitable for live deployment.

## Architectural Overview

The code is structured around a handful of cooperating classes:

- **`TradingConfig`** — Dataclass capturing all tunable parameters (strategy ranges, risk thresholds, order controls, system toggles). Defaults imply live trading (`dry_run=False`, `testnet=False`).
- **`OptionData`** — Dataclass snapshot of option contract metadata, Greeks, and order book metrics.
- **`Position`** — Enhanced position tracker with partial-fill awareness, weighted entry price computation, fill history, and human-readable summaries.
- **`StrategyState`** — Aggregated view of current strategy, holding references to both call/put positions, P&L, trailing-stop state, and lifecycle timestamps.
- **`WebSocketPriceStreamer`** — Maintains Delta Exchange WebSocket subscriptions (`mark_price` + `v2/ticker`) for fresh pricing, with reconnect logic and enriched local cache.
- **`DeltaExchangeAPI`** — Authenticated REST client with signature generation, rate limiting, retry/backoff, and convenience methods for option chain retrieval, order placement, status polling, and account hygiene (positions/orders cancellation).
- **`AdvancedOrderManager`** — Handles execution attempts, retry slippage, partial fill management, and market-order fallbacks. Ensures no overlapping orders and enforces cleanup before new submissions.
- **`ShortStrangleStrategy`** — Orchestrates the short strangle flow: prerequisite checks, strike selection via delta filtering, order execution, P&L tracking, exit logic, and optional recovery from pre-existing positions.
- **Utility Entrypoints** — Functions like `create_default_config`, `run_comprehensive_test`, and `main()` provide testing pathways and a CLI entry for live operation.

## Strategy Lifecycle

1. **Prerequisite Validation** (`validate_prerequisites`)
   - Tests API connectivity (public + authenticated endpoints).
   - Ensures no live positions or pending orders exist to avoid duplicate trades.
   - Warns on trading outside defined market hours (09:00–17:30 IST).

2. **Strike Discovery** (`get_option_chain_with_greeks`, `find_delta_options`)
   - Fetches next-day expiry option chain.
   - Filters for contracts with absolute delta in `[0.10, 0.15]` (configurable) and sufficient liquidity.
   - Chooses the call/put closest to the midpoint of the delta band.

3. **Order Execution** (`AdvancedOrderManager.execute_order_with_retries`)
   - Cancels any lingering orders before submitting new ones.
   - Attempts limit orders with price adjustments per retry (`retry_slippage_pct`).
   - Enforces monitoring loop with timeout-based cancellation and optional market-order fallback.
   - Records fill details (including partial quantities) into `Position` objects via `add_fill`.

4. **Active Monitoring** (`run_live_monitoring`)
   - Periodic P&L refresh leveraging WebSocket mark prices or REST fallbacks.
   - Maintains trailing stop targets relative to peak profit.
   - Logs concise console updates plus verbose debug logs to dedicated files.
   - Triggers detailed status reports every five minutes and performs reconciliation routines.

5. **Exit Handling** (`check_exit_conditions`, `exit_position`)
   - Exit triggers: scheduled time, max loss, max profit, trailing stop breach, or manual force exit.
   - Executes BUY orders to close short legs, re-using the order manager for symmetry.
   - Calculates realized P&L from actual fills, verifies account cleanliness, and records exit rationale.

6. **Post-Trade Hooks** (`save_trade_to_database`, `generate_trade_report`)
   - Prepares structured data for downstream persistence (database integration TBD).
   - Generates a detailed report containing configuration snapshot, P&L metrics, and metadata.

## Risk Controls & Safeguards

- **Max Loss / Profit** — Percentage thresholds based on premium received; configurable via `TradingConfig`.
- **Trailing Stop** — Tiered rules (e.g., 40% profit → breakeven stop) with incremental tightening beyond 50% profit.
- **Duplicate Prevention** — Hard checks on existing positions/orders before a new trade enters.
- **Time Gating** — Default trade/exit times are dynamic (current time offsets), ensuring the script doesn’t enter positions immediately upon boot without a buffer.
- **Partial Fill Management** — Positions preserve remaining quantities, and order retries attempt to complete unfinished legs when sensible.
- **Emergency Handling** — `force_exit_position` and robust `try/except` blocks limit runaway scenarios.

## Order Management Highlights

- **Idempotent Client IDs** — Each order uses unique `client_order_id` to guard against duplicates.
- **Retry Mechanics** — Exponential backoff between attempts; slippage applied directionally (sellers lower price, buyers raise price).
- **Market Fallback** — After exhausting limit retries, the system optionally sends market orders.
- **Cancellation Hygiene** — Before each attempt, the script cancels outstanding orders and waits for confirmation.
- **Partial Fill Reporting** — Console/file logs annotate partials with remaining quantities and recommendations.

## Monitoring & Logging

- Console logging at `INFO`, with detailed trace files:
  - `delta_trader_detailed.log` — Debug-level chronology.
  - `delta_trader_summary.log` — High-level milestone log.
- `requests`/`urllib3` verbosity suppressed on console but piped to file handler for audit trails.
- Status reports include premium totals, P&L, trailing levels, and distance to risk thresholds.
- Market alerts fire on large spot moves (placeholder logic currently tied to option symbol data).

## External Dependencies & Inputs

- **Environment Variables** (via `python-dotenv`):
  - `DELTA_API_KEY`, `DELTA_API_SECRET` for production.
  - `DELTA_TESTNET_API_KEY`, `DELTA_TESTNET_API_SECRET` for testnet fallback.
- **Libraries**: `requests`, `websocket-client`, `python-dotenv`, `asyncio`, `dataclasses`, `logging`, plus Python standard modules.
- **Timezone Handling**: IST offset management ensures trade windows align with Indian market hours.

## Running the Script

> ⚠️ **Live trading is enabled by default (`dry_run=False`)**. Switch to test credentials or set `dry_run=True` before experimenting.

Common entry points:

```bash
# Execute the main live-trading flow (be cautious: places real orders)
/home/naba/crypto_delta_algo/.venv/bin/python production_delta_trader.py

# Run the comprehensive dry-run test harness
/home/naba/crypto_delta_algo/.venv/bin/python -c "import production_delta_trader as p; p.run_comprehensive_test()"
```

Consider exporting environment variables or creating a `.env` file in the project root before running.

> ℹ️ **Configuration precedence:** when the CLI entrypoint (`production_delta_trader.py`) is launched directly, it now attempts to load the latest Streamlit UI settings. If `ui_config_snapshot.json` (or legacy `ui_overrides.json`) exists, those values are converted into the legacy `TradingConfig` before trading begins. Delete those files or relaunch via Streamlit to reset to defaults.

## Streamlit Control Panel

A browser UI is available via `streamlit_app.py`. It lets you:

- Load configuration presets and tweak parameters before launching trades.
- Choose the target options expiry date from a dropdown (or fall back to defaults) before launching trades.
- Apply overrides that persist to `ui_overrides.json` and `.env` values.
- Monitor live metrics (current P&L, premium collected, trailing SL level, leg details, and latest spot price) directly on the dashboard.
- Use **Stop & Exit Positions** to immediately close both legs and halt the running strategy; the button replaces the previous "Force Exit" label for clarity.
- Start the strategy, monitor runtime status, view trailing P&L/log output, and trigger a force exit.

### Prerequisites

- Install dependencies once (already in `requirements.txt`):

```bash
/home/naba/crypto_delta_algo/.venv/bin/pip install -r requirements.txt
```

- Ensure `.env` contains valid Delta Exchange API credentials before enabling live mode.

### Launching the UI

```bash
/home/naba/crypto_delta_algo/.venv/bin/streamlit run streamlit_app.py --server.headless true
```

Open the printed URL (default `http://localhost:8501`) to access the dashboard. The sidebar controls trade parameters; the main pane shows runner status, summary logs, and detailed logs.

> **Safety tip:** Keep `Dry Run Mode` checked while testing. Streamlit will prevent live execution if the configuration fails validation.

## Observed Limitations / Future Enhancements

- Database persistence is stubbed—needs integration with the intended Django backend.
- Risk thresholds and timing are static—could benefit from dynamic adjustments based on volatility or account size.
- WebSocket error handling is robust but lacks adaptive channel management based on active symbols.
- Strategy currently assumes symmetrical quantities for call/put legs; portfolio-level hedging or scaling is not implemented.
- No automated unit test suite is bundled; behavior is validated via live dry runs.

## Summary Snapshot

The script demonstrates a mature trading workflow suitable for production use with Delta Exchange options. Key strengths include thorough risk checks, resilient order handling, and detailed observability. Adopters should validate credentials, confirm configuration defaults (especially `dry_run`), and consider augmenting the TODO placeholders (database write, enhanced monitoring) before live deployment.
