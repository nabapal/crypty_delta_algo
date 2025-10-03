#!/usr/bin/env python3
"""Streamlit control panel for the Production Delta Trader."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config_loader import (  # noqa: E402
    apply_config_override,
    load_config,
    load_config_for_trading,
    load_config_snapshot,
    save_config_snapshot,
    validate_config,
)
from production_delta_trader import ExitReason, ShortStrangleStrategy  # noqa: E402

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

APP_STATE_KEY = "strategy_runner"
CONFIG_STATE_KEY = "active_config"
PRESET_STATE_KEY = "selected_preset"
OVERRIDES_STATE_KEY = "config_overrides"
AUTOFRESH_ENABLED_KEY = "auto_refresh_enabled"
AUTOFRESH_INTERVAL_KEY = "auto_refresh_interval"
AUTH_STATE_KEY = "is_authenticated"
AUTH_ERROR_KEY = "auth_error"
LOG_LINES = 200

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_LOG_PATH = LOG_DIR / "delta_trader_summary.log"
DETAILED_LOG_PATH = LOG_DIR / "delta_trader_detailed.log"

TRADE_LEDGER_PATH = Path(os.getenv("TRADE_LEDGER_PATH", "storage/delta_trader_trades.jsonl"))
TRADE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
OVERRIDE_PATH = Path("ui_overrides.json")
SNAPSHOT_PATH = Path("ui_config_snapshot.json")
ENV_FILE = Path(".env")
EXPIRY_LOOKAHEAD_DAYS = 30
IST_OFFSET = timedelta(hours=5, minutes=30)

load_dotenv(ENV_FILE)


def _ensure_session_state() -> None:
    """Initialise Streamlit session state containers."""
    if APP_STATE_KEY not in st.session_state:
        st.session_state[APP_STATE_KEY] = StrategyRunner()
    if PRESET_STATE_KEY not in st.session_state:
        st.session_state[PRESET_STATE_KEY] = "default"
    if CONFIG_STATE_KEY not in st.session_state:
        if SNAPSHOT_PATH.exists():
            try:
                st.session_state[CONFIG_STATE_KEY] = load_config_snapshot(SNAPSHOT_PATH)
            except Exception:
                st.session_state[CONFIG_STATE_KEY] = load_config("default")
        else:
            st.session_state[CONFIG_STATE_KEY] = load_config("default")
    if OVERRIDES_STATE_KEY not in st.session_state:
        overrides: Dict[str, Any] = {}
        if OVERRIDE_PATH.exists():
            try:
                overrides = json.loads(OVERRIDE_PATH.read_text())
            except Exception:
                overrides = {}
        st.session_state[OVERRIDES_STATE_KEY] = overrides
        if overrides:
            try:
                apply_overrides(st.session_state[CONFIG_STATE_KEY], overrides)
            except Exception:
                st.session_state[OVERRIDES_STATE_KEY] = {}
    if AUTOFRESH_ENABLED_KEY not in st.session_state:
        st.session_state[AUTOFRESH_ENABLED_KEY] = True
    if AUTOFRESH_INTERVAL_KEY not in st.session_state:
        st.session_state[AUTOFRESH_INTERVAL_KEY] = 5
    if AUTH_STATE_KEY not in st.session_state:
        st.session_state[AUTH_STATE_KEY] = False
    if AUTH_ERROR_KEY not in st.session_state:
        st.session_state[AUTH_ERROR_KEY] = None


def _get_ist_now() -> datetime:
    return datetime.now(timezone.utc) + IST_OFFSET


def _trigger_rerun() -> None:
    rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    fallback = getattr(st, "rerun", None)
    if callable(fallback):
        fallback()


def purge_old_logs(days: int = 7) -> int:
    """Delete .log files older than the specified number of days from LOG_DIR."""
    if not LOG_DIR.exists():
        return 0

    cutoff = time.time() - days * 86400
    removed = 0

    for log_file in LOG_DIR.glob("*.log"):
        try:
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink(missing_ok=True)
                removed += 1
        except FileNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - defensive logging
            logger = logging.getLogger(__name__)
            logger.warning("Failed to delete log %s: %s", log_file, exc)
    return removed


def generate_expiry_options(lookahead_days: int = EXPIRY_LOOKAHEAD_DAYS) -> Iterable[str]:
    today = _get_ist_now().date()
    options = []
    for offset in range(lookahead_days):
        candidate = today + timedelta(days=offset)
        formatted = candidate.strftime("%d-%m-%Y")
        options.append(formatted)
    return options


# ---------------------------------------------------------------------------
# Strategy runner abstraction
# ---------------------------------------------------------------------------


class StrategyRunner:
    """Manage the lifecycle of a single ShortStrangleStrategy instance."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._strategy: Optional[ShortStrangleStrategy] = None
        self._lock = threading.Lock()
        self._is_active = False
        self._status_message = "Idle"
        self._last_exit_reason: Optional[str] = None
        self._last_snapshot: Dict[str, Any] = {}
        self._last_spot_price: Optional[float] = None
        self._last_spot_timestamp: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._is_active and self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> str:
        return self._status_message

    @property
    def last_exit_reason(self) -> Optional[str]:
        return self._last_exit_reason

    def start(self, config) -> None:
        with self._lock:
            if self.is_running:
                raise RuntimeError("Strategy already running")

            self._status_message = "Initialising strategy"
            self._strategy = ShortStrangleStrategy(config)

            self._thread = threading.Thread(
                target=self._run_strategy,
                name="ShortStrangleStrategyThread",
                daemon=True,
            )
            self._is_active = True
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._status_message = "Force exit requested"
            if self._strategy:
                try:
                    self._strategy.force_exit_position()
                    self._last_exit_reason = ExitReason.FORCE_CLOSE.value
                    self._status_message = "Force exit signalled"
                except Exception as exc:  # pragma: no cover - defensive logging
                    self._status_message = f"Force exit failed: {exc}"
            self._is_active = False

    def get_snapshot(self) -> Dict[str, Any]:
        strategy_ref: Optional[ShortStrangleStrategy]
        with self._lock:
            strategy_ref = self._strategy
            snapshot: Dict[str, Any] = {
                "is_running": self.is_running,
                "status": self._status_message,
                "last_exit_reason": self._last_exit_reason,
                "spot_price": self._last_spot_price,
            }

            if strategy_ref and hasattr(strategy_ref, "state"):
                state = strategy_ref.state
                snapshot.update(
                    {
                        "current_pnl": getattr(state, "current_pnl", None),
                        "total_premium_received": getattr(state, "total_premium_received", None),
                        "current_pnl_pct": getattr(state, "current_pnl_pct", None),
                        "trailing_sl_level": getattr(state, "trailing_sl_level", None),
                    }
                )

                def _position_summary(position: Optional[Any]) -> Optional[Dict[str, Any]]:
                    if not position:
                        return None
                    side_obj = getattr(position, "side", None)
                    side_value = side_obj.value if side_obj is not None else None
                    return {
                        "symbol": getattr(position, "symbol", None),
                        "side": side_value,
                        "quantity": getattr(position, "quantity", None),
                        "entry_price": getattr(position, "entry_price", None),
                        "current_price": getattr(position, "current_price", None),
                        "unrealized_pnl": getattr(position, "unrealized_pnl", None),
                    }

                snapshot["call_position"] = _position_summary(getattr(state, "call_position", None))
                snapshot["put_position"] = _position_summary(getattr(state, "put_position", None))
            else:
                snapshot.update(
                    {
                        "current_pnl": None,
                        "current_pnl_pct": None,
                        "total_premium_received": None,
                        "trailing_sl_level": None,
                        "call_position": None,
                        "put_position": None,
                    }
                )

            self._last_snapshot = snapshot

        self._refresh_spot_price(strategy_ref)

        with self._lock:
            self._last_snapshot["spot_price"] = self._last_spot_price
            return dict(self._last_snapshot)

    def _refresh_spot_price(self, strategy_ref: Optional[ShortStrangleStrategy]) -> None:
        if not strategy_ref:
            return
        if time.time() - self._last_spot_timestamp < 15 and self._last_spot_price is not None:
            return

        try:
            symbol_candidates = []
            state = getattr(strategy_ref, "state", None)
            if state and state.call_position and getattr(state.call_position, "symbol", None):
                symbol_candidates.append(state.call_position.symbol)
            underlying = getattr(strategy_ref.config, "underlying", "BTC").upper()
            symbol_candidates.extend([f"{underlying}USD", f"{underlying}USDT"])

            for ticker_symbol in symbol_candidates:
                success, payload = strategy_ref.api._make_public_request(f"/v2/tickers/{ticker_symbol}")
                if success:
                    result = payload.get("result", {}) if isinstance(payload, dict) else {}
                    spot = result.get("spot_price") or result.get("mark_price")
                    if spot:
                        with self._lock:
                            self._last_spot_price = float(spot)
                            self._last_spot_timestamp = time.time()
                        return
        except Exception:
            # Silent fail; UI can fall back to previous spot price
            pass

    def _run_strategy(self) -> None:
        assert self._strategy is not None
        try:
            self._status_message = "Validating prerequisites"
            if not self._strategy.validate_prerequisites():
                self._status_message = "Prerequisite check failed"
                self._is_active = False
                return

            resume_existing = bool(getattr(self._strategy.state, "is_active", False))

            if resume_existing:
                self._status_message = "Resuming monitoring"
            else:
                self._status_message = "Waiting for trade window"
                self._strategy.wait_until_trade_window()

                self._status_message = "Entering short strangle"
                entered = self._strategy.enter_short_strangle()
                if not entered:
                    self._status_message = "Failed to enter position"
                    self._is_active = False
                    return

            self._status_message = "Monitoring position"
            self._strategy.run_live_monitoring(duration_minutes=60)
            state = self._strategy.state
            if state.exit_reason:
                self._last_exit_reason = state.exit_reason.value
            self._status_message = "Monitoring complete"
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            self._status_message = f"Strategy error: {exc}"
        finally:
            self._is_active = False
            # Capture latest metrics after completion
            self.get_snapshot()


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def load_preset(preset: str):
    config = load_config(preset)
    st.session_state[PRESET_STATE_KEY] = preset
    st.session_state[CONFIG_STATE_KEY] = config
    st.session_state[OVERRIDES_STATE_KEY] = {}
    try:
        save_config_snapshot(config, SNAPSHOT_PATH)
    except Exception:
        pass
    return config


def apply_overrides(config, overrides: Dict[str, Any]):
    for key, value in overrides.items():
        apply_config_override(config, key, value)


def persist_overrides(overrides: Dict[str, Any]) -> None:
    if not overrides:
        if OVERRIDE_PATH.exists():
            try:
                OVERRIDE_PATH.unlink()
            except Exception:
                pass
        return
    OVERRIDE_PATH.write_text(json.dumps(overrides, indent=2))


def read_log_tail(log_path: Path, lines: int = LOG_LINES, newest_first: bool = False) -> str:
    if not log_path.exists():
        return "Log file not found."
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        content = handle.readlines()
    tail = content[-lines:]
    if newest_first:
        tail = list(reversed(tail))
    return "".join(tail)


def load_trade_history(limit: int = 200) -> list[Dict[str, Any]]:
    """Read structured trade summaries from the ledger file."""

    if not TRADE_LEDGER_PATH.exists():
        return []

    entries: list[Dict[str, Any]] = []
    with TRADE_LEDGER_PATH.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(payload)

    if not entries:
        return []

    trimmed = entries[-limit:]
    trimmed.reverse()  # Newest first for UI display
    return trimmed


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    normalised = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalised)
    except ValueError:
        return None


def format_timestamp(value: Optional[str]) -> str:
    dt = parse_iso_datetime(value)
    if not dt:
        return "—"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"

    sign = "-" if seconds < 0 else ""
    remaining = int(abs(seconds))
    hours, rem = divmod(remaining, 3600)
    minutes, secs = divmod(rem, 60)

    if hours:
        return f"{sign}{hours}h {minutes}m"
    if minutes:
        return f"{sign}{minutes}m {secs}s"
    return f"{sign}{secs}s"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Delta Trader Control", layout="wide")
    _ensure_session_state()
    runner: StrategyRunner = st.session_state[APP_STATE_KEY]

    if not st.session_state[AUTH_STATE_KEY]:
        st.title("Delta Trader Control")
        st.subheader("Admin Login")

        if st.session_state[AUTH_ERROR_KEY]:
            st.error(st.session_state[AUTH_ERROR_KEY])

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            if username == "admin" and password == "admin":
                st.session_state[AUTH_STATE_KEY] = True
                st.session_state[AUTH_ERROR_KEY] = None
                _trigger_rerun()
            else:
                st.session_state[AUTH_ERROR_KEY] = "Invalid username or password."
                _trigger_rerun()

        st.caption("Use the admin credentials to access the control panel.")
        st.stop()

    if st.sidebar.button("Log out", key="logout_button"):
        st.session_state[AUTH_STATE_KEY] = False
        st.session_state[AUTH_ERROR_KEY] = None
        st.session_state.pop("login_username", None)
        st.session_state.pop("login_password", None)
        _trigger_rerun()

    st.sidebar.header("Configuration")
    
    # Add compact header with key metrics
    snapshot = runner.get_snapshot()
    pnl_value = snapshot.get("current_pnl") or 0.0
    pnl_pct_value = snapshot.get("current_pnl_pct") or 0.0
    
    # Compact header row
    st.markdown("### Delta Trader Control Panel")
    header_cols = st.columns([1, 1, 1, 1])
    
    with header_cols[0]:
        status_color = "🟢" if runner.is_running else "🔴"
        st.markdown(f"**Status:** {status_color} {'Running' if runner.is_running else 'Idle'}")
    with header_cols[1]:
        st.markdown(f"**P&L:** ${pnl_value:.3f}")
    with header_cols[2]:
        st.markdown(f"**P&L %:** {pnl_pct_value*100:.1f}%")
    with header_cols[3]:
        trailing_value = snapshot.get("trailing_sl_level") or 0.0
        st.markdown(f"**Trail SL:** {trailing_value*100:.1f}%")
    
    st.divider()
    
    preset = st.sidebar.selectbox(
        "Preset",
        options=["default", "conservative", "aggressive", "development"],
        index=["default", "conservative", "aggressive", "development"].index(
            st.session_state[PRESET_STATE_KEY]
        ),
        help="Choose a baseline configuration preset.",
    )

    if preset != st.session_state[PRESET_STATE_KEY]:
        config = load_preset(preset)
    else:
        config = st.session_state[CONFIG_STATE_KEY]

    overrides = st.session_state[OVERRIDES_STATE_KEY]

    with st.sidebar.expander("Monitoring Options", expanded=False):
        auto_refresh_enabled = st.checkbox(
            "Auto refresh metrics",
            value=st.session_state[AUTOFRESH_ENABLED_KEY],
            help="When enabled, the dashboard refreshes itself while the strategy is running.",
        )
        refresh_interval = st.slider(
            "Refresh interval (seconds)",
            min_value=2,
            max_value=60,
            value=int(st.session_state[AUTOFRESH_INTERVAL_KEY]),
            help="How often the dashboard should refresh live metrics.",
        )
        manual_refresh = st.button(
            "Refresh snapshot now",
            help="Force an immediate snapshot refresh without waiting for the next cycle.",
        )

    st.session_state[AUTOFRESH_ENABLED_KEY] = auto_refresh_enabled
    st.session_state[AUTOFRESH_INTERVAL_KEY] = refresh_interval

    with st.sidebar.expander("Trading Parameters", expanded=True):
        # Row 1: Core settings
        col1, col2 = st.columns(2)
        with col1:
            underlying = st.selectbox("Underlying", options=["BTC", "ETH"], index=["BTC", "ETH"].index(config.strategy.underlying))
            quantity = st.number_input("Contracts", min_value=1, max_value=25, value=int(config.strategy.quantity))
        with col2:
            delta_low = st.number_input("Delta Low", min_value=0.01, max_value=0.50, value=float(config.strategy.delta_range_low), step=0.01)
            delta_high = st.number_input("Delta High", min_value=delta_low, max_value=0.50, value=float(config.strategy.delta_range_high), step=0.01)
        
        # Row 2: Timing
        col3, col4 = st.columns(2)
        with col3:
            trade_time = st.text_input("Trade Time", value=config.timing.trade_time_ist, placeholder="HH:MM")
        with col4:
            exit_time = st.text_input("Exit Time", value=config.timing.exit_time_ist, placeholder="HH:MM")
        
        # Expiry selection
        expiry_options = list(generate_expiry_options())
        current_expiry = config.strategy.expiry_date or expiry_options[0]
        if current_expiry not in expiry_options:
            expiry_options.insert(0, current_expiry)
        expiry_date = st.selectbox("Expiry Date", options=expiry_options, index=expiry_options.index(current_expiry))

    with st.sidebar.expander("Risk Controls", expanded=False):
        # Compact risk controls in columns
        col5, col6 = st.columns(2)
        with col5:
            max_loss = st.slider("Max Loss %", min_value=10, max_value=200, value=int(config.risk.max_loss_pct * 100))
        with col6:
            max_profit = st.slider("Max Profit %", min_value=10, max_value=200, value=int(config.risk.max_profit_pct * 100))
        
        trailing_enabled = st.checkbox("Enable Trailing SL", value=config.risk.trailing_sl_enabled)
        if trailing_enabled:
            trailing_rules_json = st.text_area("Trailing Rules", value=json.dumps(config.risk.trailing_rules), height=80)
        else:
            trailing_rules_json = json.dumps(config.risk.trailing_rules)

    with st.sidebar.expander("System Settings", expanded=False):
        # Compact system settings
        col7, col8 = st.columns(2)
        with col7:
            dry_run = st.checkbox("Dry Run", value=config.system.dry_run)
            testnet = st.checkbox("Testnet", value=config.system.testnet)
        with col8:
            websocket_enabled = st.checkbox("WebSocket", value=config.system.websocket_enabled)
            order_timeout = st.number_input("Timeout(s)", min_value=10, max_value=300, value=int(config.order.order_timeout))
        
        retry_slippage = st.number_input("Retry Slippage (%)", min_value=0.0, max_value=5.0, value=float(config.order.retry_slippage_pct * 100), step=0.25)

    override_btn = st.sidebar.button("Apply Overrides", type="primary")
    if override_btn:
        overrides = {
            "strategy.underlying": underlying,
            "strategy.delta_range_low": float(delta_low),
            "strategy.delta_range_high": float(delta_high),
            "strategy.quantity": int(quantity),
            "strategy.expiry_date": expiry_date,
            "timing.trade_time_ist": trade_time,
            "timing.exit_time_ist": exit_time,
            "risk.max_loss_pct": float(max_loss) / 100.0,
            "risk.max_profit_pct": float(max_profit) / 100.0,
            "risk.trailing_sl_enabled": bool(trailing_enabled),
            "risk.trailing_rules": json.loads(trailing_rules_json or "{}"),
            "system.dry_run": bool(dry_run),
            "system.testnet": bool(testnet),
            "system.websocket_enabled": bool(websocket_enabled),
            "order.order_timeout": int(order_timeout),
            "order.retry_slippage_pct": float(retry_slippage) / 100.0,
        }
        apply_overrides(config, overrides)
        st.session_state[OVERRIDES_STATE_KEY] = overrides
        persist_overrides(overrides)
        try:
            save_config_snapshot(config, SNAPSHOT_PATH)
        except Exception:
            pass
        st.success("Overrides applied. Remember to restart the strategy to apply runtime changes.")

    def fmt_currency(value: Optional[float]) -> str:
        if value is None:
            return "—"
        return f"${value:,.2f}"

    def fmt_quantity(value: Optional[float]) -> str:
        if value is None:
            return "—"
        return f"{value:.2f}"

    def fmt_percent(value: Optional[float]) -> str:
        if value is None:
            return "—"
        return f"{value:.1%}"

    def fmt_percent_value(value: Optional[float]) -> str:
        if value is None:
            return "—"
        return f"{value:.2f}%"

    valid = validate_config(config)
    if valid:
        st.sidebar.success("Configuration valid")
    else:
        st.sidebar.error("Configuration invalid—check inputs")

    st.title("Delta Trader Control Panel")
    st.caption("Manage the Production Short Strangle strategy from the browser.")

    snapshot = runner.get_snapshot()
    pnl_value = snapshot.get("current_pnl")
    pnl_pct_value = snapshot.get("current_pnl_pct")
    premium_value = snapshot.get("total_premium_received")
    trailing_value = snapshot.get("trailing_sl_level")
    spot_value = snapshot.get("spot_price")
    call_leg = snapshot.get("call_position")
    put_leg = snapshot.get("put_position")
    underlying_label = f"{config.strategy.underlying} Spot (USD)"

    tab_dashboard, tab_trades, tab_logs = st.tabs(["Live Dashboard", "Trade History", "Logs"])

    with tab_dashboard:
        # Compact Status Row
        st.subheader("Status")
        status_cols = st.columns([1, 1, 1, 1])
        
        with status_cols[0]:
            st.metric("Runner State", "Running" if runner.is_running else "Idle")
            if runner.last_exit_reason:
                st.caption(f"Last exit: {runner.last_exit_reason}")
                
        with status_cols[1]:
            st.metric("Current P&L (USD)", fmt_currency(pnl_value))
            st.metric("Current P&L (%)", fmt_percent(pnl_pct_value))
            
        with status_cols[2]:
            st.metric("Premium Collected", fmt_currency(premium_value))
            st.metric(underlying_label, fmt_currency(spot_value))
            
        with status_cols[3]:
            st.metric("Trailing SL Level", fmt_percent(trailing_value))
            # Calculate and display max loss/profit amounts
            if premium_value:
                max_loss_amount = premium_value * config.risk.max_loss_pct
                max_profit_amount = premium_value * config.risk.max_profit_pct
                st.metric("Max Loss Amount", fmt_currency(-max_loss_amount))
                st.metric("Max Profit Amount", fmt_currency(max_profit_amount))

        # Compact Controls Section
        st.divider()
        ctrl_cols = st.columns([1, 1, 2])
        
        with ctrl_cols[0]:
            start_disabled = runner.is_running or not valid
            if st.button("Start Strategy", disabled=start_disabled, width="stretch"):
                try:
                    try:
                        save_config_snapshot(config, SNAPSHOT_PATH)
                    except Exception:
                        pass
                    legacy_config = load_config_for_trading(preset, **overrides)
                    runner.start(legacy_config)
                    st.toast("Strategy execution started", icon="✅")
                except Exception as exc:
                    st.error(f"Failed to start strategy: {exc}")
                    
        with ctrl_cols[1]:
            stop_disabled = not runner.is_running
            if st.button("Stop & Exit Positions", disabled=stop_disabled, width="stretch",
                        help="Close both legs immediately and end the strategy run."):
                runner.stop()
                st.toast("Stop requested — exit order sent", icon="🛑")
                
        with ctrl_cols[2]:
            if runner.is_running:
                st.info("To stop the automation, press **Stop & Exit Positions**. This will close open legs and halt monitoring.")
            else:
                st.caption("Strategy is idle. Adjust settings and press Start to launch a new run.")

        # Compact Position Display
        if call_leg or put_leg:
            st.divider()
            st.subheader("Open Positions")
            position_rows = []
            for leg_label, data in ("Call", call_leg), ("Put", put_leg):
                if not data:
                    continue
                position_rows.append(
                    {
                        "Leg": leg_label,
                        "Symbol": data.get("symbol") or "—",
                        "Side": (data.get("side") or "").upper() or "—",
                        "Qty": fmt_quantity(data.get("quantity")),
                        "Entry": fmt_currency(data.get("entry_price")),
                        "Mark": fmt_currency(data.get("current_price")),
                        "P&L": fmt_currency(data.get("unrealized_pnl")),
                    }
                )
            if position_rows:
                st.dataframe(position_rows, width="stretch", hide_index=True)
            else:
                st.caption("Waiting for fills — no active option legs yet.")
        else:
            st.caption("No open option legs.")

    with tab_trades:
        st.subheader("Completed Trades")
        trade_history = load_trade_history()

        if not trade_history:
            st.info("No completed trades recorded yet.")
        else:
            total_trades = len(trade_history)
            cumulative_pnl = sum(float(trade.get("realized_pnl_usd") or 0.0) for trade in trade_history)
            wins = sum(1 for trade in trade_history if float(trade.get("realized_pnl_usd") or 0.0) > 0)
            win_rate = (wins / total_trades * 100) if total_trades else None
            average_duration = (
                sum(float(trade.get("duration_seconds") or 0.0) for trade in trade_history) / total_trades
                if total_trades
                else None
            )

            metrics = st.columns(4)
            metrics[0].metric("Recorded Trades", total_trades)
            metrics[1].metric("Win Rate", fmt_percent_value(win_rate))
            metrics[2].metric("Net P&L (USD)", fmt_currency(cumulative_pnl))
            metrics[3].metric("Avg Duration", format_duration(average_duration))

            summary_rows = []
            for trade in trade_history:
                trade_id = trade.get("strategy_id") or trade.get("trade_id") or "—"
                summary_rows.append(
                    {
                        "Trade": trade_id,
                        "Entry (UTC)": format_timestamp(trade.get("entry_time_utc")),
                        "Exit (UTC)": format_timestamp(trade.get("exit_time_utc")),
                        "Exit Reason": trade.get("exit_reason") or "—",
                        "P&L (USD)": fmt_currency(trade.get("realized_pnl_usd")),
                        "Return %": fmt_percent_value(trade.get("return_pct")),
                        "Duration": format_duration(trade.get("duration_seconds")),
                    }
                )

            st.dataframe(summary_rows, width="stretch")

            trade_indices = list(range(len(trade_history)))

            def trade_label(idx: int) -> str:
                trade = trade_history[idx]
                trade_id = trade.get("strategy_id") or trade.get("trade_id") or f"Trade #{len(trade_history) - idx}"
                reason = trade.get("exit_reason") or "—"
                exit_time_label = format_timestamp(trade.get("exit_time_utc"))
                return f"{trade_id} • {reason} • {exit_time_label}"

            selected_idx = st.selectbox("Select trade for details", trade_indices, format_func=trade_label)
            selected_trade = trade_history[selected_idx]

            detail_cols = st.columns(3)
            detail_cols[0].metric("P&L (USD)", fmt_currency(selected_trade.get("realized_pnl_usd")))
            detail_cols[1].metric("Return %", fmt_percent_value(selected_trade.get("return_pct")))
            detail_cols[2].metric("Premium", fmt_currency(selected_trade.get("premium_received_usd")))

            st.write(
                f"Exit reason: **{selected_trade.get('exit_reason', '—')}**"
            )
            st.write(
                f"Entry (IST): **{selected_trade.get('entry_time_ist', '—')}** | Exit (IST): **{selected_trade.get('exit_time_ist', '—')}**"
            )
            st.write(
                f"Entry (UTC): **{format_timestamp(selected_trade.get('entry_time_utc'))}** | Exit (UTC): **{format_timestamp(selected_trade.get('exit_time_utc'))}**"
            )
            st.write(f"Duration: **{format_duration(selected_trade.get('duration_seconds'))}**")

            legs = selected_trade.get("legs", {}) or {}
            if legs:
                st.markdown("#### Leg Breakdown")
                leg_rows = []
                for leg_key in ("call", "put"):
                    leg = legs.get(leg_key)
                    if not leg:
                        continue
                    leg_rows.append(
                        {
                            "Leg": leg_key.capitalize(),
                            "Symbol": leg.get("symbol") or "—",
                            "Entry": fmt_currency(leg.get("entry_price")),
                            "Exit": fmt_currency(leg.get("exit_price")),
                            "Qty Closed": fmt_quantity(leg.get("exit_quantity") or leg.get("quantity")),
                            "P&L": fmt_currency(leg.get("realized_pnl_usd")),
                            "Status": leg.get("status", "—"),
                        }
                    )
                if leg_rows:
                    st.table(leg_rows)

                for leg_key, leg in legs.items():
                    if leg.get("status") == "failed" and leg.get("error"):
                        st.warning(f"{leg_key.capitalize()} leg error: {leg['error']}")

    with tab_logs:
        st.subheader("Logs & Maintenance")
        
        # Compact maintenance section
        col_log1, col_log2 = st.columns([1, 2])
        with col_log1:
            if st.button("Clean Old Logs", help="Delete log files older than 7 days"):
                removed = purge_old_logs(7)
                if removed:
                    st.success(f"Deleted {removed} file{'s' if removed != 1 else ''}")
                else:
                    st.info("No old files found")
                _trigger_rerun()
        with col_log2:
            current_utc = datetime.now(timezone.utc)
            st.caption(f"UTC: {current_utc.strftime('%Y-%m-%d %H:%M:%S')} • Latest entries first")

        # Compact log display
        log_tabs = st.tabs(["Summary", "Detailed"])
        
        with log_tabs[0]:
            summary_text = read_log_tail(SUMMARY_LOG_PATH, lines=50, newest_first=True)
            st.text_area("Summary Log", value=summary_text, height=250, label_visibility="collapsed")
            
        with log_tabs[1]:
            detailed_text = read_log_tail(DETAILED_LOG_PATH, lines=100, newest_first=True)
            st.text_area("Detailed Log", value=detailed_text, height=250, label_visibility="collapsed")

    st.divider()
    st.caption("Powered by Streamlit • Ensure credentials are set in .env before enabling live trading.")

    auto_refresh_enabled = st.session_state.get(AUTOFRESH_ENABLED_KEY, True)
    refresh_interval = int(st.session_state.get(AUTOFRESH_INTERVAL_KEY, 5))

    if manual_refresh:
        st.rerun()

    if runner.is_running and auto_refresh_enabled:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
