#!/usr/bin/env python3
"""Centralised configuration objects for the Production Delta Trader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Core configuration sections
# ---------------------------------------------------------------------------


@dataclass
class StrategyConfig:
    underlying: str = "BTC"
    delta_range_low: float = 0.10
    delta_range_high: float = 0.15
    quantity: int = 1
    expiry_date: str = "01-10-2025"
    expiry_buffer_hours: int = 24
    contract_size: float = 0.001


@dataclass
class TimingConfig:
    trade_time_ist: str = "09:30"
    exit_time_ist: str = "15:00"
    position_check_interval: int = 5
    status_report_interval: int = 300


@dataclass
class RiskConfig:
    max_loss_pct: float = 0.80
    max_profit_pct: float = 0.80
    trailing_sl_enabled: bool = True
    trailing_rules: Dict[float, float] = field(
        default_factory=lambda: {0.40: 0.00, 0.50: 0.25}
    )
    max_daily_loss: float = 1000.0
    max_weekly_loss: float = 5000.0


@dataclass
class OrderConfig:
    max_retries: int = 4
    retry_slippage_pct: float = 0.02
    order_timeout: int = 60
    accept_partial_fills: bool = True
    min_fill_percentage: float = 0.5


@dataclass
class SystemConfig:
    dry_run: bool = False
    testnet: bool = False
    websocket_enabled: bool = True
    max_api_retries: int = 5
    websocket_reconnect_attempts: int = 10
    error_recovery_enabled: bool = True


@dataclass
class LoggingConfig:
    file_log_level: str = "DEBUG"
    console_log_level: str = "INFO"
    enable_metrics: bool = True


@dataclass
class NotificationConfig:
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False


# ---------------------------------------------------------------------------
# Composite configuration
# ---------------------------------------------------------------------------


@dataclass
class TradingConfiguration:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    order: OrderConfig = field(default_factory=OrderConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self.strategy.underlying not in {"BTC", "ETH"}:
            raise ValueError("underlying must be BTC or ETH")
        if not 0.01 <= self.strategy.delta_range_low <= 0.50:
            raise ValueError("delta_range_low must be between 0.01 and 0.50")
        if not self.strategy.delta_range_low <= self.strategy.delta_range_high <= 0.50:
            raise ValueError("delta_range_high must be between low and 0.50")
        if self.strategy.quantity < 1:
            raise ValueError("quantity must be at least 1 option contract")
        if self.strategy.contract_size <= 0:
            raise ValueError("contract_size must be positive")
        if not 0.10 <= self.risk.max_loss_pct <= 2.0:
            raise ValueError("max_loss_pct must be between 0.10 and 2.0")
        if not 0.10 <= self.risk.max_profit_pct <= 2.0:
            raise ValueError("max_profit_pct must be between 0.10 and 2.0")
        if self.order.min_fill_percentage < 0 or self.order.min_fill_percentage > 1:
            raise ValueError("min_fill_percentage must be between 0 and 1")
        if self.system.dry_run and not self.system.testnet:
            # Informational warning—no exception
            print("⚠️  Dry run enabled on production credentials.")

    # ------------------------------------------------------------------
    # Legacy compatibility helpers
    # ------------------------------------------------------------------

    def to_legacy_config(self):
        from production_delta_trader import TradingConfig as LegacyConfig

        legacy_kwargs = {
            "underlying": self.strategy.underlying,
            "delta_range_low": self.strategy.delta_range_low,
            "delta_range_high": self.strategy.delta_range_high,
            "trade_time_ist": self.timing.trade_time_ist,
            "exit_time_ist": self.timing.exit_time_ist,
            "expiry_date": self.strategy.expiry_date,
            "expiry_buffer_hours": self.strategy.expiry_buffer_hours,
            "quantity": self.strategy.quantity,
            "contract_size": self.strategy.contract_size,
            "max_loss_pct": self.risk.max_loss_pct,
            "max_profit_pct": self.risk.max_profit_pct,
            "trailing_sl_enabled": self.risk.trailing_sl_enabled,
            "trailing_rules": self.risk.trailing_rules,
            "max_retries": self.order.max_retries,
            "retry_slippage_pct": self.order.retry_slippage_pct,
            "order_timeout": self.order.order_timeout,
            "dry_run": self.system.dry_run,
            "testnet": self.system.testnet,
            "position_check_interval": self.timing.position_check_interval,
            "websocket_enabled": self.system.websocket_enabled,
            "log_level": self.logging.file_log_level,
            "metrics_enabled": self.logging.enable_metrics,
            "max_api_retries": self.system.max_api_retries,
            "websocket_reconnect_attempts": self.system.websocket_reconnect_attempts,
            "error_recovery_enabled": self.system.error_recovery_enabled,
        }
        return LegacyConfig(**legacy_kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Presets and factory helpers
# ---------------------------------------------------------------------------


def create_default_config() -> TradingConfiguration:
    return TradingConfiguration()


def create_conservative_config() -> TradingConfiguration:
    cfg = TradingConfiguration()
    cfg.strategy.delta_range_low = 0.08
    cfg.strategy.delta_range_high = 0.12
    cfg.system.dry_run = True
    cfg.system.testnet = True
    return cfg


def create_aggressive_config() -> TradingConfiguration:
    cfg = TradingConfiguration()
    cfg.strategy.delta_range_low = 0.15
    cfg.strategy.delta_range_high = 0.25
    cfg.strategy.quantity = 2
    cfg.risk.max_loss_pct = 1.0
    cfg.risk.max_profit_pct = 1.0
    cfg.system.dry_run = False
    return cfg


def create_development_config() -> TradingConfiguration:
    cfg = TradingConfiguration()
    cfg.system.dry_run = True
    cfg.system.testnet = True
    cfg.timing.position_check_interval = 10
    cfg.timing.status_report_interval = 600
    return cfg


PRESET_FACTORIES = {
    "default": create_default_config,
    "conservative": create_conservative_config,
    "aggressive": create_aggressive_config,
    "development": create_development_config,
}
