#!/usr/bin/env python3
"""Utility helpers for loading Delta Trader configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from trading_config import (
    PRESET_FACTORIES,
    TradingConfiguration,
    create_default_config,
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def load_config(preset: str = "default") -> TradingConfiguration:
    if preset not in PRESET_FACTORIES:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESET_FACTORIES)}")
    return PRESET_FACTORIES[preset]()


def apply_config_override(config: TradingConfiguration, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) != 2:
        raise ValueError("Override key must be of the form 'section.attribute'")
    section, attribute = parts
    if not hasattr(config, section):
        raise AttributeError(f"Unknown configuration section '{section}'")
    section_obj = getattr(config, section)
    if not hasattr(section_obj, attribute):
        raise AttributeError(f"Unknown attribute '{attribute}' in section '{section}'")
    setattr(section_obj, attribute, value)


def validate_config(config: TradingConfiguration) -> bool:
    try:
        config._validate()
        return True
    except Exception as exc:  # pragma: no cover - used for UI feedback
        print(f"Configuration validation failed: {exc}")
        return False


def load_config_for_trading(preset: str = "default", **overrides: Any):
    config = load_config(preset)
    for key, value in overrides.items():
        apply_config_override(config, key, value)
    config._validate()
    return config.to_legacy_config()


def save_config_snapshot(config: TradingConfiguration, path: Path) -> None:
    snapshot = {
        "strategy": config.strategy.__dict__,
        "timing": config.timing.__dict__,
        "risk": config.risk.__dict__,
        "order": config.order.__dict__,
        "system": config.system.__dict__,
        "logging": config.logging.__dict__,
        "notifications": config.notifications.__dict__,
    }
    path.write_text(json.dumps(snapshot, indent=2))


def load_config_snapshot(path: Path) -> TradingConfiguration:
    if not path.exists():
        return create_default_config()
    data: Dict[str, Dict[str, Any]] = json.loads(path.read_text())
    config = TradingConfiguration()
    for section, values in data.items():
        if hasattr(config, section):
            section_obj = getattr(config, section)
            for key, value in values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
    config._validate()
    return config
