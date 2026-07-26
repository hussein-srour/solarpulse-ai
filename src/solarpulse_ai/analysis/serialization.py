"""Conversion helpers for standards-compliant JSON output."""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd

type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]


def to_json_value(value: object) -> JSONValue:
    """Recursively convert pandas and Python values to standard JSON values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return to_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
