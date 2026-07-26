"""Explicit feature-row eligibility metadata."""

from __future__ import annotations

import pandas as pd

from solarpulse_ai.data.schemas import REQUIRED_COLUMNS


def add_eligibility(
    dataframe: pd.DataFrame, historical_failures: dict[str, pd.Series[bool]]
) -> pd.DataFrame:
    """Mark missing required source inputs and unavailable historical context."""
    result = dataframe.copy()
    reasons: list[list[str]] = [[] for _ in range(len(result))]
    required_weather = [
        column
        for column in REQUIRED_COLUMNS
        if column not in {"timestamp", "site_id", "ac_energy_kwh"}
    ]
    for column in required_weather:
        for position in result.index[result[column].isna()]:
            reasons[int(position)].append(f"missing_source_weather:{column}")
    for reason, mask in historical_failures.items():
        for position in mask.index[mask.fillna(True)]:
            reasons[int(position)].append(reason)
    result["feature_missing_count"] = [len(items) for items in reasons]
    result["feature_missing_reasons"] = [";".join(items) for items in reasons]
    result["feature_eligible"] = result["feature_missing_count"].eq(0)
    return result
