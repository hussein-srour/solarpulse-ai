"""Validation and assignment of Phase 4 chronological split plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from solarpulse_ai.data.errors import DataValidationError, ValidationIssue

NAME_ALIASES = {"training": "train", "validation": "validation", "testing": "test"}


def load_split_plan(path: str | Path) -> dict[str, Any]:
    """Load a JSON object representing chronological periods."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataValidationError(
            [ValidationIssue("invalid_split_plan", f"Could not read split plan: {error}")]
        ) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("periods"), dict):
        raise DataValidationError(
            [ValidationIssue("invalid_split_plan", "Split plan must contain a periods object.")]
        )
    return cast(dict[str, Any], payload)


def assign_splits(dataframe: pd.DataFrame, plan: dict[str, Any]) -> pd.Series[str]:
    """Validate non-overlapping periods and label every dataset row."""
    periods = plan["periods"]
    normalized: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for source_name, output_name in NAME_ALIASES.items():
        period = periods.get(source_name)
        if not isinstance(period, dict):
            raise DataValidationError(
                [ValidationIssue("invalid_split_plan", f"Missing {source_name} period.")]
            )
        start_value = period.get("beginning_timestamp")
        end_value = period.get("ending_timestamp")
        if not isinstance(start_value, str) or not isinstance(end_value, str):
            raise DataValidationError(
                [ValidationIssue("invalid_split_plan", f"Invalid {source_name} boundaries.")]
            )
        start = pd.Timestamp(start_value)
        end = pd.Timestamp(end_value)
        if start.tzinfo is None or end.tzinfo is None:
            raise DataValidationError(
                [ValidationIssue("invalid_split_plan", "Split boundaries must be timezone-aware.")]
            )
        start = start.tz_convert("UTC")
        end = end.tz_convert("UTC")
        if pd.isna(start) or pd.isna(end) or start > end:
            raise DataValidationError(
                [ValidationIssue("invalid_split_plan", f"Invalid {source_name} boundaries.")]
            )
        normalized.append((output_name, start, end))
    if any(normalized[index][2] >= normalized[index + 1][1] for index in range(2)):
        raise DataValidationError(
            [
                ValidationIssue(
                    "invalid_split_plan", "Split periods overlap or are not chronological."
                )
            ]
        )
    labels = pd.Series(pd.NA, index=dataframe.index, dtype="string")
    for name, start, end in normalized:
        labels.loc[dataframe["timestamp"].between(start, end, inclusive="both")] = name
    if labels.isna().any():
        raise DataValidationError(
            [
                ValidationIssue(
                    "split_plan_dataset_mismatch",
                    "Every input timestamp must fall within exactly one split period.",
                )
            ]
        )
    for source_name, output_name in NAME_ALIASES.items():
        period = periods[source_name]
        selected = dataframe.loc[labels.eq(output_name)]
        planned_count = period.get("record_count")
        if planned_count is not None and (
            not isinstance(planned_count, int) or planned_count != len(selected)
        ):
            raise DataValidationError(
                [
                    ValidationIssue(
                        "split_plan_dataset_mismatch",
                        f"{source_name} record_count does not match the input dataset.",
                    )
                ]
            )
        planned_sites = period.get("counts_by_site")
        if planned_sites is not None:
            actual_sites = {
                str(key): int(value)
                for key, value in selected.groupby("site_id", sort=True).size().items()
            }
            if not isinstance(planned_sites, dict) or planned_sites != actual_sites:
                raise DataValidationError(
                    [
                        ValidationIssue(
                            "split_plan_dataset_mismatch",
                            f"{source_name} counts_by_site does not match the input dataset.",
                        )
                    ]
                )
    return labels
