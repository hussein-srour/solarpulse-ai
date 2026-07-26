"""Validation and normalization for canonical hourly datasets."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import cast

import pandas as pd

from solarpulse_ai.data.errors import DataValidationError, ValidationIssue
from solarpulse_ai.data.schemas import NUMERIC_FIELDS, REQUIRED_COLUMNS

MAX_REPORTED_ROWS = 10
type TimestampValue = str | int | float | date | datetime


def _csv_rows(mask: pd.Series[bool]) -> tuple[int, ...]:
    """Translate a boolean mask into one-based CSV row numbers including the header."""
    positions = [position + 2 for position, invalid in enumerate(mask) if invalid]
    return tuple(positions[:MAX_REPORTED_ROWS])


def _range_message(minimum: float | None, maximum: float | None) -> str:
    if minimum is not None and maximum is not None:
        return f"Values must be between {minimum:g} and {maximum:g}, inclusive."
    if minimum is not None:
        return f"Values must be greater than or equal to {minimum:g}."
    if maximum is not None:
        return f"Values must be less than or equal to {maximum:g}."
    return "Values must be finite numbers."


def _has_timezone(value: object) -> bool:
    """Return whether a timestamp value explicitly supplies timezone information."""
    try:
        timestamp = pd.Timestamp(cast(TimestampValue, value))
    except (TypeError, ValueError):
        return False
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def validate_hourly_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate, normalize, and chronologically sort hourly observations.

    A validated copy is returned. Invalid records are never deleted or corrected.
    All discoverable field-level issues are reported together.
    """
    issues: list[ValidationIssue] = []

    if dataframe.empty:
        raise DataValidationError(
            [ValidationIssue("empty_dataset", "The CSV must contain at least one data row.")]
        )

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise DataValidationError(
            [
                ValidationIssue(
                    "missing_required_column",
                    "Add the required column to the CSV.",
                    column=column,
                )
                for column in missing_columns
            ]
        )

    validated = dataframe.copy()

    raw_site_ids = validated["site_id"]
    site_ids = raw_site_ids.astype("string")
    invalid_site_ids = raw_site_ids.isna() | site_ids.str.strip().eq("")
    if invalid_site_ids.any():
        issues.append(
            ValidationIssue(
                "invalid_site_id",
                "Values must be non-empty strings.",
                column="site_id",
                rows=_csv_rows(invalid_site_ids),
            )
        )
    validated["site_id"] = site_ids

    raw_timestamps = validated["timestamp"]
    timestamps = pd.to_datetime(raw_timestamps, errors="coerce", utc=True, format="mixed")
    timezone_present = raw_timestamps.map(_has_timezone)
    invalid_timestamps = timestamps.isna() | ~timezone_present
    if invalid_timestamps.any():
        issues.append(
            ValidationIssue(
                "invalid_timestamp",
                "Use a valid timezone-aware ISO 8601 datetime; offsets are converted to UTC.",
                column="timestamp",
                rows=_csv_rows(invalid_timestamps),
            )
        )
    validated["timestamp"] = timestamps

    for field in NUMERIC_FIELDS:
        if field.name not in validated.columns:
            continue

        raw_values = validated[field.name]
        numeric_values = pd.to_numeric(raw_values, errors="coerce")
        required_missing = (
            raw_values.isna() if field.required else pd.Series(False, index=validated.index)
        )
        invalid_types = raw_values.notna() & numeric_values.isna()
        non_finite = numeric_values.notna() & ~numeric_values.map(math.isfinite)
        invalid_numeric = required_missing | invalid_types | non_finite
        if invalid_numeric.any():
            issues.append(
                ValidationIssue(
                    "invalid_numeric_value",
                    "Values must be finite numbers and required values cannot be empty.",
                    column=field.name,
                    rows=_csv_rows(invalid_numeric),
                )
            )

        out_of_range = pd.Series(False, index=validated.index)
        if field.minimum is not None:
            out_of_range |= numeric_values < field.minimum
        if field.maximum is not None:
            out_of_range |= numeric_values > field.maximum
        if out_of_range.any():
            issues.append(
                ValidationIssue(
                    "value_out_of_range",
                    _range_message(field.minimum, field.maximum),
                    column=field.name,
                    rows=_csv_rows(out_of_range),
                )
            )
        validated[field.name] = numeric_values.astype(float)

    valid_keys = validated["timestamp"].notna() & validated["site_id"].notna()
    duplicates = validated.duplicated(subset=["site_id", "timestamp"], keep=False) & valid_keys
    if duplicates.any():
        issues.append(
            ValidationIssue(
                "duplicate_observation",
                "Each site_id and timestamp combination must be unique.",
                rows=_csv_rows(duplicates),
            )
        )

    if issues:
        raise DataValidationError(issues)

    return validated.sort_values("timestamp", kind="stable").reset_index(drop=True)
