"""Validation and CSV loading for measured solar-generation records."""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from solarpulse_ai.data.errors import DataValidationError, ValidationIssue
from solarpulse_ai.data.ingestion import read_hourly_csv

GENERATION_COLUMNS: tuple[str, ...] = ("timestamp", "site_id", "ac_energy_kwh")
type TimestampValue = str | int | float | date | datetime


def _has_timezone(value: object) -> bool:
    try:
        timestamp = pd.Timestamp(cast(TimestampValue, value))
    except (TypeError, ValueError):
        return False
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def _rows(mask: pd.Series[bool]) -> tuple[int, ...]:
    return tuple(position + 2 for position, invalid in enumerate(mask) if invalid)[:10]


def validate_generation_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate measured generation without dropping or correcting invalid rows."""
    if dataframe.empty:
        raise DataValidationError(
            [ValidationIssue("empty_generation", "Generation data must contain at least one row.")]
        )

    missing = [column for column in GENERATION_COLUMNS if column not in dataframe.columns]
    if missing:
        raise DataValidationError(
            [
                ValidationIssue(
                    "missing_generation_column",
                    "Add the required column to the measured-generation CSV.",
                    column=column,
                )
                for column in missing
            ]
        )

    validated = dataframe.loc[:, GENERATION_COLUMNS].copy()
    issues: list[ValidationIssue] = []

    raw_site_ids = validated["site_id"]
    site_ids = raw_site_ids.astype("string")
    invalid_sites = raw_site_ids.isna() | site_ids.str.strip().eq("")
    if invalid_sites.any():
        issues.append(
            ValidationIssue(
                "invalid_generation_site_id",
                "Generation site_id values must be non-empty strings.",
                column="site_id",
                rows=_rows(invalid_sites),
            )
        )
    validated["site_id"] = site_ids

    raw_timestamps = validated["timestamp"]
    timestamps = pd.to_datetime(raw_timestamps, errors="coerce", utc=True, format="mixed")
    invalid_timestamps = timestamps.isna() | ~raw_timestamps.map(_has_timezone)
    if invalid_timestamps.any():
        issues.append(
            ValidationIssue(
                "invalid_generation_timestamp",
                "Use a valid timezone-aware timestamp; offsets are converted to UTC.",
                column="timestamp",
                rows=_rows(invalid_timestamps),
            )
        )
    validated["timestamp"] = timestamps

    raw_energy = validated["ac_energy_kwh"]
    energy = pd.to_numeric(raw_energy, errors="coerce")
    invalid_energy = (
        raw_energy.isna()
        | energy.isna()
        | (energy.notna() & ~energy.map(math.isfinite))
        | (energy < 0)
    )
    if invalid_energy.any():
        issues.append(
            ValidationIssue(
                "invalid_generation_energy",
                "Measured ac_energy_kwh must be finite and greater than or equal to zero.",
                column="ac_energy_kwh",
                rows=_rows(invalid_energy),
            )
        )
    validated["ac_energy_kwh"] = energy.astype(float)

    valid_keys = validated["site_id"].notna() & validated["timestamp"].notna()
    duplicates = validated.duplicated(subset=["site_id", "timestamp"], keep=False) & valid_keys
    if duplicates.any():
        issues.append(
            ValidationIssue(
                "duplicate_generation_record",
                "Each generation site_id and UTC timestamp combination must be unique.",
                rows=_rows(duplicates),
            )
        )

    if issues:
        raise DataValidationError(issues)

    return validated.sort_values(["site_id", "timestamp"], kind="stable").reset_index(drop=True)


def read_generation_csv(path: str | Path) -> pd.DataFrame:
    """Read and validate a measured-generation CSV."""
    return validate_generation_dataframe(read_hourly_csv(path))
