"""Explicit feature-dataset column contract."""

from __future__ import annotations

from dataclasses import dataclass

KEY_COLUMNS = ("timestamp", "site_id")
TARGET_COLUMN = "ac_energy_kwh"
ELIGIBILITY_COLUMNS = (
    "feature_eligible",
    "feature_missing_count",
    "feature_missing_reasons",
)


@dataclass(frozen=True, slots=True)
class FeatureContract:
    """Named column roles independent of physical CSV ordering."""

    key_columns: tuple[str, ...]
    target_column: str
    predictor_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
