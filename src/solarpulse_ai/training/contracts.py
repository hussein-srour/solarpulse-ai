"""Phase 5 manifest loading and feature-contract validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

TARGET = "ac_energy_kwh"
KEYS = ("timestamp", "site_id")
REQUIRED_METADATA = (
    "split",
    "feature_eligible",
    "feature_missing_count",
    "feature_missing_reasons",
)
FORBIDDEN_PREDICTORS = {
    *KEYS,
    TARGET,
    *REQUIRED_METADATA,
    "row_id",
    "record_id",
    "source_row",
}


@dataclass(frozen=True, slots=True)
class TrainingContract:
    """Validated column roles copied from the Phase 5 manifest."""

    predictors: tuple[str, ...]
    numerical: tuple[str, ...]
    categorical: tuple[str, ...]
    boolean: tuple[str, ...]
    metadata: tuple[str, ...]
    forecast_horizon_hours: int


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read a valid JSON object."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid feature manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("feature manifest must contain a JSON object")
    return payload


def validate_contract(frame: pd.DataFrame, manifest: dict[str, Any]) -> TrainingContract:
    """Reject contract drift, undeclared inputs, and leakage-prone predictors."""
    if manifest.get("target_column") != TARGET:
        raise ValueError(f"feature manifest target_column must be {TARGET}")
    if tuple(manifest.get("key_columns", ())) != KEYS:
        raise ValueError(f"feature manifest key_columns must be {list(KEYS)}")
    predictors = _string_tuple(manifest, "predictor_columns")
    if not predictors:
        raise ValueError("feature manifest declares no predictors")
    if len(set(predictors)) != len(predictors):
        raise ValueError("duplicate predictor names in feature manifest")
    forbidden = sorted(set(predictors) & FORBIDDEN_PREDICTORS)
    if forbidden:
        raise ValueError(f"leakage or metadata predictors are forbidden: {forbidden}")
    required = {*KEYS, TARGET, *REQUIRED_METADATA, *predictors}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"feature dataset is missing declared columns: {missing}")

    metadata = _string_tuple(manifest, "metadata_columns")
    if not set(REQUIRED_METADATA).issubset(metadata):
        raise ValueError("feature manifest metadata_columns omit required training metadata")
    if set(metadata) & set(predictors):
        raise ValueError("metadata columns must be excluded from predictors")
    horizon = manifest.get("forecast_horizon_hours")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("feature manifest forecast_horizon_hours must be a positive integer")
    if horizon != 24:
        raise ValueError("Phase 6 baseline training requires the default 24-hour horizon")

    categorical = _string_tuple(manifest, "categorical_columns")
    numerical = _string_tuple(manifest, "numerical_columns")
    boolean = tuple(str(value) for value in manifest.get("boolean_columns", ()))
    classified = [*numerical, *categorical, *boolean]
    if len(set(classified)) != len(classified):
        raise ValueError("predictor type groups overlap or contain duplicates")
    if set(classified) != set(predictors):
        absent = sorted(set(predictors) - set(classified))
        extra = sorted(set(classified) - set(predictors))
        raise ValueError(f"predictor type mismatch; unclassified={absent}, undeclared={extra}")
    undeclared_columns = set(frame.columns) - required
    expected_columns = set(manifest.get("expected_data_types", {}))
    if expected_columns and not {*KEYS, TARGET, *predictors}.issubset(expected_columns):
        raise ValueError("input feature columns do not match manifest expected_data_types")
    if int(manifest.get("output_row_count", len(frame))) != len(frame):
        raise ValueError("feature dataset row count disagrees with feature manifest")
    source_rows = manifest.get("source_row_count", len(frame))
    if not isinstance(source_rows, int) or source_rows < len(frame):
        raise ValueError("feature manifest source/output row counts are unreasonable")
    if undeclared_columns:
        allowed_extra = set(metadata) | set(REQUIRED_METADATA)
        unexpected = sorted(undeclared_columns - allowed_extra)
        if unexpected:
            raise ValueError(f"dataset contains undeclared columns: {unexpected}")
    return TrainingContract(
        predictors,
        numerical,
        categorical,
        boolean,
        metadata,
        horizon,
    )


def _string_tuple(manifest: dict[str, Any], key: str) -> tuple[str, ...]:
    value = manifest.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"feature manifest {key} must be a list of strings")
    return tuple(value)
