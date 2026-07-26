"""Feature lineage manifest and quality reporting."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from solarpulse_ai import __version__
from solarpulse_ai.data.schemas import FIELD_DEFINITIONS
from solarpulse_ai.features.config import FeatureConfig
from solarpulse_ai.features.contract import FeatureContract

UNITS = {field.name: field.unit for field in FIELD_DEFINITIONS}
UNITS.update(
    {
        "installed_capacity_kwp": "kWp",
        "panel_tilt_degrees": "degrees",
        "panel_azimuth_degrees": "degrees",
        "latitude": "degrees",
        "longitude": "degrees",
        "ghi_normalised_by_capacity": "W/m² per kWp",
    }
)


def build_manifest(
    dataframe: pd.DataFrame,
    audit_dataframe: pd.DataFrame,
    source_path: Path,
    source_rows: int,
    config: FeatureConfig,
    contract: FeatureContract,
    raw_features: list[str],
    engineered_features: list[str],
    excluded_rows: int,
) -> dict[str, Any]:
    """Build a record-free, JSON-serialisable lineage manifest."""
    descriptions = {column: _description(column) for column in contract.predictor_columns}
    return {
        "project_version": __version__,
        "generated_timestamp": datetime.now(UTC).isoformat(),
        "source_dataset_path": str(source_path.resolve()),
        "source_row_count": source_rows,
        "output_row_count": len(dataframe),
        "excluded_row_count": excluded_rows,
        "site_ids": sorted(str(value) for value in audit_dataframe["site_id"].unique()),
        "forecast_horizon_hours": config.forecast_horizon_hours,
        "feature_configuration": config.model_dump(mode="json"),
        "key_columns": list(contract.key_columns),
        "target_column": contract.target_column,
        "predictor_columns": list(contract.predictor_columns),
        "metadata_columns": list(contract.metadata_columns),
        "categorical_columns": list(contract.categorical_columns),
        "numerical_columns": list(contract.numerical_columns),
        "raw_features": raw_features,
        "engineered_features": engineered_features,
        "units": {name: UNITS.get(name) for name in contract.predictor_columns},
        "feature_descriptions": descriptions,
        "expected_data_types": {
            column: str(dataframe[column].dtype)
            for column in (
                *contract.key_columns,
                contract.target_column,
                *contract.predictor_columns,
            )
        },
        "missing_value_counts": {
            column: int(dataframe[column].isna().sum()) for column in contract.predictor_columns
        },
        "eligibility_counts": _counts(audit_dataframe, "feature_eligible"),
        "split_counts": (_counts(audit_dataframe, "split") if "split" in audit_dataframe else {}),
        "earliest_timestamp": _timestamp_or_none(audit_dataframe["timestamp"].min()),
        "latest_timestamp": _timestamp_or_none(audit_dataframe["timestamp"].max()),
        "leakage_protection_description": (
            "Generation features use exact timestamps or right-closed rolling windows ending "
            "at target timestamp minus forecast horizon; sites are isolated and the current "
            "target is never a predictor."
        ),
        "warnings_and_limitations": [
            "Development weather is historical/reanalysis proxy data; production must use "
            "forecasts available at prediction time.",
            "Historical observed weather may make evaluation optimistic.",
            "Derived ratios are descriptive and do not prove physical causality.",
            "No imputation, filling, clipping, scaling, or fitted transformation is performed.",
        ],
    }


def build_quality(
    dataframe: pd.DataFrame,
    contract: FeatureContract,
    optional_absent: list[str],
    config: FeatureConfig,
) -> dict[str, Any]:
    """Summarise feature quality without diagnosing equipment."""
    numeric = dataframe[list(contract.numerical_columns)]
    infinite = {
        column: int(pd.to_numeric(numeric[column], errors="coerce").map(_is_infinite).sum())
        for column in numeric
    }
    reasons = (
        dataframe["feature_missing_reasons"]
        .str.split(";")
        .explode()
        .loc[lambda values: values.ne("")]
        .value_counts()
    )
    lag_columns = [name for name in contract.predictor_columns if "_lag_" in name]
    rolling_columns = [name for name in contract.predictor_columns if "_rolling_" in name]
    return {
        "row_count": len(dataframe),
        "feature_count": len(contract.predictor_columns),
        "feature_groups": {
            "raw": [name for name in contract.predictor_columns if name in UNITS],
            "exact_time_lags": lag_columns,
            "rolling_history": rolling_columns,
        },
        "missing_values_by_feature": {
            name: int(dataframe[name].isna().sum()) for name in contract.predictor_columns
        },
        "constant_features": [
            name for name in contract.predictor_columns if dataframe[name].nunique(dropna=True) <= 1
        ],
        "infinite_value_counts": infinite,
        "eligibility_counts": _counts(dataframe, "feature_eligible"),
        "eligibility_reasons": {str(key): int(value) for key, value in reasons.items()},
        "counts_by_site": _counts(dataframe, "site_id"),
        "counts_by_split": _counts(dataframe, "split") if "split" in dataframe else {},
        "exact_time_lag_availability": {
            name: int(dataframe[name].notna().sum()) for name in lag_columns
        },
        "rolling_window_availability": {
            name: int(dataframe[name].notna().sum()) for name in rolling_columns
        },
        "source_optional_columns_absent": optional_absent,
        "suspicious_predictor_ranges": _suspicious_ranges(dataframe),
        "leakage_checks_performed": [
            f"target lags validated >= {config.forecast_horizon_hours}h horizon",
            "exact-time lookup by site and UTC timestamp",
            "rolling history cutoff applied",
            "target excluded from predictor contract",
        ],
        "limitations": [
            "Warnings are screening information and are not confirmed equipment faults.",
            "Historical weather is only a proxy for forecast-time weather.",
        ],
    }


def _counts(dataframe: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in dataframe[column].value_counts().items()}


def _is_infinite(value: float) -> bool:
    """Return whether a non-missing numeric value is infinite."""
    return False if pd.isna(value) else math.isinf(value)


def _timestamp_or_none(value: pd.Timestamp) -> str | None:
    """Serialise a valid timestamp and preserve an absent boundary as null."""
    return None if pd.isna(value) else value.isoformat()


def _description(name: str) -> str:
    if "_lag_" in name:
        return "Exact-time historical value for the named UTC offset, isolated by site."
    if "_rolling_" in name:
        return "Generation statistic over (forecast cutoff - window, forecast cutoff]."
    if name.startswith("local_") or name.endswith(("_sin", "_cos")):
        return "Calendar feature calculated in the site's configured IANA timezone."
    return "Raw or physically interpretable derived forecasting predictor."


def _suspicious_ranges(dataframe: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if "ghi_w_m2" in dataframe and (dataframe["ghi_w_m2"] > 1500).any():
        warnings.append("ghi_w_m2 exceeds 1500 W/m²; review source context.")
    if (
        "ambient_temperature_c" in dataframe
        and (dataframe["ambient_temperature_c"].abs() > 70).any()
    ):
        warnings.append("ambient_temperature_c exceeds a broad screening range.")
    return warnings


def eligibility_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return compact eligibility counts by site and reason."""
    exploded = dataframe[["site_id", "feature_eligible", "feature_missing_reasons"]].copy()
    exploded["reason"] = exploded.pop("feature_missing_reasons").str.split(";")
    exploded = exploded.explode("reason")
    exploded["reason"] = exploded["reason"].replace("", "eligible")
    return (
        exploded.groupby(["site_id", "feature_eligible", "reason"], dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
    )


def render_quality_markdown(quality: dict[str, Any]) -> str:
    """Render the concise human-readable feature-quality report."""
    return f"""# SolarPulse AI feature quality

- Rows: {quality["row_count"]}
- Predictors: {quality["feature_count"]}
- Eligible/ineligible: {quality["eligibility_counts"]}
- Optional source columns absent: {quality["source_optional_columns_absent"] or "none"}
- Constant features: {quality["constant_features"] or "none"}
- Suspicious ranges: {quality["suspicious_predictor_ranges"] or "none"}

Missingness is preserved rather than imputed. Eligibility reasons and exact-time
lag/rolling availability are recorded in `feature_quality.json` and
`feature_eligibility.csv`. Warnings are not confirmed equipment faults.
"""
