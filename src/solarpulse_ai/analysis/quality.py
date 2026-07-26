"""Non-destructive data-quality indicators."""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import Any

import pandas as pd

from solarpulse_ai.analysis.config import AnalysisThresholds
from solarpulse_ai.data.schemas import NUMERIC_FIELDS, OPTIONAL_COLUMNS

FLAG_COLUMNS = ("site_id", "timestamp", "indicator", "column", "value", "detail")


def _row_flags(
    dataframe: pd.DataFrame,
    mask: pd.Series[bool],
    indicator: str,
    column: str,
    detail: str,
) -> list[dict[str, object]]:
    return [
        {
            "site_id": str(row["site_id"]),
            "timestamp": row["timestamp"],
            "indicator": indicator,
            "column": column,
            "value": row.get(column, None),
            "detail": detail,
        }
        for _, row in dataframe.loc[mask].iterrows()
    ]


def _missing_timestamp_flags(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    for site_id, site in dataframe.groupby("site_id", sort=True):
        expected = pd.date_range(site["timestamp"].min(), site["timestamp"].max(), freq="h")
        for timestamp in expected.difference(pd.DatetimeIndex(site["timestamp"])):
            flags.append(
                {
                    "site_id": str(site_id),
                    "timestamp": timestamp,
                    "indicator": "missing_hourly_timestamp",
                    "column": "timestamp",
                    "value": None,
                    "detail": "Expected hourly timestamp is absent; no record was inserted.",
                }
            )
    return flags


def _long_zero_flags(
    dataframe: pd.DataFrame, thresholds: AnalysisThresholds
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    for _, site in dataframe.groupby("site_id", sort=True):
        zero = site["ac_energy_kwh"] <= thresholds.near_zero_generation_kwh
        adjacent = site["timestamp"].diff().eq(timedelta(hours=1))
        run = (zero.ne(zero.shift()) | ~adjacent).cumsum()
        lengths = site.loc[zero].groupby(run[zero])["timestamp"].transform("size")
        qualifying = pd.Series(False, index=site.index)
        qualifying.loc[lengths.index] = lengths >= thresholds.consecutive_zero_hours
        flags.extend(
            _row_flags(
                site,
                qualifying,
                "long_consecutive_near_zero_generation",
                "ac_energy_kwh",
                (
                    "Part of an adjacent near-zero run lasting at least "
                    f"{thresholds.consecutive_zero_hours} hours."
                ),
            )
        )
    return flags


def _outlier_flags(
    dataframe: pd.DataFrame, thresholds: AnalysisThresholds
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    for field in NUMERIC_FIELDS:
        column = field.name
        if column not in dataframe.columns:
            continue
        series = dataframe[column].dropna()
        if series.nunique() < 2:
            continue
        q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - thresholds.outlier_iqr_multiplier * iqr
        upper = q3 + thresholds.outlier_iqr_multiplier * iqr
        mask = dataframe[column].lt(lower) | dataframe[column].gt(upper)
        flags.extend(
            _row_flags(
                dataframe,
                mask,
                "extreme_value_iqr",
                column,
                (
                    f"Outside Tukey fences [{lower:.6g}, {upper:.6g}] using "
                    f"{thresholds.outlier_iqr_multiplier:g} × IQR."
                ),
            )
        )
    return flags


def _abrupt_change_flags(
    dataframe: pd.DataFrame, thresholds: AnalysisThresholds
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    for _, site in dataframe.groupby("site_id", sort=True):
        adjacent = site["timestamp"].diff().eq(timedelta(hours=1))
        changes = site["ac_energy_kwh"].diff().abs().where(adjacent)
        clean = changes.dropna()
        if len(clean) < 4 or clean.nunique() < 2:
            continue
        q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
        upper = q3 + thresholds.outlier_iqr_multiplier * (q3 - q1)
        mask = changes > upper
        flags.extend(
            _row_flags(
                site,
                mask.fillna(False),
                "abrupt_generation_change",
                "ac_energy_kwh",
                (f"Absolute adjacent-hour change exceeds the Tukey upper fence {upper:.6g} kWh."),
            )
        )
    return flags


def diagnose_quality(
    dataframe: pd.DataFrame, thresholds: AnalysisThresholds
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Create indicators without rejecting, correcting, or deleting observations."""
    flags: list[dict[str, object]] = []
    flags.extend(
        _row_flags(
            dataframe,
            dataframe["ac_energy_kwh"].gt(thresholds.near_zero_generation_kwh)
            & dataframe["ghi_w_m2"].lt(thresholds.low_irradiance_w_m2),
            "positive_generation_at_very_low_irradiance",
            "ac_energy_kwh",
            (
                f"Generation exceeds {thresholds.near_zero_generation_kwh:g} kWh while "
                f"GHI is below {thresholds.low_irradiance_w_m2:g} W/m²."
            ),
        )
    )
    flags.extend(
        _row_flags(
            dataframe,
            dataframe["ac_energy_kwh"].le(thresholds.near_zero_generation_kwh)
            & dataframe["ghi_w_m2"].ge(thresholds.high_irradiance_w_m2),
            "near_zero_generation_at_high_irradiance",
            "ac_energy_kwh",
            (
                f"Generation is at most {thresholds.near_zero_generation_kwh:g} kWh while "
                f"GHI is at least {thresholds.high_irradiance_w_m2:g} W/m²."
            ),
        )
    )
    flags.extend(_long_zero_flags(dataframe, thresholds))
    flags.extend(_missing_timestamp_flags(dataframe))
    for column in OPTIONAL_COLUMNS:
        if column in dataframe.columns:
            flags.extend(
                _row_flags(
                    dataframe,
                    dataframe[column].isna(),
                    "missing_optional_weather_value",
                    column,
                    "Optional weather value is missing; it was not filled.",
                )
            )
    constant_columns: list[str] = []
    for field in NUMERIC_FIELDS:
        if field.name in dataframe.columns and dataframe[field.name].dropna().nunique() <= 1:
            constant_columns.append(field.name)
            flags.append(
                {
                    "site_id": None,
                    "timestamp": None,
                    "indicator": "constant_value_sensor_column",
                    "column": field.name,
                    "value": dataframe[field.name].dropna().iloc[0]
                    if dataframe[field.name].notna().any()
                    else None,
                    "detail": "All available values are identical.",
                }
            )
    flags.extend(_outlier_flags(dataframe, thresholds))
    flags.extend(_abrupt_change_flags(dataframe, thresholds))

    ranges = dataframe.groupby("site_id")["timestamp"].agg(["min", "max"])
    different_ranges = len(ranges) > 1 and (
        ranges["min"].nunique() > 1 or ranges["max"].nunique() > 1
    )
    if different_ranges:
        flags.append(
            {
                "site_id": None,
                "timestamp": None,
                "indicator": "sites_have_different_date_ranges",
                "column": "timestamp",
                "value": None,
                "detail": "Site coverage start or end timestamps differ.",
            }
        )
    site_ranges = dataframe.groupby("site_id")["timestamp"].agg(["min", "max"])
    coverage_days = (site_ranges["max"] - site_ranges["min"]).dt.total_seconds() / 86400
    insufficient_sites = [
        str(site_id)
        for site_id, days in coverage_days.items()
        if days < thresholds.minimum_history_days
    ]
    for site_id in insufficient_sites:
        flags.append(
            {
                "site_id": site_id,
                "timestamp": None,
                "indicator": "insufficient_historical_coverage",
                "column": "timestamp",
                "value": None,
                "detail": (f"Coverage is shorter than {thresholds.minimum_history_days} days."),
            }
        )
    table = pd.DataFrame(flags, columns=FLAG_COLUMNS)
    counts = (
        {str(key): int(value) for key, value in table["indicator"].value_counts().items()}
        if not table.empty
        else {}
    )
    return (
        {
            "label": "Data-quality indicators; these are not confirmed equipment faults.",
            "thresholds": asdict(thresholds),
            "total_flags": len(table),
            "counts_by_indicator": counts,
            "constant_columns": constant_columns,
            "sites_with_insufficient_history": insufficient_sites,
            "sites_have_different_date_ranges": different_ranges,
        },
        table,
    )
