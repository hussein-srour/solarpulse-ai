"""Canonical dataset profiling."""

from __future__ import annotations

from typing import Any

import pandas as pd

from solarpulse_ai.data.schemas import OPTIONAL_COLUMNS


def _site_profile(site: pd.DataFrame) -> dict[str, Any]:
    start = site["timestamp"].min()
    end = site["timestamp"].max()
    expected = len(pd.date_range(start, end, freq="h"))
    actual = len(site)
    day_counts = site.groupby(site["timestamp"].dt.floor("D")).size()
    return {
        "earliest_timestamp": start,
        "latest_timestamp": end,
        "expected_hourly_records": expected,
        "actual_hourly_records": actual,
        "completeness_percentage": round(actual / expected * 100, 4),
        "missing_timestamp_count": expected - actual,
        "complete_days": int((day_counts == 24).sum()),
        "partial_days": int((day_counts < 24).sum()),
    }


def profile_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Build record, coverage, missingness, type, and memory metadata."""
    sites = {
        str(site_id): _site_profile(site)
        for site_id, site in dataframe.groupby("site_id", sort=True)
    }
    start = dataframe["timestamp"].min()
    end = dataframe["timestamp"].max()
    duplicate_count = int(dataframe.duplicated(["site_id", "timestamp"]).sum())
    return {
        "total_record_count": len(dataframe),
        "number_of_sites": dataframe["site_id"].nunique(),
        "site_ids": sorted(str(value) for value in dataframe["site_id"].unique()),
        "earliest_timestamp": start,
        "latest_timestamp": end,
        "total_calendar_duration": end - start,
        "records_per_site": {
            str(key): int(value)
            for key, value in dataframe.groupby("site_id", sort=True).size().items()
        },
        "expected_hourly_records_per_site": {
            key: value["expected_hourly_records"] for key, value in sites.items()
        },
        "actual_hourly_records_per_site": {
            key: value["actual_hourly_records"] for key, value in sites.items()
        },
        "completeness_percentage": {
            key: value["completeness_percentage"] for key, value in sites.items()
        },
        "missing_timestamp_count": sum(
            int(value["missing_timestamp_count"]) for value in sites.values()
        ),
        "duplicate_count": duplicate_count,
        "missing_value_count_by_column": {
            column: int(count) for column, count in dataframe.isna().sum().items()
        },
        "data_types": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        "columns_present": dataframe.columns.tolist(),
        "optional_columns_absent": [
            column for column in OPTIONAL_COLUMNS if column not in dataframe.columns
        ],
        "memory_usage_bytes": int(dataframe.memory_usage(index=True, deep=True).sum()),
        "number_of_complete_days": sum(int(value["complete_days"]) for value in sites.values()),
        "number_of_partial_days": sum(int(value["partial_days"]) for value in sites.values()),
        "by_site": sites,
    }
