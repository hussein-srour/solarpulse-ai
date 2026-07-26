"""Site-local calendar and cyclical features."""

from __future__ import annotations

import math

import pandas as pd

from solarpulse_ai.features.registry import SiteRegistry


def add_temporal_features(dataframe: pd.DataFrame, registry: SiteRegistry) -> pd.DataFrame:
    """Calculate local calendar fields without changing the UTC key."""
    result = dataframe.copy()
    feature_names = [
        "local_hour",
        "local_day_of_week",
        "local_day_of_month",
        "local_day_of_year",
        "local_week_of_year",
        "local_month",
        "local_quarter",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "month_sin",
        "month_cos",
    ]
    for name in feature_names:
        result[name] = pd.NA

    for site_id, index in result.groupby("site_id", sort=False).groups.items():
        local = result.loc[index, "timestamp"].dt.tz_convert(registry[str(site_id)].timezone)
        hour = local.dt.hour
        weekday = local.dt.dayofweek
        day_year = local.dt.dayofyear
        month = local.dt.month
        result.loc[index, "local_hour"] = hour.to_numpy()
        result.loc[index, "local_day_of_week"] = weekday.to_numpy()
        result.loc[index, "local_day_of_month"] = local.dt.day.to_numpy()
        result.loc[index, "local_day_of_year"] = day_year.to_numpy()
        result.loc[index, "local_week_of_year"] = local.dt.isocalendar().week.to_numpy()
        result.loc[index, "local_month"] = month.to_numpy()
        result.loc[index, "local_quarter"] = local.dt.quarter.to_numpy()
        result.loc[index, "is_weekend"] = (weekday >= 5).astype(int).to_numpy()
        cycles = {
            "hour": (hour, 24),
            "day_of_week": (weekday, 7),
            "day_of_year": (day_year - 1, 365.2425),
            "month": (month - 1, 12),
        }
        for prefix, (values, period) in cycles.items():
            radians = values.astype(float) * (2 * math.pi / period)
            result.loc[index, f"{prefix}_sin"] = radians.map(math.sin).to_numpy()
            result.loc[index, f"{prefix}_cos"] = radians.map(math.cos).to_numpy()
    return result.infer_objects()
