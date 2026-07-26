"""UTC temporal summaries and continuity analysis."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd


def analyse_temporal(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Summarise UTC time patterns without local-time assumptions."""
    by_hour = dataframe.groupby(dataframe["timestamp"].dt.hour)["ac_energy_kwh"].mean()
    by_weekday = dataframe.groupby(dataframe["timestamp"].dt.day_name())["ac_energy_kwh"].mean()
    month_count = dataframe["timestamp"].dt.strftime("%Y-%m").nunique()
    by_month = (
        dataframe.groupby(dataframe["timestamp"].dt.strftime("%Y-%m"))["ac_energy_kwh"].mean()
        if month_count >= 2
        else pd.Series(dtype=float)
    )
    daily = dataframe.groupby(dataframe["timestamp"].dt.floor("D"))["ac_energy_kwh"].sum()
    ghi_relationship = dataframe.groupby(dataframe["timestamp"].dt.hour).agg(
        mean_ghi_w_m2=("ghi_w_m2", "mean"),
        mean_generation_kwh=("ac_energy_kwh", "mean"),
    )
    gaps: list[dict[str, Any]] = []
    for site_id, site in dataframe.groupby("site_id", sort=True):
        differences = site["timestamp"].diff()
        previous_timestamps = site["timestamp"].shift()
        one_hour = timedelta(hours=1)
        for index in differences[differences > one_hour].index:
            current = site.loc[index, "timestamp"]
            previous = previous_timestamps.loc[index]
            gaps.append(
                {
                    "site_id": str(site_id),
                    "previous_timestamp": previous,
                    "next_timestamp": current,
                    "gap_hours": float((current - previous) / one_hour),
                    "missing_hours": int((current - previous) / one_hour) - 1,
                }
            )
    return {
        "timezone": "UTC",
        "local_time_note": (
            "Operational local-time analysis may be added later using each site's "
            "configured IANA timezone."
        ),
        "generation_by_hour_of_day_kwh": {str(key): float(value) for key, value in by_hour.items()},
        "generation_by_day_of_week_kwh": {
            str(key): float(value) for key, value in by_weekday.items()
        },
        "generation_by_month_kwh": {str(key): float(value) for key, value in by_month.items()},
        "daily_generation_trend_kwh": {str(key): float(value) for key, value in daily.items()},
        "hourly_ghi_generation_relationship": {
            str(hour): {
                "mean_ghi_w_m2": float(row["mean_ghi_w_m2"]),
                "mean_generation_kwh": float(row["mean_generation_kwh"]),
            }
            for hour, row in ghi_relationship.iterrows()
        },
        "hourly_continuity_acceptable": not gaps,
        "gaps_between_observations": gaps,
    }
