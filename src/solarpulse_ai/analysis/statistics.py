"""Descriptive statistics for generation and weather variables."""

from __future__ import annotations

from typing import Any

import pandas as pd

WEATHER_COLUMNS: tuple[str, ...] = (
    "ghi_w_m2",
    "dni_w_m2",
    "dhi_w_m2",
    "ambient_temperature_c",
    "module_temperature_c",
    "cloud_cover_pct",
    "relative_humidity_pct",
    "wind_speed_m_s",
    "precipitation_mm",
    "inverter_availability_pct",
)


def describe_series(series: pd.Series[float]) -> dict[str, float | int | None]:
    """Return the documented descriptive measures for a numeric series."""
    clean = series.dropna()
    if clean.empty:
        return {
            key: None
            for key in (
                "count",
                "minimum",
                "maximum",
                "mean",
                "median",
                "standard_deviation",
                "percentile_05",
                "percentile_25",
                "percentile_50",
                "percentile_75",
                "percentile_95",
                "total",
            )
        }
    return {
        "count": int(clean.count()),
        "minimum": float(clean.min()),
        "maximum": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "standard_deviation": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "percentile_05": float(clean.quantile(0.05)),
        "percentile_25": float(clean.quantile(0.25)),
        "percentile_50": float(clean.quantile(0.50)),
        "percentile_75": float(clean.quantile(0.75)),
        "percentile_95": float(clean.quantile(0.95)),
        "total": float(clean.sum()),
    }


def analyse_target(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Analyse hourly energy overall and by site."""
    overall = describe_series(dataframe["ac_energy_kwh"])
    overall["total_generated_energy_kwh"] = overall.pop("total")
    overall["zero_generation_percentage"] = float(dataframe["ac_energy_kwh"].eq(0).mean() * 100)
    by_site: dict[str, dict[str, float | int | None]] = {}
    for site_id, site in dataframe.groupby("site_id", sort=True):
        values = describe_series(site["ac_energy_kwh"])
        values["total_generated_energy_kwh"] = values.pop("total")
        values["zero_generation_percentage"] = float(site["ac_energy_kwh"].eq(0).mean() * 100)
        by_site[str(site_id)] = values
    hourly = dataframe.groupby(dataframe["timestamp"].dt.hour)["ac_energy_kwh"].mean()
    daily = dataframe.groupby(dataframe["timestamp"].dt.floor("D"))["ac_energy_kwh"].sum()
    month_count = dataframe["timestamp"].dt.strftime("%Y-%m").nunique()
    monthly = (
        dataframe.assign(month=dataframe["timestamp"].dt.strftime("%Y-%m"))
        .groupby("month")["ac_energy_kwh"]
        .sum()
        if month_count >= 2
        else pd.Series(dtype=float)
    )
    aggregations_by_site: dict[str, dict[str, dict[str, float]]] = {}
    for site_id, site in dataframe.groupby("site_id", sort=True):
        site_daily = site.groupby(site["timestamp"].dt.floor("D"))["ac_energy_kwh"].sum()
        site_monthly = (
            site.assign(month=site["timestamp"].dt.strftime("%Y-%m"))
            .groupby("month")["ac_energy_kwh"]
            .sum()
            if site["timestamp"].dt.strftime("%Y-%m").nunique() >= 2
            else pd.Series(dtype=float)
        )
        site_hourly = site.groupby(site["timestamp"].dt.hour)["ac_energy_kwh"].mean()
        aggregations_by_site[str(site_id)] = {
            "hourly_production_profile_kwh": {
                str(key): float(value) for key, value in site_hourly.items()
            },
            "daily_energy_totals_kwh": {
                str(key): float(value) for key, value in site_daily.items()
            },
            "monthly_energy_totals_kwh": {
                str(key): float(value) for key, value in site_monthly.items()
            },
        }
    return {
        "overall": overall,
        "by_site": by_site,
        "hourly_production_profile_kwh": {str(key): float(value) for key, value in hourly.items()},
        "daily_energy_totals_kwh": {str(key): float(value) for key, value in daily.items()},
        "monthly_energy_totals_kwh": {str(key): float(value) for key, value in monthly.items()},
        "aggregations_by_site": aggregations_by_site,
    }


def analyse_weather(dataframe: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """Describe only weather fields present in the canonical dataset."""
    return {
        column: describe_series(dataframe[column])
        for column in WEATHER_COLUMNS
        if column in dataframe.columns
    }
