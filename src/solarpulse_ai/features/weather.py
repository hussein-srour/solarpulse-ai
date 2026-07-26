"""Raw, derived, and exact-time historical weather features."""

from __future__ import annotations

import pandas as pd

from solarpulse_ai.data.schemas import OPTIONAL_COLUMNS
from solarpulse_ai.features.config import FeatureConfig
from solarpulse_ai.features.registry import SiteRegistry

RAW_WEATHER = (
    "ghi_w_m2",
    "ambient_temperature_c",
    "cloud_cover_pct",
    "relative_humidity_pct",
    "wind_speed_m_s",
    *OPTIONAL_COLUMNS,
)
WEATHER_LAG_BASES = {
    "ghi_w_m2": "ghi",
    "cloud_cover_pct": "cloud_cover",
    "ambient_temperature_c": "ambient_temperature",
    "relative_humidity_pct": "relative_humidity",
    "wind_speed_m_s": "wind_speed",
    "dni_w_m2": "dni",
    "dhi_w_m2": "dhi",
    "module_temperature_c": "module_temperature",
    "precipitation_mm": "precipitation",
    "inverter_availability_pct": "inverter_availability",
}


def _safe_ratio(
    numerator: pd.Series[float], denominator: pd.Series[float], epsilon: float
) -> pd.Series[float]:
    """Return a ratio only when irradiance is physically defined."""
    safe_denominator = denominator.where(denominator.abs() > epsilon)
    return numerator / safe_denominator


def add_weather_features(
    dataframe: pd.DataFrame, registry: SiteRegistry, config: FeatureConfig
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add interpretable weather features, preserving source values."""
    result = dataframe.copy()
    engineered: list[str] = []
    result["is_daylight"] = (result["ghi_w_m2"] >= config.daylight_ghi_threshold_w_m2).astype(int)
    engineered.append("is_daylight")
    capacity = result["site_id"].map(
        {site_id: site.installed_capacity_kwp for site_id, site in registry.sites.items()}
    )
    result["ghi_normalised_by_capacity"] = result["ghi_w_m2"] / capacity
    result["temperature_ghi_interaction"] = result["ambient_temperature_c"] * result["ghi_w_m2"]
    result["cloud_ghi_interaction"] = result["cloud_cover_pct"] * result["ghi_w_m2"]
    result["humidity_temperature_interaction"] = (
        result["relative_humidity_pct"] * result["ambient_temperature_c"]
    )
    engineered.extend(
        [
            "ghi_normalised_by_capacity",
            "temperature_ghi_interaction",
            "cloud_ghi_interaction",
            "humidity_temperature_interaction",
        ]
    )
    if "dhi_w_m2" in result:
        result["diffuse_fraction"] = _safe_ratio(
            result["dhi_w_m2"], result["ghi_w_m2"], config.irradiance_epsilon
        )
        engineered.append("diffuse_fraction")
    if "dni_w_m2" in result:
        result["direct_to_global_ratio"] = _safe_ratio(
            result["dni_w_m2"], result["ghi_w_m2"], config.irradiance_epsilon
        )
        engineered.append("direct_to_global_ratio")
    if "precipitation_mm" in result:
        result["precipitation_indicator"] = (result["precipitation_mm"] > 0).astype(int)
        engineered.append("precipitation_indicator")
    if "inverter_availability_pct" in result:
        result["availability_fraction"] = result["inverter_availability_pct"] / 100
        engineered.append("availability_fraction")
    if "module_temperature_c" in result:
        result["apparent_temperature_difference"] = (
            result["module_temperature_c"] - result["ambient_temperature_c"]
        )
        engineered.append("apparent_temperature_difference")

    lag_features: list[str] = []
    if config.include_weather_lags:
        source_columns = [column for column in WEATHER_LAG_BASES if column in result]
        lookup = result.set_index(["site_id", "timestamp"])[source_columns]
        keys = pd.MultiIndex.from_arrays([result["site_id"], result["timestamp"]])
        for lag in config.weather_lag_hours:
            shifted_keys = pd.MultiIndex.from_arrays(
                [result["site_id"], result["timestamp"] - pd.Timedelta(lag, unit="h")]
            )
            matched = lookup.reindex(shifted_keys)
            matched.index = result.index
            for source in source_columns:
                name = f"{WEATHER_LAG_BASES[source]}_lag_{lag}h"
                result[name] = matched[source]
                lag_features.append(name)
        del keys
    return result, engineered, lag_features
