"""Weather, exact-time history, rolling, and leakage tests."""

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from solarpulse_ai.features.config import FeatureConfig
from solarpulse_ai.features.history import add_generation_history
from solarpulse_ai.features.registry import SiteRegistry
from solarpulse_ai.features.weather import add_weather_features


def _registry(tmp_path: Path) -> SiteRegistry:
    path = tmp_path / "site.json"
    path.write_text(
        json.dumps(
            {
                "site_id": "a",
                "latitude": 0,
                "longitude": 0,
                "timezone": "UTC",
                "installed_capacity_kwp": 10,
                "panel_tilt_degrees": 5,
                "panel_azimuth_degrees": 0,
            }
        ),
        encoding="utf-8",
    )
    return SiteRegistry.from_paths([path])


def _history(hours: int = 200, sites: tuple[str, ...] = ("a",)) -> pd.DataFrame:
    rows = []
    for site_index, site in enumerate(sites):
        for hour, timestamp in enumerate(
            pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
        ):
            rows.append(
                {
                    "timestamp": timestamp,
                    "site_id": site,
                    "ac_energy_kwh": float(hour + site_index * 1000),
                }
            )
    return pd.DataFrame(rows)


def test_exact_lags_and_rolling_cutoff() -> None:
    """Lags use exact offsets and rolling values end at the 24-hour cutoff."""
    result, features, failures = add_generation_history(
        _history(), FeatureConfig(rolling_window_hours=(3,))
    )
    assert cast(Any, result.loc[48, "ac_energy_lag_24h"]) == 24
    assert cast(Any, result.loc[48, "ac_energy_lag_48h"]) == 0
    assert pd.isna(cast(Any, result.loc[48, "ac_energy_lag_168h"]))
    assert cast(Any, result.loc[48, "ac_energy_rolling_3h_count"]) == 3
    assert cast(Any, result.loc[48, "ac_energy_rolling_3h_mean"]) == pytest.approx(23)
    assert cast(Any, result.loc[48, "ac_energy_rolling_3h_std"]) == pytest.approx(1)
    assert cast(Any, result.loc[48, "ac_energy_rolling_3h_min"]) == 22
    assert cast(Any, result.loc[48, "ac_energy_rolling_3h_max"]) == 24
    assert cast(Any, result.loc[48, "ac_energy_rolling_3h_median"]) == 23
    assert "ac_energy_kwh" not in features
    assert failures["missing_exact_target_lag_168h"].loc[48]


def test_missing_timestamp_is_not_row_lag() -> None:
    """An absent exact timestamp remains missing despite enough preceding rows."""
    frame = _history().drop(index=24).reset_index(drop=True)
    result, _, _ = add_generation_history(frame, FeatureConfig(rolling_window_hours=(3,)))
    row = result.loc[result["timestamp"].eq(pd.Timestamp("2025-01-03T00:00:00Z"))].iloc[0]
    assert pd.isna(row["ac_energy_lag_24h"])


def test_history_is_site_isolated() -> None:
    """One site's values never enter another site's lag or rolling feature."""
    result, _, _ = add_generation_history(
        _history(sites=("a", "b")), FeatureConfig(rolling_window_hours=(3,))
    )
    b = result.loc[(result["site_id"] == "b") & (result["timestamp"] == "2025-01-03")].iloc[0]
    assert b["ac_energy_lag_24h"] == 1024
    assert b["ac_energy_rolling_3h_mean"] == 1023


def test_current_and_future_targets_cannot_change_past_features() -> None:
    """Mutation tests enforce the prediction-time information boundary."""
    frame = _history()
    config = FeatureConfig(rolling_window_hours=(24,))
    baseline, features, _ = add_generation_history(frame, config)
    changed = frame.copy()
    changed.loc[100:, "ac_energy_kwh"] = 999999
    mutated, _, _ = add_generation_history(changed, config)
    pd.testing.assert_frame_equal(
        baseline.loc[:99, features],
        mutated.loc[:99, features],
    )
    current = frame.copy()
    current.loc[100, "ac_energy_kwh"] = -999
    current_result, _, _ = add_generation_history(current, config)
    pd.testing.assert_frame_equal(
        baseline.loc[[100], features],
        current_result.loc[[100], features],
    )


def test_irregular_rolling_uses_time_window_and_count() -> None:
    """Rolling windows count available timestamps and do not fill gaps."""
    frame = _history(60).drop(index=[22, 23]).reset_index(drop=True)
    result, _, _ = add_generation_history(frame, FeatureConfig(rolling_window_hours=(3,)))
    row = result.loc[result["timestamp"].eq(pd.Timestamp("2025-01-03T00:00:00Z"))].iloc[0]
    assert row["ac_energy_rolling_3h_count"] == 1
    assert row["ac_energy_rolling_3h_mean"] == 24


def test_require_complete_history_adds_failures() -> None:
    """Completeness is an explicit eligibility policy."""
    _, _, default = add_generation_history(
        _history(30), FeatureConfig(target_lag_hours=(24,), rolling_window_hours=(3,))
    )
    _, _, complete = add_generation_history(
        _history(30),
        FeatureConfig(
            target_lag_hours=(24,), rolling_window_hours=(3,), require_complete_history=True
        ),
    )
    assert "incomplete_rolling_history_3h" not in default
    assert complete["incomplete_rolling_history_3h"].iloc[24]


def test_weather_features_ratios_interactions_and_lags(tmp_path: Path) -> None:
    """Weather derivations preserve undefined ratios and exact-time lag gaps."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=25, freq="h", tz="UTC"),
            "site_id": ["a"] * 25,
            "ghi_w_m2": [0.0] + [100.0] * 24,
            "ambient_temperature_c": [20.0] * 25,
            "cloud_cover_pct": [50.0] * 25,
            "relative_humidity_pct": [60.0] * 25,
            "wind_speed_m_s": [2.0] * 25,
            "dni_w_m2": [0.0] + [50.0] * 24,
            "dhi_w_m2": [0.0] + [25.0] * 24,
            "module_temperature_c": [25.0] * 25,
            "precipitation_mm": [0.0, *([1.0] * 24)],
            "inverter_availability_pct": [90.0] * 25,
        }
    )
    result, engineered, lags = add_weather_features(
        frame, _registry(tmp_path), FeatureConfig(weather_lag_hours=(24,))
    )
    assert pd.isna(cast(Any, result.loc[0, "diffuse_fraction"]))
    assert cast(Any, result.loc[1, "diffuse_fraction"]) == 0.25
    assert cast(Any, result.loc[1, "direct_to_global_ratio"]) == 0.5
    assert cast(Any, result.loc[1, "temperature_ghi_interaction"]) == 2000
    assert cast(Any, result.loc[1, "cloud_ghi_interaction"]) == 5000
    assert cast(Any, result.loc[1, "humidity_temperature_interaction"]) == 1200
    assert cast(Any, result.loc[1, "precipitation_indicator"]) == 1
    assert cast(Any, result.loc[1, "availability_fraction"]) == 0.9
    assert cast(Any, result.loc[1, "apparent_temperature_difference"]) == 5
    assert cast(Any, result.loc[24, "ghi_lag_24h"]) == 0
    assert pd.isna(cast(Any, result.loc[0, "ghi_lag_24h"]))
    assert "diffuse_fraction" in engineered
    assert "ghi_lag_24h" in lags


def test_optional_weather_absent_and_lags_disabled(tmp_path: Path) -> None:
    """Absent optional sources are not invented."""
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")],
            "site_id": ["a"],
            "ghi_w_m2": [20.0],
            "ambient_temperature_c": [20.0],
            "cloud_cover_pct": [10.0],
            "relative_humidity_pct": [50.0],
            "wind_speed_m_s": [1.0],
        }
    )
    result, engineered, lags = add_weather_features(
        frame, _registry(tmp_path), FeatureConfig(include_weather_lags=False)
    )
    assert "diffuse_fraction" not in result
    assert "precipitation_indicator" not in engineered
    assert lags == []
