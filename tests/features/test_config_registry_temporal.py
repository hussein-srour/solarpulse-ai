"""Configuration, registry, and site-local calendar tests."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from solarpulse_ai.data.errors import SiteConfigurationError
from solarpulse_ai.features.config import FeatureConfig
from solarpulse_ai.features.registry import SiteRegistry
from solarpulse_ai.features.temporal import add_temporal_features


def _site(path: Path, site_id: str, timezone: str = "Africa/Dar_es_Salaam") -> Path:
    path.write_text(
        json.dumps(
            {
                "site_id": site_id,
                "latitude": -6.8,
                "longitude": 39.2,
                "timezone": timezone,
                "installed_capacity_kwp": 10,
                "panel_tilt_degrees": 10,
                "panel_azimuth_degrees": 0,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_default_config_is_json_serialisable() -> None:
    """Defaults express the 24-hour day-ahead objective."""
    config = FeatureConfig()
    assert config.forecast_horizon_hours == 24
    assert config.target_lag_hours == (24, 48, 168)
    assert json.loads(config.model_dump_json())["rolling_window_hours"] == [3, 6, 24, 72, 168]


@pytest.mark.parametrize("horizon", [0, -1])
def test_invalid_horizon(horizon: int) -> None:
    """A forecast horizon must be positive."""
    with pytest.raises(ValidationError):
        FeatureConfig(forecast_horizon_hours=horizon)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_lag_hours", (23,)),
        ("rolling_window_hours", (0,)),
        ("weather_lag_hours", (-1,)),
        ("target_lag_hours", (24, 24)),
        ("rolling_window_hours", (3, 3)),
    ],
)
def test_invalid_hour_lists(field: str, value: tuple[int, ...]) -> None:
    """Short target lags, non-positive values, and duplicates are rejected."""
    with pytest.raises(ValidationError):
        FeatureConfig.model_validate({field: value})


@pytest.mark.parametrize("field", ["daylight_ghi_threshold_w_m2", "irradiance_epsilon"])
def test_non_finite_threshold(field: str) -> None:
    """Thresholds cannot contain infinity."""
    with pytest.raises(ValidationError):
        FeatureConfig.model_validate({field: math.inf})


def test_registry_one_and_multiple_sites(tmp_path: Path) -> None:
    """Registry accepts distinct configurations and exact dataset matches."""
    registry = SiteRegistry.from_paths(
        [_site(tmp_path / "a.json", "a"), _site(tmp_path / "b.json", "b", "UTC")]
    )
    registry.validate_dataset_sites(["a", "b"])
    assert registry["a"].installed_capacity_kwp == 10


def test_registry_rejects_duplicate_missing_and_unused(tmp_path: Path) -> None:
    """Every dataset site has exactly one configuration."""
    first = _site(tmp_path / "a.json", "a")
    duplicate = _site(tmp_path / "duplicate.json", "a")
    with pytest.raises(SiteConfigurationError, match="Duplicate"):
        SiteRegistry.from_paths([first, duplicate])
    registry = SiteRegistry.from_paths([first])
    with pytest.raises(SiteConfigurationError, match="Missing"):
        registry.validate_dataset_sites(["a", "b"])
    with pytest.raises(SiteConfigurationError, match="not represented"):
        registry.validate_dataset_sites([])
    registry.validate_dataset_sites([], allow_unused=True)


def test_registry_requires_a_path() -> None:
    """An empty registry is invalid."""
    with pytest.raises(SiteConfigurationError):
        SiteRegistry.from_paths([])


def test_temporal_features_preserve_utc_and_isolate_timezones(tmp_path: Path) -> None:
    """Local calendars use each site's IANA timezone without mutating UTC."""
    registry = SiteRegistry.from_paths(
        [_site(tmp_path / "tz.json", "tz"), _site(tmp_path / "utc.json", "utc", "UTC")]
    )
    timestamp = pd.Timestamp("2025-01-04T21:00:00Z")
    frame = pd.DataFrame(
        {"timestamp": [timestamp, timestamp], "site_id": ["tz", "utc"], "ac_energy_kwh": [1, 2]}
    )
    featured = add_temporal_features(frame, registry)
    assert featured["timestamp"].tolist() == [timestamp, timestamp]
    assert featured["local_hour"].tolist() == [0, 21]
    assert featured["local_day_of_week"].tolist() == [6, 5]
    assert featured["is_weekend"].tolist() == [1, 1]
    assert featured.loc[0, "hour_sin"] == pytest.approx(0)
    assert featured.loc[0, "hour_cos"] == pytest.approx(1)


def test_temporal_features_handle_dst(tmp_path: Path) -> None:
    """IANA conversion distinguishes repeated DST hours."""
    registry = SiteRegistry.from_paths([_site(tmp_path / "ny.json", "ny", "America/New_York")])
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-11-02T05:00:00Z", "2025-11-02T06:00:00Z"]),
            "site_id": ["ny", "ny"],
        }
    )
    featured = add_temporal_features(frame, registry)
    assert featured["local_hour"].tolist() == [1, 1]
