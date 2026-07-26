"""Pipeline, eligibility, split, manifest, report, and CLI tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from solarpulse_ai.data.errors import DataValidationError
from solarpulse_ai.features.build import main
from solarpulse_ai.features.config import FeatureConfig
from solarpulse_ai.features.contract import TARGET_COLUMN
from solarpulse_ai.features.eligibility import add_eligibility
from solarpulse_ai.features.pipeline import run_feature_pipeline
from solarpulse_ai.features.splits import assign_splits, load_split_plan


def _inputs(tmp_path: Path, hours: int = 200) -> tuple[Path, Path]:
    timestamps = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    dataframe = pd.DataFrame(
        {
            "timestamp": timestamps,
            "site_id": ["a"] * hours,
            "ac_energy_kwh": [float(index) for index in range(hours)],
            "ghi_w_m2": [100.0] * hours,
            "ambient_temperature_c": [25.0] * hours,
            "cloud_cover_pct": [20.0] * hours,
            "relative_humidity_pct": [60.0] * hours,
            "wind_speed_m_s": [2.0] * hours,
        }
    )
    source = tmp_path / "source.csv"
    dataframe.to_csv(source, index=False)
    site = tmp_path / "site.json"
    site.write_text(
        json.dumps(
            {
                "site_id": "a",
                "latitude": -6.8,
                "longitude": 39.2,
                "timezone": "Africa/Dar_es_Salaam",
                "installed_capacity_kwp": 10,
                "panel_tilt_degrees": 10,
                "panel_azimuth_degrees": 0,
            }
        ),
        encoding="utf-8",
    )
    return source, site


def _plan(tmp_path: Path) -> Path:
    path = tmp_path / "split_plan.json"
    path.write_text(
        json.dumps(
            {
                "periods": {
                    "training": {
                        "beginning_timestamp": "2025-01-01T00:00:00+00:00",
                        "ending_timestamp": "2025-01-05T23:00:00+00:00",
                    },
                    "validation": {
                        "beginning_timestamp": "2025-01-06T00:00:00+00:00",
                        "ending_timestamp": "2025-01-07T15:00:00+00:00",
                    },
                    "testing": {
                        "beginning_timestamp": "2025-01-07T16:00:00+00:00",
                        "ending_timestamp": "2025-01-09T07:00:00+00:00",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_pipeline_writes_contract_manifest_and_reports(tmp_path: Path) -> None:
    """Default output preserves rows and separates target from predictors."""
    source, site = _inputs(tmp_path)
    result = run_feature_pipeline(
        source,
        [site],
        tmp_path / "model_features.csv",
        tmp_path / "reports",
        FeatureConfig(rolling_window_hours=(3, 24)),
    )
    assert len(result.dataframe) == 200
    assert TARGET_COLUMN not in result.contract.predictor_columns
    assert result.dataframe["timestamp"].is_monotonic_increasing
    assert result.dataframe["feature_eligible"].sum() == 32
    assert result.manifest["source_row_count"] == 200
    assert result.manifest["forecast_horizon_hours"] == 24
    assert "complete input records" not in json.dumps(result.manifest)
    assert result.quality["source_optional_columns_absent"]
    for name in (
        "feature_manifest.json",
        "feature_quality.json",
        "feature_quality.md",
        "feature_eligibility.csv",
    ):
        assert (tmp_path / "reports" / name).is_file()


def test_only_eligible_is_the_only_filter(tmp_path: Path) -> None:
    """Filtering occurs only when explicitly requested and reports exclusions."""
    source, site = _inputs(tmp_path)
    result = run_feature_pipeline(
        source,
        [site],
        tmp_path / "eligible.csv",
        tmp_path / "reports",
        FeatureConfig(rolling_window_hours=(3,)),
        only_eligible=True,
    )
    assert len(result.dataframe) == 32
    assert result.excluded_rows == 168
    assert result.dataframe["feature_eligible"].all()


def test_split_assignment_and_optional_files(tmp_path: Path) -> None:
    """Phase 4 boundaries label rows chronologically and optionally create CSVs."""
    source, site = _inputs(tmp_path)
    result = run_feature_pipeline(
        source,
        [site],
        tmp_path / "features.csv",
        tmp_path / "reports",
        FeatureConfig(rolling_window_hours=(3,)),
        split_plan_path=_plan(tmp_path),
        write_splits=True,
    )
    assert result.dataframe["split"].value_counts().to_dict() == {
        "train": 120,
        "validation": 40,
        "test": 40,
    }
    for name in ("train_features.csv", "validation_features.csv", "test_features.csv"):
        assert (tmp_path / name).is_file()


def test_write_splits_requires_plan(tmp_path: Path) -> None:
    """Split files cannot be requested without boundaries."""
    source, site = _inputs(tmp_path)
    with pytest.raises(ValueError, match="requires"):
        run_feature_pipeline(
            source,
            [site],
            tmp_path / "features.csv",
            tmp_path / "reports",
            write_splits=True,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"periods": {}},
        {
            "periods": {
                "training": {
                    "beginning_timestamp": "2025-01-01T00:00:00Z",
                    "ending_timestamp": "2025-01-06T00:00:00Z",
                },
                "validation": {
                    "beginning_timestamp": "2025-01-05T00:00:00Z",
                    "ending_timestamp": "2025-01-07T00:00:00Z",
                },
                "testing": {
                    "beginning_timestamp": "2025-01-07T01:00:00Z",
                    "ending_timestamp": "2025-01-09T00:00:00Z",
                },
            }
        },
    ],
)
def test_invalid_split_plans(tmp_path: Path, payload: dict[str, object]) -> None:
    """Missing structure and overlapping periods fail explicitly."""
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    source, _ = _inputs(tmp_path)
    frame = pd.read_csv(source)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    with pytest.raises(DataValidationError):
        assign_splits(frame, load_split_plan(path))


def test_split_plan_must_cover_dataset(tmp_path: Path) -> None:
    """A plan that leaves timestamps unlabelled does not apply to the input."""
    source, _ = _inputs(tmp_path, 201)
    frame = pd.read_csv(source)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    with pytest.raises(DataValidationError, match="Every input timestamp"):
        assign_splits(frame, load_split_plan(_plan(tmp_path)))


def test_eligibility_distinguishes_source_and_history() -> None:
    """Eligibility reasons identify source weather separately from history."""
    frame = pd.DataFrame(
        {
            "ghi_w_m2": [pd.NA],
            "ambient_temperature_c": [1.0],
            "cloud_cover_pct": [1.0],
            "relative_humidity_pct": [1.0],
            "wind_speed_m_s": [1.0],
        }
    )
    result = add_eligibility(frame, {"missing_exact_target_lag_24h": pd.Series([True])})
    reasons = str(result.loc[0, "feature_missing_reasons"])
    assert "missing_source_weather:ghi_w_m2" in reasons
    assert "missing_exact_target_lag_24h" in reasons
    assert not result.loc[0, "feature_eligible"]


def test_cli_success_repeated_sites_and_failure(tmp_path: Path) -> None:
    """CLI supports repeated site arguments and process-compatible failures."""
    source, site = _inputs(tmp_path)
    assert (
        main(
            [
                "--input",
                str(source),
                "--site-config",
                str(site),
                "--output",
                str(tmp_path / "features.csv"),
                "--report-dir",
                str(tmp_path / "reports"),
                "--target-lag-hours",
                "24",
                "--rolling-window-hours",
                "3",
                "--disable-weather-lags",
            ]
        )
        == 0
    )
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(site.read_text(encoding="utf-8"), encoding="utf-8")
    assert (
        main(
            [
                "--input",
                str(source),
                "--site-config",
                str(site),
                "--site-config",
                str(duplicate),
                "--output",
                str(tmp_path / "bad.csv"),
            ]
        )
        == 1
    )


def test_disable_feature_groups(tmp_path: Path) -> None:
    """Configuration switches remove predictor groups without deleting source rows."""
    source, site = _inputs(tmp_path)
    result = run_feature_pipeline(
        source,
        [site],
        tmp_path / "minimal.csv",
        tmp_path / "reports",
        FeatureConfig(
            include_raw_weather_features=False,
            include_site_metadata_features=False,
            include_weather_lags=False,
            include_target_history=False,
        ),
    )
    assert "ghi_w_m2" not in result.contract.predictor_columns
    assert "installed_capacity_kwp" not in result.dataframe
    assert not any("_lag_" in name for name in result.contract.predictor_columns)
