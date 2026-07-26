"""Tests for Phase 4 exploratory analysis and readiness reporting."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from solarpulse_ai.analysis.charts import CHART_FILENAMES, generate_charts
from solarpulse_ai.analysis.config import AnalysisThresholds, SplitProportions
from solarpulse_ai.analysis.correlation import analyse_correlations
from solarpulse_ai.analysis.eda import main
from solarpulse_ai.analysis.pipeline import run_analysis
from solarpulse_ai.analysis.profile import profile_dataset
from solarpulse_ai.analysis.quality import diagnose_quality
from solarpulse_ai.analysis.readiness import assess_readiness
from solarpulse_ai.analysis.serialization import to_json_value
from solarpulse_ai.analysis.splits import plan_chronological_splits
from solarpulse_ai.analysis.statistics import (
    analyse_target,
    analyse_weather,
    describe_series,
)
from solarpulse_ai.analysis.temporal import analyse_temporal
from solarpulse_ai.data.validation import validate_hourly_dataframe


def _canonical(
    hours: int = 72,
    sites: tuple[str, ...] = ("site-a",),
    start: str = "2025-01-01T00:00:00Z",
    *,
    optional: bool = True,
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=hours, freq="h")
    rows: list[dict[str, object]] = []
    for site_number, site_id in enumerate(sites):
        for position, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            daylight = 6 <= hour <= 18
            ghi = float(max(0, 800 - abs(12 - hour) * 120)) if daylight else 0.0
            generation = ghi / 80 + site_number * 0.1
            row: dict[str, object] = {
                "timestamp": timestamp,
                "site_id": site_id,
                "ac_energy_kwh": generation,
                "ghi_w_m2": ghi,
                "ambient_temperature_c": 20.0 + hour / 2,
                "cloud_cover_pct": float(position % 100),
                "relative_humidity_pct": float(80 - position % 30),
                "wind_speed_m_s": 1.0 + position % 5,
            }
            if optional:
                row.update(
                    {
                        "dni_w_m2": ghi * 0.7,
                        "dhi_w_m2": ghi * 0.3,
                        "module_temperature_c": 22.0 + hour,
                        "precipitation_mm": 0.0,
                        "inverter_availability_pct": 100.0,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _validated(
    hours: int = 72,
    sites: tuple[str, ...] = ("site-a",),
    start: str = "2025-01-01T00:00:00Z",
    *,
    optional: bool = True,
) -> pd.DataFrame:
    return validate_hourly_dataframe(
        _canonical(hours=hours, sites=sites, start=start, optional=optional)
    )


def test_profile_covers_single_multi_site_gaps_and_partial_days() -> None:
    """Profiles report site coverage, gaps, missingness, and complete/partial days."""
    single = _validated(hours=30, optional=False)
    single = single.drop(index=5).reset_index(drop=True)
    profile = profile_dataset(single)

    assert profile["number_of_sites"] == 1
    assert profile["missing_timestamp_count"] == 1
    assert profile["number_of_complete_days"] == 0
    assert profile["number_of_partial_days"] == 2
    assert "dni_w_m2" in profile["optional_columns_absent"]
    assert profile["duplicate_count"] == 0
    assert profile["memory_usage_bytes"] > 0

    multiple = profile_dataset(_validated(hours=24, sites=("site-b", "site-a")))
    assert multiple["site_ids"] == ["site-a", "site-b"]
    assert multiple["records_per_site"] == {"site-a": 24, "site-b": 24}
    assert multiple["number_of_complete_days"] == 2


def test_target_weather_and_temporal_statistics_are_correct() -> None:
    """Descriptive measures, percentiles, aggregations, and UTC summaries are exact."""
    measures = describe_series(pd.Series([0.0, 1.0, 2.0, 3.0]))
    assert measures["mean"] == pytest.approx(1.5)
    assert measures["median"] == pytest.approx(1.5)
    assert measures["percentile_05"] == pytest.approx(0.15)
    assert measures["percentile_25"] == pytest.approx(0.75)
    assert measures["percentile_50"] == pytest.approx(1.5)
    assert measures["percentile_75"] == pytest.approx(2.25)
    assert measures["percentile_95"] == pytest.approx(2.85)
    assert describe_series(pd.Series(dtype=float))["count"] is None

    dataframe = _validated(hours=48, optional=False)
    target = analyse_target(dataframe)
    assert target["overall"]["count"] == 48
    assert sum(target["daily_energy_totals_kwh"].values()) == pytest.approx(
        target["overall"]["total_generated_energy_kwh"]
    )
    assert len(target["daily_energy_totals_kwh"]) == 2
    assert len(target["hourly_production_profile_kwh"]) == 24
    assert target["monthly_energy_totals_kwh"] == {}
    assert set(analyse_weather(dataframe)) == {
        "ghi_w_m2",
        "ambient_temperature_c",
        "cloud_cover_pct",
        "relative_humidity_pct",
        "wind_speed_m_s",
    }

    temporal = analyse_temporal(dataframe.drop(index=12).reset_index(drop=True))
    assert temporal["timezone"] == "UTC"
    assert not temporal["hourly_continuity_acceptable"]
    assert temporal["gaps_between_observations"][0]["missing_hours"] == 1
    assert len(temporal["generation_by_hour_of_day_kwh"]) == 24


def test_quality_diagnostics_find_requested_indicators_without_modifying_data() -> None:
    """Suspicious observations become indicators while source values remain unchanged."""
    dataframe = _validated(hours=18)
    dataframe.loc[:, "ambient_temperature_c"] = 25.0
    dataframe.loc[0, ["ac_energy_kwh", "ghi_w_m2"]] = [5.0, 0.0]
    dataframe.loc[1:6, "ac_energy_kwh"] = 0.0
    dataframe.loc[1:6, "ghi_w_m2"] = 700.0
    dataframe.loc[7, "ac_energy_kwh"] = 1000.0
    dataframe.loc[8, "dni_w_m2"] = None
    dataframe = dataframe.drop(index=10).reset_index(drop=True)
    original = dataframe.copy(deep=True)

    summary, flags = diagnose_quality(
        dataframe,
        AnalysisThresholds(
            consecutive_zero_hours=4,
            outlier_iqr_multiplier=1.0,
            minimum_history_days=2,
        ),
    )
    indicators = set(flags["indicator"])
    assert {
        "positive_generation_at_very_low_irradiance",
        "near_zero_generation_at_high_irradiance",
        "long_consecutive_near_zero_generation",
        "missing_hourly_timestamp",
        "missing_optional_weather_value",
        "constant_value_sensor_column",
        "extreme_value_iqr",
        "abrupt_generation_change",
        "insufficient_historical_coverage",
    } <= indicators
    assert summary["total_flags"] == len(flags)
    pd.testing.assert_frame_equal(dataframe, original)

    second_site = dataframe.copy()
    second_site["site_id"] = "site-b"
    second_site["timestamp"] += timedelta(hours=2)
    combined = pd.concat([dataframe, second_site], ignore_index=True)
    different_summary, different_flags = diagnose_quality(
        combined, AnalysisThresholds(minimum_history_days=1)
    )
    assert different_summary["sites_have_different_date_ranges"]
    assert "sites_have_different_date_ranges" in set(different_flags["indicator"])


def test_correlations_handle_constant_and_available_columns_safely() -> None:
    """Pearson output clearly distinguishes available and constant correlations."""
    dataframe = _validated(hours=48)
    dataframe["ambient_temperature_c"] = 25.0
    summary, table = analyse_correlations(dataframe)

    constant = table.loc[table["weather_variable"] == "ambient_temperature_c"].iloc[0]
    ghi = table.loc[table["weather_variable"] == "ghi_w_m2"].iloc[0]
    assert constant["availability"] == "unavailable"
    assert constant["reason"] == "weather column is constant"
    assert ghi["availability"] == "available"
    assert ghi["correlation_with_ac_energy_kwh"] == pytest.approx(1.0)
    assert "does not prove causation" in summary["causation_note"]

    dataframe["ac_energy_kwh"] = 1.0
    _, constant_target = analyse_correlations(dataframe)
    assert set(constant_target["availability"]) == {"unavailable"}


def test_chronological_split_order_overlap_counts_and_validation() -> None:
    """Split planning preserves global order, reports sites, and rejects bad plans."""
    dataframe = _validated(hours=20, sites=("site-a", "site-b"))
    plan, labels = plan_chronological_splits(
        dataframe, SplitProportions(training=0.6, validation=0.2, testing=0.2)
    )
    periods = plan["periods"]
    assert plan["non_overlapping"]
    assert periods["training"]["ending_timestamp"] < periods["validation"]["beginning_timestamp"]
    assert periods["validation"]["ending_timestamp"] < periods["testing"]["beginning_timestamp"]
    assert set(periods["training"]["counts_by_site"]) == {"site-a", "site-b"}
    assert labels.value_counts().sum() == len(dataframe)

    with pytest.raises(ValueError, match="total 1.0"):
        SplitProportions(training=0.7, validation=0.2, testing=0.2)
    with pytest.raises(ValueError, match="greater than zero"):
        SplitProportions(training=1.0, validation=0.0, testing=0.0)
    with pytest.raises(ValueError, match="empty period"):
        plan_chronological_splits(_validated(hours=3), SplitProportions())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"low_irradiance_w_m2": -1.0}, "non-negative"),
        (
            {"low_irradiance_w_m2": 20.0, "high_irradiance_w_m2": 10.0},
            "must exceed",
        ),
        ({"near_zero_generation_kwh": -1.0}, "non-negative"),
        ({"consecutive_zero_hours": 1}, "at least 2"),
        ({"outlier_iqr_multiplier": 0.0}, "positive"),
        ({"minimum_history_days": 0}, "at least one"),
    ],
)
def test_invalid_quality_thresholds_are_rejected(kwargs: dict[str, object], message: str) -> None:
    """Diagnostic thresholds reject nonsensical configuration."""
    with pytest.raises(ValueError, match=message):
        AnalysisThresholds(**kwargs)  # type: ignore[arg-type]


def test_readiness_categories_and_rules_are_transparent() -> None:
    """Ready, warning, and blocking states follow documented target/daylight rules."""
    multi_month = _validated(
        hours=48,
        sites=("site-a", "site-b"),
        start="2025-01-31T00:00:00Z",
    )
    ready = assess_readiness(multi_month, 0, AnalysisThresholds())
    assert ready["category"] == "ready"

    warning = assess_readiness(
        _validated(hours=48, start="2025-01-31T00:00:00Z"),
        1,
        AnalysisThresholds(),
    )
    assert warning["category"] == "ready_with_warnings"
    assert warning["major_limitations_before_training"]

    constant = _validated(hours=48)
    constant["ac_energy_kwh"] = 0.0
    not_ready = assess_readiness(constant, 0, AnalysisThresholds())
    assert not_ready["category"] == "not_ready"
    assert not not_ready["target_has_sufficient_variation"]


def test_charts_are_created_and_all_figures_are_closed(tmp_path: Path) -> None:
    """Every deterministic chart is CI-safe and matplotlib figures do not leak."""
    chart_paths = generate_charts(_validated(hours=48), tmp_path)

    assert [path.name for path in chart_paths] == list(CHART_FILENAMES)
    assert all(path.is_file() and path.stat().st_size > 0 for path in chart_paths)
    assert plt.get_fignums() == []


def test_full_pipeline_writes_serialisable_reports_and_optional_splits(tmp_path: Path) -> None:
    """A private local CSV produces the complete ignored report bundle."""
    input_path = tmp_path / "private.csv"
    output_path = tmp_path / "reports" / "eda"
    _canonical(hours=48, sites=("site-b", "site-a")).to_csv(input_path, index=False)

    result = run_analysis(input_path, output_path, write_splits=True)

    assert result.dataframe[["site_id", "timestamp"]].equals(
        result.dataframe[["site_id", "timestamp"]].sort_values(["site_id", "timestamp"])
    )
    expected = {
        "dataset_profile.json",
        "dataset_report.md",
        "split_plan.json",
        "data_quality_flags.csv",
        "correlations.csv",
        "training_split.csv",
        "validation_split.csv",
        "testing_split.csv",
    }
    assert expected <= {path.name for path in output_path.iterdir()}
    assert len(list((output_path / "charts").glob("*.png"))) == len(CHART_FILENAMES)
    payload = json.loads((output_path / "dataset_profile.json").read_text())
    assert payload["identity"]["record_count"] == 96
    assert payload["model_readiness"]["category"] in {
        "ready",
        "ready_with_warnings",
        "not_ready",
    }
    assert "plant performance" in (output_path / "dataset_report.md").read_text()
    assert to_json_value(pd.Timestamp("2025-01-01", tz="UTC")) == ("2025-01-01T00:00:00+00:00")
    assert to_json_value(float("nan")) is None
    assert to_json_value(datetime(2025, 1, 1, tzinfo=UTC)) == "2025-01-01T00:00:00+00:00"


def test_cli_success_and_validation_failure(tmp_path: Path) -> None:
    """CLI returns zero for valid local input and non-zero for invalid canonical data."""
    valid_path = tmp_path / "valid.csv"
    _canonical(hours=24, optional=False).to_csv(valid_path, index=False)
    output_path = tmp_path / "output"

    assert main(["--input", str(valid_path), "--output-dir", str(output_path)]) == 0
    assert (output_path / "dataset_report.md").exists()

    invalid_path = tmp_path / "invalid.csv"
    invalid = _canonical(hours=24)
    invalid.loc[0, "ac_energy_kwh"] = -1
    invalid.to_csv(invalid_path, index=False)
    invalid_output = tmp_path / "invalid-output"
    assert main(["--input", str(invalid_path), "--output-dir", str(invalid_output)]) == 1
    assert not invalid_output.exists()
