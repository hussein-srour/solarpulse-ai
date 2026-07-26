"""Tests for strict measured-generation and weather joining."""

from pathlib import Path

import pandas as pd
import pytest

from solarpulse_ai.data.errors import DatasetJoinError, DataValidationError
from solarpulse_ai.data.join import join_csv_files, join_generation_weather, main
from solarpulse_ai.data.schemas import CANONICAL_COLUMNS


def _generation(hours: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [f"2025-01-01T{hour:02d}:00:00Z" for hour in hours],
            "site_id": ["example-site"] * len(hours),
            "ac_energy_kwh": [float(hour) for hour in hours],
        }
    )


def _weather(hours: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    count = len(hours)
    return pd.DataFrame(
        {
            "timestamp": [f"2025-01-01T{hour:02d}:00:00Z" for hour in hours],
            "site_id": ["example-site"] * count,
            "ambient_temperature_c": [25.0] * count,
            "relative_humidity_pct": [75.0] * count,
            "precipitation_mm": [0.0] * count,
            "cloud_cover_pct": [30.0] * count,
            "wind_speed_m_s": [2.0] * count,
            "ghi_w_m2": [100.0] * count,
            "dni_w_m2": [60.0] * count,
            "dhi_w_m2": [40.0] * count,
        }
    )


def test_successful_one_to_one_join_returns_canonical_structure() -> None:
    """Matching validated inputs produce Phase 2 canonical output."""
    result = join_generation_weather(_generation(), _weather())

    assert result.columns.tolist() == [
        column for column in CANONICAL_COLUMNS if column in result.columns
    ]
    assert result["ac_energy_kwh"].tolist() == [0.0, 1.0]
    assert str(result["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_missing_weather_hour_is_reported() -> None:
    """A gap inside the generation range is never silently filled."""
    with pytest.raises(DatasetJoinError, match="Missing weather hours"):
        join_generation_weather(_generation((0, 1, 2)), _weather((0, 2)))


def test_unmatched_generation_timestamp_is_reported() -> None:
    """Generation without a corresponding weather key is explicit."""
    with pytest.raises(DatasetJoinError, match="cannot be matched"):
        join_generation_weather(_generation((0, 1)), _weather((0,)))


def test_duplicate_join_keys_are_rejected_on_both_sides() -> None:
    """One-to-one input uniqueness is mandatory."""
    duplicate_generation = pd.concat([_generation((0,)), _generation((0,))])
    with pytest.raises(DataValidationError, match="duplicate_generation_record"):
        join_generation_weather(duplicate_generation, _weather((0,)))

    duplicate_weather = pd.concat([_weather((0,)), _weather((0,))])
    with pytest.raises(DataValidationError, match="duplicate_weather_record"):
        join_generation_weather(_generation((0,)), duplicate_weather)


def test_final_output_is_checked_by_phase_two_validator() -> None:
    """Canonical range rules still apply after the join."""
    weather = _weather()
    weather["ghi_w_m2"] = [-1.0, 100.0]

    with pytest.raises(DataValidationError, match="invalid_weather_value"):
        join_generation_weather(_generation(), weather)


def test_join_csv_command_writes_valid_output(tmp_path: Path) -> None:
    """The join service and command persist only fully validated output."""
    generation_path = tmp_path / "generation.csv"
    weather_path = tmp_path / "weather.csv"
    output_path = tmp_path / "processed" / "training.csv"
    _generation().to_csv(generation_path, index=False)
    _weather().to_csv(weather_path, index=False)

    result = join_csv_files(generation_path, weather_path, output_path)
    exit_code = main(
        [
            "--generation",
            str(generation_path),
            "--weather",
            str(weather_path),
            "--output",
            str(tmp_path / "command.csv"),
        ]
    )

    assert len(result) == 2
    assert output_path.exists()
    assert exit_code == 0


def test_join_cli_returns_nonzero_for_invalid_input(tmp_path: Path) -> None:
    """The command fails safely and does not write invalid output."""
    generation_path = tmp_path / "generation.csv"
    weather_path = tmp_path / "weather.csv"
    generation = _generation()
    generation.loc[0, "ac_energy_kwh"] = -1
    generation.to_csv(generation_path, index=False)
    _weather().to_csv(weather_path, index=False)

    exit_code = main(
        [
            "--generation",
            str(generation_path),
            "--weather",
            str(weather_path),
            "--output",
            str(tmp_path / "output.csv"),
        ]
    )

    assert exit_code == 1
    assert not (tmp_path / "output.csv").exists()
