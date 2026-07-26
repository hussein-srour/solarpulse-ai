"""Tests for hourly CSV ingestion and validation."""

from pathlib import Path

import pandas as pd
import pytest

from solarpulse_ai.data.errors import CSVIngestionError, DataValidationError
from solarpulse_ai.data.ingestion import ingest_hourly_csv, main, read_hourly_csv


@pytest.fixture
def valid_data() -> pd.DataFrame:
    """Return two illustrative, deliberately unsorted hourly observations."""
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01T11:00:00+03:00", "2026-01-01T07:00:00Z"],
            "site_id": ["site-a", "site-a"],
            "ac_energy_kwh": [14.2, 10.5],
            "ghi_w_m2": [710.0, 540.0],
            "ambient_temperature_c": [28.4, 25.1],
            "cloud_cover_pct": [18.0, 24.0],
            "relative_humidity_pct": [61.0, 68.0],
            "wind_speed_m_s": [3.2, 2.5],
        }
    )


def _write_csv(path: Path, dataframe: pd.DataFrame) -> Path:
    dataframe.to_csv(path, index=False)
    return path


def test_valid_dataset_is_normalized_and_sorted(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """Valid records are UTC-normalized, numeric, and sorted by timestamp."""
    source = _write_csv(tmp_path / "input.csv", valid_data)
    output = tmp_path / "processed" / "output.csv"

    result = ingest_hourly_csv(source, output)

    assert output.exists()
    assert str(result["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].dt.hour.tolist() == [7, 8]
    assert result["ac_energy_kwh"].dtype == float


def test_missing_required_column_is_rejected(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """A missing required field produces a named validation issue."""
    source = _write_csv(tmp_path / "input.csv", valid_data.drop(columns="ghi_w_m2"))

    with pytest.raises(DataValidationError, match=r"missing_required_column.*ghi_w_m2"):
        ingest_hourly_csv(source, tmp_path / "output.csv")


def test_negative_energy_is_rejected(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """AC energy cannot be negative."""
    valid_data.loc[0, "ac_energy_kwh"] = -0.1
    source = _write_csv(tmp_path / "input.csv", valid_data)

    with pytest.raises(DataValidationError, match=r"value_out_of_range.*ac_energy_kwh"):
        ingest_hourly_csv(source, tmp_path / "output.csv")


def test_cloud_cover_above_100_is_rejected(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """Cloud cover is constrained to a percentage range."""
    valid_data.loc[0, "cloud_cover_pct"] = 100.1
    source = _write_csv(tmp_path / "input.csv", valid_data)

    with pytest.raises(DataValidationError, match=r"value_out_of_range.*cloud_cover_pct"):
        ingest_hourly_csv(source, tmp_path / "output.csv")


def test_non_numeric_measurement_is_rejected(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """Measurement fields must contain numeric values."""
    valid_data["ghi_w_m2"] = pd.Series(["unavailable", "540.0"], dtype="string")
    source = _write_csv(tmp_path / "input.csv", valid_data)

    with pytest.raises(DataValidationError, match=r"invalid_numeric_value.*ghi_w_m2"):
        ingest_hourly_csv(source, tmp_path / "output.csv")


def test_invalid_timestamp_is_rejected(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """Unparseable timestamps identify the affected CSV row."""
    valid_data.loc[0, "timestamp"] = "not-a-timestamp"
    source = _write_csv(tmp_path / "input.csv", valid_data)

    with pytest.raises(DataValidationError, match=r"invalid_timestamp.*CSV rows=2"):
        ingest_hourly_csv(source, tmp_path / "output.csv")


def test_timezone_naive_timestamp_is_rejected(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """A timestamp without an explicit timezone is not silently assumed to be UTC."""
    valid_data.loc[0, "timestamp"] = "2026-01-01T08:00:00"
    source = _write_csv(tmp_path / "input.csv", valid_data)

    with pytest.raises(DataValidationError, match="timezone-aware"):
        ingest_hourly_csv(source, tmp_path / "output.csv")


def test_duplicate_site_timestamp_is_rejected(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """Duplicate UTC instants are detected even when source offsets differ."""
    duplicate = pd.concat([valid_data.iloc[[0]], valid_data.iloc[[0]]], ignore_index=True)
    duplicate.loc[1, "timestamp"] = "2026-01-01T08:00:00Z"
    source = _write_csv(tmp_path / "input.csv", duplicate)

    with pytest.raises(DataValidationError, match="duplicate_observation"):
        ingest_hourly_csv(source, tmp_path / "output.csv")


def test_completely_empty_csv_is_rejected(tmp_path: Path) -> None:
    """A zero-byte file fails with a clear ingestion error."""
    source = tmp_path / "empty.csv"
    source.touch()

    with pytest.raises(CSVIngestionError, match="completely empty"):
        read_hourly_csv(source)


def test_header_only_csv_is_rejected(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """A CSV with the canonical header but no records is still empty."""
    source = _write_csv(tmp_path / "empty.csv", valid_data.iloc[0:0])

    with pytest.raises(CSVIngestionError, match="no data rows"):
        read_hourly_csv(source)


def test_optional_columns_may_be_absent(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """The required-only canonical shape is valid."""
    source = _write_csv(tmp_path / "input.csv", valid_data)

    result = ingest_hourly_csv(source, tmp_path / "output.csv")

    assert "dni_w_m2" not in result.columns
    assert len(result) == 2


def test_invalid_optional_value_is_rejected(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """Optional fields obey their constraints whenever supplied."""
    valid_data["precipitation_mm"] = [0.0, -0.1]
    source = _write_csv(tmp_path / "input.csv", valid_data)

    with pytest.raises(DataValidationError, match=r"value_out_of_range.*precipitation_mm"):
        ingest_hourly_csv(source, tmp_path / "output.csv")


def test_processed_file_contains_validated_records(
    tmp_path: Path, valid_data: pd.DataFrame
) -> None:
    """Successful ingestion creates a readable processed CSV."""
    source = _write_csv(tmp_path / "input.csv", valid_data)
    output = tmp_path / "nested" / "validated.csv"

    ingest_hourly_csv(source, output)
    persisted = pd.read_csv(output)

    assert len(persisted) == len(valid_data)
    assert persisted["timestamp"].iloc[0].endswith("+00:00")


def test_missing_input_file_is_rejected(tmp_path: Path) -> None:
    """The ingestion layer checks input existence before reading."""
    with pytest.raises(CSVIngestionError, match="does not exist"):
        read_hourly_csv(tmp_path / "missing.csv")


def test_cli_returns_zero_on_success(tmp_path: Path, valid_data: pd.DataFrame) -> None:
    """The CLI uses exit code zero and writes output for valid input."""
    source = _write_csv(tmp_path / "input.csv", valid_data)
    output = tmp_path / "output.csv"

    exit_code = main(["--input", str(source), "--output", str(output)])

    assert exit_code == 0
    assert output.exists()


def test_cli_returns_nonzero_on_validation_failure(
    tmp_path: Path, valid_data: pd.DataFrame
) -> None:
    """The CLI returns a non-zero exit code for invalid input."""
    source = _write_csv(tmp_path / "input.csv", valid_data.drop(columns="site_id"))

    exit_code = main(["--input", str(source), "--output", str(tmp_path / "output.csv")])

    assert exit_code != 0
