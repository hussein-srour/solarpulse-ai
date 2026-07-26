"""Tests for the measured-generation input contract."""

from pathlib import Path

import pandas as pd
import pytest

from solarpulse_ai.data.errors import DataValidationError
from solarpulse_ai.data.generation import read_generation_csv, validate_generation_dataframe


def _generation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2025-01-01T03:00:00+03:00", "2025-01-01T01:00:00Z"],
            "site_id": ["example-site", "example-site"],
            "ac_energy_kwh": [0.0, 4.2],
        }
    )


def test_valid_generation_csv_is_utc_normalized(tmp_path: Path) -> None:
    """Valid measured records load, normalize to UTC, and sort."""
    path = tmp_path / "generation.csv"
    _generation().to_csv(path, index=False)

    result = read_generation_csv(path)

    assert result["timestamp"].dt.tz is not None
    assert result["timestamp"].dt.hour.tolist() == [0, 1]
    assert result["ac_energy_kwh"].tolist() == [0.0, 4.2]


@pytest.mark.parametrize("value", [-0.1, float("inf"), "not-energy", None])
def test_invalid_generation_energy_is_rejected(value: object) -> None:
    """Energy must be present, finite, numeric, and non-negative."""
    dataframe = _generation()
    dataframe["ac_energy_kwh"] = pd.Series([value, 4.2], dtype="object")

    with pytest.raises(DataValidationError, match="invalid_generation_energy"):
        validate_generation_dataframe(dataframe)


def test_duplicate_generation_records_are_rejected_after_utc_conversion() -> None:
    """Equivalent instants with different offsets are duplicate keys."""
    dataframe = _generation()
    dataframe.loc[1, "timestamp"] = "2025-01-01T00:00:00Z"

    with pytest.raises(DataValidationError, match="duplicate_generation_record"):
        validate_generation_dataframe(dataframe)


def test_invalid_generation_shape_and_values_are_reported() -> None:
    """Missing fields, empty data, blank sites, and naive times are invalid."""
    with pytest.raises(DataValidationError, match="empty_generation"):
        validate_generation_dataframe(pd.DataFrame())

    with pytest.raises(DataValidationError, match="missing_generation_column"):
        validate_generation_dataframe(_generation().drop(columns="site_id"))

    dataframe = _generation()
    dataframe.loc[0, "site_id"] = ""
    dataframe.loc[1, "timestamp"] = "2025-01-01T01:00:00"
    with pytest.raises(DataValidationError) as captured:
        validate_generation_dataframe(dataframe)

    codes = {issue.code for issue in captured.value.issues}
    assert codes == {"invalid_generation_site_id", "invalid_generation_timestamp"}
