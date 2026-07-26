"""Feature contract, eligibility, and chronological-split tests."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from solarpulse_ai.training.config import TrainingConfig
from solarpulse_ai.training.contracts import load_manifest, validate_contract
from solarpulse_ai.training.dataset import load_feature_dataset, prepare_dataset


def _validated(frame: pd.DataFrame, manifest: dict[str, object], minimum: int = 1) -> None:
    contract = validate_contract(frame, manifest)
    prepare_dataset(
        frame,
        contract,
        TrainingConfig(
            minimum_train_rows=minimum,
            minimum_validation_rows=minimum,
            minimum_test_rows=minimum,
            enabled_models=("persistence",),
        ),
    )


def test_valid_contract_and_eligibility_audit(
    training_inputs: tuple[Path, Path, pd.DataFrame, dict[str, object]],
) -> None:
    """Only eligible rows survive and exclusion summaries retain reasons."""
    feature_path, _, frame, manifest = training_inputs
    frame.loc[[1, 35, 70], "feature_eligible"] = False
    frame.loc[[1, 35, 70], "feature_missing_reasons"] = "missing_weather"
    contract = validate_contract(frame, manifest)
    prepared = prepare_dataset(
        frame,
        contract,
        TrainingConfig(
            minimum_train_rows=1,
            minimum_validation_rows=1,
            minimum_test_rows=1,
        ),
    )
    assert len(load_feature_dataset(feature_path)) == 90
    assert len(prepared.frame) == 87
    assert prepared.eligibility["excluded_counts_by_split"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert prepared.eligibility["eligibility_reason_counts"] == {"missing_weather": 3}
    assert "ac_energy_lag_24h" in prepared.constant_predictors


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns=["ac_energy_kwh"]), "missing declared"),
        (lambda frame: frame.drop(columns=["split"]), "missing declared"),
        (lambda frame: frame.assign(split="unknown"), "split must contain"),
        (
            lambda frame: frame.assign(
                timestamp=pd.date_range("2025-01-01", periods=len(frame), freq="h", tz="UTC")
                .to_series()
                .sample(frac=1, random_state=1)
                .to_numpy()
            ),
            "chronologically sorted",
        ),
        (
            lambda frame: frame.assign(
                timestamp=frame["timestamp"].mask(frame.index == 1, frame.loc[0, "timestamp"])
            ),
            "duplicate site_id",
        ),
        (
            lambda frame: frame.assign(
                ac_energy_kwh=frame["ac_energy_kwh"].mask(frame.index == 0, -1)
            ),
            "finite and non-negative",
        ),
        (
            lambda frame: frame.assign(
                weather_signal=frame["weather_signal"].mask(frame.index == 0, np.inf)
            ),
            "infinity",
        ),
    ],
)
def test_invalid_datasets_are_rejected(
    training_inputs: tuple[Path, Path, pd.DataFrame, dict[str, object]],
    mutation: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    """Contract, identity, target, infinity, and time violations fail clearly."""
    _, _, frame, manifest = training_inputs
    with pytest.raises(ValueError, match=message):
        _validated(mutation(frame.copy()), manifest)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"target_column": "wrong"}, "target_column"),
        ({"key_columns": ["site_id", "timestamp"]}, "key_columns"),
        (
            {
                "predictor_columns": ["weather_signal", "weather_signal"],
                "numerical_columns": ["weather_signal", "weather_signal"],
            },
            "duplicate predictor",
        ),
        (
            {
                "predictor_columns": ["weather_signal", "ac_energy_kwh"],
                "numerical_columns": ["weather_signal", "ac_energy_kwh"],
            },
            "forbidden",
        ),
        ({"forecast_horizon_hours": 12}, "24-hour"),
        ({"numerical_columns": ["weather_signal"]}, "type mismatch"),
        ({"output_row_count": 12}, "row count"),
    ],
)
def test_manifest_disagreement_is_rejected(
    training_inputs: tuple[Path, Path, pd.DataFrame, dict[str, object]],
    change: dict[str, object],
    message: str,
) -> None:
    """The manifest is authoritative but must agree exactly with the CSV."""
    _, _, frame, manifest = training_inputs
    changed = {**manifest, **change}
    with pytest.raises(ValueError, match=message):
        validate_contract(frame, changed)


def test_invalid_json_manifest(tmp_path: Path) -> None:
    """Malformed and non-object JSON fail as manifest errors."""
    path = tmp_path / "manifest.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid feature manifest"):
        load_manifest(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_manifest(path)


@pytest.mark.parametrize("split_index", [0, 30, 60])
def test_empty_split_after_eligibility_fails(
    training_inputs: tuple[Path, Path, pd.DataFrame, dict[str, object]],
    split_index: int,
) -> None:
    """No required period may disappear after eligibility filtering."""
    _, _, frame, manifest = training_inputs
    split = ("train", "validation", "test")[split_index // 30]
    frame.loc[frame["split"].eq(split), "feature_eligible"] = False
    with pytest.raises(ValueError, match=split):
        _validated(frame, manifest)


def test_categorical_unknown_and_boolean_contract(
    training_inputs: tuple[Path, Path, pd.DataFrame, dict[str, object]],
) -> None:
    """Explicit categorical and boolean predictors validate without raw object leakage."""
    _, _, frame, manifest = training_inputs
    frame["weather_class"] = ["clear"] * 30 + ["cloudy"] * 60
    frame["flag"] = ["true"] * len(frame)
    predictors = [*manifest["predictor_columns"], "weather_class", "flag"]  # type: ignore[misc]
    expected_types = cast(dict[str, str], manifest["expected_data_types"])
    changed = {
        **manifest,
        "predictor_columns": predictors,
        "numerical_columns": manifest["numerical_columns"],
        "categorical_columns": ["weather_class"],
        "boolean_columns": ["flag"],
        "expected_data_types": {
            **expected_types,
            "weather_class": "object",
            "flag": "object",
        },
    }
    _validated(frame, changed)
