"""Phase 7 configuration, rolling-origin splitting, and search tests."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from solarpulse_ai.training.advanced_config import AdvancedTrainingConfig, XGBoostSearchSpace
from solarpulse_ai.training.cross_validation import RollingOriginSplitter
from solarpulse_ai.training.experiment import make_run_id, prepare_run_directory
from solarpulse_ai.training.hyperparameters import generate_candidates, rank_candidates


def test_advanced_configuration_defaults_and_json_round_trip(tmp_path: Path) -> None:
    """Defaults are safe, strict, and reproducibly serialised."""
    config = AdvancedTrainingConfig()
    assert config.forecast_horizon_hours == 24
    assert config.cross_validation_gap_hours == 24
    assert config.search_budget == 24
    path = tmp_path / "config.json"
    config.to_json(path)
    assert AdvancedTrainingConfig.from_json(path) == config
    assert json.loads(path.read_text(encoding="utf-8"))["selection_metric"] == "mae"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cross_validation_splits", 1, "greater than or equal"),
        ("search_budget", 0, "greater than"),
        ("cross_validation_gap_hours", 23, "at least"),
        ("validation_window_hours", 0, "greater than"),
        ("n_jobs", 0, "must not be zero"),
        ("selection_metric", "r2", "must be one of"),
    ],
)
def test_invalid_advanced_configuration(field: str, value: object, message: str) -> None:
    """Invalid durations, counts, and policy combinations fail clearly."""
    with pytest.raises(ValidationError, match=message):
        AdvancedTrainingConfig.model_validate({field: value})


def _hourly_frame(hours: int = 20, sites: int = 2) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame(
        [
            {"timestamp": timestamp, "site_id": f"site-{site}", "split": "train"}
            for timestamp in timestamps
            for site in range(sites)
        ]
    )


def test_rolling_origin_expands_and_keeps_timestamps_together() -> None:
    """Folds expand, keep fixed validation windows, preserve gaps, and group sites."""
    frame = _hourly_frame()
    folds = RollingOriginSplitter(
        n_splits=2,
        gap_hours=2,
        minimum_training_hours=8,
        validation_window_hours=4,
    ).split(frame)
    assert len(folds[0].training_indices) < len(folds[1].training_indices)
    assert len(folds[0].validation_indices) == len(folds[1].validation_indices) == 8
    for fold in folds:
        training = frame.loc[list(fold.training_indices)]
        validation = frame.loc[list(fold.validation_indices)]
        assert training["timestamp"].max() < validation["timestamp"].min()
        assert validation["timestamp"].min() - training["timestamp"].max() == pd.Timedelta(hours=3)
        assert validation.groupby("timestamp")["site_id"].nunique().eq(2).all()
        assert fold.boundary.training_row_count == len(training)
        assert fold.boundary.validation_site_count == 2


def test_rolling_origin_rejects_unsorted_insufficient_and_nontrain_leakage() -> None:
    """Invalid chronology/history fails and tuning input remains train-labelled."""
    splitter = RollingOriginSplitter(
        n_splits=2, gap_hours=2, minimum_training_hours=8, validation_window_hours=4
    )
    with pytest.raises(ValueError, match="insufficient"):
        splitter.split(_hourly_frame(10))
    unsorted = _hourly_frame().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        splitter.split(unsorted)
    train_only = _hourly_frame()
    folds = splitter.split(train_only)
    assert all(
        set(train_only.loc[list(fold.validation_indices), "split"]) == {"train"} for fold in folds
    )


def test_candidate_generation_and_ranking_are_deterministic() -> None:
    """A fixed seed fixes candidate order and metric/stability/complexity ranking."""
    space = XGBoostSearchSpace(
        n_estimators=(10, 20),
        max_depth=(2, 3),
        learning_rate=(0.05,),
        min_child_weight=(1,),
        subsample=(1,),
        colsample_bytree=(1,),
        reg_alpha=(0,),
        reg_lambda=(1,),
        gamma=(0,),
    )
    first = generate_candidates(space, budget=3, random_seed=42)
    assert first == generate_candidates(space, budget=3, random_seed=42)
    assert len(first) == 3
    ranked = rank_candidates(
        [
            {
                **first[0],
                "status": "succeeded",
                "mean_mae": 1.0,
                "std_mae": 0.2,
            },
            {
                **first[1],
                "status": "succeeded",
                "mean_mae": 1.0,
                "std_mae": 0.1,
            },
            {**first[2], "status": "failed", "failure_message": "expected test failure"},
        ],
        "mae",
    )
    assert ranked[0]["candidate_id"] == first[1]["candidate_id"]
    assert ranked[-1]["candidate_rank"] is None
    with pytest.raises(ValueError, match="all hyperparameter candidates failed"):
        rank_candidates([ranked[-1]], "mae")


def test_run_identifiers_are_safe_and_duplicate_protected(tmp_path: Path) -> None:
    """Generated and explicit run IDs are readable without silent overwrites."""
    run_id = make_run_id({"seed": 42}, now=datetime(2025, 1, 2, tzinfo=UTC))
    assert run_id.startswith("20250102T000000000000Z-")
    prepare_run_directory(tmp_path, "controlled-run")
    with pytest.raises(ValueError, match="already exists"):
        prepare_run_directory(tmp_path, "controlled-run")
    with pytest.raises(ValueError, match="unsafe"):
        prepare_run_directory(tmp_path, "../escape")
