"""Lightweight synthetic Phase 7 estimator, tracking, leakage, and CLI tests."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarpulse_ai.training.advanced import main
from solarpulse_ai.training.advanced_config import AdvancedTrainingConfig, XGBoostSearchSpace
from solarpulse_ai.training.advanced_estimators import build_xgboost_model
from solarpulse_ai.training.advanced_pipeline import run_advanced_training
from solarpulse_ai.training.contracts import TrainingContract


def _space() -> XGBoostSearchSpace:
    return XGBoostSearchSpace(
        n_estimators=(5, 8),
        max_depth=(2,),
        learning_rate=(0.1,),
        min_child_weight=(1,),
        subsample=(1,),
        colsample_bytree=(1,),
        reg_alpha=(0,),
        reg_lambda=(1,),
        gamma=(0,),
    )


def _config(**changes: object) -> AdvancedTrainingConfig:
    values: dict[str, object] = {
        "cross_validation_splits": 2,
        "cross_validation_gap_hours": 24,
        "minimum_training_hours": 24,
        "validation_window_hours": 4,
        "search_budget": 2,
        "n_jobs": 1,
        "compare_with_phase6_baselines": False,
        "refit_selected_model": False,
        "search_space": _space(),
    }
    values.update(changes)
    return AdvancedTrainingConfig.model_validate(values)


def _advanced_inputs(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    hours = 84
    timestamps = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    hour = np.arange(hours, dtype=float)
    daylight = ((hour % 24 >= 6) & (hour % 24 <= 17)).astype(float)
    target = daylight * (2.0 + hour / 20)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "site_id": ["site-a"] * hours,
            "ac_energy_kwh": target,
            "split": ["train"] * 60 + ["validation"] * 12 + ["test"] * 12,
            "feature_eligible": [True] * hours,
            "feature_missing_count": [0] * hours,
            "feature_missing_reasons": [""] * hours,
            "weather_signal": target + np.sin(hour) * 0.01,
            "ac_energy_lag_24h": np.maximum(target - 0.2, 0),
            "is_daylight": daylight,
            "installed_capacity_kwp": np.full(hours, 20.0),
        }
    )
    predictors = [
        "weather_signal",
        "ac_energy_lag_24h",
        "is_daylight",
        "installed_capacity_kwp",
    ]
    manifest = {
        "source_row_count": hours,
        "output_row_count": hours,
        "forecast_horizon_hours": 24,
        "key_columns": ["timestamp", "site_id"],
        "target_column": "ac_energy_kwh",
        "predictor_columns": predictors,
        "metadata_columns": [
            "split",
            "feature_eligible",
            "feature_missing_count",
            "feature_missing_reasons",
        ],
        "categorical_columns": [],
        "numerical_columns": predictors,
        "expected_data_types": {
            "timestamp": "datetime64[ns, UTC]",
            "site_id": "object",
            "ac_energy_kwh": "float64",
            **dict.fromkeys(predictors, "float64"),
        },
    }
    feature_path = tmp_path / "features.csv"
    manifest_path = tmp_path / "manifest.json"
    frame.to_csv(feature_path, index=False)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return feature_path, manifest_path, frame


def test_xgboost_estimator_is_deterministic_and_handles_missing() -> None:
    """Fixed seeds reproduce finite predictions with explicit train-fitted imputation."""
    contract = TrainingContract(
        predictors=("value",),
        numerical=("value",),
        categorical=(),
        boolean=(),
        metadata=("split",),
        forecast_horizon_hours=24,
    )
    features = pd.DataFrame({"value": [1.0, np.nan, 2.0, 4.0]})
    target = np.array([1.0, 1.5, 2.0, 4.0])
    parameters = _space().model_dump(mode="python")
    scalar = {key: values[0] for key, values in parameters.items()}
    first = build_xgboost_model(contract, scalar, random_seed=42, n_jobs=1).fit(features, target)
    second = build_xgboost_model(contract, scalar, random_seed=42, n_jobs=1).fit(features, target)
    np.testing.assert_allclose(first.predict(features), second.predict(features))
    assert np.isfinite(first.predict(features)).all()


def test_advanced_run_tracks_outputs_registry_and_reload(tmp_path: Path) -> None:
    """A synthetic controlled run creates verified outputs without real-performance claims."""
    feature_path, manifest_path, _ = _advanced_inputs(tmp_path)
    result = run_advanced_training(
        feature_path,
        manifest_path,
        tmp_path / "reports",
        tmp_path / "artifacts",
        tmp_path / "registry.json",
        config=_config(),
        run_id="synthetic-controlled",
    )
    assert result.selected_model == "tuned_xgboost"
    assert joblib.load(result.model_path) is not None
    expected = {
        "run_config.json",
        "environment.json",
        "dataset_fingerprint.json",
        "feature_manifest_snapshot.json",
        "cross_validation_folds.json",
        "hyperparameter_candidates.csv",
        "cross_validation_results.csv",
        "validation_leaderboard.csv",
        "validation_predictions.csv",
        "test_predictions.csv",
        "final_metrics.json",
        "model_selection.json",
        "feature_importance.csv",
        "experiment_summary.md",
        "model_card.md",
    }
    assert expected <= {path.name for path in result.experiment_directory.iterdir()}
    assert len(list((result.experiment_directory / "charts").glob("*.png"))) >= 10
    assert not plt.get_fignums()
    fingerprint = json.loads(
        (result.experiment_directory / "dataset_fingerprint.json").read_text(encoding="utf-8")
    )
    assert "records" not in fingerprint
    summary = (result.experiment_directory / "experiment_summary.md").read_text(encoding="utf-8")
    assert "not real plant performance" in summary


def test_persistence_can_remain_champion_after_advanced_tuning(tmp_path: Path) -> None:
    """Complexity never grants XGBoost an automatic validation win."""
    feature_path, manifest_path, frame = _advanced_inputs(tmp_path)
    frame["ac_energy_lag_24h"] = frame["ac_energy_kwh"]
    frame.to_csv(feature_path, index=False)
    result = run_advanced_training(
        feature_path,
        manifest_path,
        tmp_path / "reports",
        tmp_path / "artifacts",
        tmp_path / "registry.json",
        config=_config(
            compare_with_phase6_baselines=True,
            search_budget=1,
            register_model=False,
        ),
        run_id="persistence-wins",
    )
    assert result.selected_model == "persistence"
    leaderboard = pd.read_csv(result.experiment_directory / "validation_leaderboard.csv")
    assert leaderboard.iloc[0]["model_identifier"] == "persistence"
    assert bool(leaderboard.iloc[0]["selected"])


def test_test_target_mutation_does_not_change_cv_or_selection(tmp_path: Path) -> None:
    """Untouched test targets cannot alter tuning candidates or validation selection."""
    feature_path, manifest_path, frame = _advanced_inputs(tmp_path)
    first = run_advanced_training(
        feature_path,
        manifest_path,
        tmp_path / "reports-a",
        tmp_path / "artifacts-a",
        tmp_path / "registry-a.json",
        config=_config(register_model=False),
        run_id="first",
    )
    frame.loc[frame["split"].eq("test"), "ac_energy_kwh"] = 9999.0
    frame.to_csv(feature_path, index=False)
    second = run_advanced_training(
        feature_path,
        manifest_path,
        tmp_path / "reports-b",
        tmp_path / "artifacts-b",
        tmp_path / "registry-b.json",
        config=_config(register_model=False),
        run_id="second",
    )
    for name in ("hyperparameter_candidates.csv", "cross_validation_results.csv"):
        first_frame = pd.read_csv(first.experiment_directory / name)
        second_frame = pd.read_csv(second.experiment_directory / name)
        duration_columns = [
            column for column in first_frame if str(column).endswith("duration_seconds")
        ]
        first_frame = first_frame.drop(columns=duration_columns)
        second_frame = second_frame.drop(columns=duration_columns)
        pd.testing.assert_frame_equal(first_frame, second_frame)
    first_selection = json.loads(
        (first.experiment_directory / "model_selection.json").read_text(encoding="utf-8")
    )
    second_selection = json.loads(
        (second.experiment_directory / "model_selection.json").read_text(encoding="utf-8")
    )
    assert (
        first_selection["selected_model_identifier"]
        == second_selection["selected_model_identifier"]
    )


def test_advanced_cli_success_failure_and_registry_disabled(tmp_path: Path) -> None:
    """CLI accepts JSON config, protects duplicates, and reports bad input non-zero."""
    feature_path, manifest_path, _ = _advanced_inputs(tmp_path)
    config_path = tmp_path / "advanced.json"
    _config(register_model=False).to_json(config_path)
    arguments = [
        "--features",
        str(feature_path),
        "--feature-manifest",
        str(manifest_path),
        "--output-dir",
        str(tmp_path / "reports"),
        "--artifact-dir",
        str(tmp_path / "artifacts"),
        "--registry",
        str(tmp_path / "registry.json"),
        "--config",
        str(config_path),
        "--run-id",
        "cli-synthetic",
        "--no-register-model",
    ]
    assert main(arguments) == 0
    assert not (tmp_path / "registry.json").exists()
    assert main(arguments) == 1
    bad = arguments.copy()
    bad[1] = str(tmp_path / "missing.csv")
    bad[bad.index("cli-synthetic")] = "missing-input"
    assert main(bad) == 1
