"""End-to-end training, persistence, reporting, chart, and CLI tests."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from solarpulse_ai.training.config import TrainingConfig
from solarpulse_ai.training.pipeline import load_training_config, run_training
from solarpulse_ai.training.train import main


def _small_config(**changes: object) -> TrainingConfig:
    values: dict[str, object] = {
        "minimum_train_rows": 10,
        "minimum_validation_rows": 10,
        "minimum_test_rows": 10,
        "enabled_models": ("persistence", "dummy_mean", "ridge"),
        "random_forest_estimators": 10,
        "histogram_boosting_max_iter": 10,
        "permutation_importance_repeats": 2,
    }
    values.update(changes)
    return TrainingConfig.model_validate(values)


def test_full_training_run_persists_and_reports(
    training_inputs: tuple[Path, Path, pd.DataFrame, dict[str, object]],
    tmp_path: Path,
) -> None:
    """A synthetic run writes only generated artifacts and verifies reload predictions."""
    feature_path, manifest_path, _, _ = training_inputs
    result = run_training(
        feature_path,
        manifest_path,
        tmp_path / "models",
        tmp_path / "reports",
        "baseline-test",
        _small_config(
            enabled_models=(
                "persistence",
                "dummy_mean",
                "ridge",
                "random_forest",
                "histogram_gradient_boosting",
            )
        ),
    )
    assert result.selected_model in {
        "ridge",
        "random_forest",
        "histogram_gradient_boosting",
    }
    assert result.model_path.is_file()
    assert joblib.load(result.model_path) is not None
    model_dir = tmp_path / "models" / "baseline-test"
    for name in (
        "selected_model.joblib",
        "selected_model_metadata.json",
        "training_configuration.json",
        "training_manifest.json",
        "model_card.md",
        "feature_manifest_snapshot.json",
    ):
        assert (model_dir / name).is_file()
    report_dir = tmp_path / "reports"
    for name in (
        "training_summary.json",
        "training_summary.md",
        "validation_model_comparison.csv",
        "validation_metrics_by_site.csv",
        "test_metrics.csv",
        "test_metrics_by_site.csv",
        "feature_importance.csv",
        "validation_predictions.csv",
        "test_predictions.csv",
    ):
        assert (report_dir / name).is_file()
    assert len(list((report_dir / "charts").glob("*.png"))) == 10
    assert not plt.get_fignums()
    predictions = pd.read_csv(report_dir / "test_predictions.csv")
    assert "weather_signal" not in predictions
    assert set(predictions) <= {
        "timestamp",
        "site_id",
        "split",
        "actual_ac_energy_kwh",
        "prediction",
        "residual",
        "absolute_error",
        "squared_error",
        "is_daylight",
        "model_identifier",
    }
    metadata = json.loads((model_dir / "selected_model_metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["model_artifact_sha256"]) == 64
    assert metadata["software_versions"]["scikit_learn"]
    card = (model_dir / "model_card.md").read_text(encoding="utf-8")
    assert "does not guarantee future performance" in card
    assert "historical observations or reanalysis" in card


def test_test_values_do_not_change_validation_winner(
    training_inputs: tuple[Path, Path, pd.DataFrame, dict[str, object]],
    tmp_path: Path,
) -> None:
    """Selection is complete before the test target is inspected."""
    feature_path, manifest_path, frame, manifest = training_inputs
    first = run_training(
        feature_path,
        manifest_path,
        tmp_path / "models-a",
        tmp_path / "reports-a",
        "first",
        _small_config(),
    )
    frame.loc[frame["split"].eq("test"), "ac_energy_kwh"] = 9999.0
    changed_feature = tmp_path / "changed.csv"
    changed_manifest = tmp_path / "changed_manifest.json"
    frame.to_csv(changed_feature, index=False)
    changed_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    second = run_training(
        changed_feature,
        changed_manifest,
        tmp_path / "models-b",
        tmp_path / "reports-b",
        "second",
        _small_config(),
    )
    assert first.selected_model == second.selected_model
    assert (
        first.training_manifest["validation_metrics"]
        == second.training_manifest["validation_metrics"]
    )


def test_persistence_winner_gets_serialisable_specification(
    training_inputs: tuple[Path, Path, pd.DataFrame, dict[str, object]],
    tmp_path: Path,
) -> None:
    """Persistence can win honestly without pretending a learned estimator won."""
    feature_path, manifest_path, frame, _ = training_inputs
    frame["ac_energy_lag_24h"] = frame["ac_energy_kwh"]
    frame.to_csv(feature_path, index=False)
    result = run_training(
        feature_path,
        manifest_path,
        tmp_path / "models",
        tmp_path / "reports",
        "persistence",
        _small_config(enabled_models=("persistence", "dummy_mean")),
    )
    assert result.selected_model == "persistence"
    loaded = joblib.load(result.model_path)
    assert loaded.predict(frame.iloc[:2]).tolist() == frame["ac_energy_kwh"].iloc[:2].tolist()
    assert pd.read_csv(tmp_path / "reports" / "feature_importance.csv").empty
    assert len(list((tmp_path / "reports" / "charts").glob("*.png"))) == 9


def test_cli_success_config_and_failure(
    training_inputs: tuple[Path, Path, pd.DataFrame, dict[str, object]],
    tmp_path: Path,
) -> None:
    """CLI merges strict JSON configuration, succeeds, and returns non-zero on bad data."""
    feature_path, manifest_path, _, _ = training_inputs
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(_small_config().model_dump(mode="json")),
        encoding="utf-8",
    )
    assert load_training_config(config_path).primary_metric == "mae"
    arguments = [
        "--features",
        str(feature_path),
        "--feature-manifest",
        str(manifest_path),
        "--model-dir",
        str(tmp_path / "models"),
        "--report-dir",
        str(tmp_path / "reports"),
        "--run-name",
        "cli-run",
        "--config",
        str(config_path),
        "--enabled-models",
        "persistence,ridge",
        "--no-refit-on-train-validation",
    ]
    assert main(arguments) == 0
    bad_arguments = arguments.copy()
    bad_arguments[1] = str(tmp_path / "missing.csv")
    assert main(bad_arguments) == 1
