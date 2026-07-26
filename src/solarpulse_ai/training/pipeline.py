"""End-to-end baseline training, selection, untouched test evaluation, and reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from solarpulse_ai import __version__
from solarpulse_ai.training.charts import generate_charts
from solarpulse_ai.training.config import TrainingConfig
from solarpulse_ai.training.contracts import (
    REQUIRED_METADATA,
    TARGET,
    TrainingContract,
    load_manifest,
    validate_contract,
)
from solarpulse_ai.training.dataset import load_feature_dataset, prepare_dataset, split_frame
from solarpulse_ai.training.estimators import (
    PersistencePredictor,
    build_model,
    fixed_parameters,
)
from solarpulse_ai.training.metrics import calculate_metrics, daylight_metrics, metrics_by_site
from solarpulse_ai.training.model_card import render_model_card
from solarpulse_ai.training.persistence import (
    Predictor,
    git_commit,
    save_and_verify,
    sha256_file,
    software_versions,
)
from solarpulse_ai.training.predictions import postprocess, prediction_frame
from solarpulse_ai.training.reports import render_summary, write_csv, write_json
from solarpulse_ai.training.selection import common_cohort, rank_models


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Paths and selection facts from one completed run."""

    selected_model: str
    model_path: Path
    report_directory: Path
    training_manifest: dict[str, Any]


def run_training(
    feature_path: str | Path,
    feature_manifest_path: str | Path,
    model_directory: str | Path,
    report_directory: str | Path,
    run_name: str,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Run fixed candidates and evaluate the validation-selected model once on test."""
    configured = config or TrainingConfig()
    if not run_name.strip() or any(character in run_name for character in ("/", "\\", "\0")):
        raise ValueError("run_name must be a non-empty path-safe name")
    feature_source = Path(feature_path)
    manifest_source = Path(feature_manifest_path)
    source = load_feature_dataset(feature_source)
    feature_manifest = load_manifest(manifest_source)
    contract = validate_contract(source, feature_manifest)
    dataset = prepare_dataset(source, contract, configured)
    persistence = PersistencePredictor(configured.persistence_lag_hours)
    if persistence.predictor_column not in contract.predictors:
        raise ValueError(
            f"feature manifest must declare persistence predictor {persistence.predictor_column}"
        )
    train = split_frame(dataset, "train")
    validation = split_frame(dataset, "validation")
    test = split_frame(dataset, "test")
    predictors = list(contract.predictors)

    fitted: dict[str, Any] = {}
    validation_raw: dict[str, np.ndarray] = {}
    for model_id in configured.enabled_models:
        if model_id == "persistence":
            validation_raw[model_id] = persistence.predict(validation)
        else:
            pipeline = build_model(model_id, contract, configured)
            pipeline.fit(train[predictors], train[TARGET])
            fitted[model_id] = pipeline
            validation_raw[model_id] = np.asarray(
                pipeline.predict(validation[predictors]), dtype=float
            )
    validation_cohort, validation_aligned, validation_cohort_details = common_cohort(
        validation, validation_raw, persistence.predictor_column
    )
    validation_records: list[dict[str, Any]] = []
    validation_processed: dict[str, np.ndarray] = {}
    negative_counts: dict[str, dict[str, int]] = {"validation": {}, "test": {}}
    for model_id, raw in validation_aligned.items():
        processed = postprocess(raw, clip_negative=configured.clip_negative_predictions)
        validation_processed[model_id] = processed.values
        negative_counts["validation"][model_id] = processed.raw_negative_count
        validation_records.append(
            {
                "model_identifier": model_id,
                **calculate_metrics(
                    validation_cohort[TARGET].to_numpy(dtype=float),
                    processed.values,
                    full_row_count=len(validation),
                    capacity_kwp=_capacity(validation_cohort),
                ),
            }
        )
    ranking = rank_models(validation_records)
    selected = str(ranking[0]["model_identifier"])
    selection_model = persistence if selected == "persistence" else fitted[selected]

    importance = _feature_importance(
        selected,
        selection_model,
        validation_cohort,
        contract,
        configured,
    )
    selected_validation_predictions = validation_processed[selected]
    validation_prediction_report = prediction_frame(
        validation_cohort, selected_validation_predictions, selected
    )

    final_model: Any
    fit_rows = train
    if selected == "persistence":
        final_model = persistence
    else:
        final_model = build_model(selected, contract, configured)
        if configured.refit_selected_model_on_train_validation:
            fit_rows = pd.concat([train, validation], ignore_index=True)
        final_model.fit(fit_rows[predictors], fit_rows[TARGET])

    selected_test_raw = np.asarray(
        final_model.predict(test if selected == "persistence" else test[predictors]),
        dtype=float,
    )
    test_raw = {
        selected: selected_test_raw,
        "persistence": persistence.predict(test),
    }
    test_cohort, test_aligned, test_cohort_details = common_cohort(
        test, test_raw, persistence.predictor_column
    )
    final_test_metrics: dict[str, dict[str, Any]] = {}
    test_processed: dict[str, np.ndarray] = {}
    for model_id, raw in test_aligned.items():
        processed = postprocess(raw, clip_negative=configured.clip_negative_predictions)
        test_processed[model_id] = processed.values
        negative_counts["test"][model_id] = processed.raw_negative_count
        key = "selected_model" if model_id == selected else "persistence"
        final_test_metrics[key] = {
            "model_identifier": model_id,
            **calculate_metrics(
                test_cohort[TARGET].to_numpy(dtype=float),
                processed.values,
                full_row_count=len(test),
                capacity_kwp=_capacity(test_cohort),
            ),
        }
    if selected == "persistence":
        final_test_metrics["persistence"] = dict(final_test_metrics["selected_model"])
    selected_test_predictions = test_processed[selected]
    test_prediction_report = prediction_frame(test_cohort, selected_test_predictions, selected)
    validation_by_site = metrics_by_site(
        validation_cohort, selected_validation_predictions, full_row_count=len(validation)
    )
    test_by_site = metrics_by_site(test_cohort, selected_test_predictions, full_row_count=len(test))

    run_timestamp = datetime.now(UTC).isoformat()
    versions = software_versions()
    feature_checksum = sha256_file(feature_source)
    feature_manifest_checksum = sha256_file(manifest_source)
    model_dir = Path(model_directory) / run_name
    report_dir = Path(report_directory)
    model_path = model_dir / "selected_model.joblib"
    verification_input = test_cohort if selected == "persistence" else test_cohort[predictors]
    expected_raw = np.asarray(final_model.predict(verification_input), dtype=float)
    model_checksum = save_and_verify(final_model, model_path, verification_input, expected_raw)
    validation_metrics = {
        str(record["model_identifier"]): {
            key: value for key, value in record.items() if key != "model_identifier"
        }
        for record in validation_records
    }
    warnings = [
        "Historical/reanalysis weather is a proxy and may make evaluation optimistic.",
        "Baseline test performance does not guarantee future or production performance.",
        "Permutation importance does not prove causality.",
    ]
    daylight = {
        "validation_selected_model": (
            daylight_metrics(validation_cohort, selected_validation_predictions)
            if configured.daylight_only_secondary_metrics
            else None
        ),
        "test_selected_model": (
            daylight_metrics(test_cohort, selected_test_predictions)
            if configured.daylight_only_secondary_metrics
            else None
        ),
    }
    training_manifest: dict[str, Any] = {
        "project_version": __version__,
        "run_name": run_name,
        "run_timestamp_utc": run_timestamp,
        "source_feature_file_path": str(feature_source.resolve()),
        "source_feature_file_sha256": feature_checksum,
        "feature_manifest_path": str(manifest_source.resolve()),
        "feature_manifest_sha256": feature_manifest_checksum,
        "forecast_horizon_hours": contract.forecast_horizon_hours,
        "input_row_counts": {"source": len(source), "eligible": len(dataset.frame)},
        "eligibility_filtering": dataset.eligibility,
        "split_counts": dataset.split_counts,
        "split_timestamp_boundaries": dataset.split_boundaries,
        "site_ids": sorted(str(value) for value in dataset.frame["site_id"].unique()),
        "target_column": TARGET,
        "predictor_columns": predictors,
        "excluded_metadata_columns": [*REQUIRED_METADATA, "timestamp", "site_id", TARGET],
        "constant_predictors_from_training_only": list(dataset.constant_predictors),
        "preprocessing_steps": {
            "numeric": "training-fitted median imputation with missing indicators",
            "ridge": "training-fitted StandardScaler after imputation",
            "categorical": "training-fitted most-frequent imputation and unknown-safe one-hot",
            "tree_scaling": "none",
        },
        "candidate_models": list(configured.enabled_models),
        "candidate_fixed_parameters": {
            key: value
            for key, value in fixed_parameters(configured).items()
            if key in configured.enabled_models
        },
        "random_seed": configured.random_seed,
        "validation_comparison_cohort": validation_cohort_details,
        "validation_metrics": validation_metrics,
        "model_ranking": ranking,
        "selected_model": selected,
        "selection_rule": (
            "validation MAE ascending; validation RMSE; fixed simplicity order; identifier"
        ),
        "refit_policy": {
            "enabled": configured.refit_selected_model_on_train_validation,
            "fit_splits": sorted(str(value) for value in fit_rows["split"].unique()),
        },
        "test_comparison_cohort": test_cohort_details,
        "final_test_metrics": final_test_metrics,
        "daylight_metrics": daylight,
        "negative_prediction_counts": negative_counts,
        "software_versions": versions,
        "git_commit": git_commit(),
        "limitations_and_warnings": warnings,
    }
    summary = {
        "run_name": run_name,
        "feature_file_sha256": feature_checksum,
        "feature_manifest_sha256": feature_manifest_checksum,
        "forecast_horizon_hours": contract.forecast_horizon_hours,
        "predictor_count": len(predictors),
        "eligibility": dataset.eligibility,
        "split_boundaries": dataset.split_boundaries,
        "selected_model": selected,
        "model_ranking": ranking,
        "validation_comparison_cohort": validation_cohort_details,
        "test_comparison_cohort": test_cohort_details,
        "validation_metrics": validation_metrics,
        "final_test_metrics": final_test_metrics,
        "daylight_metrics": daylight,
        "clip_negative_predictions": configured.clip_negative_predictions,
        "negative_prediction_counts": negative_counts,
        "warnings": warnings,
    }
    comparison = pd.DataFrame(ranking)
    test_metrics_frame = pd.DataFrame(list(final_test_metrics.values()))

    write_json(configured.model_dump(mode="json"), model_dir / "training_configuration.json")
    write_json(training_manifest, model_dir / "training_manifest.json")
    write_json(feature_manifest, model_dir / "feature_manifest_snapshot.json")
    metadata = {
        "selected_model_identifier": selected,
        "target_column": TARGET,
        "predictor_names_in_order": predictors,
        "forecast_horizon_hours": contract.forecast_horizon_hours,
        "training_timestamp_ranges": {
            key: dataset.split_boundaries[key] for key in ("train", "validation")
        },
        "validation_selection_metrics": validation_metrics[selected],
        "final_test_metrics": final_test_metrics["selected_model"],
        "software_versions": versions,
        "model_artifact_sha256": model_checksum,
        "postprocessing": {
            "clip_negative_predictions": configured.clip_negative_predictions,
            "no_maximum_clipping": True,
        },
    }
    write_json(metadata, model_dir / "selected_model_metadata.json")
    (model_dir / "model_card.md").write_text(
        render_model_card(training_manifest, model_checksum), encoding="utf-8"
    )
    write_json(summary, report_dir / "training_summary.json")
    (report_dir / "training_summary.md").write_text(render_summary(summary), encoding="utf-8")
    write_csv(comparison, report_dir / "validation_model_comparison.csv")
    write_csv(validation_by_site, report_dir / "validation_metrics_by_site.csv")
    write_csv(test_metrics_frame, report_dir / "test_metrics.csv")
    write_csv(test_by_site, report_dir / "test_metrics_by_site.csv")
    write_csv(importance, report_dir / "feature_importance.csv")
    write_csv(validation_prediction_report, report_dir / "validation_predictions.csv")
    write_csv(test_prediction_report, report_dir / "test_predictions.csv")
    chart_warnings = generate_charts(
        comparison,
        validation_prediction_report,
        test_prediction_report,
        test_by_site,
        importance,
        report_dir / "charts",
    )
    if chart_warnings:
        training_manifest["limitations_and_warnings"].extend(chart_warnings)
        write_json(training_manifest, model_dir / "training_manifest.json")
        write_json(summary, report_dir / "training_summary.json")
    return TrainingResult(selected, model_path, report_dir, training_manifest)


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a strict optional JSON configuration."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid training configuration: {error}") from error
    return TrainingConfig.model_validate(payload)


def _capacity(rows: pd.DataFrame) -> np.ndarray | None:
    return (
        rows["installed_capacity_kwp"].to_numpy(dtype=float)
        if "installed_capacity_kwp" in rows
        else None
    )


def _feature_importance(
    selected: str,
    model: Predictor,
    validation: pd.DataFrame,
    contract: TrainingContract,
    config: TrainingConfig,
) -> pd.DataFrame:
    if selected == "persistence":
        return pd.DataFrame(
            columns=["predictor", "importance_mean", "importance_std", "interpretation"]
        )
    result = permutation_importance(
        model,
        validation[list(contract.predictors)],
        validation[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=config.permutation_importance_repeats,
        random_state=config.random_seed,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "predictor": list(contract.predictors),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "interpretation": "validation permutation importance; not causal",
        }
    ).sort_values("importance_mean", ascending=False, kind="stable")
