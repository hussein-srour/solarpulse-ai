"""Phase 7 tuning, fair baseline comparison, tracking, and model versioning."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost
from sklearn.pipeline import Pipeline

from solarpulse_ai.training.advanced_config import AdvancedTrainingConfig
from solarpulse_ai.training.advanced_estimators import build_xgboost_model
from solarpulse_ai.training.config import TrainingConfig
from solarpulse_ai.training.contracts import (
    TARGET,
    TrainingContract,
    load_manifest,
    validate_contract,
)
from solarpulse_ai.training.cross_validation import RollingFold, RollingOriginSplitter
from solarpulse_ai.training.cv_reports import generate_advanced_charts
from solarpulse_ai.training.dataset import (
    PreparedDataset,
    load_feature_dataset,
    prepare_dataset,
    split_frame,
)
from solarpulse_ai.training.estimators import PersistencePredictor, build_model
from solarpulse_ai.training.experiment import (
    canonical_checksum,
    dataset_fingerprint,
    make_run_id,
    prepare_run_directory,
    write_artifact_checksums,
)
from solarpulse_ai.training.hyperparameters import generate_candidates, rank_candidates
from solarpulse_ai.training.metrics import calculate_metrics, daylight_metrics, metrics_by_site
from solarpulse_ai.training.persistence import (
    Predictor,
    git_commit,
    save_and_verify,
    sha256_file,
    software_versions,
)
from solarpulse_ai.training.predictions import postprocess, prediction_frame
from solarpulse_ai.training.registry import ModelRegistry
from solarpulse_ai.training.reports import write_csv, write_json
from solarpulse_ai.training.selection import common_cohort

BASELINE_MODELS = ("persistence", "ridge", "random_forest", "histogram_gradient_boosting")
COMPLEXITY = {
    "persistence": 0,
    "ridge": 1,
    "histogram_gradient_boosting": 2,
    "random_forest": 3,
    "tuned_xgboost": 4,
}
WEATHER_LIMITATION = (
    "Historical or reanalysis weather is only a development proxy for forecast inputs. "
    "Production forecasts require weather values available before prediction time; observed "
    "target-time weather can make evaluation optimistic. The model is not production validated "
    "until evaluated with genuine archived forecast weather, and tuning does not remove this limit."
)


@dataclass(frozen=True, slots=True)
class AdvancedTrainingResult:
    """Completed experiment identifiers and generated artifact paths."""

    run_id: str
    selected_model: str
    model_version: str
    experiment_directory: Path
    model_path: Path
    registry_status: str | None


def run_advanced_training(
    feature_path: str | Path,
    feature_manifest_path: str | Path,
    output_directory: str | Path,
    artifact_directory: str | Path,
    registry_path: str | Path,
    *,
    config: AdvancedTrainingConfig | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
    register_candidate_only: bool = False,
    promote_selected_champion: bool = False,
) -> AdvancedTrainingResult:
    """Tune on train only, select on validation only, then evaluate test once."""
    started = time.perf_counter()
    configured = config or AdvancedTrainingConfig()
    config_payload = configured.model_dump(mode="json")
    effective_run_id = run_id or make_run_id(config_payload)
    run_dir = prepare_run_directory(output_directory, effective_run_id, overwrite=overwrite)
    feature_source = Path(feature_path)
    manifest_source = Path(feature_manifest_path)
    source = load_feature_dataset(feature_source)
    manifest = load_manifest(manifest_source)
    contract = validate_contract(source, manifest)
    if contract.forecast_horizon_hours != configured.forecast_horizon_hours:
        raise ValueError("configuration forecast horizon disagrees with feature manifest")
    baseline_config = TrainingConfig(
        minimum_train_rows=1,
        minimum_validation_rows=1,
        minimum_test_rows=1,
        random_seed=configured.random_seed,
        clip_negative_predictions=configured.clip_negative_predictions,
        refit_selected_model_on_train_validation=configured.refit_selected_model,
    )
    dataset = prepare_dataset(source, contract, baseline_config)
    train = split_frame(dataset, "train")
    if configured.maximum_training_rows is not None:
        timestamps = train["timestamp"].drop_duplicates()
        keep = timestamps.iloc[-configured.maximum_training_rows :]
        train = train.loc[train["timestamp"].isin(keep)].copy()
    validation = split_frame(dataset, "validation")
    test = split_frame(dataset, "test")
    splitter = RollingOriginSplitter(
        n_splits=configured.cross_validation_splits,
        gap_hours=configured.cross_validation_gap_hours,
        minimum_training_hours=configured.minimum_training_hours,
        validation_window_hours=configured.validation_window_hours,
    )
    folds = splitter.split(train)
    candidates = generate_candidates(
        configured.search_space,
        budget=configured.search_budget,
        random_seed=configured.random_seed,
    )
    candidate_records, fold_records = _tune(train, contract, configured, folds, candidates, run_dir)
    ranked_candidates = rank_candidates(candidate_records, configured.selection_metric)
    best_candidate = ranked_candidates[0]
    best_parameters = dict(best_candidate["parameters"])
    predictors = list(contract.predictors)
    advanced_model = build_xgboost_model(
        contract,
        best_parameters,
        random_seed=configured.random_seed,
        n_jobs=configured.n_jobs,
    )
    advanced_fit_start = time.perf_counter()
    advanced_model.fit(train[predictors], train[TARGET])
    advanced_fit_seconds = time.perf_counter() - advanced_fit_start

    validation_raw: dict[str, np.ndarray] = {}
    predict_start = time.perf_counter()
    validation_raw["tuned_xgboost"] = np.asarray(
        advanced_model.predict(validation[predictors]), dtype=float
    )
    inference_seconds: dict[str, float] = {"tuned_xgboost": time.perf_counter() - predict_start}
    training_seconds: dict[str, float] = {"tuned_xgboost": advanced_fit_seconds}
    persistence = PersistencePredictor(24)
    if configured.compare_with_phase6_baselines:
        for model_id in BASELINE_MODELS:
            if model_id == "persistence":
                start = time.perf_counter()
                validation_raw[model_id] = persistence.predict(validation)
                inference_seconds[model_id] = time.perf_counter() - start
                training_seconds[model_id] = 0.0
            else:
                baseline = build_model(model_id, contract, baseline_config)
                start = time.perf_counter()
                baseline.fit(train[predictors], train[TARGET])
                training_seconds[model_id] = time.perf_counter() - start
                start = time.perf_counter()
                validation_raw[model_id] = np.asarray(
                    baseline.predict(validation[predictors]), dtype=float
                )
                inference_seconds[model_id] = time.perf_counter() - start
    cohort, aligned, cohort_details = common_cohort(
        validation, validation_raw, persistence.predictor_column
    )
    leaderboard, processed_validation = _validation_leaderboard(
        cohort,
        aligned,
        len(validation),
        configured,
        training_seconds,
        inference_seconds,
        len(predictors),
        float(best_candidate[f"std_{configured.selection_metric}"]),
    )
    selected = str(leaderboard.iloc[0]["model_identifier"])
    leaderboard["selected"] = leaderboard["model_identifier"].eq(selected)
    selected_validation_predictions = processed_validation[selected]
    validation_report = prediction_frame(cohort, selected_validation_predictions, selected)

    fit_rows = (
        pd.concat([train, validation], ignore_index=True)
        if configured.refit_selected_model and selected != "persistence"
        else train
    )
    if selected == "tuned_xgboost":
        final_model: Predictor = build_xgboost_model(
            contract,
            best_parameters,
            random_seed=configured.random_seed,
            n_jobs=configured.n_jobs,
        )
    elif selected == "persistence":
        final_model = persistence
    else:
        final_model = build_model(selected, contract, baseline_config)
    if selected != "persistence":
        final_model.fit(fit_rows[predictors], fit_rows[TARGET])  # type: ignore[attr-defined]
    test_input = test if selected == "persistence" else test[predictors]
    raw_test = np.asarray(final_model.predict(test_input), dtype=float)
    test_cohort, test_aligned, test_cohort_details = common_cohort(
        test,
        {selected: raw_test, "persistence": persistence.predict(test)},
        persistence.predictor_column,
    )
    selected_test = postprocess(
        test_aligned[selected], clip_negative=configured.clip_negative_predictions
    )
    persistence_test = postprocess(
        test_aligned["persistence"], clip_negative=configured.clip_negative_predictions
    )
    final_metrics = {
        "selected_model": {
            "model_identifier": selected,
            **calculate_metrics(
                test_cohort[TARGET].to_numpy(float),
                selected_test.values,
                full_row_count=len(test),
                capacity_kwp=_capacity(test_cohort),
            ),
        },
        "persistence": {
            "model_identifier": "persistence",
            **calculate_metrics(
                test_cohort[TARGET].to_numpy(float),
                persistence_test.values,
                full_row_count=len(test),
                capacity_kwp=_capacity(test_cohort),
            ),
        },
    }
    test_report = prediction_frame(test_cohort, selected_test.values, selected)
    robustness = _robustness_analysis(test_cohort, selected_test.values)
    importance = _xgboost_importance(advanced_model)
    config_checksum = canonical_checksum(config_payload)
    dataset_checksum = sha256_file(feature_source)
    manifest_checksum = sha256_file(manifest_source)
    version_hash = canonical_checksum(
        {
            "config": config_checksum,
            "dataset": dataset_checksum,
            "manifest": manifest_checksum,
            "selected": selected,
        }
    )[:10]
    family = "xgb" if selected == "tuned_xgboost" else selected.replace("_", "-")
    model_version = f"solarpulse-{family}-{datetime.now(UTC):%Y%m%d}-{version_hash}"
    model_dir = Path(artifact_directory) / model_version
    model_path = model_dir / "model.joblib"
    verification = test_cohort if selected == "persistence" else test_cohort[predictors]
    expected = np.asarray(final_model.predict(verification), dtype=float)
    artifact_checksum = save_and_verify(final_model, model_path, verification, expected)
    _write_outputs(
        run_dir=run_dir,
        config=configured,
        source=source,
        manifest=manifest,
        feature_source=feature_source,
        manifest_source=manifest_source,
        dataset=dataset,
        folds=folds,
        candidates=ranked_candidates,
        fold_records=fold_records,
        leaderboard=leaderboard,
        validation_report=validation_report,
        test_report=test_report,
        final_metrics=final_metrics,
        robustness=robustness,
        importance=importance,
        selected=selected,
        best_candidate=best_candidate,
        cohort_details=cohort_details,
        test_cohort_details=test_cohort_details,
        model_version=model_version,
        model_path=model_path,
        artifact_checksum=artifact_checksum,
        started=started,
    )
    write_json(
        {
            "model_version": model_version,
            "run_id": effective_run_id,
            "model_family": selected,
            "target": TARGET,
            "forecast_horizon_hours": configured.forecast_horizon_hours,
            "predictor_names_in_order": predictors,
            "configuration_checksum": config_checksum,
            "dataset_checksum": dataset_checksum,
            "feature_manifest_checksum": manifest_checksum,
            "artifact_checksum": artifact_checksum,
            "artifact_security": "Load joblib/pickle artifacts only from trusted sources.",
            "software_versions": {**software_versions(), "xgboost": xgboost.__version__},
        },
        model_dir / "metadata.json",
    )
    registry_status: str | None = None
    if configured.register_model:
        registry = ModelRegistry(registry_path)
        record = registry.register_candidate(
            {
                "model_version": model_version,
                "run_id": effective_run_id,
                "model_family": selected,
                "configuration_checksum": config_checksum,
                "dataset_checksum": dataset_checksum,
                "feature_manifest_checksum": manifest_checksum,
                "training_timestamp": datetime.now(UTC).isoformat(),
                "forecast_horizon_hours": configured.forecast_horizon_hours,
                "validation_metrics": leaderboard.iloc[0].to_dict(),
                "test_metrics": final_metrics["selected_model"],
                "artifact_path": str(model_path.resolve()),
                "artifact_checksum": artifact_checksum,
            }
        )
        registry_status = str(record["status"])
        if promote_selected_champion and not register_candidate_only:
            promoted = registry.promote(model_version, selection_based_on_validation=True)
            registry_status = str(promoted["status"])
    return AdvancedTrainingResult(
        effective_run_id,
        selected,
        model_version,
        run_dir,
        model_path,
        registry_status,
    )


def _tune(
    train: pd.DataFrame,
    contract: TrainingContract,
    config: AdvancedTrainingConfig,
    folds: list[RollingFold],
    candidates: list[dict[str, Any]],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictors = list(contract.predictors)
    candidate_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_started = time.perf_counter()
        candidate_folds: list[dict[str, Any]] = []
        try:
            for fold in folds:
                training = train.loc[list(fold.training_indices)]
                validation = train.loc[list(fold.validation_indices)]
                model = build_xgboost_model(
                    contract,
                    dict(candidate["parameters"]),
                    random_seed=config.random_seed,
                    n_jobs=config.n_jobs,
                )
                fit_started = time.perf_counter()
                model.fit(training[predictors], training[TARGET])
                training_duration = time.perf_counter() - fit_started
                predict_started = time.perf_counter()
                raw = np.asarray(model.predict(validation[predictors]), dtype=float)
                prediction_duration = time.perf_counter() - predict_started
                processed = postprocess(raw, clip_negative=config.clip_negative_predictions)
                metrics = calculate_metrics(validation[TARGET].to_numpy(float), processed.values)
                by_site = metrics_by_site(validation, processed.values).to_dict(orient="records")
                row = {
                    "candidate_id": candidate["candidate_id"],
                    "fold_number": fold.boundary.fold_number,
                    "mae": metrics["mae_kwh"],
                    "rmse": metrics["rmse_kwh"],
                    "median_absolute_error": metrics["median_absolute_error_kwh"],
                    "r2": metrics["r2"],
                    "mean_bias_error": metrics["mean_bias_error_kwh"],
                    "wape": metrics["wape"],
                    "prediction_count": metrics["prediction_count"],
                    "actual_energy_total": metrics["actual_total_energy_kwh"],
                    "predicted_energy_total": metrics["predicted_total_energy_kwh"],
                    "negative_raw_prediction_count": processed.raw_negative_count,
                    "prediction_coverage": metrics["prediction_coverage_pct"],
                    "daylight_metrics": json.dumps(
                        daylight_metrics(validation, processed.values), sort_keys=True
                    ),
                    "metrics_by_site": json.dumps(by_site, sort_keys=True),
                    "training_duration_seconds": training_duration,
                    "prediction_duration_seconds": prediction_duration,
                }
                candidate_folds.append(row)
                fold_records.append(row)
                if config.save_fold_predictions:
                    write_csv(
                        prediction_frame(
                            validation, processed.values, str(candidate["candidate_id"])
                        ),
                        run_dir
                        / "fold_predictions"
                        / f"{candidate['candidate_id']}-fold-{fold.boundary.fold_number}.csv",
                    )
            metrics_record: dict[str, Any] = {}
            for metric in ("mae", "rmse"):
                values = np.asarray([float(row[metric]) for row in candidate_folds])
                metrics_record.update(
                    {
                        f"mean_{metric}": float(values.mean()),
                        f"std_{metric}": float(values.std(ddof=0)),
                        f"min_{metric}": float(values.min()),
                        f"max_{metric}": float(values.max()),
                    }
                )
            candidate_records.append(
                {
                    **candidate,
                    "status": "succeeded",
                    **metrics_record,
                    "training_duration_seconds": sum(
                        float(row["training_duration_seconds"]) for row in candidate_folds
                    ),
                    "prediction_duration_seconds": sum(
                        float(row["prediction_duration_seconds"]) for row in candidate_folds
                    ),
                    "total_duration_seconds": time.perf_counter() - candidate_started,
                    "failure_message": None,
                }
            )
        except (TypeError, ValueError, RuntimeError) as error:
            candidate_records.append(
                {
                    **candidate,
                    "status": "failed",
                    "failure_message": f"{type(error).__name__}: {error}",
                    "total_duration_seconds": time.perf_counter() - candidate_started,
                }
            )
    return candidate_records, fold_records


def _validation_leaderboard(
    cohort: pd.DataFrame,
    aligned: dict[str, np.ndarray],
    full_count: int,
    config: AdvancedTrainingConfig,
    training_seconds: dict[str, float],
    inference_seconds: dict[str, float],
    predictor_count: int,
    xgb_stability: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    records: list[dict[str, Any]] = []
    processed: dict[str, np.ndarray] = {}
    for model_id, raw in aligned.items():
        result = postprocess(raw, clip_negative=config.clip_negative_predictions)
        processed[model_id] = result.values
        records.append(
            {
                "model_identifier": model_id,
                "model_type": "advanced" if model_id == "tuned_xgboost" else "phase6_baseline",
                **calculate_metrics(
                    cohort[TARGET].to_numpy(float),
                    result.values,
                    full_row_count=full_count,
                    capacity_kwp=_capacity(cohort),
                ),
                "negative_raw_prediction_count": result.raw_negative_count,
                "site_coverage": int(cohort["site_id"].nunique()),
                "training_duration_seconds": training_seconds[model_id],
                "inference_duration_seconds": inference_seconds[model_id],
                "predictor_count": 1 if model_id == "persistence" else predictor_count,
                "cross_validation_stability": xgb_stability if model_id == "tuned_xgboost" else 0.0,
                "model_complexity_rank": COMPLEXITY[model_id],
            }
        )
    ordered = sorted(
        records,
        key=lambda record: (
            float(record["mae_kwh"]),
            float(record["rmse_kwh"]),
            float(record["cross_validation_stability"]),
            int(record["model_complexity_rank"]),
            str(record["model_identifier"]),
        ),
    )
    return pd.DataFrame([{"rank": index, **row} for index, row in enumerate(ordered, 1)]), processed


def _xgboost_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["estimator"]
    names = list(preprocessor.get_feature_names_out())
    gain = estimator.get_booster().get_score(importance_type="gain")
    weight = estimator.get_booster().get_score(importance_type="weight")
    return pd.DataFrame(
        {
            "feature": names,
            "gain": [float(gain.get(f"f{index}", 0.0)) for index in range(len(names))],
            "weight": [float(weight.get(f"f{index}", 0.0)) for index in range(len(names))],
            "interpretation": "XGBoost model importance; association, not causality",
        }
    ).sort_values(["gain", "feature"], ascending=[False, True], kind="stable")


def _write_outputs(
    *,
    run_dir: Path,
    config: AdvancedTrainingConfig,
    source: pd.DataFrame,
    manifest: dict[str, Any],
    feature_source: Path,
    manifest_source: Path,
    dataset: PreparedDataset,
    folds: list[RollingFold],
    candidates: list[dict[str, Any]],
    fold_records: list[dict[str, Any]],
    leaderboard: pd.DataFrame,
    validation_report: pd.DataFrame,
    test_report: pd.DataFrame,
    final_metrics: dict[str, Any],
    robustness: pd.DataFrame,
    importance: pd.DataFrame,
    selected: str,
    best_candidate: dict[str, Any],
    cohort_details: dict[str, Any],
    test_cohort_details: dict[str, Any],
    model_version: str,
    model_path: Path,
    artifact_checksum: str,
    started: float,
) -> None:
    config_payload = config.model_dump(mode="json")
    write_json(config_payload, run_dir / "run_config.json")
    write_json(
        {
            **software_versions(),
            "xgboost": xgboost.__version__,
            "git_commit_sha": git_commit(),
        },
        run_dir / "environment.json",
    )
    write_json(
        dataset_fingerprint(
            dataset_path=feature_source,
            dataset_checksum=sha256_file(feature_source),
            frame_rows=len(source),
            split_counts=dataset.split_counts,
        ),
        run_dir / "dataset_fingerprint.json",
    )
    write_json(manifest, run_dir / "feature_manifest_snapshot.json")
    write_json(
        {"folds": [fold.boundary.to_dict() for fold in folds]},
        run_dir / "cross_validation_folds.json",
    )
    flat_candidates = pd.DataFrame(
        [
            {
                **{key: value for key, value in row.items() if key != "parameters"},
                **{f"parameter_{key}": value for key, value in row["parameters"].items()},
            }
            for row in candidates
        ]
    )
    write_csv(flat_candidates, run_dir / "hyperparameter_candidates.csv")
    write_csv(pd.DataFrame(fold_records), run_dir / "cross_validation_results.csv")
    write_csv(leaderboard, run_dir / "validation_leaderboard.csv")
    write_csv(validation_report, run_dir / "validation_predictions.csv")
    write_csv(test_report, run_dir / "test_predictions.csv")
    write_csv(importance, run_dir / "feature_importance.csv")
    write_csv(robustness, run_dir / "robustness_analysis.csv")
    selection = {
        "selected_model_identifier": selected,
        "selected_model_family": "xgboost" if selected == "tuned_xgboost" else selected,
        "selection_metric": config.selection_metric,
        "selection_reason": (
            "Shared validation cohort only: MAE, RMSE, CV stability, simplicity, identifier. "
            "Test metrics were unavailable to selection."
        ),
        "validation_ranking": leaderboard.to_dict(orient="records"),
        "comparison_against_persistence": _comparison(leaderboard, selected, "persistence"),
        "advanced_model_change": _comparison(leaderboard, "tuned_xgboost", "persistence"),
        "selected_candidate": best_candidate["candidate_id"],
        "model_version": model_version,
        "refit_policy": (
            "selected learned model refitted on train plus validation before test"
            if config.refit_selected_model
            else "selected model retained training-only fit before test"
        ),
        "model_limitations": [WEATHER_LIMITATION, "Feature importance does not prove causality."],
    }
    write_json(selection, run_dir / "model_selection.json")
    summary = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "configuration_checksum": canonical_checksum(config_payload),
        "dataset_checksum": sha256_file(feature_source),
        "feature_manifest_checksum": sha256_file(manifest_source),
        "random_seed": config.random_seed,
        "search_budget": config.search_budget,
        "selected_candidate": best_candidate["candidate_id"],
        "selected_model": selected,
        "validation_comparison_cohort": cohort_details,
        "test_comparison_cohort": test_cohort_details,
        "test_metrics_after_selection": final_metrics,
        "model_version": model_version,
        "model_artifact_path": str(model_path.resolve()),
        "model_artifact_checksum": artifact_checksum,
        "training_duration_seconds": time.perf_counter() - started,
        "warnings_and_limitations": [
            WEATHER_LIMITATION,
            "No production-readiness claim is made from model selection.",
            "Synthetic fixtures must never be interpreted as measured plant performance.",
            "Feature importance does not prove causality.",
        ],
        "robustness_cohort_definitions": {
            "site": "one cohort per site_id",
            "hour": "UTC hour of target timestamp",
            "month": "UTC calendar month; small cohorts are labelled",
            "daylight": "Phase 5 is_daylight flag when available",
            "irradiance": "low/high split at finite GHI median when available",
            "cloud_cover": "low/high split at finite cloud-cover median when available",
            "target_magnitude": "zero and nonzero target terciles",
            "small_cohort_threshold": 30,
        },
    }
    write_json(summary, run_dir / "final_metrics.json")
    report = _summary_markdown(summary, selection)
    (run_dir / "experiment_summary.md").write_text(report, encoding="utf-8")
    (run_dir / "model_card.md").write_text(_model_card(summary, selection), encoding="utf-8")
    charts = generate_advanced_charts(
        flat_candidates,
        pd.DataFrame(fold_records),
        leaderboard,
        validation_report,
        test_report,
        metrics_by_site(
            test_report.rename(columns={"actual_ac_energy_kwh": TARGET}),
            test_report["prediction"].to_numpy(float),
        ),
        importance,
        run_dir / "charts",
    )
    summary["chart_warnings"] = charts
    write_artifact_checksums(run_dir, summary)


def _comparison(leaderboard: pd.DataFrame, first: str, second: str) -> dict[str, Any] | None:
    left = leaderboard.loc[leaderboard["model_identifier"].eq(first)]
    right = leaderboard.loc[leaderboard["model_identifier"].eq(second)]
    if left.empty or right.empty:
        return None
    first_mae = float(left.iloc[0]["mae_kwh"])
    second_mae = float(right.iloc[0]["mae_kwh"])
    return {
        "first": first,
        "second": second,
        "mae_difference_kwh": first_mae - second_mae,
        "interpretation": "negative means the first model has lower validation MAE",
    }


def _robustness_analysis(rows: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Summarise descriptive cohorts without using them for model selection."""
    working = rows.reset_index(drop=True).copy()
    working["_prediction"] = predictions
    timestamps = pd.to_datetime(working["timestamp"], utc=True)
    cohorts: list[tuple[str, str, pd.Series]] = []
    for site in sorted(working["site_id"].astype(str).unique()):
        cohorts.append(("site", site, working["site_id"].astype(str).eq(site)))
    for hour in sorted(timestamps.dt.hour.unique()):
        cohorts.append(("utc_hour", str(hour), timestamps.dt.hour.eq(hour)))
    for month in sorted(timestamps.dt.month.unique()):
        cohorts.append(("utc_month", str(month), timestamps.dt.month.eq(month)))
    if "is_daylight" in working:
        daylight = working["is_daylight"].astype(bool)
        cohorts.extend(
            [("daylight", "daylight", daylight), ("daylight", "non_daylight", ~daylight)]
        )
    for column, cohort_name in (
        ("ghi_w_m2", "irradiance"),
        ("cloud_cover_pct", "cloud_cover"),
    ):
        if column in working:
            values = pd.to_numeric(working[column], errors="coerce")
            finite = pd.Series(
                np.isfinite(values.to_numpy(float)),
                index=working.index,
                dtype=bool,
            )
            if finite.any():
                threshold = float(values.loc[finite].median())
                cohorts.extend(
                    [
                        (cohort_name, f"low_le_{threshold:g}", finite & values.le(threshold)),
                        (cohort_name, f"high_gt_{threshold:g}", finite & values.gt(threshold)),
                    ]
                )
    target = working[TARGET]
    positive = target.gt(0)
    if positive.any():
        lower, upper = target.loc[positive].quantile([1 / 3, 2 / 3])
        cohorts.extend(
            [
                ("target_magnitude", "zero", ~positive),
                ("target_magnitude", "low", positive & target.le(lower)),
                ("target_magnitude", "medium", target.gt(lower) & target.le(upper)),
                ("target_magnitude", "high", target.gt(upper)),
            ]
        )
    records: list[dict[str, Any]] = []
    for cohort_type, cohort_value, mask in cohorts:
        selected_rows = working.loc[mask]
        if selected_rows.empty:
            continue
        metrics = calculate_metrics(
            selected_rows[TARGET].to_numpy(float),
            selected_rows["_prediction"].to_numpy(float),
        )
        records.append(
            {
                "cohort_type": cohort_type,
                "cohort_value": cohort_value,
                "small_cohort": len(selected_rows) < 30,
                **metrics,
            }
        )
    return pd.DataFrame(records)


def _summary_markdown(summary: dict[str, Any], selection: dict[str, Any]) -> str:
    return f"""# SolarPulse AI advanced forecasting experiment

Run `{summary["run_id"]}` tuned XGBoost with expanding-window rolling-origin validation using
training-labelled rows only. The validation gap protects the 24-hour forecast horizon.

The validation-selected model is `{summary["selected_model"]}`. This is an experiment record,
not a production-readiness or plant-performance claim. Test data was evaluated only after
selection and never changed the winner. {selection["refit_policy"]}.

## Limitations

{WEATHER_LIMITATION}

Feature importance describes model use, not causality. Meaningful conclusions require genuine
measured solar-generation data; synthetic test outcomes are not real plant performance.
"""


def _model_card(summary: dict[str, Any], selection: dict[str, Any]) -> str:
    return f"""# Model card: {summary["model_version"]}

- Objective: predict hourly `ac_energy_kwh` 24 hours ahead.
- Selected model: `{summary["selected_model"]}`.
- Selection: validation only; test metrics did not affect promotion.
- Refit: {selection["refit_policy"]}.
- Artifact SHA-256: `{summary["model_artifact_checksum"]}`.

## Intended use and security

Development evaluation of day-ahead solar-energy forecasts. Load joblib/pickle artifacts only
from trusted sources and verify the recorded checksum and feature order.

## Limitations

{WEATHER_LIMITATION}

Advanced tuning does not prove production readiness. Feature importance is not causal.
"""


def _capacity(rows: pd.DataFrame) -> np.ndarray | None:
    return (
        rows["installed_capacity_kwp"].to_numpy(float) if "installed_capacity_kwp" in rows else None
    )
