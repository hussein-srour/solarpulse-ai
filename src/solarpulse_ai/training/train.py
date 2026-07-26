"""Command-line entry point for Phase 6 baseline training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from solarpulse_ai.logging_config import configure_logging, get_logger
from solarpulse_ai.training.config import TrainingConfig
from solarpulse_ai.training.pipeline import load_training_config, run_training

LOGGER = get_logger(__name__)


def _models(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    """Build the training argument parser."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate deterministic day-ahead solar baselines."
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--enabled-models", type=_models)
    parser.add_argument("--primary-metric", choices=("mae",))
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--persistence-lag", type=int)
    parser.add_argument("--minimum-train-rows", type=int)
    parser.add_argument("--minimum-validation-rows", type=int)
    parser.add_argument("--minimum-test-rows", type=int)
    parser.add_argument("--random-forest-estimators", type=int)
    parser.add_argument("--random-forest-max-depth", type=int)
    parser.add_argument("--random-forest-min-samples-leaf", type=int)
    parser.add_argument("--histogram-boosting-max-iter", type=int)
    parser.add_argument("--histogram-boosting-learning-rate", type=float)
    parser.add_argument("--histogram-boosting-max-leaf-nodes", type=int)
    parser.add_argument("--permutation-importance-repeats", type=int)
    parser.add_argument(
        "--clip-negative-predictions",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--refit-on-train-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run training and return a process-compatible status code."""
    configure_logging()
    arguments = build_parser().parse_args(argv)
    LOGGER.info("Training run: %s", arguments.run_name)
    LOGGER.info(
        "Inputs: features=%s; feature_manifest=%s",
        arguments.features,
        arguments.feature_manifest,
    )
    try:
        base = (
            load_training_config(arguments.config)
            if arguments.config is not None
            else TrainingConfig()
        )
        overrides = {
            "enabled_models": arguments.enabled_models,
            "primary_metric": arguments.primary_metric,
            "random_seed": arguments.random_seed,
            "persistence_lag_hours": arguments.persistence_lag,
            "minimum_train_rows": arguments.minimum_train_rows,
            "minimum_validation_rows": arguments.minimum_validation_rows,
            "minimum_test_rows": arguments.minimum_test_rows,
            "random_forest_estimators": arguments.random_forest_estimators,
            "random_forest_max_depth": arguments.random_forest_max_depth,
            "random_forest_min_samples_leaf": arguments.random_forest_min_samples_leaf,
            "histogram_boosting_max_iter": arguments.histogram_boosting_max_iter,
            "histogram_boosting_learning_rate": arguments.histogram_boosting_learning_rate,
            "histogram_boosting_max_leaf_nodes": arguments.histogram_boosting_max_leaf_nodes,
            "permutation_importance_repeats": arguments.permutation_importance_repeats,
            "clip_negative_predictions": arguments.clip_negative_predictions,
            "refit_selected_model_on_train_validation": arguments.refit_on_train_validation,
        }
        config = base.model_copy(
            update={key: value for key, value in overrides.items() if value is not None}
        )
        config = TrainingConfig.model_validate(config.model_dump())
        LOGGER.info("Candidate models: %s", ", ".join(config.enabled_models))
        result = run_training(
            arguments.features,
            arguments.feature_manifest,
            arguments.model_dir,
            arguments.report_dir,
            arguments.run_name,
            config,
        )
    except (OSError, ValueError, ValidationError) as error:
        LOGGER.error("Training failed: %s", error)
        return 1
    LOGGER.info("Eligible split counts: %s", result.training_manifest["split_counts"])
    LOGGER.info("Validation winner: %s", result.selected_model)
    LOGGER.info("Final untouched test evaluation complete")
    LOGGER.info("Model artifact: %s", result.model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
