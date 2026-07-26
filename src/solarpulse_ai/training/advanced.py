"""Command-line entry point for Phase 7 advanced forecasting."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from solarpulse_ai.logging_config import configure_logging, get_logger
from solarpulse_ai.training.advanced_config import AdvancedTrainingConfig
from solarpulse_ai.training.advanced_pipeline import run_advanced_training

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the advanced training CLI."""
    parser = argparse.ArgumentParser(
        description="Tune and evaluate leakage-safe advanced solar forecasts."
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--search-budget", type=int)
    parser.add_argument("--cv-splits", type=int)
    parser.add_argument("--cv-gap-hours", type=int)
    parser.add_argument("--minimum-training-hours", type=int)
    parser.add_argument("--validation-window-hours", type=int)
    parser.add_argument("--selection-metric", choices=("mae", "rmse"))
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--maximum-training-rows", type=int)
    parser.add_argument(
        "--compare-with-phase6-baselines",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--register-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--refit-selected-model", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--clip-negative-predictions",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--register-candidate-only", action="store_true")
    parser.add_argument("--promote-selected-champion", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a complete experiment and return a process-compatible status."""
    configure_logging()
    arguments = build_parser().parse_args(argv)
    try:
        base = (
            AdvancedTrainingConfig.from_json(arguments.config)
            if arguments.config
            else AdvancedTrainingConfig()
        )
        overrides = {
            "search_budget": arguments.search_budget,
            "cross_validation_splits": arguments.cv_splits,
            "cross_validation_gap_hours": arguments.cv_gap_hours,
            "minimum_training_hours": arguments.minimum_training_hours,
            "validation_window_hours": arguments.validation_window_hours,
            "selection_metric": arguments.selection_metric,
            "random_seed": arguments.random_seed,
            "n_jobs": arguments.n_jobs,
            "maximum_training_rows": arguments.maximum_training_rows,
            "compare_with_phase6_baselines": arguments.compare_with_phase6_baselines,
            "register_model": arguments.register_model,
            "refit_selected_model": arguments.refit_selected_model,
            "clip_negative_predictions": arguments.clip_negative_predictions,
        }
        config = AdvancedTrainingConfig.model_validate(
            base.model_copy(
                update={key: value for key, value in overrides.items() if value is not None}
            ).model_dump()
        )
        LOGGER.info("Advanced forecasting run ID: %s", arguments.run_id or "auto")
        LOGGER.info(
            "Rolling-origin folds=%s gap=%sh search_budget=%s",
            config.cross_validation_splits,
            config.cross_validation_gap_hours,
            config.search_budget,
        )
        result = run_advanced_training(
            arguments.features,
            arguments.feature_manifest,
            arguments.output_dir,
            arguments.artifact_dir,
            arguments.registry,
            config=config,
            run_id=arguments.run_id,
            overwrite=arguments.overwrite,
            register_candidate_only=arguments.register_candidate_only,
            promote_selected_champion=arguments.promote_selected_champion,
        )
    except (OSError, ValueError, ValidationError, RuntimeError) as error:
        LOGGER.error("Advanced training failed: %s", error)
        return 1
    LOGGER.info("Completed run ID: %s", result.run_id)
    LOGGER.info("Validation-selected model: %s", result.selected_model)
    LOGGER.info("Final untouched test evaluation complete after selection")
    LOGGER.info("Model version: %s", result.model_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
