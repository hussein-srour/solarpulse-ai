"""Command-line entry point for exploratory dataset analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from solarpulse_ai.analysis.config import AnalysisThresholds, SplitProportions
from solarpulse_ai.analysis.pipeline import run_analysis
from solarpulse_ai.data.errors import DataLayerError
from solarpulse_ai.logging_config import configure_logging, get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the exploratory-analysis argument parser."""
    parser = argparse.ArgumentParser(
        description="Profile a validated canonical hourly dataset without training a model."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--training-proportion", type=float, default=0.70)
    parser.add_argument("--validation-proportion", type=float, default=0.15)
    parser.add_argument("--testing-proportion", type=float, default=0.15)
    parser.add_argument("--low-irradiance-threshold", type=float, default=20.0)
    parser.add_argument("--high-irradiance-threshold", type=float, default=500.0)
    parser.add_argument("--near-zero-generation-threshold", type=float, default=0.01)
    parser.add_argument("--consecutive-zero-duration", type=int, default=6)
    parser.add_argument("--iqr-outlier-multiplier", type=float, default=1.5)
    parser.add_argument("--minimum-history-days", type=int, default=60)
    parser.add_argument("--write-splits", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run EDA and return zero on success or non-zero on expected failure."""
    configure_logging()
    arguments = build_parser().parse_args(argv)
    LOGGER.info("Analysing canonical dataset: %s", arguments.input)
    LOGGER.info("Writing EDA reports to: %s", arguments.output_dir)
    try:
        thresholds = AnalysisThresholds(
            low_irradiance_w_m2=arguments.low_irradiance_threshold,
            high_irradiance_w_m2=arguments.high_irradiance_threshold,
            near_zero_generation_kwh=arguments.near_zero_generation_threshold,
            consecutive_zero_hours=arguments.consecutive_zero_duration,
            outlier_iqr_multiplier=arguments.iqr_outlier_multiplier,
            minimum_history_days=arguments.minimum_history_days,
        )
        proportions = SplitProportions(
            training=arguments.training_proportion,
            validation=arguments.validation_proportion,
            testing=arguments.testing_proportion,
        )
        result = run_analysis(
            arguments.input,
            arguments.output_dir,
            thresholds,
            proportions,
            write_splits=arguments.write_splits,
        )
    except (DataLayerError, OSError, ValueError) as error:
        LOGGER.error("EDA failed: %s", error)
        return 1
    LOGGER.info("Analysed %d records", len(result.dataframe))
    LOGGER.info("Readiness result: %s", result.report["model_readiness"]["category"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
