"""Command-line execution for leakage-safe feature engineering."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from solarpulse_ai.data.errors import DataLayerError
from solarpulse_ai.features.config import FeatureConfig
from solarpulse_ai.features.pipeline import run_feature_pipeline
from solarpulse_ai.logging_config import configure_logging, get_logger

LOGGER = get_logger(__name__)


def _hours(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive hour values."""
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("hours must be comma-separated integers") from error


def build_parser() -> argparse.ArgumentParser:
    """Build the feature-engineering argument parser."""
    parser = argparse.ArgumentParser(
        description="Build leakage-safe day-ahead solar forecasting features."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--site-config", required=True, action="append", type=Path)
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/features"))
    parser.add_argument("--forecast-horizon-hours", type=int, default=24)
    parser.add_argument("--target-lag-hours", type=_hours, default=(24, 48, 168))
    parser.add_argument("--rolling-window-hours", type=_hours, default=(3, 6, 24, 72, 168))
    parser.add_argument("--weather-lag-hours", type=_hours, default=(24, 48))
    parser.add_argument("--daylight-ghi-threshold", type=float, default=10.0)
    parser.add_argument("--require-complete-history", action="store_true")
    parser.add_argument("--only-eligible", action="store_true")
    parser.add_argument("--write-splits", action="store_true")
    parser.add_argument("--disable-weather-lags", action="store_true")
    parser.add_argument("--disable-target-history", action="store_true")
    parser.add_argument("--disable-raw-weather-features", action="store_true")
    parser.add_argument("--disable-site-metadata-features", action="store_true")
    parser.add_argument("--allow-unused-site-configs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build features and return a process-compatible status code."""
    configure_logging()
    arguments = build_parser().parse_args(argv)
    LOGGER.info("Reading canonical dataset: %s", arguments.input)
    LOGGER.info("Writing feature dataset: %s", arguments.output)
    try:
        config = FeatureConfig(
            forecast_horizon_hours=arguments.forecast_horizon_hours,
            target_lag_hours=arguments.target_lag_hours,
            rolling_window_hours=arguments.rolling_window_hours,
            weather_lag_hours=arguments.weather_lag_hours,
            daylight_ghi_threshold_w_m2=arguments.daylight_ghi_threshold,
            include_raw_weather_features=not arguments.disable_raw_weather_features,
            include_site_metadata_features=not arguments.disable_site_metadata_features,
            include_weather_lags=not arguments.disable_weather_lags,
            include_target_history=not arguments.disable_target_history,
            require_complete_history=arguments.require_complete_history,
        )
        result = run_feature_pipeline(
            arguments.input,
            arguments.site_config,
            arguments.output,
            arguments.report_dir,
            config,
            split_plan_path=arguments.split_plan,
            only_eligible=arguments.only_eligible,
            write_splits=arguments.write_splits,
            allow_unused_site_configs=arguments.allow_unused_site_configs,
        )
    except (DataLayerError, OSError, ValueError, ValidationError) as error:
        LOGGER.error("Feature generation failed: %s", error)
        return 1
    counts = result.dataframe["feature_eligible"].value_counts()
    LOGGER.info("Processed %d rows", len(result.dataframe) + result.excluded_rows)
    LOGGER.info(
        "Eligible rows: %d; ineligible rows: %d; excluded rows: %d",
        int(counts.get(True, 0)),
        int(counts.get(False, 0)),
        result.excluded_rows,
    )
    LOGGER.info("Forecast horizon: %d hours", config.forecast_horizon_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
