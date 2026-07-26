"""Strict generation-weather dataset joining and command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd

from solarpulse_ai.data.errors import DataLayerError, DatasetJoinError
from solarpulse_ai.data.generation import read_generation_csv, validate_generation_dataframe
from solarpulse_ai.data.ingestion import read_hourly_csv, write_processed_csv
from solarpulse_ai.data.schemas import CANONICAL_COLUMNS
from solarpulse_ai.data.validation import validate_hourly_dataframe
from solarpulse_ai.data.weather import validate_weather_dataframe
from solarpulse_ai.logging_config import configure_logging, get_logger

LOGGER = get_logger(__name__)
JOIN_KEYS = ["site_id", "timestamp"]


def _describe_keys(dataframe: pd.DataFrame, limit: int = 10) -> str:
    records = dataframe.loc[:, JOIN_KEYS].head(limit)
    return ", ".join(
        f"{site_id}@{cast(datetime, timestamp).isoformat()}"
        for site_id, timestamp in zip(records["site_id"], records["timestamp"], strict=True)
    )


def _missing_weather_hours(generation: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    missing: list[pd.DataFrame] = []
    for site_id, site_generation in generation.groupby("site_id", sort=False):
        site_weather = weather[weather["site_id"] == site_id]
        if site_weather.empty:
            continue
        expected = pd.date_range(
            site_generation["timestamp"].min(),
            site_generation["timestamp"].max(),
            freq="h",
        )
        missing_timestamps = expected.difference(pd.DatetimeIndex(site_weather["timestamp"]))
        if not missing_timestamps.empty:
            missing.append(pd.DataFrame({"site_id": site_id, "timestamp": missing_timestamps}))
    if not missing:
        return pd.DataFrame(columns=JOIN_KEYS)
    return pd.concat(missing, ignore_index=True)


def join_generation_weather(generation: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Strictly join validated generation and weather and validate canonical output."""
    validated_generation = validate_generation_dataframe(generation)
    validated_weather = validate_weather_dataframe(weather)

    missing_hours = _missing_weather_hours(validated_generation, validated_weather)
    unmatched = validated_generation.merge(
        validated_weather.loc[:, JOIN_KEYS],
        on=JOIN_KEYS,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"]

    problems: list[str] = []
    if not missing_hours.empty:
        problems.append(
            "Missing weather hours within the generation range: " + _describe_keys(missing_hours)
        )
    if not unmatched.empty:
        problems.append(
            "Generation timestamps cannot be matched to weather: " + _describe_keys(unmatched)
        )
    if problems:
        raise DatasetJoinError("\n".join(problems))

    combined = validated_generation.merge(
        validated_weather,
        on=JOIN_KEYS,
        how="inner",
        validate="one_to_one",
    )
    ordered_columns = [column for column in CANONICAL_COLUMNS if column in combined.columns]
    return validate_hourly_dataframe(combined.loc[:, ordered_columns])


def join_csv_files(
    generation_path: str | Path,
    weather_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Read, join, validate, and write generation and weather CSV files."""
    generation = read_generation_csv(generation_path)
    weather = validate_weather_dataframe(read_hourly_csv(weather_path))
    combined = join_generation_weather(generation, weather)
    write_processed_csv(combined, output_path)
    LOGGER.info("Wrote %d joined records to %s", len(combined), Path(output_path))
    return combined


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset join argument parser."""
    parser = argparse.ArgumentParser(
        description="Join measured generation with Open-Meteo hourly weather."
    )
    parser.add_argument("--generation", required=True, type=Path)
    parser.add_argument("--weather", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset joining command."""
    configure_logging()
    arguments = build_parser().parse_args(argv)
    try:
        join_csv_files(arguments.generation, arguments.weather, arguments.output)
    except DataLayerError as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
