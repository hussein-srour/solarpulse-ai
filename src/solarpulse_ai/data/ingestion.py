"""CSV ingestion command for canonical hourly solar and weather data."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from solarpulse_ai.data.errors import CSVIngestionError, DataLayerError
from solarpulse_ai.data.validation import validate_hourly_dataframe
from solarpulse_ai.logging_config import configure_logging, get_logger

LOGGER = get_logger(__name__)


def read_hourly_csv(input_path: str | Path) -> pd.DataFrame:
    """Read an hourly CSV after checking that it is an accessible regular file."""
    path = Path(input_path)
    if not path.exists():
        raise CSVIngestionError(f"Input CSV does not exist: {path}")
    if not path.is_file():
        raise CSVIngestionError(f"Input CSV path is not a file: {path}")

    try:
        dataframe = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise CSVIngestionError(
            f"Input CSV is completely empty and has no header: {path}"
        ) from error
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise CSVIngestionError(f"Could not read input CSV {path}: {error}") from error

    if dataframe.empty:
        raise CSVIngestionError(f"Input CSV contains no data rows: {path}")
    return dataframe


def write_processed_csv(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    """Write validated observations to a processed CSV, creating its parent directory."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)
    except (OSError, ValueError) as error:
        raise CSVIngestionError(f"Could not write processed CSV {path}: {error}") from error
    return path


def ingest_hourly_csv(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Read, validate, normalize, sort, and save an hourly CSV dataset."""
    input_location = Path(input_path)
    output_location = Path(output_path)
    LOGGER.info("Reading hourly data from %s", input_location)
    raw_dataframe = read_hourly_csv(input_location)
    validated_dataframe = validate_hourly_dataframe(raw_dataframe)
    write_processed_csv(validated_dataframe, output_location)
    LOGGER.info(
        "Processed %d records from %s to %s",
        len(validated_dataframe),
        input_location,
        output_location,
    )
    return validated_dataframe


def build_parser() -> argparse.ArgumentParser:
    """Build the ingestion command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate and normalize hourly SolarPulse CSV data."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to the source CSV.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the validated processed CSV.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CSV ingestion command and return a process-compatible exit code."""
    configure_logging()
    arguments = build_parser().parse_args(argv)
    try:
        ingest_hourly_csv(arguments.input, arguments.output)
    except DataLayerError as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
