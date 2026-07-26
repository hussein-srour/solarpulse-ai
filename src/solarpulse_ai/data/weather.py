"""Open-Meteo historical-weather adapter and download command."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from solarpulse_ai.config import SiteConfig, load_site_config
from solarpulse_ai.data.errors import (
    DataLayerError,
    DataValidationError,
    ValidationIssue,
    WeatherAPIError,
)
from solarpulse_ai.data.ingestion import write_processed_csv
from solarpulse_ai.logging_config import configure_logging, get_logger

LOGGER = get_logger(__name__)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "SolarPulse-AI/0.1 (+https://github.com/hussein-srour/solarpulse-ai)"
MAX_REQUEST_DAYS = 366
CHUNK_DAYS = 31
TEMPORARY_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

WEATHER_FIELD_MAP: Mapping[str, str] = {
    "temperature_2m": "ambient_temperature_c",
    "relative_humidity_2m": "relative_humidity_pct",
    "precipitation": "precipitation_mm",
    "cloud_cover": "cloud_cover_pct",
    "wind_speed_10m": "wind_speed_m_s",
    "shortwave_radiation": "ghi_w_m2",
    "direct_normal_irradiance": "dni_w_m2",
    "diffuse_radiation": "dhi_w_m2",
}
WEATHER_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "site_id",
    *WEATHER_FIELD_MAP.values(),
)


def validate_date_range(start_date: date, end_date: date) -> None:
    """Reject reversed and excessively large historical-weather requests."""
    if end_date < start_date:
        raise WeatherAPIError("end-date must be on or after start-date")
    requested_days = (end_date - start_date).days + 1
    if requested_days > MAX_REQUEST_DAYS:
        raise WeatherAPIError(
            f"Date range is limited to {MAX_REQUEST_DAYS} days per command invocation"
        )


def _rows(mask: pd.Series[bool]) -> tuple[int, ...]:
    return tuple(position + 2 for position, invalid in enumerate(mask) if invalid)[:10]


def validate_weather_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate weather-only canonical records without inventing generation values."""
    if dataframe.empty:
        raise DataValidationError(
            [ValidationIssue("empty_weather", "Weather data must contain at least one row.")]
        )
    missing = [column for column in WEATHER_COLUMNS if column not in dataframe.columns]
    if missing:
        raise DataValidationError(
            [
                ValidationIssue(
                    "missing_weather_column",
                    "The Open-Meteo weather field is required.",
                    column=column,
                )
                for column in missing
            ]
        )

    validated = dataframe.loc[:, WEATHER_COLUMNS].copy()
    issues: list[ValidationIssue] = []
    site_ids = validated["site_id"].astype("string")
    invalid_sites = validated["site_id"].isna() | site_ids.str.strip().eq("")
    if invalid_sites.any():
        issues.append(
            ValidationIssue(
                "invalid_weather_site_id",
                "Weather site_id values must be non-empty strings.",
                column="site_id",
                rows=_rows(invalid_sites),
            )
        )
    validated["site_id"] = site_ids

    timestamps = pd.to_datetime(validated["timestamp"], errors="coerce", utc=True, format="mixed")
    invalid_timestamps = timestamps.isna()
    if invalid_timestamps.any():
        issues.append(
            ValidationIssue(
                "invalid_weather_timestamp",
                "Weather timestamps must be valid UTC instants.",
                column="timestamp",
                rows=_rows(invalid_timestamps),
            )
        )
    validated["timestamp"] = timestamps

    ranged_fields = {
        "relative_humidity_pct": (0.0, 100.0),
        "cloud_cover_pct": (0.0, 100.0),
        "precipitation_mm": (0.0, None),
        "wind_speed_m_s": (0.0, None),
        "ghi_w_m2": (0.0, None),
        "dni_w_m2": (0.0, None),
        "dhi_w_m2": (0.0, None),
    }
    for column in WEATHER_FIELD_MAP.values():
        raw_values = validated[column]
        values = pd.to_numeric(raw_values, errors="coerce")
        invalid = raw_values.isna() | values.isna() | (values.notna() & ~values.map(math.isfinite))
        minimum, maximum = ranged_fields.get(column, (None, None))
        if minimum is not None:
            invalid |= values < minimum
        if maximum is not None:
            invalid |= values > maximum
        if invalid.any():
            issues.append(
                ValidationIssue(
                    "invalid_weather_value",
                    "Weather values must be finite and within the canonical field range.",
                    column=column,
                    rows=_rows(invalid),
                )
            )
        validated[column] = values.astype(float)

    valid_keys = validated["site_id"].notna() & validated["timestamp"].notna()
    duplicates = validated.duplicated(subset=["site_id", "timestamp"], keep=False) & valid_keys
    if duplicates.any():
        issues.append(
            ValidationIssue(
                "duplicate_weather_record",
                "Each weather site_id and UTC timestamp combination must be unique.",
                rows=_rows(duplicates),
            )
        )
    if issues:
        raise DataValidationError(issues)

    return validated.sort_values(["site_id", "timestamp"], kind="stable").reset_index(drop=True)


class OpenMeteoHistoricalClient:
    """Bounded, retrying client for Open-Meteo historical hourly weather."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Configure an injected or internally owned HTTP client."""
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(30.0, connect=5.0),
            headers={"User-Agent": USER_AGENT},
        )
        self._owns_client = client is None
        self._max_retries = max_retries
        self._sleep = sleep

    def __enter__(self) -> OpenMeteoHistoricalClient:
        """Return the client for context-managed use."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close internally owned HTTP resources."""
        self.close()

    def close(self) -> None:
        """Close only an internally created HTTP client."""
        if self._owns_client:
            self._client.close()

    def fetch(self, site: SiteConfig, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch, map, combine, and validate bounded date chunks."""
        validate_date_range(start_date, end_date)
        chunks: list[pd.DataFrame] = []
        chunk_start = start_date
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end_date)
            chunks.append(self._fetch_chunk(site, chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(days=1)
        return validate_weather_dataframe(pd.concat(chunks, ignore_index=True))

    def _fetch_chunk(self, site: SiteConfig, start_date: date, end_date: date) -> pd.DataFrame:
        parameters: dict[str, str | float] = {
            "latitude": site.latitude,
            "longitude": site.longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(WEATHER_FIELD_MAP),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        }
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(
                    ARCHIVE_URL,
                    params=parameters,
                    headers={"User-Agent": USER_AGENT},
                )
            except httpx.TimeoutException as error:
                if attempt == self._max_retries:
                    raise WeatherAPIError(
                        f"Open-Meteo request timed out after {attempt + 1} attempts"
                    ) from error
                self._sleep(0.25 * (2**attempt))
                continue
            except httpx.RequestError as error:
                if attempt == self._max_retries:
                    raise WeatherAPIError(
                        f"Open-Meteo request failed after {attempt + 1} attempts: {error}"
                    ) from error
                self._sleep(0.25 * (2**attempt))
                continue

            if response.status_code in TEMPORARY_STATUS_CODES and attempt < self._max_retries:
                self._sleep(0.25 * (2**attempt))
                continue
            if response.is_error:
                raise WeatherAPIError(
                    f"Open-Meteo returned HTTP {response.status_code} for "
                    f"{start_date.isoformat()} through {end_date.isoformat()}"
                )
            break

        if response is None:
            raise WeatherAPIError("Open-Meteo request produced no response")
        return self._decode_response(response, site.site_id)

    @staticmethod
    def _decode_response(response: httpx.Response, site_id: str) -> pd.DataFrame:
        try:
            payload: Any = response.json()
        except ValueError as error:
            raise WeatherAPIError("Open-Meteo returned malformed JSON") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("hourly"), dict):
            raise WeatherAPIError("Open-Meteo response is missing the hourly object")

        hourly = payload["hourly"]
        required_fields = ("time", *WEATHER_FIELD_MAP)
        missing = [field for field in required_fields if field not in hourly]
        if missing:
            raise WeatherAPIError(
                "Open-Meteo response is missing hourly fields: " + ", ".join(missing)
            )
        arrays = {field: hourly[field] for field in required_fields}
        if any(not isinstance(values, list) for values in arrays.values()):
            raise WeatherAPIError("Open-Meteo hourly fields must be arrays")
        lengths = {len(values) for values in arrays.values()}
        if len(lengths) != 1:
            raise WeatherAPIError("Open-Meteo hourly arrays have mismatched lengths")
        if lengths == {0}:
            raise WeatherAPIError("Open-Meteo returned no hourly weather records")

        dataframe = pd.DataFrame(
            {
                "timestamp": arrays["time"],
                "site_id": site_id,
                **{canonical: arrays[source] for source, canonical in WEATHER_FIELD_MAP.items()},
            }
        )
        return dataframe


def download_weather(
    site_config_path: str | Path,
    start_date: date,
    end_date: date,
    output_path: str | Path,
) -> pd.DataFrame:
    """Download validated weather data and write it as CSV."""
    site = load_site_config(site_config_path)
    with OpenMeteoHistoricalClient() as client:
        dataframe = client.fetch(site, start_date, end_date)
    write_processed_csv(dataframe, output_path)
    LOGGER.info("Wrote %d weather records to %s", len(dataframe), Path(output_path))
    return dataframe


def _date_argument(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use an ISO date in YYYY-MM-DD format") from error


def build_parser() -> argparse.ArgumentParser:
    """Build the weather download argument parser."""
    parser = argparse.ArgumentParser(description="Download Open-Meteo historical weather.")
    parser.add_argument("--site-config", required=True, type=Path)
    parser.add_argument("--start-date", required=True, type=_date_argument)
    parser.add_argument("--end-date", required=True, type=_date_argument)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the weather command."""
    configure_logging()
    arguments = build_parser().parse_args(argv)
    try:
        download_weather(
            arguments.site_config,
            arguments.start_date,
            arguments.end_date,
            arguments.output,
        )
    except DataLayerError as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
