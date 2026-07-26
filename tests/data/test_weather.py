"""Tests for the Open-Meteo historical-weather adapter."""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from solarpulse_ai.config import SiteConfig
from solarpulse_ai.data.errors import DataValidationError, WeatherAPIError
from solarpulse_ai.data.weather import (
    OpenMeteoHistoricalClient,
    main,
    validate_date_range,
    validate_weather_dataframe,
)


def _site() -> SiteConfig:
    return SiteConfig(
        site_id="example-site",
        latitude=-6.7924,
        longitude=39.2083,
        timezone="Africa/Dar_es_Salaam",
        installed_capacity_kwp=10,
        panel_tilt_degrees=10,
        panel_azimuth_degrees=0,
    )


def _payload() -> dict[str, Any]:
    return {
        "hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
            "temperature_2m": [25.0, 25.5],
            "relative_humidity_2m": [80.0, 78.0],
            "precipitation": [0.0, 0.2],
            "cloud_cover": [40.0, 35.0],
            "wind_speed_10m": [2.0, 2.5],
            "shortwave_radiation": [0.0, 10.0],
            "direct_normal_irradiance": [0.0, 5.0],
            "diffuse_radiation": [0.0, 5.0],
        }
    }


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_successful_response_maps_fields_and_uses_utc_options() -> None:
    """The adapter requests explicit units and maps all hourly fields."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_payload(), request=request)

    with _client(handler) as http_client:
        result = OpenMeteoHistoricalClient(http_client).fetch(
            _site(), date(2025, 1, 1), date(2025, 1, 1)
        )

    assert result["ambient_temperature_c"].tolist() == [25.0, 25.5]
    assert result["relative_humidity_pct"].tolist() == [80.0, 78.0]
    assert result["site_id"].tolist() == ["example-site", "example-site"]
    assert str(result["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert "ac_energy_kwh" not in result.columns
    query = requests[0].url.params
    assert query["timezone"] == "UTC"
    assert query["wind_speed_unit"] == "ms"
    assert query["precipitation_unit"] == "mm"
    assert requests[0].headers["user-agent"].startswith("SolarPulse-AI/")


def test_timeout_is_retried_then_reported() -> None:
    """Repeated read timeouts stop after the configured bound."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    with _client(handler) as http_client:
        client = OpenMeteoHistoricalClient(http_client, max_retries=2, sleep=lambda _: None)
        with pytest.raises(WeatherAPIError, match="timed out after 3 attempts"):
            client.fetch(_site(), date(2025, 1, 1), date(2025, 1, 1))

    assert calls == 3


def test_http_4xx_is_not_retried() -> None:
    """Permanent client errors fail immediately."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, request=request)

    with _client(handler) as http_client, pytest.raises(WeatherAPIError, match="HTTP 400"):
        OpenMeteoHistoricalClient(http_client).fetch(_site(), date(2025, 1, 1), date(2025, 1, 1))
    assert calls == 1


def test_http_5xx_retries_and_can_recover() -> None:
    """Temporary server errors use bounded retries before succeeding."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json=_payload(), request=request)

    with _client(handler) as http_client:
        result = OpenMeteoHistoricalClient(http_client, max_retries=2, sleep=lambda _: None).fetch(
            _site(), date(2025, 1, 1), date(2025, 1, 1)
        )

    assert calls == 3
    assert len(result) == 2


def test_exhausted_http_5xx_has_clear_error() -> None:
    """A temporary response still fails clearly after the retry limit."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request)

    with _client(handler) as http_client, pytest.raises(WeatherAPIError, match="HTTP 500"):
        OpenMeteoHistoricalClient(http_client, max_retries=1, sleep=lambda _: None).fetch(
            _site(), date(2025, 1, 1), date(2025, 1, 1)
        )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, content=b"{"), "malformed JSON"),
        (httpx.Response(200, json={}), "missing the hourly object"),
        (
            httpx.Response(200, json={"hourly": {"time": ["2025-01-01T00:00"]}}),
            "missing hourly fields",
        ),
    ],
)
def test_malformed_or_missing_response_fields_fail(response: httpx.Response, message: str) -> None:
    """Malformed JSON and missing response objects or fields are rejected."""

    def handler(request: httpx.Request) -> httpx.Response:
        response.request = request
        return response

    with _client(handler) as http_client, pytest.raises(WeatherAPIError, match=message):
        OpenMeteoHistoricalClient(http_client).fetch(_site(), date(2025, 1, 1), date(2025, 1, 1))


def test_mismatched_array_lengths_are_rejected() -> None:
    """Hourly variables must align exactly with the time array."""
    payload = _payload()
    payload["hourly"]["temperature_2m"] = [25.0]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    with (
        _client(handler) as http_client,
        pytest.raises(WeatherAPIError, match="mismatched lengths"),
    ):
        OpenMeteoHistoricalClient(http_client).fetch(_site(), date(2025, 1, 1), date(2025, 1, 1))


def test_date_range_validation_rejects_reversed_and_oversized_ranges() -> None:
    """Date ranges are ordered and bounded."""
    with pytest.raises(WeatherAPIError, match="on or after"):
        validate_date_range(date(2025, 1, 2), date(2025, 1, 1))
    with pytest.raises(WeatherAPIError, match="limited to 366 days"):
        validate_date_range(date(2024, 1, 1), date(2025, 1, 1))


def test_long_requests_are_split_into_documented_chunks() -> None:
    """Valid multi-month ranges are divided into bounded API requests."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = _payload()
        start = request.url.params["start_date"]
        payload["hourly"]["time"] = [f"{start}T00:00", f"{start}T01:00"]
        return httpx.Response(200, json=payload, request=request)

    with _client(handler) as http_client:
        OpenMeteoHistoricalClient(http_client).fetch(_site(), date(2025, 1, 1), date(2025, 2, 2))

    assert len(requests) == 2
    assert requests[0].url.params["end_date"] == "2025-01-31"
    assert requests[1].url.params["start_date"] == "2025-02-01"


def test_weather_validation_rejects_bad_values_duplicates_and_shape() -> None:
    """Weather-only validation enforces fields, values, sites, and unique keys."""
    with pytest.raises(DataValidationError, match="empty_weather"):
        validate_weather_dataframe(pd.DataFrame())

    dataframe = pd.DataFrame(_decode_payload_as_canonical())
    with pytest.raises(DataValidationError, match="missing_weather_column"):
        validate_weather_dataframe(dataframe.drop(columns="ghi_w_m2"))

    dataframe = pd.concat([dataframe, dataframe.iloc[[1]]], ignore_index=True)
    dataframe.loc[0, "cloud_cover_pct"] = 101
    dataframe.loc[0, "site_id"] = ""
    with pytest.raises(DataValidationError) as captured:
        validate_weather_dataframe(dataframe)
    codes = {issue.code for issue in captured.value.issues}
    assert {
        "invalid_weather_site_id",
        "invalid_weather_value",
        "duplicate_weather_record",
    }.issubset(codes)


def _decode_payload_as_canonical() -> dict[str, object]:
    hourly = _payload()["hourly"]
    return {
        "timestamp": hourly["time"],
        "site_id": ["example-site", "example-site"],
        "ambient_temperature_c": hourly["temperature_2m"],
        "relative_humidity_pct": hourly["relative_humidity_2m"],
        "precipitation_mm": hourly["precipitation"],
        "cloud_cover_pct": hourly["cloud_cover"],
        "wind_speed_m_s": hourly["wind_speed_10m"],
        "ghi_w_m2": hourly["shortwave_radiation"],
        "dni_w_m2": hourly["direct_normal_irradiance"],
        "dhi_w_m2": hourly["diffuse_radiation"],
    }


def test_weather_cli_returns_nonzero_for_invalid_range(tmp_path: Path) -> None:
    """The command reports validation failures without attempting HTTP."""
    exit_code = main(
        [
            "--site-config",
            str(Path("config/example_site.json")),
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-01",
            "--output",
            str(tmp_path / "weather.csv"),
        ]
    )
    assert exit_code == 1
