# Historical weather integration

Phase 3 adds an adapter for the Open-Meteo Historical Weather API and a strict
service for joining external weather with measured solar generation. It does
not train a model and does not infer plant production from weather.

## Architecture and provenance

```text
site JSON ─> Open-Meteo Historical Weather API ─> weather validation ─┐
                                                                     ├─> one-to-one UTC join
measured generation CSV ─────────────────> generation validation ────┘
                                                                          │
                                                        Phase 2 canonical validation
```

The weather source is
[Open-Meteo's Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).
Open-Meteo API data are offered under the
[Creative Commons Attribution 4.0 licence](https://open-meteo.com/en/license).
Historical values are external reanalysis/model data assembled from numerical
models and observations; they are not on-site sensor measurements. Users of
persisted or redistributed weather data must preserve the required attribution.

No API credential is needed by this adapter. Do not place confidential
coordinates, credentials, environment dumps, or production data in source
control.

## Site configuration

The JSON object has these required fields:

| Field | Validation |
| --- | --- |
| `site_id` | Non-empty string |
| `latitude` | `-90` to `90` |
| `longitude` | `-180` to `180` |
| `timezone` | Valid IANA timezone |
| `installed_capacity_kwp` | Greater than zero |
| `panel_tilt_degrees` | `0` to `90` |
| `panel_azimuth_degrees` | Numeric orientation metadata |

[`config/example_site.json`](../config/example_site.json) contains an
illustrative Dar es Salaam location only. It is not an AG Energies installation
and must not be treated as confidential or operational configuration:

```json
{
  "site_id": "example-dar-es-salaam-site",
  "latitude": -6.7924,
  "longitude": 39.2083,
  "timezone": "Africa/Dar_es_Salaam",
  "installed_capacity_kwp": 10.0,
  "panel_tilt_degrees": 10.0,
  "panel_azimuth_degrees": 0.0
}
```

The site timezone documents the physical location. API timestamps are requested
and stored in UTC so join keys are unambiguous.

## Weather field mapping

| Open-Meteo hourly field | Canonical field | Unit |
| --- | --- | --- |
| `temperature_2m` | `ambient_temperature_c` | °C |
| `relative_humidity_2m` | `relative_humidity_pct` | % |
| `precipitation` | `precipitation_mm` | mm |
| `cloud_cover` | `cloud_cover_pct` | % |
| `wind_speed_10m` | `wind_speed_m_s` | m/s |
| `shortwave_radiation` | `ghi_w_m2` | W/m² |
| `direct_normal_irradiance` | `dni_w_m2` | W/m² |
| `diffuse_radiation` | `dhi_w_m2` | W/m² |

Every weather record receives the configured `site_id`. Weather output never
contains `ac_energy_kwh`: energy must come from measured plant records.

## Measured-generation CSV

The measured input has exactly three required values per record:

```csv
timestamp,site_id,ac_energy_kwh
2025-01-01T00:00:00Z,example-dar-es-salaam-site,0.0
2025-01-01T01:00:00Z,example-dar-es-salaam-site,1.7
```

Timestamps must explicitly include a timezone and are converted to UTC. Energy
must be finite and non-negative. A site/timestamp key must be unique. Invalid
rows are reported and are never silently removed, clipped, or corrected.

## Commands

Download historical weather:

```bash
python -m solarpulse_ai.data.weather \
  --site-config config/example_site.json \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --output data/external/weather.csv
```

Date order is validated. Each invocation is limited to 366 days and valid
ranges are divided into non-overlapping 31-day API requests. The client uses
explicit connection/read timeouts and bounded retries for temporary failures.
It reports HTTP, timeout, malformed JSON, missing-field, and array-alignment
errors without logging complete API responses.

Join generation and weather:

```bash
python -m solarpulse_ai.data.join \
  --generation data/raw/generation.csv \
  --weather data/external/weather.csv \
  --output data/processed/training_dataset.csv
```

The service validates both inputs, rejects duplicate keys, reports missing
weather hours and unmatched generation timestamps, and uses a strict one-to-one
join on `site_id` plus UTC timestamp. It never interpolates or fills data. The
combined output follows the Phase 2 canonical column order and is passed
through the Phase 2 validator before it is written.

## Limitations and data quality

- Reanalysis is gridded model output and may differ from conditions at a
  specific array, particularly near coasts, complex terrain, or local shading.
- Radiation fields are not a substitute for calibrated on-site irradiance
  sensors, and weather-derived estimates are not measured production.
- Grid resolution, source-model updates, latency, and biases can affect
  consistency. Record retrieval dates and source/model choices in downstream
  experiments.
- Exact hourly alignment exposes missing data instead of hiding it. Investigate
  clock drift, daylight-saving conversions, meter aggregation intervals, and
  site identifiers at the source.
- Panel tilt, azimuth, and capacity are configuration metadata in Phase 3; the
  adapter does not transform horizontal radiation to plane-of-array irradiance.
