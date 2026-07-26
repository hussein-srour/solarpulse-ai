# Hourly data foundation

The data layer defines and enforces the canonical contract shared by future
solar-generation and weather integrations. It reads local CSV files, converts
timestamps to UTC, validates every record, reports invalid rows without
silently changing or deleting them, and writes valid data to the processed
area. It does not call external APIs or perform modelling.

## Directory layout

- `data/raw/` is for source files kept only on the local system.
- `data/external/` is for externally supplied reference data kept locally.
- `data/processed/` is for validated, generated outputs.

Git tracks only each directory's `.gitkeep` marker. Operational and generated
datasets are ignored.

## Data dictionary

| Field | Required | Type | Unit | Validation |
| --- | --- | --- | --- | --- |
| `timestamp` | Yes | Timezone-aware datetime | UTC | Must parse as a datetime; offsets are converted to UTC |
| `site_id` | Yes | String | — | Must be non-empty |
| `ac_energy_kwh` | Yes | Float | kWh | `>= 0` |
| `ghi_w_m2` | Yes | Float | W/m² | `>= 0` |
| `ambient_temperature_c` | Yes | Float | °C | Must be finite |
| `cloud_cover_pct` | Yes | Float | % | `0–100`, inclusive |
| `relative_humidity_pct` | Yes | Float | % | `0–100`, inclusive |
| `wind_speed_m_s` | Yes | Float | m/s | `>= 0` |
| `dni_w_m2` | No | Float | W/m² | If present, `>= 0` |
| `dhi_w_m2` | No | Float | W/m² | If present, `>= 0` |
| `module_temperature_c` | No | Float | °C | If present, must be finite |
| `precipitation_mm` | No | Float | mm | If present, `>= 0` |
| `inverter_availability_pct` | No | Float | % | If present, `0–100`, inclusive |

Optional columns may be omitted. If an optional column is supplied, every
non-empty value must satisfy its type and range constraints. Required fields
cannot be empty. Positive or negative infinity is not accepted.

The combination of `site_id` and UTC-normalized `timestamp` is unique. This
means timestamps that describe the same instant using different offsets are
duplicates for the same site.

## Illustrative CSV

These rows demonstrate formatting only and do not represent real production
or measured solar performance:

```csv
timestamp,site_id,ac_energy_kwh,ghi_w_m2,ambient_temperature_c,cloud_cover_pct,relative_humidity_pct,wind_speed_m_s
2026-01-01T07:00:00Z,example-site,10.5,540.0,25.1,24.0,68.0,2.5
2026-01-01T08:00:00Z,example-site,14.2,710.0,28.4,18.0,61.0,3.2
```

## Running ingestion

Install the project and development dependencies, then run:

```bash
python -m solarpulse_ai.data.ingestion \
  --input data/raw/hourly_data.csv \
  --output data/processed/validated_hourly_data.csv
```

The command logs its input and output paths and the number of processed
records. Exit code `0` means the output was validated and saved. A non-zero
exit code means the input was not processed; the log describes missing
columns, invalid values, duplicate keys, invalid timestamps, empty files, or
file-system errors. At most the first ten affected CSV row numbers are shown
for each validation rule.

Records are sorted chronologically after validation. Invalid records are never
silently dropped, imputed, clipped, or otherwise corrected.

## Phase 3 sources

Phase 3 keeps measured generation separate from external Open-Meteo historical
weather until both have been independently validated. The strict join produces
this canonical contract without interpolation. Weather is reanalysis data and
must never be described as measured plant production; `ac_energy_kwh` comes
only from the measured-generation CSV. See
[Historical weather integration](weather.md) for provenance, commands,
attribution, and limitations.
