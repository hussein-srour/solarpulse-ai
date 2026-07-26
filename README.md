# SolarPulse AI

SolarPulse AI is an open-source platform for forecasting solar photovoltaic (PV)
energy production and identifying abnormal system underperformance. The platform
will combine historical generation measurements with weather observations to
support reliable operations, maintenance, and energy planning.

This repository currently contains the professional project foundation, API
service, and development tooling. Forecasting and anomaly-detection models have
not yet been implemented.

## Planned system architecture

```text
Generation data ─┐
                 ├─> Ingestion ─> Validation ─> Feature engineering
Weather data ────┘                                  │
                                                    ├─> Forecast training
                                                    ├─> Prediction service
                                                    └─> Anomaly detection
                                                               │
                                              FastAPI <────────┤
                                                 │             │
                                                 └─> Dashboard ┘
```

The initial boundaries deliberately separate data processing, modelling, API,
and presentation concerns. Concrete storage, orchestration, model, and
dashboard choices will be introduced as requirements and representative data
become available.

## Technology stack

- Python 3.12
- FastAPI and Uvicorn for the HTTP API
- pandas for canonical tabular ingestion and validation
- Pydantic Settings for typed environment configuration
- pytest for automated testing
- Ruff for linting and formatting
- mypy for static type checking
- pre-commit for local quality checks
- Docker and Docker Compose for a consistent runtime
- GitHub Actions for continuous integration

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md
│   ├── data.md
│   └── development.md
├── data/
│   ├── external/            # Local external reference data
│   ├── processed/           # Validated generated data
│   └── raw/                 # Local source data
├── src/solarpulse_ai/
│   ├── anomaly/             # Underperformance detection boundary
│   ├── api/                 # FastAPI routing and response schemas
│   ├── config/              # Environment-backed application settings
│   ├── dashboard/           # Future dashboard boundary
│   ├── data/                # Ingestion and validation boundaries
│   ├── features/            # Feature engineering boundary
│   ├── models/              # Training and prediction boundaries
│   └── main.py              # FastAPI application factory and entry point
├── tests/
│   └── api/                 # API endpoint tests
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Installation

### Prerequisites

- Python 3.12
- `make` (optional, but recommended)
- Docker with Docker Compose (optional)

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
pre-commit install
```

## Running the API

Run the development server:

```bash
make run
```

Or invoke Uvicorn directly:

```bash
uvicorn solarpulse_ai.main:app --reload --host 0.0.0.0 --port 8000
```

The service exposes:

- `GET /` — service metadata
- `GET /health` — liveness status
- `/docs` — interactive OpenAPI documentation

To run the API in containers:

```bash
docker compose up --build
```

## Hourly data ingestion

The Phase 2 data layer provides a single canonical contract for hourly solar
generation and weather observations. It validates complete input files,
normalizes timestamps to UTC, rejects duplicate site/timestamp keys, sorts
valid records chronologically, and writes processed CSV output. It never
silently deletes or corrects an invalid record.

Run ingestion with:

```bash
python -m solarpulse_ai.data.ingestion \
  --input data/raw/hourly_data.csv \
  --output data/processed/validated_hourly_data.csv
```

The command returns exit code `0` after valid output is written and a non-zero
code with an actionable error report when ingestion or validation fails.

### Canonical data dictionary

| Field | Required | Unit | Rule |
| --- | --- | --- | --- |
| `timestamp` | Yes | UTC datetime | Valid timezone-aware instant; offsets convert to UTC |
| `site_id` | Yes | — | Non-empty string |
| `ac_energy_kwh` | Yes | kWh | Float `>= 0` |
| `ghi_w_m2` | Yes | W/m² | Float `>= 0` |
| `ambient_temperature_c` | Yes | °C | Finite float |
| `cloud_cover_pct` | Yes | % | Float from `0` to `100` |
| `relative_humidity_pct` | Yes | % | Float from `0` to `100` |
| `wind_speed_m_s` | Yes | m/s | Float `>= 0` |
| `dni_w_m2` | No | W/m² | Float `>= 0` when present |
| `dhi_w_m2` | No | W/m² | Float `>= 0` when present |
| `module_temperature_c` | No | °C | Finite float when present |
| `precipitation_mm` | No | mm | Float `>= 0` when present |
| `inverter_availability_pct` | No | % | Float from `0` to `100` when present |

Illustrative formatting only (not real solar-performance data):

```csv
timestamp,site_id,ac_energy_kwh,ghi_w_m2,ambient_temperature_c,cloud_cover_pct,relative_humidity_pct,wind_speed_m_s
2026-01-01T07:00:00Z,example-site,10.5,540.0,25.1,24.0,68.0,2.5
2026-01-01T08:00:00Z,example-site,14.2,710.0,28.4,18.0,61.0,3.2
```

See the [hourly data foundation](docs/data.md) for the complete behaviour,
directory policy, validation rules, and error reporting. Real data-source
integrations are intentionally deferred to a later phase.

## Quality checks and tests

Run all required checks:

```bash
make check
```

Or run each tool independently:

```bash
ruff check .
ruff format --check .
mypy src tests
pytest
```

## Future development roadmap

1. Integrate approved real generation and weather sources.
2. Establish reproducible feature-engineering pipelines and data versioning.
3. Add baseline forecasting experiments and evaluation methodology.
4. Build a versioned model-training and prediction workflow.
5. Introduce anomaly detection with explainable alert thresholds.
6. Expand the API with authenticated prediction and monitoring endpoints.
7. Build the operational dashboard and observability integrations.
8. Add deployment environments, model monitoring, and retraining automation.

No production dataset or simulated model output is included. The documentation
rows are illustrative formatting only. Future data additions must follow
applicable privacy, security, and licensing requirements.

## Documentation

- [Architecture](docs/architecture.md)
- [Hourly data foundation](docs/data.md)
- [Development guide](docs/development.md)

## Licence

SolarPulse AI is available under the [MIT License](LICENSE).
