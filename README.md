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
│   └── development.md
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

1. Define canonical schemas for generation, weather, site, and equipment data.
2. Implement source-specific ingestion adapters and data-quality validation.
3. Establish reproducible feature-engineering pipelines and data versioning.
4. Add baseline forecasting experiments and evaluation methodology.
5. Build a versioned model-training and prediction workflow.
6. Introduce anomaly detection with explainable alert thresholds.
7. Expand the API with authenticated prediction and monitoring endpoints.
8. Build the operational dashboard and observability integrations.
9. Add deployment environments, model monitoring, and retraining automation.

No sample data or simulated model output is included. Future data additions
must follow applicable privacy, security, and licensing requirements.

## Documentation

- [Architecture](docs/architecture.md)
- [Development guide](docs/development.md)

## Licence

SolarPulse AI is available under the [MIT License](LICENSE).
