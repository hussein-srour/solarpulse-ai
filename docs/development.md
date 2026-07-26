# Development guide

## Environment

SolarPulse AI targets Python 3.12 exclusively during this initial phase. Use an
isolated virtual environment and install the editable project with its
development dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` for local overrides. The application runs with
safe defaults, and `.env` is excluded from version control.

## Workflow

Before opening a pull request, run:

```bash
make check
```

Ruff enforces lint and formatting rules, mypy runs in strict mode, and pytest
executes the automated test suite with a minimum coverage threshold.

Hourly CSV ingestion can be exercised locally with illustrative data using the
command documented in [Hourly data foundation](data.md). Do not commit files
placed in `data/raw`, `data/external`, or `data/processed`; only their
`.gitkeep` markers belong in version control.

Historical-weather download and generation joining are documented in
[Historical weather integration](weather.md). HTTP tests must use an injected
mock transport and must never call the live Open-Meteo service. Use only
illustrative site metadata in tests and documentation.

Install the pre-commit hooks once per clone:

```bash
pre-commit install
```

## Scope rules

- Do not commit credentials, raw operational data, generated model artifacts,
  or reports.
- Add source adapters without coupling domain logic to a vendor.
- Add tests with every behaviour change.
- Record major architecture decisions before selecting storage, orchestration,
  modelling, or dashboard technologies.
