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
