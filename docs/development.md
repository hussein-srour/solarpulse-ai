# Development guide

## Phase 6 verification

Feature and training tests use generated temporary data only and require no
internet or private site records. Run:

```bash
ruff format --check .
ruff check .
mypy src tests
pytest
python -m pip check
```

The project targets Python 3.12. Phase 7 adds bounded `xgboost>=3.0,<4.0`
alongside scikit-learn and joblib; it does not add a remote tracker, cloud SDK,
LightGBM, CatBoost, TensorFlow, or PyTorch. XGBoost wheels require OpenMP on
macOS (`brew install libomp`). Linux CI wheels provide their normal runtime
path.

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

Run Phase 4 analysis against a private local processed CSV:

```bash
python -m solarpulse_ai.analysis.eda \
  --input data/processed/training_dataset.csv \
  --output-dir reports/eda
```

Do not stage anything from `reports/` or the data directories. Phase 4 tests
construct temporary datasets and use matplotlib's non-interactive `Agg`
backend, so CI requires neither a display nor internet access. See
[Exploratory analysis and readiness](analysis.md) for configurable thresholds
and split options.

Phase 7 tests use only small generated solar-like fixtures, `n_jobs=1`, bounded
trees, and no network. Their metrics validate software behaviour and must never
be described as real plant performance. Experiment and model outputs belong
under ignored `reports/` and `artifacts/` paths. Run the workflow documented in
[Advanced forecasting](advanced-forecasting.md).

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
