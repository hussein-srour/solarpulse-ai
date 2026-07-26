# Planned architecture

## Feature-engineering boundary

`solarpulse_ai.features` sits after canonical validation and Phase 4 split
planning, and before future model training. Typed modules separate
configuration, site registry, temporal/weather/history transformations,
eligibility, split integration, contract/lineage, reports, orchestration, and
CLI execution. Generation predictors respect `target timestamp - forecast
horizon`; the target is never part of the predictor contract.

SolarPulse AI is organised as a modular pipeline with explicit boundaries
between data acquisition, analytical workflows, and delivery surfaces. The
foundation does not prescribe a model family or data platform before real
operational requirements and representative datasets are available.

## Components

### Data ingestion

The local CSV adapters keep measured generation separate from weather. The
Open-Meteo adapter retrieves external historical reanalysis in bounded date
chunks with explicit timeouts and retries. Source records are mapped into
canonical weather names without fabricating measured plant production.

### Data validation

Validation enforces the documented hourly schema, UTC timestamps, numeric
types, ranges, and unique site/timestamp keys. Invalid observations are
reported explicitly rather than silently corrected. See
[Hourly data foundation](data.md) for the field-level contract.

### Dataset alignment

Measured generation and validated weather are joined one-to-one using
`site_id` and UTC timestamp. Duplicate keys, missing weather hours, and
unmatched generation timestamps fail explicitly. No gap filling or
interpolation occurs, and the Phase 2 validator checks the final canonical
dataset. See [Historical weather integration](weather.md).

### Exploratory analysis

The `solarpulse_ai.analysis` package consumes only Phase 2-valid canonical CSV
data. Small typed modules separate profiling, descriptive statistics, temporal
analysis, Pearson correlation, non-destructive quality indicators, readiness
rules, chronological split planning, charts, report writing, and CLI
orchestration. All internal timestamps are UTC. The analysis does not mutate
records or train models, and generated reports remain local under the ignored
`reports/` directory. See [Exploratory analysis and readiness](analysis.md).

### Feature engineering

Future versioned transformations will derive only features justified by the
forecasting and monitoring use cases from the already aligned canonical data.

### Model training and prediction

`solarpulse_ai.training` separates configuration, feature-contract/data
validation, preprocessing, fixed estimators, cohorts, metrics, selection,
persistence, manifests, model cards, charts, and CLI execution. Train-only
pipelines feed a validation-only selector; the untouched test edge is reached
after selection. Generated artifacts remain outside version control. The
`models` package remains the future online inference boundary.

Phase 7 adds focused advanced modules without creating a parallel training
system: an expanding rolling-origin splitter, deterministic XGBoost candidate
generator and tuner, cross-validation reports, local experiment tracking,
versioned trusted artifacts, and an atomic champion/challenger registry.
Training-only folds tune parameters; a shared validation cohort selects among
advanced and Phase 6 candidates; test results cannot change selection or
promotion.

### Anomaly detection

Underperformance detection will compare observed behaviour with an appropriate
expected-production baseline and retain evidence for each alert. Detection
logic is intentionally deferred until baseline and alert requirements exist.

### API and dashboard

FastAPI provides the service boundary. The initial application exposes only
service metadata and liveness endpoints. A dashboard package is reserved for a
future operational interface; no dashboard framework or simulated output is
included yet.

## Cross-cutting concerns

- Settings use typed environment variables with the `SOLARPULSE_` prefix.
- Secrets must be supplied at runtime and never committed.
- Reusable standard-library logging is configured without environment or
  credential values. Persistence, orchestration, authentication, telemetry,
  and model governance will be designed alongside production requirements.
