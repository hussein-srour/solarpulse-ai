# Planned architecture

SolarPulse AI is organised as a modular pipeline with explicit boundaries
between data acquisition, analytical workflows, and delivery surfaces. The
foundation does not prescribe a model family or data platform before real
operational requirements and representative datasets are available.

## Components

### Data ingestion

The current local CSV adapter reads hourly PV generation and weather
observations into a source-independent canonical schema, then writes validated
processed output. Future source-specific adapters will preserve source
metadata and make acquisition repeatable without embedding credentials in the
repository.

### Data validation

Validation enforces the documented hourly schema, UTC timestamps, numeric
types, ranges, and unique site/timestamp keys. Invalid observations are
reported explicitly rather than silently corrected. See
[Hourly data foundation](data.md) for the field-level contract.

### Feature engineering

Versioned transformations will align weather and generation time series and
derive only features justified by the forecasting and monitoring use cases.

### Model training and prediction

Training will be reproducible and evaluated against transparent baselines.
Prediction will load versioned, approved artifacts and expose a stable
application boundary. Neither workflow is implemented in this foundation.

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
