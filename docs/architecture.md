# Planned architecture

SolarPulse AI is organised as a modular pipeline with explicit boundaries
between data acquisition, analytical workflows, and delivery surfaces. The
foundation does not prescribe a model family or data platform before real
operational requirements and representative datasets are available.

## Components

### Data ingestion

Source-specific adapters will acquire historical PV generation and weather
observations. Adapters will preserve source metadata and make ingestion
repeatable without embedding credentials in the repository.

### Data validation

Validation will enforce canonical schemas, timestamps, units, ranges, and
cross-field constraints. Invalid observations will be reported explicitly
rather than silently corrected.

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
- Structured logging, persistence, orchestration, authentication, telemetry,
  and model governance will be designed alongside their production
  requirements.
