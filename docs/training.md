# Baseline model training

## Objective and inputs

Phase 6 trains reproducible fixed baselines for hourly `ac_energy_kwh` 24 hours
ahead. It consumes the Phase 5 feature CSV and `feature_manifest.json`, plus an
optional strict JSON configuration, and writes a verified model and evaluation
reports to Git-ignored directories.

The CSV must contain UTC `timestamp`, `site_id`, target `ac_energy_kwh`,
`split`, the three `feature_*` eligibility fields, and every predictor declared
by the manifest. Predictors are never inferred from leftover columns.

## Contract, eligibility, and leakage protection

The manifest must declare the target, ordered keys/predictors, metadata,
numerical/categorical roles, sensible row counts, and the 24-hour horizon.
Duplicate, missing, unclassified, undeclared, or metadata predictors fail.
Keys, current target, split/eligibility fields, report fields, and row
identities cannot enter `X`. Timestamps support auditing and plots only.

Phase 5 exact lags and forecast-cutoff rolling generation are allowed because
they were constructed at the prediction-time boundary. Persistence reads the
exact `ac_energy_lag_24h` column by default—never the current target or previous
row.

Only `feature_eligible == true` rows can be fitted or evaluated. Reports retain
original, eligible, excluded, by-split, by-site, and by-reason counts. Training
fails when any split is empty or below its configured minimum.

Timestamps must parse as UTC. Duplicate `(site_id, timestamp)` records,
negative/non-finite targets, infinite predictors, invalid split labels, and
overlapping or nonchronological periods are rejected. The order is train fit,
validation selection, optional train-plus-validation refit, then one untouched
test evaluation. There is no shuffle, random split, time-mixing
cross-validation, tuning, or test-driven replacement.

## Preprocessing and candidates

Scikit-learn `Pipeline` and `ColumnTransformer` save transformations with the
estimator. Numeric fields use training-fitted median imputation and missing
indicators; Ridge also uses training-fitted standardisation. Declared
categoricals use training-fitted most-frequent imputation and dense, unknown-
safe one-hot encoding. Booleans are converted consistently. Trees are not
scaled.

Fixed candidates are exact-lag persistence, training-only mean
`DummyRegressor`, Ridge, deterministic bounded random forest, and deterministic
bounded histogram gradient boosting. No search occurs. These are baselines,
not final production models.

## Common cohorts and selection

Every candidate shares the same validation rows: eligible rows with valid
target, finite persistence lag, and finite candidate outputs. Reports include
full/common rows, coverage, exclusions, and reasons. Selection uses validation
MAE only, followed by validation RMSE, fixed simplicity order, and identifier
for deterministic ties. Training and test metrics cannot select the winner.

The selected final model and persistence share a common test cohort. If
persistence wins, the run states that no learned baseline won and serialises an
exact-lag prediction specification.

## Metrics and post-processing

Reports include overall and per-site MAE, RMSE, median absolute error, R², mean
bias (predicted minus actual), WAPE, count, actual/predicted total energy, and
coverage. Energy errors use kWh. WAPE is null when actual total energy is zero.
When positive `installed_capacity_kwp` exists, normalised errors use kWh/kWp.
Daylight-only secondary metrics are included when `is_daylight` exists.

With negative clipping enabled, raw-negative counts remain reported and values
become zero before metrics. There is no maximum clip; clipping can be disabled.

## Artifacts, reports, and reproducibility

Each run writes `selected_model.joblib`, model metadata, strict configuration,
training manifest, feature-manifest snapshot, and model card. The model is
reloaded and raw predictions must match before completion. Provenance includes
ordered predictors, SHA-256 input/model checksums, split boundaries, fixed
parameters, metrics, seed, UTC time, Git commit, and Python, pandas, NumPy,
scikit-learn, and joblib versions—never source records.

Reports include JSON/Markdown summaries; comparison, site, test, compact
prediction/residual, and validation-only permutation-importance CSVs; and
headless charts. A persistence winner has no estimator importance, so that
chart is skipped with a warning. Importance is not proof of causality.

Generated datasets, reports, charts, model cards, manifests, predictions, and
joblib files remain ignored by Git.

## CLI

```bash
python -m solarpulse_ai.training.train \
  --features data/processed/model_features.csv \
  --feature-manifest reports/features/feature_manifest.json \
  --model-dir artifacts/models/phase-06 \
  --report-dir reports/training \
  --run-name baseline-v1
```

`--config` accepts the complete strict configuration. CLI overrides cover
models, MAE policy, seed, persistence lag, clipping, split minima, fixed forest
and boosting parameters, permutation repeats, and refitting. Logs contain paths,
counts, candidates, winner, test completion, and artifact location—not full
records, secrets, or environment variables.

## Limitations

Historical observations/reanalysis weather are proxies for forecasts available
24 hours ahead and may make offline results optimistic. Synthetic tests validate
software, not plant performance. Test results do not guarantee future
performance. Phase 7 builds on these exact interfaces; see
[Advanced forecasting](advanced-forecasting.md).
