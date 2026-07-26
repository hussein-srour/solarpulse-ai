# Advanced forecasting

## Purpose and boundary

Phase 7 extends the Phase 6 training package; it does not introduce a second
training system. The objective remains hourly `ac_energy_kwh` 24 hours ahead
from the validated Phase 5 feature contract. Advanced models follow baselines
because extra complexity must earn its place on the same validation cohort.
Persistence, Ridge, random forest, or histogram gradient boosting can honestly
remain champion.

Meaningful performance conclusions require genuine measured generation data.
The generated test fixtures prove software behaviour only and are never plant
performance evidence or a production-readiness claim.

## XGBoost and preprocessing

`XGBRegressor` uses squared-error regression, CPU histogram trees, an explicit
missing-value marker, a fixed seed, and bounded `n_jobs`. The dependency is
`xgboost>=3.0,<4.0`, compatible with Python 3.12 and the current scikit-learn
range. macOS installations need OpenMP, commonly installed with
`brew install libomp`; Linux CI uses the wheel's normal runtime dependencies.

The existing Phase 6 `ColumnTransformer` remains inside the persisted pipeline.
Numeric medians, missing indicators, categorical modes, and one-hot categories
are learned independently from each fold's training rows. Unknown categories
are ignored safely. XGBoost does not need scaling. Keys, target, split,
eligibility, and undeclared columns cannot enter the predictor matrix; infinite
numeric values fail before training.

## Rolling-origin validation and tuning

Only `split == "train"` is given to the tuner. Unique UTC timestamps remain in
order, and every site's rows at one timestamp stay together. Each fold expands
the training period, leaves a configurable gap, and evaluates a fixed-size
validation window. The default 24-hour gap is at least the forecast horizon,
which prevents target-adjacent information crossing the prediction boundary.
Empty folds, overlapping periods, unsorted records, and insufficient history
fail explicitly.

The bounded search covers `n_estimators`, `max_depth`, `learning_rate`,
`min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`,
`reg_lambda`, and `gamma`. A fixed Python random seed shuffles the finite
Cartesian space and takes at most the search budget. Dataset, configuration,
manifest, package versions, budget, and seed therefore reproduce candidate
order. Candidate failures are recorded; the run fails if every candidate
fails.

Each candidate records fold MAE, RMSE, median absolute error, R², bias, WAPE,
count, totals, negative raw predictions, coverage, daylight metrics, per-site
metrics, and fit/prediction durations. Ranking minimises mean selection metric,
then variability, tree-complexity proxy, and candidate identifier.

## Selection and untouched test

The best tuned configuration is fitted on the full training partition. Its
predictions and the Phase 6 candidates use exactly one shared validation
cohort. Champion selection defaults to validation MAE, followed by validation
RMSE, cross-validation stability, simpler model, and identifier. Test targets
are not read by tuning and test metrics never replace the validation winner.

When `refit_selected_model` is enabled, a learned validation winner is rebuilt
on train plus validation before its one final test evaluation. Persistence
needs no fit. The policy is recorded in the experiment summary and model card.
Promotion is authorised from validation selection only; test metrics are
descriptive.

## Experiments, versions, and registry

Generated runs live under `reports/experiments/<run_id>/`. Automatic IDs contain
a UTC timestamp, configuration hash, and `xgb`; `--run-id` supports controlled
tests. IDs reject path separators and existing runs are protected unless
`--overwrite` targets a directory marked as a SolarPulse experiment.

Each run contains:

- `run_config.json`, `environment.json`, and privacy-preserving dataset
  fingerprint;
- manifest snapshot, fold boundaries, candidates, and fold results;
- validation leaderboard and compact validation/test predictions;
- final metrics, selection record, feature importance, summaries, model card,
  and one chart per file.

Dataset, manifest, configuration, model, and generated artifact checksums
support reproducibility without storing complete private input rows. XGBoost
gain and split-count importance describe model use, not causality; SHAP is not
included.

Model versions such as `solarpulse-xgb-20260726-1a2b3c4d5e` describe a training
artifact, not an application release. The local JSON registry supports
candidate, champion, archived, and rejected states. Registration verifies the
artifact path and checksum. Promotion atomically archives the prior champion,
so an objective has at most one champion. Incomplete or invalid artifacts
cannot be promoted. Registry methods list models, show the champion, register,
promote, archive, and verify.

Joblib and pickle can execute code while loading. Load only trusted local
artifacts after verifying checksum, dependency metadata, manifest compatibility,
and ordered features.

## Command-line workflow

```bash
python -m solarpulse_ai.training.advanced \
  --features data/processed/model_features.csv \
  --feature-manifest reports/features/feature_manifest.json \
  --output-dir reports/experiments \
  --artifact-dir artifacts/models \
  --registry artifacts/model_registry.json \
  --search-budget 24 \
  --cv-splits 5 \
  --cv-gap-hours 24 \
  --validation-window-hours 336 \
  --selection-metric mae \
  --random-seed 42 \
  --n-jobs 1 \
  --promote-selected-champion
```

`--config` loads strict JSON. Other switches set an explicit run ID, history
cap, fold and search controls, disable baseline comparison, registry, refit, or
negative clipping, protect/overwrite a run, and choose candidate-only
registration. Success returns zero; validation or training failure returns
non-zero. Logs contain progress and identifiers, never datasets, credentials,
or environment secrets.

## Robustness and limitations

Reports retain overall, site, UTC-hour, daylight, irradiance, cloud-cover, and
target-magnitude context where those inputs exist; small cohorts require
caution. Future runs can extend cohort tables without changing selection.

Historical or reanalysis weather is only a development proxy for weather
forecasts. Production forecasts require values available before prediction
time. Using observed target-time weather can make evaluation optimistic. The
model is not production validated until it is evaluated with genuine archived
forecast weather. Advanced tuning does not remove this limitation.

Operational monitoring, remote registry/storage, deployment, automated
promotion, and production forecast-weather acquisition remain outside Phase 7.
