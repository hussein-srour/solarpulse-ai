# Feature engineering

SolarPulse AI builds a tabular dataset for predicting `ac_energy_kwh` at target
timestamp `t`. The default horizon is 24 hours. Target-time weather is allowed
because it represents the forecast that production must supply for that hour;
generation history is limited to information available at or before `t - 24h`.

Historical/reanalysis weather is only a development proxy. Production must use
weather forecasts that were actually available at prediction time. Evaluation
with observed historical weather can be optimistic.

## Prediction-time information boundary

For horizon `h`, no generation newer than `t - h` can influence a feature for
`t`. Exact lags join on `(site_id, timestamp - lag)` rather than row position.
Rolling windows are time-aware, site-isolated, and use the right-closed,
left-open interval `(t - h - window, t - h]`. The current target is retained as
the label but is never a predictor.

## Feature groups

- Raw weather: GHI, ambient temperature, cloud cover, relative humidity, wind
  speed, and whichever canonical optional weather columns are present.
- Derived weather: daylight, GHI per installed capacity, temperature/GHI,
  cloud/GHI and humidity/temperature interactions, and source-dependent diffuse,
  direct/global, precipitation, availability, and module/ambient features.
- Local time: calendar fields plus sine/cosine encodings for hour, weekday,
  day-of-year, and month.
- Exact history: configured target lags (24, 48, and 168 hours by default) and
  configured weather lags (24 and 48 hours by default).
- Rolling history: mean, sample standard deviation, minimum, maximum, median,
  and observation count for 3, 6, 24, 72, and 168-hour windows by default.
- Site metadata: installed capacity, panel tilt/azimuth, latitude, longitude,
  and lagged generation per capacity. Timezone remains configuration, not a
  string predictor.

UTC remains the canonical timestamp. Calendar features are calculated per site
with its validated IANA timezone, including daylight-saving rules. Cyclical
encodings preserve adjacency such as hour 23 being close to hour 0. Ratio
features are missing when GHI is at or below `irradiance_epsilon`; undefined
ratios are never replaced with zero. Derived relationships aid prediction and
do not establish physical causality.

## Eligibility and missing values

All validated source rows are preserved by default. `feature_eligible`,
`feature_missing_count`, and `feature_missing_reasons` distinguish unavailable
historical context from missing required source weather. Absent optional source
columns are reported but not failures. `require_complete_history` additionally
requires each rolling window's nominal hourly count. Only `--only-eligible`
filters rows, and the excluded count is reported.

The pipeline never fills, interpolates, clips, scales, standardises, encodes, or
imputes. Fitted transformations belong in model training and must be fitted on
training data only.

## Split plans, lineage, and output

A Phase 4 `split_plan.json` is validated for timezone-aware, chronological,
non-overlapping periods that cover the input. Rows receive `train`,
`validation`, or `test`; they are never shuffled. `--write-splits` additionally
writes those three feature CSVs.

The feature contract explicitly identifies keys, target, predictors, metadata,
categorical columns, and numerical columns. The JSON manifest records source
lineage, configuration, column roles, units, descriptions, dtypes, missingness,
eligibility, split counts, timestamp range, and limitations without embedding
source records. Quality JSON, Markdown, and eligibility CSV reports describe
availability, constants, infinities, broad range warnings, and leakage checks.
Warnings are not confirmed equipment faults.

Generated outputs are ignored by Git:

- the requested feature CSV and optional split feature CSVs;
- `reports/features/feature_manifest.json`;
- `reports/features/feature_quality.json`;
- `reports/features/feature_quality.md`;
- `reports/features/feature_eligibility.csv`.

## CLI

```bash
python -m solarpulse_ai.features.build \
  --input data/processed/training_dataset.csv \
  --site-config config/example_site.json \
  --split-plan reports/eda/split_plan.json \
  --output data/processed/model_features.csv \
  --report-dir reports/features \
  --forecast-horizon-hours 24
```

Repeat `--site-config` for multiple sites. Comma-separated options configure
target lags, rolling windows, and weather lags. Switches can disable raw weather,
weather lags, target history, or site metadata. `config/example_site.json` is
illustrative Dar es Salaam metadata only, not a production or confidential site.

The future training interface should consume the feature contract, select only
predictors, fit missing-value and scaling policies on the training split, and
evaluate chronologically. Phase 5 intentionally trains no model.

