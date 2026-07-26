# Exploratory analysis and dataset readiness

## Handoff to feature engineering

The optional Phase 4 `split_plan.json` supplies chronological UTC boundaries to
Phase 5. Feature generation validates that training, validation, and testing
periods are ordered, non-overlapping, and cover every input row, then emits
`train`, `validation`, and `test` labels without shuffling. Backward-looking
features may be calculated before assignment because exact lags and rolling
windows strictly respect the forecast cutoff.

Phase 6 consumes these periods: candidates fit on train, validation MAE selects
on a shared cohort, and the untouched test period is evaluated once after the
winner is fixed. Test metrics never replace the validation winner.

## Why this phase comes before training

Exploratory data analysis (EDA) checks whether the canonical hourly dataset is
understood well enough to design a defensible modelling experiment. Schema
validation is necessary but cannot reveal every gap, constant sensor,
suspicious generation/irradiance combination, narrow date range, or leakage
risk. Phase 4 therefore profiles and reports the evidence without training a
model or changing any record.

Passing this screening does not prove that a dataset, installation, or future
model is production-ready. Weather may be external reanalysis rather than a
site measurement, and generation results are not described as plant
performance unless the input provenance establishes measured operational data.

## Private local workflow

Keep the canonical CSV beneath an ignored data directory and run:

```bash
python -m solarpulse_ai.analysis.eda \
  --input data/processed/training_dataset.csv \
  --output-dir reports/eda
```

The command logs paths, record count, and readiness category, but never logs
complete rows, environment-variable values, or credentials. It reads the CSV
with the Phase 2 reader, validates it with the canonical validator, normalises
timezone-aware instants to UTC, and sorts a validated copy by site and
timestamp. Missing required fields, invalid numbers, range violations,
timezone-naive timestamps, and duplicate site/timestamp keys fail with a
non-zero exit. No correction, filling, clipping, interpolation, or deletion is
performed.

Useful options are:

```text
--training-proportion
--validation-proportion
--testing-proportion
--low-irradiance-threshold
--high-irradiance-threshold
--near-zero-generation-threshold
--consecutive-zero-duration
--iqr-outlier-multiplier
--minimum-history-days
--write-splits
```

## Report outputs

All outputs are generated locally and ignored by Git:

- `dataset_profile.json` contains dataset identity, generation time, profile,
  target/weather statistics, temporal summaries, correlations, diagnostic
  summary, readiness assessment, and split recommendation. Values are converted
  to standards-compliant JSON scalars.
- `dataset_report.md` is the human-readable report with limitations and next
  steps.
- `split_plan.json` gives the exact UTC beginning, ending, record count, unique
  timestamp count, and per-site count for each split.
- `data_quality_flags.csv` contains row-level or dataset-level indicators but no
  complete confidential input record.
- `correlations.csv` reports Pearson results, paired counts, availability, and
  unavailable reasons.
- `charts/*.png` contains deterministic, one-chart-per-file headless figures.
- `training_split.csv`, `validation_split.csv`, and `testing_split.csv` are
  created only with `--write-splits`.

The profile includes record/site counts, site identifiers, UTC range and
duration, expected and actual hourly records per site, completeness, missing
hours, duplicates, missing values, types, columns, absent optional fields,
memory use, and complete/partial site-days.

Target summaries include count, minimum, maximum, mean, median, sample standard
deviation, 5th/25th/50th/75th/95th percentiles, total energy, zero percentage,
hourly profile, daily totals, and monthly totals when at least two months are
represented. Available weather fields receive the same descriptive measures.

## Charts

- `actual_generation_over_time.png`: hourly energy by site in UTC.
- `daily_energy_totals.png`: daily energy totals by site.
- `hourly_generation_profile.png`: mean generation by UTC hour.
- `target_distribution.png`: hourly target histogram.
- `generation_vs_ghi.png`: generation/GHI scatter by site.
- `correlation_heatmap.png`: Pearson matrix for varying available columns.
- `missing_values_by_field.png`: missing count for every present field.
- `data_availability_by_site.png`: actual versus expected hourly records.
- `weather_trends.png`: standardised daily means for available weather fields.

Figures use matplotlib with the `Agg` backend, close after saving, and require
no display. A missing or constant optional field is skipped or clearly marked
unavailable; no field is fabricated. Standardising a weather trend is only a
visual scale transformation and does not modify input or reported values.

## Data-quality indicators and thresholds

Indicators are evidence for review, not confirmed equipment faults. Defaults:

| Threshold | Default | Use |
| --- | ---: | --- |
| Low irradiance | 20 W/m² | Positive generation below this GHI is flagged |
| High irradiance | 500 W/m² | Near-zero generation at/above this GHI is flagged |
| Near-zero generation | 0.01 kWh | Defines near-zero observations and runs |
| Consecutive zero duration | 6 hours | Minimum adjacent UTC run length |
| IQR multiplier | 1.5 | Tukey fences for extremes and abrupt changes |
| Minimum history | 60 days | Shorter per-site coverage is flagged |

Additional indicators cover missing hourly timestamps, partially missing
optional weather values, constant numeric sensor columns, different site date
ranges, and insufficient coverage. Extreme values fall outside
`Q1 - multiplier × IQR` or `Q3 + multiplier × IQR`. Abrupt generation changes
use the same upper Tukey fence on absolute adjacent-hour changes within each
site. A time gap breaks a consecutive run and an abrupt-change comparison.

These rules are deliberately transparent and configurable. A flag can reflect
time alignment, reanalysis limitations, curtailment, maintenance, sensor
behaviour, site design, or another context; it is not an automatic diagnosis.

## Temporal and correlation analysis

Temporal summaries cover generation by UTC hour, weekday, month when at least
two months exist, daily trend, hourly mean GHI/generation, continuity, and gap
lengths. Local-time operational interpretation may be added later only after a
site's configured IANA timezone is available.

Correlation uses the Pearson product-moment coefficient on complete target and
weather pairs. Constant columns and fewer than two pairs are reported as
unavailable. Pearson correlation describes linear association. It does not
prove that a weather variable caused generation to change, and it can be
affected by daylight cycles, seasonality, site differences, missingness, and
confounding conditions.

## Readiness rules

Every validated run reports:

- schema validity;
- hourly continuity;
- any present-field missingness;
- target variation;
- coverage of at least two calendar months;
- at least one fully populated record at/above the high-irradiance threshold;
- single-site or multi-site scope;
- limitations that should be resolved before training.

Categories are:

- `ready`: continuous, no missingness, varied target, multiple months, complete
  daylight evidence, and multiple sites.
- `ready_with_warnings`: target variation and complete daylight evidence exist,
  but at least one other limitation remains.
- `not_ready`: the target is constant or no complete high-irradiance record
  exists.

These are experiment-readiness screening categories, not production approval.

## Chronological split strategy

Defaults are 70% training, 15% validation, and 15% testing. Proportions must be
positive and total exactly 1 within floating-point tolerance. Globally sorted
unique UTC timestamps are cut in sequence; every record at a timestamp receives
the same split. This ensures:

- no random shuffle;
- training ends before validation begins;
- validation ends before testing begins;
- no overlapping period;
- exact boundaries and per-site counts are reported;
- any empty period is rejected.

Future observations therefore cannot leak into an earlier training period.
Random time-series splitting would inflate evaluation by giving training access
to information from the future. Separate CSVs are opt-in because the split plan
is sufficient for Phase 4 and no model is trained.
