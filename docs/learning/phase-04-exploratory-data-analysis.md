# Phase 4: exploratory data analysis

## A. Simple explanation

Phase 4 is a health check for the dataset before teaching a model. The command
opens a private hourly solar-and-weather CSV, confirms that it follows the
project's rules, and describes what is actually present. It counts sites and
hours, finds time gaps and missing optional values, summarises generation and
weather, creates charts, and marks unusual records for human review.

Nothing is repaired automatically. An unusual row might have a legitimate
operational explanation, so the system keeps the original evidence intact. It
also suggests time-ordered training, validation, and testing periods, but does
not train a model.

## B. Technical explanation

`solarpulse_ai.analysis` separates concerns into typed modules:

- `profile` computes coverage, completeness, schema, missingness, memory, and
  complete/partial day metadata.
- `statistics` computes target and available-weather descriptive measures plus
  hourly, daily, and conditional monthly aggregates.
- `temporal` works in UTC and reports calendar patterns, GHI/generation hourly
  relationships, continuity, and exact gaps.
- `correlation` computes complete-pair Pearson correlations and safely reports
  constants or insufficient pairs.
- `quality` creates non-destructive flags using named irradiance, near-zero,
  consecutive-run, Tukey-IQR, and history thresholds.
- `readiness` applies documented `ready`, `ready_with_warnings`, and
  `not_ready` screening rules.
- `splits` assigns globally ordered unique UTC timestamps to non-overlapping
  70/15/15 periods without shuffling.
- `charts` writes nine matplotlib `Agg` PNGs and closes each figure.
- `reporting` converts pandas values to standard JSON and writes Markdown/CSV.
- `pipeline` and `eda` provide the reusable orchestration and CLI.

The Phase 2 validator remains the authority for required columns, timezone
awareness, numeric finiteness/ranges, and unique site/timestamp keys. Reports
are local ignored artifacts. Tests construct data in temporary directories and
cover calculations, indicators, charts, figure cleanup, serialisation, splits,
readiness, CLI success/failure, and previous phases without internet access.

## C. Interview explanation

“In Phase 4 I built a reusable EDA and model-readiness pipeline around our
canonical hourly generation-and-weather dataset. I reused strict schema
validation, kept all analysis in UTC, computed site and temporal profiles,
descriptive statistics and safe Pearson correlations, and produced headless
matplotlib reports. I added transparent, non-destructive data-quality
indicators rather than auto-correcting operational records. I also designed
global chronological train/validation/test splits to prevent temporal leakage,
documented every readiness rule, and tested the workflow with programmatically
generated datasets so no company data or live API was required.”

## D. Interview questions

### What is exploratory data analysis?

EDA is the structured inspection of a dataset's shape, distributions,
relationships, time coverage, missingness, and unusual observations before
choosing or training a model.

### Why perform EDA before training?

It exposes data limitations and invalid assumptions early. A model can run on
poorly understood data and still produce misleading metrics, so EDA makes the
later experiment defensible.

### Why should time-series data not be randomly shuffled?

Random shuffling can put future conditions into training and earlier
conditions into testing. That does not represent forecasting from the past
into the future.

### What is data leakage?

Leakage occurs when training receives information that would not be available
at prediction time. Temporal leakage is a common form where future
observations influence an earlier model or feature.

### Why use training, validation, and test sets?

Training fits candidate models, validation supports model and configuration
choices, and the held-back test period gives a final less-biased estimate.

### What is correlation?

Correlation measures the direction and strength of an association. Phase 4
uses Pearson correlation for linear association between generation and each
available numerical weather field.

### Why does correlation not prove causation?

Two values can move together because of a shared driver, time pattern, site
difference, or confounder. Association alone does not establish a causal
mechanism.

### How were missing timestamps detected?

For each site, the system creates the expected hourly UTC range between its
first and last observation and compares that range with actual timestamps.
Every absent hour is reported without inserting a row.

### How were suspicious records handled?

They were retained and written as row-level data-quality indicators with a
named rule and explanation. Dataset-level indicators cover constants,
coverage, and differing site ranges.

### Why were records not automatically corrected?

Without source-system and operational context, filling, clipping, or deleting
could destroy real evidence and introduce bias. Correction needs an explicit,
reviewed policy in a later phase.

### How was the analysis tested?

Tests generate temporary single-site and multi-site canonical datasets and
exercise gaps, partial days, optional fields, missing values, constants,
irradiance/generation indicators, zero runs, abrupt changes, outliers,
statistics, correlations, all charts, figure closure, JSON/Markdown output,
split ordering and overlap, readiness categories, and CLI exit codes. The
complete offline suite also reruns all Phase 1–3 tests.
