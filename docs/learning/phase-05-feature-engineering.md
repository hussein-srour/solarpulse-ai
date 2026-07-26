# Phase 5 learning guide: feature engineering

## A. Beginner explanation

A feature is an input that helps a model make a prediction. Raw timestamps,
weather readings, and generation measurements are useful, but their structure
is not always obvious to a model. Feature engineering turns them into clearer
signals: local hour, whether sunlight is present, yesterday's generation at the
same UTC hour, or the average generation available before a forecast was made.
The target (`ac_energy_kwh`) stays unchanged as the answer the model will learn
to predict.

## B. Technical explanation

The pipeline groups predictors into raw and derived weather, site-local
calendar/cyclical fields, exact-time target and weather lags, cutoff-anchored
rolling target statistics, and numerical site metadata. Sine/cosine pairs encode
cycles without an artificial break between the end and beginning of a period.

The prediction horizon is the distance between prediction time and target time.
For the 24-hour objective, the forecast cutoff for target `t` is `t - 24h`.
Exact-time lags join a site at a specific UTC offset; they never mean “the 24th
previous row.” Rolling windows use timestamps and end at the forecast cutoff,
so neither the current target nor newer generation can leak into features.

UTC is retained for unique keys, chronological joins, and split boundaries.
Each site's IANA timezone is used independently for calendar meaning and DST.
Eligibility metadata preserves early rows that lack history and explains why
they are incomplete. The manifest records configuration, column roles, units,
types, missingness, and source lineage. Phase 4 split periods are assigned after
backward-looking features are built, never shuffled, and verified to cover the
dataset without overlap.

Historical observed weather is treated as a forecast proxy during development.
Because it contains the weather that actually occurred, it can make results
look better than forecasts available in production would.

## C. Interview explanation

“I built a leakage-safe feature-engineering pipeline for 24-hour-ahead solar
forecasting. It creates site-local cyclical time features, weather interactions,
exact-time target and weather lags, and rolling generation statistics. All
history features respect the forecast cutoff, are isolated per site, and are
tested to ensure future target values cannot influence training features. It
also produces explicit eligibility metadata, chronological split labels, and a
lineage manifest without imputing or training a model.”

## D. Interview questions and answers

### What is feature engineering?

It transforms validated source fields into consistent, informative predictors
while preserving the target and the prediction-time information boundary.

### Why use cyclical encoding for hour and month?

Numeric values alone make the end and beginning of a cycle look far apart.
Sine/cosine coordinates retain their circular relationship.

### Why is hour 23 close to hour 0?

They are one hour apart on a daily cycle, even though their raw numeric
difference is 23.

### What is a lag feature?

It is a value from a precise earlier timestamp, such as generation at exactly
`t - 24h`.

### Why use exact timestamp lags instead of previous rows?

Missing hours make row position unreliable. Timestamp joins leave the feature
missing rather than silently using the wrong hour.

### What is a rolling feature?

It summarises observations in a time interval, such as the mean or median over
the 24 hours ending at the forecast cutoff.

### What is the forecast horizon?

It is how far ahead the target is predicted; the default is 24 hours.

### What is the forecast cutoff?

It is the newest time whose generation can be known: target time minus forecast
horizon.

### What is target leakage?

It occurs when a predictor uses target or future information unavailable when a
real prediction would be made.

### How did you prevent leakage?

Target lags must be at least the horizon; history uses exact site/time keys;
rolling windows end at the cutoff; the target is excluded from the predictor
contract; and mutation tests prove current/future targets cannot alter past
features.

### Why did you avoid random shuffling?

Random shuffling can put future conditions in training and past conditions in
evaluation. Chronological periods better represent deployment.

### Why were missing lag values not automatically filled?

Filling can invent history or propagate information across the forecast
boundary. Missing context is retained and explained by eligibility metadata.

### Why calculate local-time features while preserving UTC keys?

Local time represents solar and human calendar cycles, while UTC provides
unambiguous joins and chronological boundaries across sites.

### Why can historical weather create optimistic results?

It records what actually happened, whereas production only has an imperfect
forecast available before the target hour.

### What is feature lineage?

It records where a feature dataset came from, the configuration and horizon
used, column roles, units, types, missingness, and limitations.

### How did you test the pipeline?

Programmatically generated data covers configuration, timezones/DST, multi-site
isolation, gaps, exact lags, cutoff windows, weather ratios, eligibility,
splits, reports, CLI behavior, and mutation-based leakage checks.

