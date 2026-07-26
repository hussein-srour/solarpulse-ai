# Phase 6 learning and interview guide

## A. Beginner explanation

Model training shows an algorithm historical examples so it can learn a
repeatable relationship. `X` is information known when forecasting—weather,
calendar values, site properties, and safe historical lags. `y` is the answer:
hourly AC energy in kWh.

Simple baselines matter. If a complex method cannot beat “use the value from 24
hours ago” or “use the training mean,” its complexity has not earned its cost.
Training data teaches. Validation data chooses. Test data is a sealed final exam
used once after the choice; using it to choose makes its score less trustworthy.

## B. Technical explanation

The ordered feature matrix contains only Phase 5 manifest predictors; the target
vector is `ac_energy_kwh`. Chronological splits prevent future observations from
teaching the past. Eligibility filters unusable history explicitly.

Preprocessing stays inside each scikit-learn pipeline. Median imputation,
missing indicators, Ridge scaling, categorical imputation, and vocabularies fit
on training only. Validation/test call `transform`, never `fit`, so future
mutations cannot change medians, means, scales, or categories.

Persistence reads the exact Phase 5 lag. `DummyRegressor` predicts the training
mean. Ridge is regularised linear regression. A random forest averages varied
decision trees. Histogram gradient boosting adds binned small trees
sequentially to correct errors. Parameters are fixed and bounded, with no
tuning.

MAE is average absolute error in kWh. RMSE squares errors first, emphasizing
large misses. R² compares squared error with a mean reference and can be
negative. WAPE divides total absolute error by total actual energy and is null
when that total is zero. MAPE is unsuitable as primary because night generation
is zero or tiny. Residual is actual minus prediction; bias here is predicted
minus actual.

Candidates share one validation cohort with finite persistence and model
outputs. Validation MAE selects; RMSE, fixed simplicity, and identifier only
break ties. Test targets cannot select. A learned winner may refit on train plus
validation, then receives one shared-cohort test comparison with persistence.
If persistence wins, an exact-lag specification is saved honestly.

Joblib serialises preprocessing and estimator together. SHA-256 checksums,
versions, Git commit, seed, parameters, predictor order, boundaries, and UTC
time support reproducibility. Reloaded predictions must match pre-save output.
Residuals describe misses. Validation permutation importance shuffles one
original predictor and observes score change; correlation and confounding mean
it is not causal.

Historical observed/reanalysis weather knows what happened, while production
only knows a forecast available at issue time. This can make offline estimates
optimistic even with leakage-safe target history.

## C. Interview explanation

> I built a reproducible baseline-training pipeline for 24-hour-ahead solar
> generation forecasting. I compared persistence, mean, linear, random-forest,
> and histogram-gradient-boosting models using a shared chronological validation
> cohort. All preprocessing was fitted only on training data, the model was
> selected by validation MAE, and the untouched test period was evaluated once
> after selection. I persisted the complete preprocessing and model pipeline—or
> an exact-lag specification if persistence won—with a training manifest, model
> card, checksums, residual reports, validation-only permutation importance, and
> overall, daylight, and per-site metrics.

## D. Interview questions and answers

### What are features and targets?

Features are inputs known at forecast time. The target is hourly AC energy.

### What does fitting a model mean?

Estimating model parameters from training examples under a defined learning
rule.

### Why compare against persistence?

Yesterday’s same-hour solar output is a credible, cheap reference that learned
models should justify beating.

### What is DummyRegressor?

A baseline that ignores `X`; here it always predicts the training-target mean.

### What is Ridge regression?

Linear regression with an L2 penalty that stabilises correlated coefficients.

### How does a random forest work?

It fits many varied decision trees and averages them to reduce instability.

### What is gradient boosting?

It adds small trees sequentially, each correcting remaining ensemble errors.

### Why use a preprocessing pipeline?

It binds transformations to inference and prevents training/serving drift.

### Why fit imputation only on training data?

A future-derived median transfers validation/test distribution information into
training and biases evaluation.

### What is MAE?

Average absolute actual-versus-predicted difference, in kWh.

### What is RMSE?

The square root of average squared error, also in kWh.

### Why is RMSE more sensitive to large errors?

Squaring gives large misses disproportionate weight.

### What is R²?

Squared-error performance relative to predicting the actual mean: one is
perfect, zero matches it, and negative is worse.

### What is WAPE?

Total absolute error divided by total actual energy; null when the denominator
is zero.

### Why avoid MAPE for solar generation?

Zero/tiny night actuals make percentage errors undefined or misleadingly huge.

### What is a residual?

Actual minus prediction; positive means underprediction.

### What is overfitting?

Learning training quirks that do not generalise to later periods.

### Why select models using validation data?

It supports one governed choice without rewarding training memorisation.

### Why keep test data untouched?

It preserves an independent final estimate; repeated choices turn it into
validation.

### What is data leakage?

Using information unavailable at prediction time or future split information
during fitting/selection.

### Why not randomly shuffle time-series data?

Later conditions could teach models evaluated on earlier timestamps.

### Why use the same comparison cohort?

It prevents row-difficulty differences from masquerading as model differences.

### What happens when persistence beats learned models?

The run reports it and saves the exact-lag specification without claiming a
learned estimator won.

### Why save preprocessing together with the model?

Inference needs the exact learned medians, scales, vocabulary, and ordering.

### What is reproducibility?

Repeating governed code, inputs, configuration, seeds, and versions with
equivalent predictions.

### What is permutation importance?

The validation-score change when one original predictor is shuffled.

### Why does importance not prove causation?

Correlation, confounding, proxies, sampling, and model structure affect it.

### How was the training system tested?

Small synthetic tests cover contracts, eligibility, chronology, leakage,
training-only preprocessing, unknown categories, deterministic models, exact
persistence, formulas, cohorts, ties, clipping, test-isolated selection,
refitting, save/reload equivalence, checksums, reports, charts, CLI outcomes,
and earlier regressions without network access. Synthetic metrics are not plant
performance evidence.
