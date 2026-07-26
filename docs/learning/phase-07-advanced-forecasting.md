# Phase 7 learning guide: advanced forecasting

## A. Beginner explanation

XGBoost builds many small decision trees. Each new tree concentrates on errors
left by the trees before it, so the group can learn nonlinear relationships
such as changing solar output across weather and time. More flexible does not
automatically mean better: a persistence forecast or simple Ridge model may be
more accurate, stable, and easier to operate. That is why every advanced
candidate is compared with the Phase 6 baselines.

Model tuning means choosing settings such as the number and depth of trees or
how quickly each tree changes the forecast. Solar observations have an order,
so ordinary random cross-validation would let later conditions help predict
earlier ones. Rolling-origin validation trains on the past, leaves a safety
gap, and validates on the next period. The training history grows as time moves
forward.

## B. Technical explanation

Gradient-boosted decision trees minimise loss stage by stage. Learning rate
shrinks each tree's contribution; smaller values commonly require more trees.
Maximum depth and minimum child weight control tree capacity. Row
`subsample` and feature `colsample_bytree` can reduce variance. `reg_alpha`,
`reg_lambda`, and `gamma` penalise complexity.

Phase 7 deterministically samples a finite bounded parameter space with a fixed
seed. Every candidate runs over all expanding-window folds. A 24-hour gap
protects the 24-hour forecast horizon. Candidate ranking uses cross-validation
mean, variability, a complexity proxy, and a stable ID. Variability matters
because a low mean from erratic folds may not generalise.

The tuner receives training-labelled rows only. After tuning, the chosen
XGBoost setup is fitted on all training rows and compared with persistence,
Ridge, random forest, and histogram gradient boosting on one shared validation
cohort. Validation selects the champion. The untouched test partition is
evaluated only after selection and cannot replace the winner. If configured,
the learned winner is then refitted on train plus validation before test; the
same policy is recorded.

Experiment tracking stores configuration, environment versions, checksums,
folds, candidates, predictions, metrics, charts, selection, limitations, and
artifact checksums locally. Model versioning names a trained artifact rather
than an application release. The champion/challenger registry verifies
artifacts, permits one champion per objective, archives the prior champion, and
requires validation-governed promotion.

Feature gain and split count show how XGBoost used transformed predictors, but
importance is not causality. Historical/reanalysis weather also remains a
development proxy: observed target-time weather may be unavailable to a real
day-ahead forecast and can make results optimistic.

## C. Interview explanation

“I extended the baseline forecasting system with a tuned XGBoost model. I used
expanding-window rolling-origin cross-validation with a 24-hour temporal gap,
and optimised hyperparameters using training data only. The tuned candidate was
then compared with persistence and simpler baseline models on a shared
validation cohort. The test partition remained untouched until final
evaluation. I also implemented reproducible experiment tracking, artifact
checksums, model versioning and a local champion/challenger registry.”

## D. Interview questions and answers

### What is XGBoost?

An efficient implementation of regularised gradient-boosted decision trees
that can model nonlinearities and interactions in tabular data.

### How does gradient boosting work?

Trees are added sequentially. Each tree reduces errors represented by the
current loss gradients, and the learning rate controls its contribution.

### Why compare an advanced model with persistence?

Persistence is cheap, interpretable, and often strong for solar time series.
An advanced model is useful only if it improves the governed validation result.

### What is hyperparameter tuning?

It evaluates model settings not learned directly during fitting, such as tree
depth, learning rate, regularisation, and sampling ratios.

### What is rolling-origin cross-validation?

It repeatedly trains on an expanding past window and validates on a later fixed
window, reproducing the direction of real forecasting.

### Why not use ordinary random cross-validation?

Random folds mix past and future, breaking temporal dependence and creating
optimistic leakage.

### Why use a temporal gap?

The gap separates training targets from the validation prediction boundary. A
24-hour gap protects this 24-hour-ahead objective.

### Why tune using training data only?

Using validation or test targets would adapt hyperparameters to cohorts needed
for unbiased selection and reporting.

### What is the role of the validation partition?

It fairly compares the tuned advanced candidate and fixed baselines and selects
the champion.

### Why must the test set remain untouched?

It is the final estimate after all tuning and selection decisions. Looking
early turns it into another validation set.

### What is overfitting?

Learning noise or dataset-specific detail that reduces training error but harms
future performance.

### What do learning rate and tree depth control?

Learning rate controls the size of each boosting step; depth controls how
complex each tree's interactions can be.

### What are subsample and colsample_bytree?

They select fractions of rows and features per tree, which can reduce
correlation and variance.

### What is regularisation?

Penalties such as alpha, lambda, and gamma discourage unnecessary model
complexity.

### Why examine cross-validation variability?

A stable candidate is less dependent on one lucky time window than an erratic
candidate with the same mean.

### How was the champion selected?

Validation MAE, validation RMSE, cross-validation stability, simpler model, and
identifier—in that order. Test metrics never participate.

### What is experiment tracking?

A reproducible local record connecting inputs by checksum, configuration,
environment, folds, candidates, outputs, decisions, warnings, and artifacts.

### What is model versioning?

An immutable identity for one trained artifact and its data/configuration
lineage, separate from software release versions.

### What is a champion/challenger registry?

A governed catalogue where candidates can challenge the active model and a
validated promotion archives the previous champion.

### Why does feature importance not prove causality?

It measures how a fitted model uses correlations, not what physically causes
solar production to change.

### Why might historical weather produce optimistic results?

Observed target-time weather can be more accurate than forecasts that were
actually available 24 hours earlier.

### How was reproducibility ensured?

Fixed seeds and threads, sorted UTC rows, deterministic candidate IDs/order,
bounded parameters, dependency versions, configuration snapshots, and
checksums.

### How was leakage tested?

Tests verify chronological non-overlap, grouped timestamps, train-only fold
fitting, forbidden predictors, and unchanged tuning/selection after test-target
mutation.
