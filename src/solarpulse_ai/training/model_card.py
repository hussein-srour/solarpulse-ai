"""Professional, record-free model-card generation."""

from __future__ import annotations

from typing import Any


def render_model_card(manifest: dict[str, Any], model_checksum: str) -> str:
    """Render facts from an actual run without inventing performance."""
    selected = manifest["selected_model"]
    persistence_note = (
        "Persistence won; no learned estimator outperformed it on validation MAE."
        if selected == "persistence"
        else "The selected model is a fixed baseline, not a final production model."
    )
    return f"""# SolarPulse AI model card

## Purpose and version

This artifact forecasts hourly `ac_energy_kwh` at a {manifest["forecast_horizon_hours"]}-hour
horizon for configured solar sites. Selected model: `{selected}`. Project version:
`{manifest["project_version"]}`. Artifact SHA-256: `{model_checksum}`.

## Intended use

Use for reproducible baseline research, offline comparison, and pipeline validation. Do not use
as an autonomous operational-control, safety, financial, contractual, or maintenance decision
system. {persistence_note}

## Data and features

Training expects the validated Phase 5 feature contract. Predictor groups are numerical,
boolean, and explicitly declared categorical fields; key, target, eligibility, split, report,
and row-identity metadata are excluded. Rows must be eligible. Splits are chronological:
train precedes validation, which precedes the untouched test period.

Weather in development data may be historical observations or reanalysis. That is only a proxy
for forecasts available at prediction time and can make results optimistic. Sites, seasons,
weather regimes, outages, curtailment, sensor faults, and cleaning states may be
under-represented.

## Training and selection

Candidates: {", ".join(manifest["candidate_models"])}. Imputation, scaling, and category
vocabularies are fitted only on training rows inside the saved pipeline. All candidates share
one validation cohort. Selection uses validation MAE only, then RMSE, fixed simplicity order,
and identifier. The test split is evaluated once after selection and never replaces the winner.

Validation result: `{manifest["validation_metrics"][selected]}`.
Final test result: `{manifest["final_test_metrics"]["selected_model"]}`.

## Risks, representativeness, and monitoring

Test performance does not guarantee future performance. Metrics are not evidence of fairness
across unobserved sites or conditions. Monitor input-contract drift, missingness, coverage,
residual bias, daylight/site error, weather provenance, and data latency. Retrain only with a
new chronological validation/test design and versioned evidence. Permutation importance is
validation-only and does not prove causality.
"""
