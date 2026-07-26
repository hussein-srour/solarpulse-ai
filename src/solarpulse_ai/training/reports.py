"""Training report serialization and concise Markdown summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write deterministic, human-readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Write a generated report CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


def render_summary(summary: dict[str, Any]) -> str:
    """Render an evidence-based training report."""
    validation_coverage = summary["validation_comparison_cohort"]["comparison_coverage_pct"]
    test_coverage = summary["test_comparison_cohort"]["comparison_coverage_pct"]
    clipping = "clipped to zero" if summary["clip_negative_predictions"] else "not clipped"
    return f"""# SolarPulse AI baseline training summary

- Run: `{summary["run_name"]}`
- Feature dataset SHA-256: `{summary["feature_file_sha256"]}`
- Feature manifest SHA-256: `{summary["feature_manifest_sha256"]}`
- Forecast horizon: {summary["forecast_horizon_hours"]} hours
- Predictors: {summary["predictor_count"]}
- Eligible/excluded rows: {summary["eligibility"]["eligible_row_count"]} /
  {summary["eligibility"]["excluded_row_count"]}
- Selected by validation MAE: `{summary["selected_model"]}`
- Validation common-cohort coverage: {validation_coverage:.2f}%
- Test common-cohort coverage: {test_coverage:.2f}%

## Method

Train, validation, and test ranges are chronological and non-overlapping. Candidate
preprocessing is fitted on training rows only. Median imputation and missing indicators are
used for numeric data; Ridge also standardises. Declared categorical inputs use training-fitted
imputation and unknown-safe one-hot encoding. Tree models are not scaled. Model selection uses
the shared validation cohort and MAE only; test data is untouched until the winner is fixed.
Negative predictions are counted and {clipping}.

## Results

Validation ranking: `{summary["model_ranking"]}`.

Final test selected-model metrics: `{summary["final_test_metrics"]["selected_model"]}`.

Persistence comparison: `{summary["final_test_metrics"]["persistence"]}`.
Daylight metrics: `{summary["daylight_metrics"]}`. Per-site metrics are in the generated CSVs.

## Limitations and next steps

These are baseline results from the supplied dataset, not a claim of production accuracy.
Historical or reanalysis weather is a proxy for forecasts available at prediction time and may
produce optimistic estimates. Test performance does not guarantee future performance.
Permutation importance is validation-only and is not proof of causality. Next steps are
forecast-time weather validation, monitoring design, and a separately governed advanced-model
phase.
"""
