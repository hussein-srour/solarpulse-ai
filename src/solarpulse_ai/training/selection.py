"""Common-cohort construction and deterministic validation selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

COMPLEXITY_ORDER = {
    "persistence": 0,
    "dummy_mean": 1,
    "ridge": 2,
    "histogram_gradient_boosting": 3,
    "random_forest": 4,
}


def common_cohort(
    rows: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    persistence_column: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Use exactly the rows on which every candidate and persistence are available."""
    mask = np.isfinite(pd.to_numeric(rows[persistence_column], errors="coerce").to_numpy(float))
    reasons: dict[str, int] = {"persistence_unavailable": int((~mask).sum())}
    for model_id, values in predictions.items():
        finite = np.isfinite(values)
        reasons[f"{model_id}_prediction_unavailable"] = int((~finite).sum())
        mask &= finite
    cohort = rows.loc[mask].reset_index(drop=True)
    aligned = {model_id: values[mask] for model_id, values in predictions.items()}
    details = {
        "full_row_count": len(rows),
        "common_comparison_row_count": len(cohort),
        "comparison_coverage_pct": 100.0 * len(cohort) / len(rows) if len(rows) else 0.0,
        "excluded_comparison_row_count": int((~mask).sum()),
        "excluded_reasons": {key: value for key, value in reasons.items() if value},
    }
    if cohort.empty:
        raise ValueError("common comparison cohort is empty")
    return cohort, aligned, details


def rank_models(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank solely by validation MAE, with documented deterministic tie-breaks."""
    ranked = sorted(
        records,
        key=lambda item: (
            float(item["mae_kwh"]),
            float(item["rmse_kwh"]),
            COMPLEXITY_ORDER[str(item["model_identifier"])],
            str(item["model_identifier"]),
        ),
    )
    return [{"rank": index, **record} for index, record in enumerate(ranked, start=1)]
