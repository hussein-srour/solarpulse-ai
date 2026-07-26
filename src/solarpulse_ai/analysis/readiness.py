"""Transparent dataset model-readiness assessment."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from solarpulse_ai.analysis.config import AnalysisThresholds

type ReadinessCategory = Literal["ready", "ready_with_warnings", "not_ready"]


def assess_readiness(
    dataframe: pd.DataFrame,
    missing_timestamp_count: int,
    thresholds: AnalysisThresholds,
) -> dict[str, Any]:
    """Apply documented readiness rules without claiming production readiness."""
    target_varies = dataframe["ac_energy_kwh"].nunique() > 1
    multiple_months = dataframe["timestamp"].dt.strftime("%Y-%m").nunique() >= 2
    missingness = bool(dataframe.isna().any().any())
    continuity = missing_timestamp_count == 0
    daylight = dataframe["ghi_w_m2"] >= thresholds.high_irradiance_w_m2
    complete_daylight = bool(daylight.any() and dataframe.loc[daylight].notna().all(axis=1).any())
    multiple_sites = dataframe["site_id"].nunique() > 1

    limitations: list[str] = []
    if not continuity:
        limitations.append("Resolve missing hourly timestamps or document an explicit gap policy.")
    if missingness:
        limitations.append("Review optional-field missingness before feature selection.")
    if not target_varies:
        limitations.append("The target lacks variation required for supervised learning.")
    if not multiple_months:
        limitations.append("Collect multiple months to represent broader temporal conditions.")
    if not complete_daylight:
        limitations.append("No fully populated high-irradiance observation is available.")
    if not multiple_sites:
        limitations.append("Single-site evidence does not establish cross-site generalisation.")

    if not target_varies or not complete_daylight:
        category: ReadinessCategory = "not_ready"
    elif limitations:
        category = "ready_with_warnings"
    else:
        category = "ready"
    return {
        "category": category,
        "dataset_valid": True,
        "chronological_continuity_acceptable": continuity,
        "missingness_present": missingness,
        "target_has_sufficient_variation": target_varies,
        "history_covers_multiple_months": multiple_months,
        "complete_daylight_records_present": complete_daylight,
        "site_scope": "multiple_sites" if multiple_sites else "single_site",
        "major_limitations_before_training": limitations,
        "production_readiness_note": (
            "Schema validity and this screening do not by themselves establish "
            "production readiness."
        ),
        "rules": {
            "ready": (
                "Valid, continuous, no missingness, varied target, multiple months, "
                "complete daylight records, and multiple sites."
            ),
            "ready_with_warnings": (
                "Target varies and complete daylight records exist, but one or more "
                "non-blocking limitations remain."
            ),
            "not_ready": ("Target is constant or no complete high-irradiance record exists."),
        },
    }
