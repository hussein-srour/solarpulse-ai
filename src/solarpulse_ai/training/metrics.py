"""Forecast evaluation metrics with explicit solar-energy units."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    full_row_count: int | None = None,
    capacity_kwp: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Calculate aggregate regression metrics; WAPE is null for zero actual energy."""
    y = np.asarray(actual, dtype=float)
    y_hat = np.asarray(predicted, dtype=float)
    count = len(y)
    total = float(y.sum())
    result: dict[str, float | int | None] = {
        "mae_kwh": float(mean_absolute_error(y, y_hat)),
        "rmse_kwh": float(mean_squared_error(y, y_hat) ** 0.5),
        "median_absolute_error_kwh": float(median_absolute_error(y, y_hat)),
        "r2": float(r2_score(y, y_hat)) if count >= 2 else None,
        "mean_bias_error_kwh": float(np.mean(y_hat - y)),
        "wape": float(np.abs(y - y_hat).sum() / total) if total > 0 else None,
        "prediction_count": count,
        "actual_total_energy_kwh": total,
        "predicted_total_energy_kwh": float(y_hat.sum()),
        "prediction_coverage_pct": 100.0,
    }
    if full_row_count is not None and full_row_count > 0:
        result["prediction_coverage_pct"] = 100.0 * count / full_row_count
    if capacity_kwp is not None:
        capacity = np.asarray(capacity_kwp, dtype=float)
        valid = np.isfinite(capacity) & (capacity > 0)
        if valid.any():
            errors = (y_hat[valid] - y[valid]) / capacity[valid]
            result["capacity_normalised_mae_kwh_per_kwp"] = float(np.abs(errors).mean())
            result["capacity_normalised_rmse_kwh_per_kwp"] = float(np.mean(errors**2) ** 0.5)
    return result


def metrics_by_site(
    rows: pd.DataFrame, predictions: np.ndarray, *, full_row_count: int | None = None
) -> pd.DataFrame:
    """Calculate the same metrics independently for each site."""
    working = rows.reset_index(drop=True).copy()
    working["_prediction"] = predictions
    records: list[dict[str, Any]] = []
    for site_id, group in working.groupby("site_id", sort=True):
        capacity = (
            group["installed_capacity_kwp"].to_numpy(dtype=float)
            if "installed_capacity_kwp" in group
            else None
        )
        metrics = calculate_metrics(
            group["ac_energy_kwh"].to_numpy(dtype=float),
            group["_prediction"].to_numpy(dtype=float),
            full_row_count=full_row_count,
            capacity_kwp=capacity,
        )
        records.append({"site_id": str(site_id), **metrics})
    return pd.DataFrame(records)


def daylight_metrics(rows: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any] | None:
    """Calculate secondary daylight-only metrics when the Phase 5 flag exists."""
    if "is_daylight" not in rows:
        return None
    mask = rows["is_daylight"].astype(bool).to_numpy()
    if not mask.any():
        return None
    return calculate_metrics(
        rows.loc[mask, "ac_energy_kwh"].to_numpy(dtype=float),
        predictions[mask],
        full_row_count=len(rows),
    )
