"""Prediction post-processing and compact residual records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ProcessedPredictions:
    """Final predictions plus transparent clipping diagnostics."""

    values: np.ndarray
    raw_negative_count: int


def postprocess(predictions: np.ndarray, *, clip_negative: bool) -> ProcessedPredictions:
    """Reject non-finite output and optionally clip negative solar energy to zero."""
    raw = np.asarray(predictions, dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("model generated non-finite predictions")
    negative_count = int((raw < 0).sum())
    final = np.maximum(raw, 0.0) if clip_negative else raw.copy()
    return ProcessedPredictions(final, negative_count)


def prediction_frame(rows: pd.DataFrame, predictions: np.ndarray, model_id: str) -> pd.DataFrame:
    """Create the deliberately narrow prediction-report schema."""
    actual = rows["ac_energy_kwh"].to_numpy(dtype=float)
    residual = actual - predictions
    output = pd.DataFrame(
        {
            "timestamp": rows["timestamp"].astype(str).to_numpy(),
            "site_id": rows["site_id"].astype(str).to_numpy(),
            "split": rows["split"].astype(str).to_numpy(),
            "actual_ac_energy_kwh": actual,
            "prediction": predictions,
            "residual": residual,
            "absolute_error": np.abs(residual),
            "squared_error": residual**2,
        }
    )
    if "is_daylight" in rows:
        output["is_daylight"] = rows["is_daylight"].to_numpy()
    output["model_identifier"] = model_id
    return output
