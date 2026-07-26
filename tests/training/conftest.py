"""Small deterministic Phase 5 fixtures for training tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def training_inputs(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame, dict[str, object]]:
    """Write a compact synthetic feature CSV and matching Phase 5 manifest."""
    rows = 90
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    hour = np.arange(rows, dtype=float)
    daylight = ((hour % 24 >= 6) & (hour % 24 <= 17)).astype(float)
    signal = daylight * (5.0 + 0.08 * hour + np.sin(hour / 4))
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "site_id": ["site-a"] * rows,
            "ac_energy_kwh": np.maximum(signal, 0),
            "split": ["train"] * 30 + ["validation"] * 30 + ["test"] * 30,
            "feature_eligible": [True] * rows,
            "feature_missing_count": [0] * rows,
            "feature_missing_reasons": [""] * rows,
            "weather_signal": signal + np.cos(hour) * 0.05,
            "ac_energy_lag_24h": np.full(rows, 2.0),
            "is_daylight": daylight,
            "installed_capacity_kwp": np.full(rows, 20.0),
        }
    )
    predictors = [
        "weather_signal",
        "ac_energy_lag_24h",
        "is_daylight",
        "installed_capacity_kwp",
    ]
    manifest: dict[str, object] = {
        "project_version": "0.1.0",
        "source_row_count": rows,
        "output_row_count": rows,
        "forecast_horizon_hours": 24,
        "key_columns": ["timestamp", "site_id"],
        "target_column": "ac_energy_kwh",
        "predictor_columns": predictors,
        "metadata_columns": [
            "split",
            "feature_eligible",
            "feature_missing_count",
            "feature_missing_reasons",
        ],
        "categorical_columns": [],
        "numerical_columns": predictors,
        "expected_data_types": {
            "timestamp": "datetime64[ns, UTC]",
            "site_id": "object",
            "ac_energy_kwh": "float64",
            **dict.fromkeys(predictors, "float64"),
        },
        "warnings_and_limitations": ["synthetic fixture"],
    }
    feature_path = tmp_path / "features.csv"
    manifest_path = tmp_path / "feature_manifest.json"
    frame.to_csv(feature_path, index=False)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return feature_path, manifest_path, frame, manifest
