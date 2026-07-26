"""Fixed deterministic baseline estimators and persistence specification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from solarpulse_ai.training.config import TrainingConfig
from solarpulse_ai.training.contracts import TrainingContract
from solarpulse_ai.training.preprocessing import build_preprocessor


@dataclass(frozen=True, slots=True)
class PersistencePredictor:
    """Serializable exact-lag prediction specification."""

    lag_hours: int

    @property
    def predictor_column(self) -> str:
        """Return the Phase 5 exact-lag feature name."""
        return f"ac_energy_lag_{self.lag_hours}h"

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict from the declared exact lag, never row position or current target."""
        if self.predictor_column not in frame:
            raise ValueError(f"persistence requires {self.predictor_column}")
        return pd.to_numeric(frame[self.predictor_column], errors="coerce").to_numpy(dtype=float)


def build_model(model_id: str, contract: TrainingContract, config: TrainingConfig) -> Pipeline:
    """Construct one trainable fixed baseline with preprocessing included."""
    if model_id == "dummy_mean":
        estimator: Any = DummyRegressor(strategy="mean")
        scale = False
    elif model_id == "ridge":
        estimator = Ridge(alpha=1.0)
        scale = True
    elif model_id == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=config.random_forest_estimators,
            max_depth=config.random_forest_max_depth,
            min_samples_leaf=config.random_forest_min_samples_leaf,
            random_state=config.random_seed,
            n_jobs=1,
        )
        scale = False
    elif model_id == "histogram_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=config.histogram_boosting_max_iter,
            learning_rate=config.histogram_boosting_learning_rate,
            max_leaf_nodes=config.histogram_boosting_max_leaf_nodes,
            random_state=config.random_seed,
        )
        scale = False
    else:
        raise ValueError(f"{model_id} is not a trainable model")
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(contract, scale=scale)),
            ("estimator", estimator),
        ]
    )


def fixed_parameters(config: TrainingConfig) -> dict[str, dict[str, Any]]:
    """Return the reportable fixed candidate configurations."""
    return {
        "persistence": {"lag_hours": config.persistence_lag_hours},
        "dummy_mean": {"strategy": "mean"},
        "ridge": {"alpha": 1.0, "scale": True},
        "random_forest": {
            "n_estimators": config.random_forest_estimators,
            "max_depth": config.random_forest_max_depth,
            "min_samples_leaf": config.random_forest_min_samples_leaf,
            "random_state": config.random_seed,
            "n_jobs": 1,
        },
        "histogram_gradient_boosting": {
            "max_iter": config.histogram_boosting_max_iter,
            "learning_rate": config.histogram_boosting_learning_rate,
            "max_leaf_nodes": config.histogram_boosting_max_leaf_nodes,
            "random_state": config.random_seed,
        },
    }
