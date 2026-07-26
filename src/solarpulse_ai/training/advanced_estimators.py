"""Deterministic CPU-compatible XGBoost estimator construction."""

from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from solarpulse_ai.training.contracts import TrainingContract
from solarpulse_ai.training.preprocessing import build_preprocessor


def build_xgboost_model(
    contract: TrainingContract,
    parameters: dict[str, Any],
    *,
    random_seed: int,
    n_jobs: int,
) -> Pipeline:
    """Build preprocessing and XGBoost as one persistable fitted pipeline."""
    estimator = XGBRegressor(
        **parameters,
        objective="reg:squarederror",
        random_state=random_seed,
        n_jobs=n_jobs,
        missing=float("nan"),
        tree_method="hist",
        device="cpu",
        verbosity=0,
    )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(contract, scale=False)),
            ("estimator", estimator),
        ]
    )
