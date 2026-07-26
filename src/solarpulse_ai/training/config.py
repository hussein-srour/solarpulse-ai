"""Validated and serialisable training configuration."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator

MODEL_IDS = (
    "persistence",
    "dummy_mean",
    "ridge",
    "random_forest",
    "histogram_gradient_boosting",
)


class TrainingConfig(BaseModel):
    """Fixed baseline choices; no tuning is performed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    random_seed: int = 42
    primary_metric: str = "mae"
    clip_negative_predictions: bool = True
    minimum_train_rows: int = Field(default=24, gt=0)
    minimum_validation_rows: int = Field(default=12, gt=0)
    minimum_test_rows: int = Field(default=12, gt=0)
    enabled_models: tuple[str, ...] = MODEL_IDS
    persistence_lag_hours: int = Field(default=24, gt=0)
    random_forest_estimators: int = Field(default=100, gt=0, le=500)
    random_forest_max_depth: int | None = Field(default=12, gt=0)
    random_forest_min_samples_leaf: int = Field(default=2, gt=0)
    histogram_boosting_max_iter: int = Field(default=100, gt=0, le=500)
    histogram_boosting_learning_rate: float = Field(default=0.1, gt=0, le=1)
    histogram_boosting_max_leaf_nodes: int = Field(default=15, gt=1)
    permutation_importance_repeats: int = Field(default=5, gt=0, le=50)
    daylight_only_secondary_metrics: bool = True
    refit_selected_model_on_train_validation: bool = True

    @field_validator("primary_metric")
    @classmethod
    def only_mae(cls, value: str) -> str:
        """Keep the model-selection policy explicit."""
        if value != "mae":
            raise ValueError("primary_metric must be 'mae'")
        return value

    @field_validator("enabled_models")
    @classmethod
    def supported_unique_models(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject unsupported, absent, or repeated candidates."""
        if not value:
            raise ValueError("at least one model must be enabled")
        unsupported = sorted(set(value) - set(MODEL_IDS))
        if unsupported:
            raise ValueError(f"unsupported models: {unsupported}")
        if len(set(value)) != len(value):
            raise ValueError("enabled_models must not contain duplicates")
        return tuple(model for model in MODEL_IDS if model in value)

    @field_validator("histogram_boosting_learning_rate")
    @classmethod
    def finite_learning_rate(cls, value: float) -> float:
        """Reject NaN and infinity."""
        if not math.isfinite(value):
            raise ValueError("learning rate must be finite")
        return value
