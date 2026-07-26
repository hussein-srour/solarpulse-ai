"""Typed Phase 7 training and XGBoost search configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_SELECTION_METRICS = ("mae", "rmse")


class XGBoostSearchSpace(BaseModel):
    """Bounded values sampled deterministically by the tuning runner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_estimators: tuple[int, ...] = (200, 350, 500, 750, 1000)
    max_depth: tuple[int, ...] = (2, 3, 4, 5, 6, 8)
    learning_rate: tuple[float, ...] = (0.01, 0.03, 0.05, 0.08, 0.1, 0.15)
    min_child_weight: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0, 15.0)
    subsample: tuple[float, ...] = (0.65, 0.75, 0.85, 1.0)
    colsample_bytree: tuple[float, ...] = (0.65, 0.75, 0.85, 1.0)
    reg_alpha: tuple[float, ...] = (0.0, 0.1, 1.0, 5.0, 10.0)
    reg_lambda: tuple[float, ...] = (0.1, 1.0, 5.0, 10.0, 20.0)
    gamma: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 5.0)

    @model_validator(mode="after")
    def bounded_values(self) -> XGBoostSearchSpace:
        """Reject empty, duplicated, non-finite, or out-of-range choices."""
        bounds: dict[str, tuple[float, float]] = {
            "n_estimators": (1, 1000),
            "max_depth": (1, 8),
            "learning_rate": (0.001, 0.15),
            "min_child_weight": (0, 15),
            "subsample": (0.65, 1),
            "colsample_bytree": (0.65, 1),
            "reg_alpha": (0, 10),
            "reg_lambda": (0.1, 20),
            "gamma": (0, 5),
        }
        for name, (lower, upper) in bounds.items():
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} must contain at least one value")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicate values")
            if any(
                not math.isfinite(float(value)) or not lower <= value <= upper for value in values
            ):
                raise ValueError(f"{name} values must be finite and within [{lower}, {upper}]")
        return self


class AdvancedTrainingConfig(BaseModel):
    """JSON-serialisable Phase 7 orchestration configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_horizon_hours: int = Field(default=24, gt=0)
    cross_validation_splits: int = Field(default=5, ge=2, le=20)
    cross_validation_gap_hours: int = Field(default=24, gt=0)
    minimum_training_hours: int = Field(default=24 * 60, gt=0)
    validation_window_hours: int = Field(default=24 * 14, gt=0)
    search_budget: int = Field(default=24, gt=0, le=100)
    selection_metric: str = "mae"
    random_seed: int = 42
    n_jobs: int = Field(default=1, ge=-64, le=64)
    clip_negative_predictions: bool = True
    refit_selected_model: bool = True
    compare_with_phase6_baselines: bool = True
    save_fold_predictions: bool = False
    register_model: bool = True
    maximum_training_rows: int | None = Field(default=None, gt=0)
    search_space: XGBoostSearchSpace = Field(default_factory=XGBoostSearchSpace)

    @field_validator("selection_metric")
    @classmethod
    def supported_metric(cls, value: str) -> str:
        """Limit selection to metrics with deterministic minimisation semantics."""
        if value not in SUPPORTED_SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {SUPPORTED_SELECTION_METRICS}")
        return value

    @field_validator("n_jobs")
    @classmethod
    def nonzero_jobs(cls, value: int) -> int:
        """Match common thread-count semantics while rejecting zero."""
        if value == 0:
            raise ValueError("n_jobs must not be zero")
        return value

    @model_validator(mode="after")
    def valid_temporal_policy(self) -> AdvancedTrainingConfig:
        """Enforce the leakage buffer protecting the forecast horizon."""
        if self.cross_validation_gap_hours < self.forecast_horizon_hours:
            raise ValueError("cross_validation_gap_hours must be at least forecast_horizon_hours")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> AdvancedTrainingConfig:
        """Restore strict configuration from a JSON object."""
        try:
            payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid advanced training configuration: {error}") from error
        return cls.model_validate(payload)

    def to_json(self, path: str | Path) -> None:
        """Write deterministic JSON for reproducibility."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
