"""Validated configuration for leakage-safe feature generation."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FeatureConfig(BaseModel):
    """User-controlled feature choices and prediction-time boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_horizon_hours: int = Field(default=24, gt=0)
    target_lag_hours: tuple[int, ...] = (24, 48, 168)
    rolling_window_hours: tuple[int, ...] = (3, 6, 24, 72, 168)
    weather_lag_hours: tuple[int, ...] = (24, 48)
    daylight_ghi_threshold_w_m2: float = Field(default=10.0, ge=0)
    irradiance_epsilon: float = Field(default=1e-6, ge=0)
    include_raw_weather_features: bool = True
    include_site_metadata_features: bool = True
    include_weather_lags: bool = True
    include_target_history: bool = True
    require_complete_history: bool = False

    @field_validator("target_lag_hours", "rolling_window_hours", "weather_lag_hours")
    @classmethod
    def positive_unique_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require positive, non-duplicated hour values."""
        if any(isinstance(item, bool) or item <= 0 for item in value):
            raise ValueError("lags and windows must contain positive integers")
        if len(set(value)) != len(value):
            raise ValueError("duplicate lag and window values are not allowed")
        return value

    @field_validator("daylight_ghi_threshold_w_m2", "irradiance_epsilon")
    @classmethod
    def finite_threshold(cls, value: float) -> float:
        """Reject infinite and NaN thresholds."""
        if not math.isfinite(value):
            raise ValueError("thresholds must be finite")
        return value

    @model_validator(mode="after")
    def lags_respect_horizon(self) -> FeatureConfig:
        """Prevent target history from crossing the forecast cutoff."""
        if any(lag < self.forecast_horizon_hours for lag in self.target_lag_hours):
            raise ValueError("target lags must be at least the forecast horizon")
        return self
