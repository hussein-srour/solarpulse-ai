"""Typed configuration for exploratory analysis."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnalysisThresholds:
    """Transparent thresholds used to produce non-destructive quality indicators."""

    low_irradiance_w_m2: float = 20.0
    high_irradiance_w_m2: float = 500.0
    near_zero_generation_kwh: float = 0.01
    consecutive_zero_hours: int = 6
    outlier_iqr_multiplier: float = 1.5
    minimum_history_days: int = 60

    def __post_init__(self) -> None:
        """Reject thresholds that cannot produce meaningful diagnostics."""
        if self.low_irradiance_w_m2 < 0:
            raise ValueError("Low irradiance threshold must be non-negative.")
        if self.high_irradiance_w_m2 <= self.low_irradiance_w_m2:
            raise ValueError("High irradiance threshold must exceed the low threshold.")
        if self.near_zero_generation_kwh < 0:
            raise ValueError("Near-zero generation threshold must be non-negative.")
        if self.consecutive_zero_hours < 2:
            raise ValueError("Consecutive-zero duration must be at least 2 hours.")
        if self.outlier_iqr_multiplier <= 0:
            raise ValueError("IQR multiplier must be positive.")
        if self.minimum_history_days < 1:
            raise ValueError("Minimum history must be at least one day.")


@dataclass(frozen=True, slots=True)
class SplitProportions:
    """Chronological training, validation, and testing proportions."""

    training: float = 0.70
    validation: float = 0.15
    testing: float = 0.15

    def __post_init__(self) -> None:
        """Ensure proportions are positive and total one."""
        values = (self.training, self.validation, self.testing)
        if any(value <= 0 for value in values):
            raise ValueError("All split proportions must be greater than zero.")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("Training, validation, and testing proportions must total 1.0.")
