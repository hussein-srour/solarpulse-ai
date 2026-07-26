"""Leakage-safe expanding-window rolling-origin cross-validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class FoldBoundary:
    """Auditable timestamp and cohort boundaries for one fold."""

    fold_number: int
    training_start: str
    training_end: str
    gap_start: str
    gap_end: str
    validation_start: str
    validation_end: str
    training_row_count: int
    validation_row_count: int
    training_site_count: int
    validation_site_count: int
    eligible_rows_by_site: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable audit record."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RollingFold:
    """Integer row indices plus their audit boundary."""

    training_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    boundary: FoldBoundary


class RollingOriginSplitter:
    """Split sorted unique UTC timestamps into expanding training windows."""

    def __init__(
        self,
        *,
        n_splits: int,
        gap_hours: int,
        minimum_training_hours: int,
        validation_window_hours: int,
    ) -> None:
        """Configure a bounded expanding-window splitter."""
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if min(gap_hours, minimum_training_hours, validation_window_hours) <= 0:
            raise ValueError("rolling-origin durations must be positive")
        self.n_splits = n_splits
        self.gap_hours = gap_hours
        self.minimum_training_hours = minimum_training_hours
        self.validation_window_hours = validation_window_hours

    def split(self, frame: pd.DataFrame) -> list[RollingFold]:
        """Create fixed validation windows ending at the final training timestamp."""
        if frame.empty:
            raise ValueError("cannot split an empty training frame")
        if "timestamp" not in frame or "site_id" not in frame:
            raise ValueError("rolling-origin input requires timestamp and site_id")
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if timestamps.isna().any():
            raise ValueError("rolling-origin timestamps must parse as UTC")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("rolling-origin rows must be sorted chronologically")
        unique = pd.DatetimeIndex(timestamps.drop_duplicates())
        required = (
            self.minimum_training_hours
            + self.gap_hours
            + self.n_splits * self.validation_window_hours
        )
        available = int((unique[-1] - unique[0]).total_seconds() // 3600) + 1
        if available < required:
            raise ValueError(
                f"insufficient training history: need at least {required} hourly periods; "
                f"found {available}"
            )
        final_end = unique[-1]
        first_validation_start = final_end - pd.Timedelta(
            hours=self.n_splits * self.validation_window_hours - 1
        )
        folds: list[RollingFold] = []
        for offset in range(self.n_splits):
            validation_start = first_validation_start + pd.Timedelta(
                hours=offset * self.validation_window_hours
            )
            validation_end = validation_start + pd.Timedelta(hours=self.validation_window_hours - 1)
            training_end = validation_start - pd.Timedelta(hours=self.gap_hours + 1)
            gap_start = training_end + pd.Timedelta(hours=1)
            gap_end = validation_start - pd.Timedelta(hours=1)
            train_mask = timestamps <= training_end
            validation_mask = timestamps.between(validation_start, validation_end)
            training_indices = tuple(frame.index[train_mask].tolist())
            validation_indices = tuple(frame.index[validation_mask].tolist())
            if not training_indices or not validation_indices:
                raise ValueError(f"fold {offset + 1} has an empty training or validation cohort")
            training = frame.loc[list(training_indices)]
            validation = frame.loc[list(validation_indices)]
            boundary = FoldBoundary(
                fold_number=offset + 1,
                training_start=timestamps.loc[list(training_indices)].min().isoformat(),
                training_end=training_end.isoformat(),
                gap_start=gap_start.isoformat(),
                gap_end=gap_end.isoformat(),
                validation_start=validation_start.isoformat(),
                validation_end=validation_end.isoformat(),
                training_row_count=len(training),
                validation_row_count=len(validation),
                training_site_count=int(training["site_id"].nunique()),
                validation_site_count=int(validation["site_id"].nunique()),
                eligible_rows_by_site={
                    str(site): {
                        "training": int((training["site_id"] == site).sum()),
                        "validation": int((validation["site_id"] == site).sum()),
                    }
                    for site in sorted(set(training["site_id"]) | set(validation["site_id"]))
                },
            )
            folds.append(RollingFold(training_indices, validation_indices, boundary))
        return folds
