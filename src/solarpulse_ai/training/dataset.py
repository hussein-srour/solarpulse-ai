"""Feature-dataset loading, eligibility filtering, and chronological validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solarpulse_ai.training.config import TrainingConfig
from solarpulse_ai.training.contracts import KEYS, TARGET, TrainingContract

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Eligible, validated rows and record-free audit summaries."""

    frame: pd.DataFrame
    eligibility: dict[str, Any]
    split_counts: dict[str, int]
    split_boundaries: dict[str, dict[str, str]]
    constant_predictors: tuple[str, ...]


def load_feature_dataset(path: str | Path) -> pd.DataFrame:
    """Read the feature CSV without transforming predictor missingness."""
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"invalid feature dataset: {error}") from error


def prepare_dataset(
    source: pd.DataFrame, contract: TrainingContract, config: TrainingConfig
) -> PreparedDataset:
    """Validate all rows before returning the eligible chronological subset."""
    frame = source.copy()
    parsed = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError("timestamps must parse as UTC")
    frame["timestamp"] = parsed
    if any(
        not group["timestamp"].is_monotonic_increasing
        for _, group in frame.groupby("site_id", sort=False)
    ):
        raise ValueError("records must remain chronologically sorted within each site")
    if frame.duplicated(list(KEYS)).any():
        raise ValueError("duplicate site_id/timestamp rows are not allowed")
    labels = set(frame["split"].dropna().astype(str))
    if labels != set(SPLITS):
        raise ValueError(f"split must contain exactly {list(SPLITS)}; found {sorted(labels)}")
    target = pd.to_numeric(frame[TARGET], errors="coerce")
    if not np.isfinite(target.to_numpy(dtype=float)).all() or (target < 0).any():
        raise ValueError("target values must be finite and non-negative")
    frame[TARGET] = target
    for column in contract.numerical:
        values = pd.to_numeric(frame[column], errors="coerce")
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"predictor {column} contains positive or negative infinity")
        frame[column] = values
    for column in contract.boolean:
        frame[column] = _boolean_series(frame[column], column).astype(float)
    for column in contract.categorical:
        if not (
            pd.api.types.is_object_dtype(frame[column])
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
        ):
            frame[column] = frame[column].astype("string")
    classified = set(contract.numerical) | set(contract.categorical) | set(contract.boolean)
    for column in contract.predictors:
        if column not in classified:
            raise ValueError(f"unsupported unclassified predictor: {column}")

    eligible_mask = _boolean_series(frame["feature_eligible"], "feature_eligible")
    reasons = (
        frame.loc[~eligible_mask, "feature_missing_reasons"]
        .fillna("")
        .astype(str)
        .str.split(";")
        .explode()
    )
    eligibility = {
        "original_row_count": len(frame),
        "eligible_row_count": int(eligible_mask.sum()),
        "excluded_row_count": int((~eligible_mask).sum()),
        "excluded_counts_by_split": _counts(frame.loc[~eligible_mask], "split"),
        "excluded_counts_by_site": _counts(frame.loc[~eligible_mask], "site_id"),
        "eligibility_reason_counts": {
            str(key): int(value)
            for key, value in reasons.loc[reasons.ne("")].value_counts().items()
        },
    }
    eligible = frame.loc[eligible_mask].copy()
    eligible = eligible.sort_values(["timestamp", "site_id"], kind="stable").reset_index(drop=True)
    counts = _counts(eligible, "split")
    minima = {
        "train": config.minimum_train_rows,
        "validation": config.minimum_validation_rows,
        "test": config.minimum_test_rows,
    }
    for split, minimum in minima.items():
        if counts.get(split, 0) < minimum:
            raise ValueError(
                f"{split} has {counts.get(split, 0)} eligible rows; minimum is {minimum}"
            )
    boundaries: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        subset = eligible.loc[eligible["split"].eq(split), "timestamp"]
        boundaries[split] = {"start": subset.min().isoformat(), "end": subset.max().isoformat()}
    if not (
        pd.Timestamp(boundaries["train"]["end"])
        < pd.Timestamp(boundaries["validation"]["start"])
        <= pd.Timestamp(boundaries["validation"]["end"])
        < pd.Timestamp(boundaries["test"]["start"])
    ):
        raise ValueError("split timestamp ranges must be chronological and non-overlapping")
    train = eligible.loc[eligible["split"].eq("train")]
    constants = tuple(
        column for column in contract.predictors if train[column].nunique(dropna=True) <= 1
    )
    return PreparedDataset(eligible, eligibility, counts, boundaries, constants)


def split_frame(dataset: PreparedDataset, split: str) -> pd.DataFrame:
    """Return one chronological split without shuffling."""
    return dataset.frame.loc[dataset.frame["split"].eq(split)].copy()


def _boolean_series(series: pd.Series, name: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "1": True,
        "0": False,
    }
    converted = series.map(mapping)
    if converted.isna().any():
        raise ValueError(f"{name} must contain boolean values")
    return converted.astype(bool)


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}
