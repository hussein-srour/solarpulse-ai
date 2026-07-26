"""Leakage-resistant chronological split planning."""

from __future__ import annotations

from typing import Any

import pandas as pd

from solarpulse_ai.analysis.config import SplitProportions

SPLIT_NAMES = ("training", "validation", "testing")


def plan_chronological_splits(
    dataframe: pd.DataFrame, proportions: SplitProportions
) -> tuple[dict[str, Any], pd.Series[str]]:
    """Assign globally ordered timestamps to non-overlapping periods without shuffling."""
    timestamps = pd.DatetimeIndex(dataframe["timestamp"].drop_duplicates().sort_values())
    count = len(timestamps)
    training_count = int(count * proportions.training)
    validation_count = int(count * proportions.validation)
    testing_count = count - training_count - validation_count
    if min(training_count, validation_count, testing_count) < 1:
        raise ValueError(
            "Chronological split would contain an empty period; provide more unique timestamps."
        )

    training_end = timestamps[training_count - 1]
    validation_end = timestamps[training_count + validation_count - 1]
    labels = pd.Series("testing", index=dataframe.index, dtype="string")
    labels.loc[dataframe["timestamp"] <= training_end] = "training"
    labels.loc[
        (dataframe["timestamp"] > training_end) & (dataframe["timestamp"] <= validation_end)
    ] = "validation"

    periods: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        selected = dataframe.loc[labels == name]
        if selected.empty:
            raise ValueError(f"Chronological {name} split is empty.")
        periods[name] = {
            "beginning_timestamp": selected["timestamp"].min(),
            "ending_timestamp": selected["timestamp"].max(),
            "record_count": len(selected),
            "unique_timestamp_count": selected["timestamp"].nunique(),
            "counts_by_site": {
                str(key): int(value)
                for key, value in selected.groupby("site_id", sort=True).size().items()
            },
        }
    return (
        {
            "strategy": "Global UTC chronological timestamps; records are never shuffled.",
            "proportions": {
                "training": proportions.training,
                "validation": proportions.validation,
                "testing": proportions.testing,
            },
            "periods": periods,
            "non_overlapping": (
                periods["training"]["ending_timestamp"]
                < periods["validation"]["beginning_timestamp"]
                and periods["validation"]["ending_timestamp"]
                < periods["testing"]["beginning_timestamp"]
            ),
        },
        labels,
    )
