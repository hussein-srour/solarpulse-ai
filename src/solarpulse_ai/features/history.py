"""Leakage-safe exact-time and rolling generation history."""

from __future__ import annotations

import pandas as pd

from solarpulse_ai.features.config import FeatureConfig

ROLLING_STATS = ("mean", "std", "min", "max", "median")


def add_generation_history(
    dataframe: pd.DataFrame, config: FeatureConfig
) -> tuple[pd.DataFrame, list[str], dict[str, pd.Series[bool]]]:
    """Add history available at or before ``t - horizon`` independently per site.

    Rolling windows are right-closed and left-open: ``(cutoff - window, cutoff]``.
    """
    result = dataframe.copy()
    feature_names: list[str] = []
    unavailable: dict[str, pd.Series[bool]] = {}
    if not config.include_target_history:
        return result, feature_names, unavailable

    lookup = result.set_index(["site_id", "timestamp"])["ac_energy_kwh"]
    for lag in config.target_lag_hours:
        shifted_keys = pd.MultiIndex.from_arrays(
            [result["site_id"], result["timestamp"] - pd.Timedelta(lag, unit="h")]
        )
        name = f"ac_energy_lag_{lag}h"
        values = lookup.reindex(shifted_keys)
        values.index = result.index
        result[name] = values
        feature_names.append(name)
        unavailable[f"missing_exact_target_lag_{lag}h"] = values.isna()

    for window in config.rolling_window_hours:
        count_name = f"ac_energy_rolling_{window}h_count"
        for name in [
            count_name,
            *(f"ac_energy_rolling_{window}h_{stat}" for stat in ROLLING_STATS),
        ]:
            result[name] = pd.NA
            feature_names.append(name)
        for _site_id, indices in result.groupby("site_id", sort=False).groups.items():
            site = result.loc[indices, ["timestamp", "ac_energy_kwh"]].sort_values("timestamp")
            cutoff_index = pd.DatetimeIndex(
                site["timestamp"] + pd.Timedelta(config.forecast_horizon_hours, unit="h"),
                name="timestamp",
            )
            available = pd.Series(site["ac_energy_kwh"].to_numpy(), index=cutoff_index)
            targets = pd.DatetimeIndex(result.loc[indices, "timestamp"])
            evaluation_index = available.index.union(targets).sort_values()
            rolling = available.reindex(evaluation_index).rolling(f"{window}h", closed="right")
            stats: dict[str, pd.Series[float]] = {
                "count": rolling.count(),
                "mean": rolling.mean(),
                "std": rolling.std(),
                "min": rolling.min(),
                "max": rolling.max(),
                "median": rolling.median(),
            }
            for statistic, values in stats.items():
                column = f"ac_energy_rolling_{window}h_{statistic}"
                result.loc[indices, column] = values.reindex(targets).to_numpy()
        counts = pd.to_numeric(result[count_name], errors="coerce")
        unavailable[f"missing_rolling_history_{window}h"] = counts.fillna(0).eq(0)
        if config.require_complete_history:
            unavailable[f"incomplete_rolling_history_{window}h"] = counts.lt(window)
    return result.infer_objects(), feature_names, unavailable
