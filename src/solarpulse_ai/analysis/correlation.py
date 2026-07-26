"""Pearson correlation analysis for generation and weather."""

from __future__ import annotations

from typing import Any

import pandas as pd

from solarpulse_ai.analysis.statistics import WEATHER_COLUMNS


def analyse_correlations(dataframe: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate Pearson correlations, marking constant or insufficient fields unavailable."""
    rows: list[dict[str, object]] = []
    usable = ["ac_energy_kwh"]
    for column in WEATHER_COLUMNS:
        if column not in dataframe.columns:
            continue
        pair = dataframe[["ac_energy_kwh", column]].dropna()
        reason: str | None = None
        correlation: float | None = None
        if len(pair) < 2:
            reason = "fewer than two complete paired observations"
        elif pair[column].nunique() < 2:
            reason = "weather column is constant"
        elif pair["ac_energy_kwh"].nunique() < 2:
            reason = "target column is constant"
        else:
            correlation = float(pair["ac_energy_kwh"].corr(pair[column], method="pearson"))
            usable.append(column)
        rows.append(
            {
                "weather_variable": column,
                "method": "Pearson",
                "paired_observations": len(pair),
                "correlation_with_ac_energy_kwh": correlation,
                "availability": "available" if correlation is not None else "unavailable",
                "reason": reason,
            }
        )
    table = pd.DataFrame(rows)
    matrix = dataframe[usable].corr(method="pearson") if len(usable) > 1 else pd.DataFrame()
    return (
        {
            "method": "Pearson product-moment correlation",
            "causation_note": "Correlation describes association and does not prove causation.",
            "table": rows,
            "matrix": matrix.to_dict() if not matrix.empty else {},
        },
        table,
    )
