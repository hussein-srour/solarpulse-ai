"""Orchestration for reusable exploratory analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from solarpulse_ai.analysis.charts import generate_charts
from solarpulse_ai.analysis.config import AnalysisThresholds, SplitProportions
from solarpulse_ai.analysis.correlation import analyse_correlations
from solarpulse_ai.analysis.profile import profile_dataset
from solarpulse_ai.analysis.quality import diagnose_quality
from solarpulse_ai.analysis.readiness import assess_readiness
from solarpulse_ai.analysis.reporting import write_csv, write_json, write_markdown_report
from solarpulse_ai.analysis.splits import SPLIT_NAMES, plan_chronological_splits
from solarpulse_ai.analysis.statistics import analyse_target, analyse_weather
from solarpulse_ai.analysis.temporal import analyse_temporal
from solarpulse_ai.data.ingestion import read_hourly_csv
from solarpulse_ai.data.validation import validate_hourly_dataframe


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Paths and key outputs produced by one analysis run."""

    dataframe: pd.DataFrame
    report: dict[str, Any]
    output_directory: Path
    chart_paths: tuple[Path, ...]


def run_analysis(
    input_path: str | Path,
    output_directory: str | Path,
    thresholds: AnalysisThresholds | None = None,
    proportions: SplitProportions | None = None,
    *,
    write_splits: bool = False,
) -> AnalysisResult:
    """Validate a canonical CSV and create local reports without changing its records."""
    configured_thresholds = thresholds or AnalysisThresholds()
    configured_proportions = proportions or SplitProportions()
    source = Path(input_path)
    output = Path(output_directory)
    raw = read_hourly_csv(source)
    validated = (
        validate_hourly_dataframe(raw)
        .sort_values(["site_id", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )

    profile = profile_dataset(validated)
    target = analyse_target(validated)
    weather = analyse_weather(validated)
    temporal = analyse_temporal(validated)
    correlations, correlation_table = analyse_correlations(validated)
    quality, quality_flags = diagnose_quality(validated, configured_thresholds)
    readiness = assess_readiness(
        validated,
        int(profile["missing_timestamp_count"]),
        configured_thresholds,
    )
    split_plan, split_labels = plan_chronological_splits(validated, configured_proportions)
    generated_at = datetime.now(UTC)
    report: dict[str, Any] = {
        "identity": {
            "source_path": str(source.resolve()),
            "generated_at_utc": generated_at,
            "record_count": len(validated),
        },
        "profile": profile,
        "target_analysis": target,
        "weather_analysis": weather,
        "temporal_analysis": temporal,
        "correlation_analysis": correlations,
        "data_quality": quality,
        "model_readiness": readiness,
        "split_plan": split_plan,
    }

    output.mkdir(parents=True, exist_ok=True)
    write_json(report, output / "dataset_profile.json")
    write_json(split_plan, output / "split_plan.json")
    write_csv(quality_flags, output / "data_quality_flags.csv")
    write_csv(correlation_table, output / "correlations.csv")
    charts = tuple(generate_charts(validated, output / "charts"))
    write_markdown_report(report, output / "dataset_report.md")

    if write_splits:
        for name in SPLIT_NAMES:
            write_csv(validated.loc[split_labels == name], output / f"{name}_split.csv")

    return AnalysisResult(validated, report, output, charts)
