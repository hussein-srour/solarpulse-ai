"""Markdown and JSON report writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from solarpulse_ai.analysis.serialization import to_json_value


def write_json(data: dict[str, Any], path: Path) -> Path:
    """Write standards-compliant, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_json_value(data), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _mapping_table(mapping: dict[str, object], value_label: str) -> str:
    rows = [f"| {key} | {value} |" for key, value in mapping.items()]
    return "\n".join([f"| Item | {value_label} |", "| --- | --- |", *rows])


def build_markdown_report(report: dict[str, Any]) -> str:
    """Render the full analysis into a concise Markdown report."""
    profile = report["profile"]
    target = report["target_analysis"]["overall"]
    quality = report["data_quality"]
    readiness = report["model_readiness"]
    split_periods = report["split_plan"]["periods"]
    correlation_rows = report["correlation_analysis"]["table"]
    weather = report["weather_analysis"]
    limitations = readiness["major_limitations_before_training"]
    quality_rows = [
        f"| {name} | {count} |" for name, count in quality["counts_by_indicator"].items()
    ] or ["| None | 0 |"]
    correlation_table = [
        (
            f"| {row['weather_variable']} | {row['paired_observations']} | "
            f"{row['correlation_with_ac_energy_kwh']} | {row['availability']} | "
            f"{row['reason'] or ''} |"
        )
        for row in correlation_rows
    ]
    split_rows = [
        (
            f"| {name} | {period['beginning_timestamp']} | "
            f"{period['ending_timestamp']} | {period['record_count']} |"
        )
        for name, period in split_periods.items()
    ]
    target_display = {key: value for key, value in target.items() if key not in {"total"}}
    limitation_lines = (
        "\n".join(f"- {item}" for item in limitations)
        if limitations
        else "- None identified by these screening rules."
    )
    return f"""# SolarPulse AI exploratory dataset report

Generated: {report["identity"]["generated_at_utc"]}

Source: `{report["identity"]["source_path"]}`

This report describes dataset structure and data-quality indicators. It does not
claim plant performance, causal effects, confirmed equipment faults, or production readiness.

## Dataset profile

- Records: {profile["total_record_count"]}
- Sites: {profile["number_of_sites"]} ({", ".join(profile["site_ids"])})
- UTC coverage: {profile["earliest_timestamp"]} to {profile["latest_timestamp"]}
- Calendar duration: {profile["total_calendar_duration"]}
- Missing hourly timestamps: {profile["missing_timestamp_count"]}
- Complete site-days: {profile["number_of_complete_days"]}
- Partial site-days: {profile["number_of_partial_days"]}
- Memory usage: {profile["memory_usage_bytes"]} bytes
- Optional columns absent: {", ".join(profile["optional_columns_absent"]) or "None"}

### Records per site

{_mapping_table(profile["records_per_site"], "Records")}

### Missing values

{_mapping_table(profile["missing_value_count_by_column"], "Missing")}

## Target analysis: `ac_energy_kwh`

{_mapping_table(target_display, "Value")}

Daily and monthly aggregate values are available in `dataset_profile.json`. Monthly
totals are emitted only when at least two calendar months are represented.

## Available weather analysis

Descriptive statistics were calculated for: {", ".join(weather) or "none"}.
No absent optional field was fabricated.

## Temporal analysis

All calculations use UTC. Generation was summarised by hour, weekday, available
months, and day. Gaps use adjacent hourly timestamps per site. Local-time
operational analysis may be added later using each site's configured IANA timezone.

## Correlation analysis

Pearson product-moment correlation is used on complete numeric pairs. Constant
columns and insufficient pairs are marked unavailable. Correlation describes
association and does not prove causation.

| Weather variable | Paired records | Correlation | Availability | Reason |
| --- | ---: | ---: | --- | --- |
{chr(10).join(correlation_table)}

## Data-quality indicators

These are non-destructive indicators, not confirmed equipment faults. No record
was filled, clipped, interpolated, deleted, or otherwise corrected.

| Indicator | Flag rows |
| --- | ---: |
{chr(10).join(quality_rows)}

Thresholds: `{quality["thresholds"]}`

## Model-readiness assessment

Category: **{readiness["category"]}**

Passing schema validation alone does not establish production readiness.

Rules:

- `ready`: {readiness["rules"]["ready"]}
- `ready_with_warnings`: {readiness["rules"]["ready_with_warnings"]}
- `not_ready`: {readiness["rules"]["not_ready"]}

Major limitations:

{limitation_lines}

## Chronological split recommendation

Global unique UTC timestamps are assigned in order and never shuffled. Periods do
not overlap, preventing future observations from leaking into earlier splits.

| Split | Beginning | Ending | Records |
| --- | --- | --- | ---: |
{chr(10).join(split_rows)}

## Charts

See `charts/` for actual generation over time, daily totals, hourly profile,
target distribution, generation versus GHI, Pearson heatmap, missingness,
site availability, and available weather trends.

## Limitations and next steps

- Review every indicator against source-system context before taking operational action.
- Decide and document a gap/missingness policy before feature engineering.
- Add site IANA timezones before local-time operational interpretation.
- Validate seasonality and split adequacy with representative private history.
- Model training is intentionally outside Phase 4.
"""


def write_markdown_report(report: dict[str, Any], path: Path) -> Path:
    """Write the complete Markdown analysis report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown_report(report), encoding="utf-8")
    return path


def write_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    """Write a generated analysis table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path
