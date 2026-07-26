"""End-to-end feature-pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from solarpulse_ai.analysis.reporting import write_csv, write_json
from solarpulse_ai.data.ingestion import read_hourly_csv
from solarpulse_ai.data.schemas import OPTIONAL_COLUMNS
from solarpulse_ai.data.validation import validate_hourly_dataframe
from solarpulse_ai.features.config import FeatureConfig
from solarpulse_ai.features.contract import (
    ELIGIBILITY_COLUMNS,
    KEY_COLUMNS,
    TARGET_COLUMN,
    FeatureContract,
)
from solarpulse_ai.features.eligibility import add_eligibility
from solarpulse_ai.features.history import add_generation_history
from solarpulse_ai.features.manifest import (
    build_manifest,
    build_quality,
    eligibility_table,
    render_quality_markdown,
)
from solarpulse_ai.features.registry import SiteRegistry
from solarpulse_ai.features.splits import assign_splits, load_split_plan
from solarpulse_ai.features.temporal import add_temporal_features
from solarpulse_ai.features.weather import RAW_WEATHER, add_weather_features


@dataclass(frozen=True, slots=True)
class FeatureResult:
    """Outputs and metadata from one pipeline run."""

    dataframe: pd.DataFrame
    contract: FeatureContract
    manifest: dict[str, Any]
    quality: dict[str, Any]
    excluded_rows: int


def run_feature_pipeline(
    input_path: str | Path,
    site_config_paths: list[str | Path],
    output_path: str | Path,
    report_directory: str | Path,
    config: FeatureConfig | None = None,
    *,
    split_plan_path: str | Path | None = None,
    only_eligible: bool = False,
    write_splits: bool = False,
    allow_unused_site_configs: bool = False,
) -> FeatureResult:
    """Validate source data, build features, and write requested artifacts."""
    configured = config or FeatureConfig()
    source = Path(input_path)
    output = Path(output_path)
    report_dir = Path(report_directory)
    raw = read_hourly_csv(source)
    optional_absent = [column for column in OPTIONAL_COLUMNS if column not in raw]
    validated = (
        validate_hourly_dataframe(raw)
        .sort_values(["site_id", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )
    registry = SiteRegistry.from_paths(site_config_paths)
    registry.validate_dataset_sites(
        (str(value) for value in validated["site_id"].unique()),
        allow_unused=allow_unused_site_configs,
    )

    featured = add_temporal_features(validated, registry)
    featured, weather_engineered, weather_lags = add_weather_features(
        featured, registry, configured
    )
    featured, history_features, historical_failures = add_generation_history(featured, configured)
    featured = add_eligibility(featured, historical_failures)

    site_fields: list[str] = []
    if configured.include_site_metadata_features:
        site_fields = [
            "installed_capacity_kwp",
            "panel_tilt_degrees",
            "panel_azimuth_degrees",
            "latitude",
            "longitude",
        ]
        for field in site_fields:
            featured[field] = featured["site_id"].map(
                {site_id: getattr(site, field) for site_id, site in registry.sites.items()}
            )
        if configured.include_target_history:
            history_source = f"ac_energy_lag_{min(configured.target_lag_hours)}h"
            featured["historical_generation_per_capacity"] = (
                featured[history_source] / featured["installed_capacity_kwp"]
            )
            site_fields.append("historical_generation_per_capacity")

    if split_plan_path is not None:
        featured["split"] = assign_splits(featured, load_split_plan(split_plan_path))
    raw_features = [
        column
        for column in RAW_WEATHER
        if column in featured and configured.include_raw_weather_features
    ]
    temporal_features = [
        str(column)
        for column in featured.columns
        if column.startswith("local_")
        or column == "is_weekend"
        or column.endswith(("_sin", "_cos"))
    ]
    site_features = site_fields
    predictors = tuple(
        dict.fromkeys(
            [
                *raw_features,
                *weather_engineered,
                *temporal_features,
                *history_features,
                *weather_lags,
                *site_features,
            ]
        )
    )
    metadata = (*ELIGIBILITY_COLUMNS, *(("split",) if "split" in featured else ()))
    contract = FeatureContract(
        KEY_COLUMNS,
        TARGET_COLUMN,
        predictors,
        metadata,
        (),
        predictors,
    )
    ordered = featured[[*KEY_COLUMNS, TARGET_COLUMN, *metadata, *predictors]].copy()
    source_rows = len(ordered)
    excluded_rows = 0
    if only_eligible:
        excluded_rows = int((~ordered["feature_eligible"]).sum())
        ordered = ordered.loc[ordered["feature_eligible"]].reset_index(drop=True)

    manifest = build_manifest(
        ordered,
        featured,
        source,
        source_rows,
        configured,
        contract,
        raw_features,
        [*weather_engineered, *temporal_features, *history_features, *weather_lags, *site_features],
        excluded_rows,
    )
    quality = build_quality(featured, contract, optional_absent, configured)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(ordered, output)
    write_json(manifest, report_dir / "feature_manifest.json")
    write_json(quality, report_dir / "feature_quality.json")
    (report_dir / "feature_quality.md").write_text(
        render_quality_markdown(quality), encoding="utf-8"
    )
    write_csv(eligibility_table(featured), report_dir / "feature_eligibility.csv")
    if write_splits:
        if "split" not in ordered:
            raise ValueError("--write-splits requires --split-plan")
        for split in ("train", "validation", "test"):
            write_csv(
                ordered.loc[ordered["split"].eq(split)],
                output.parent / f"{split}_features.csv",
            )
    return FeatureResult(ordered, contract, manifest, quality, excluded_rows)
