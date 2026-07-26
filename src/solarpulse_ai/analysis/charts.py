"""Headless matplotlib chart generation for exploratory analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from solarpulse_ai.analysis.statistics import WEATHER_COLUMNS

CHART_FILENAMES: tuple[str, ...] = (
    "actual_generation_over_time.png",
    "daily_energy_totals.png",
    "hourly_generation_profile.png",
    "target_distribution.png",
    "generation_vs_ghi.png",
    "correlation_heatmap.png",
    "missing_values_by_field.png",
    "data_availability_by_site.png",
    "weather_trends.png",
)


def _save(figure: Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _axes(title: str, xlabel: str, ylabel: str) -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(figsize=(10, 5))
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(alpha=0.25)
    return figure, axes


def generate_charts(dataframe: pd.DataFrame, output_directory: Path) -> list[Path]:
    """Write deterministic one-chart-per-file PNGs and close every figure."""
    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    figure, axes = _axes("Actual generation over time (UTC)", "Timestamp (UTC)", "Energy (kWh)")
    for site_id, site in dataframe.groupby("site_id", sort=True):
        axes.plot(site["timestamp"], site["ac_energy_kwh"], label=str(site_id), linewidth=1)
    axes.legend()
    written.append(output_directory / CHART_FILENAMES[0])
    _save(figure, written[-1])

    daily = (
        dataframe.assign(day=dataframe["timestamp"].dt.floor("D"))
        .groupby(["day", "site_id"])["ac_energy_kwh"]
        .sum()
        .unstack(fill_value=0)
    )
    figure, axes = _axes("Daily energy totals", "Date (UTC)", "Energy (kWh)")
    daily.plot(ax=axes)
    written.append(output_directory / CHART_FILENAMES[1])
    _save(figure, written[-1])

    hourly = dataframe.groupby(dataframe["timestamp"].dt.hour)["ac_energy_kwh"].mean()
    figure, axes = _axes(
        "Mean hourly generation profile (UTC)", "Hour of day (UTC)", "Mean energy (kWh)"
    )
    axes.plot(hourly.index.to_numpy(), hourly.to_numpy(dtype=float), marker="o")
    axes.set_xticks(range(0, 24, 2))
    written.append(output_directory / CHART_FILENAMES[2])
    _save(figure, written[-1])

    figure, axes = _axes("Target distribution", "Hourly energy (kWh)", "Record count")
    axes.hist(dataframe["ac_energy_kwh"], bins=min(30, max(5, len(dataframe) // 2)))
    written.append(output_directory / CHART_FILENAMES[3])
    _save(figure, written[-1])

    figure, axes = _axes(
        "Generation versus global horizontal irradiance",
        "GHI (W/m²)",
        "Hourly energy (kWh)",
    )
    for site_id, site in dataframe.groupby("site_id", sort=True):
        axes.scatter(site["ghi_w_m2"], site["ac_energy_kwh"], label=str(site_id), alpha=0.55)
    axes.legend()
    written.append(output_directory / CHART_FILENAMES[4])
    _save(figure, written[-1])

    correlation_columns = [
        "ac_energy_kwh",
        *(column for column in WEATHER_COLUMNS if column in dataframe.columns),
    ]
    varying = [
        column
        for column in correlation_columns
        if column == "ac_energy_kwh" or dataframe[column].nunique() > 1
    ]
    figure, axes = plt.subplots(figsize=(9, 7))
    if dataframe["ac_energy_kwh"].nunique() > 1 and len(varying) >= 2:
        matrix = dataframe[varying].corr(method="pearson")
        image = axes.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        axes.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        axes.set_yticks(range(len(matrix.index)), matrix.index)
        for row in range(len(matrix.index)):
            for column in range(len(matrix.columns)):
                axes.text(
                    column,
                    row,
                    f"{matrix.iloc[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        figure.colorbar(image, ax=axes, label="Pearson correlation")
    else:
        axes.text(0.5, 0.5, "Correlation unavailable: insufficient varying columns.", ha="center")
        axes.set_axis_off()
    axes.set_title("Pearson correlation heatmap")
    written.append(output_directory / CHART_FILENAMES[5])
    _save(figure, written[-1])

    missing = dataframe.isna().sum()
    figure, axes = _axes("Missing values by field", "Field", "Missing record count")
    axes.bar(missing.index.to_numpy(), missing.to_numpy(dtype=float))
    axes.tick_params(axis="x", rotation=45)
    written.append(output_directory / CHART_FILENAMES[6])
    _save(figure, written[-1])

    actual = dataframe.groupby("site_id").size()
    expected = dataframe.groupby("site_id")["timestamp"].agg(
        lambda value: len(pd.date_range(value.min(), value.max(), freq="h"))
    )
    availability = pd.DataFrame({"Actual": actual, "Expected": expected})
    figure, axes = _axes("Hourly data availability by site", "Site", "Record count")
    availability.plot.bar(ax=axes)
    written.append(output_directory / CHART_FILENAMES[7])
    _save(figure, written[-1])

    weather = [column for column in WEATHER_COLUMNS if column in dataframe.columns]
    daily_weather = (
        dataframe.assign(day=dataframe["timestamp"].dt.floor("D")).groupby("day")[weather].mean()
    )
    normalized = (daily_weather - daily_weather.mean()) / daily_weather.std().replace(0, 1)
    figure, axes = _axes(
        "Available weather trends (standardised)", "Date (UTC)", "Standard deviations"
    )
    normalized.plot(ax=axes)
    axes.legend(fontsize=7, ncol=2)
    written.append(output_directory / CHART_FILENAMES[8])
    _save(figure, written[-1])

    return written
