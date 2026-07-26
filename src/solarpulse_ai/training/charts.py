"""Headless deterministic training charts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_charts(
    comparison: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_by_site: pd.DataFrame,
    importance: pd.DataFrame,
    directory: str | Path,
) -> list[str]:
    """Create one chart per file and return non-fatal skip warnings."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    def attempt(name: str, draw: Callable[[], None]) -> None:
        try:
            draw()
            plt.tight_layout()
            plt.savefig(output / name, dpi=140)
        except (KeyError, TypeError, ValueError) as error:
            warnings.append(f"Skipped {name}: {error}")
        finally:
            plt.close()

    attempt(
        "validation_mae_comparison.png",
        lambda: _bar(comparison, "model_identifier", "mae_kwh", "Validation MAE", "MAE (kWh)"),
    )
    attempt(
        "validation_rmse_comparison.png",
        lambda: _bar(comparison, "model_identifier", "rmse_kwh", "Validation RMSE", "RMSE (kWh)"),
    )
    attempt(
        "validation_actual_vs_predicted_over_time.png",
        lambda: _time_series(validation_predictions, "Selected model validation"),
    )
    attempt(
        "test_actual_vs_predicted_over_time.png",
        lambda: _time_series(test_predictions, "Selected model test"),
    )
    attempt("predicted_vs_actual_scatter.png", lambda: _scatter(test_predictions))
    attempt("residuals_over_time.png", lambda: _residual_time(test_predictions))
    attempt("residual_distribution.png", lambda: _residual_hist(test_predictions))
    attempt(
        "per_site_test_mae.png",
        lambda: _bar(test_by_site, "site_id", "mae_kwh", "Per-site test MAE", "MAE (kWh)"),
    )
    if importance.empty:
        warnings.append("Permutation-importance chart not applicable to persistence.")
    else:
        attempt("permutation_importance.png", lambda: _importance(importance))
    attempt("daily_actual_vs_predicted_energy.png", lambda: _daily(test_predictions))
    return warnings


def _bar(frame: pd.DataFrame, x: str, y: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(frame[x].astype(str), frame[y].astype(float))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")


def _time_series(frame: pd.DataFrame, title: str) -> None:
    working = frame.assign(timestamp=pd.to_datetime(frame["timestamp"], utc=True))
    aggregated = working.groupby("timestamp", as_index=False)[
        ["actual_ac_energy_kwh", "prediction"]
    ].sum()
    plt.figure(figsize=(10, 4.5))
    plt.plot(aggregated["timestamp"], aggregated["actual_ac_energy_kwh"], label="Actual")
    plt.plot(aggregated["timestamp"], aggregated["prediction"], label="Predicted")
    plt.title(title)
    plt.ylabel("Energy (kWh)")
    plt.legend()


def _scatter(frame: pd.DataFrame) -> None:
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(frame["actual_ac_energy_kwh"], frame["prediction"], alpha=0.55)
    maximum = max(float(frame["actual_ac_energy_kwh"].max()), float(frame["prediction"].max()))
    plt.plot([0, maximum], [0, maximum], linestyle="--", color="black")
    plt.xlabel("Actual (kWh)")
    plt.ylabel("Predicted (kWh)")
    plt.title("Selected model: predicted versus actual")


def _residual_time(frame: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 4.5))
    plt.scatter(pd.to_datetime(frame["timestamp"], utc=True), frame["residual"], s=10)
    plt.axhline(0, linestyle="--", color="black")
    plt.ylabel("Residual: actual - predicted (kWh)")
    plt.title("Test residuals over time")


def _residual_hist(frame: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.hist(frame["residual"], bins=min(30, max(5, len(frame) // 3)))
    plt.xlabel("Residual: actual - predicted (kWh)")
    plt.ylabel("Count")
    plt.title("Test residual distribution")


def _importance(frame: pd.DataFrame) -> None:
    top = frame.sort_values("importance_mean", ascending=True).tail(20)
    plt.figure(figsize=(8, max(4.5, len(top) * 0.28)))
    plt.barh(top["predictor"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Decrease in negative MAE score")
    plt.title("Validation permutation importance (not causal)")


def _daily(frame: pd.DataFrame) -> None:
    working = frame.assign(date=pd.to_datetime(frame["timestamp"], utc=True).dt.date)
    daily = working.groupby("date")[["actual_ac_energy_kwh", "prediction"]].sum()
    plt.figure(figsize=(9, 4.5))
    plt.plot(daily.index.astype(str), daily["actual_ac_energy_kwh"], marker="o", label="Actual")
    plt.plot(daily.index.astype(str), daily["prediction"], marker="o", label="Predicted")
    plt.ylabel("Daily energy (kWh)")
    plt.title("Daily aggregated test energy")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
