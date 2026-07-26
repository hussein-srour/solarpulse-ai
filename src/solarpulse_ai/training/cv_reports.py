"""Deterministic headless Phase 7 cross-validation and evaluation charts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_advanced_charts(
    candidates: pd.DataFrame,
    folds: pd.DataFrame,
    leaderboard: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    per_site: pd.DataFrame,
    importance: pd.DataFrame,
    directory: str | Path,
) -> list[str]:
    """Create one deterministic chart per file and always close figures."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    def draw(name: str, callback: Callable[[], None]) -> None:
        try:
            callback()
            plt.tight_layout()
            plt.savefig(output / name, dpi=140)
        except (KeyError, TypeError, ValueError) as error:
            warnings.append(f"Skipped {name}: {error}")
        finally:
            plt.close()

    draw("cross_validation_mae_by_fold.png", lambda: _folds(folds))
    draw("candidate_mean_mae.png", lambda: _candidate_error(candidates))
    draw("validation_leaderboard.png", lambda: _bar(leaderboard, "model_identifier", "mae_kwh"))
    draw(
        "validation_actual_vs_predicted.png",
        lambda: _time(validation_predictions, "Validation actual versus predicted"),
    )
    draw(
        "test_actual_vs_predicted.png",
        lambda: _time(test_predictions, "Test actual versus predicted"),
    )
    draw("residual_distribution.png", lambda: _hist(test_predictions))
    draw("residuals_vs_predicted.png", lambda: _residual_scatter(test_predictions))
    draw("hourly_residual_profile.png", lambda: _hourly(test_predictions))
    draw("per_site_mae.png", lambda: _bar(per_site, "site_id", "mae_kwh"))
    draw("cumulative_actual_vs_predicted.png", lambda: _cumulative(test_predictions))
    if not importance.empty:
        draw("feature_importance_gain.png", lambda: _importance(importance, "gain"))
        draw("top_features.png", lambda: _importance(importance, "gain"))
    return warnings


def _bar(frame: pd.DataFrame, x: str, y: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(frame[x].astype(str), frame[y].astype(float))
    plt.ylabel(y)
    plt.xticks(rotation=25, ha="right")


def _folds(frame: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4.5))
    for candidate, group in frame.groupby("candidate_id", sort=True):
        plt.plot(group["fold_number"], group["mae"], marker="o", alpha=0.65, label=candidate)
    plt.xlabel("Fold")
    plt.ylabel("MAE (kWh)")
    if frame["candidate_id"].nunique() <= 8:
        plt.legend(fontsize="small")


def _candidate_error(frame: pd.DataFrame) -> None:
    successful = frame.loc[frame["status"].eq("succeeded")]
    plt.figure(figsize=(9, 4.5))
    plt.errorbar(
        successful["candidate_id"],
        successful["mean_mae"],
        yerr=successful["std_mae"],
        fmt="o",
    )
    plt.ylabel("Cross-validation MAE (kWh)")
    plt.xticks(rotation=30, ha="right")


def _time(frame: pd.DataFrame, title: str) -> None:
    working = frame.assign(timestamp=pd.to_datetime(frame["timestamp"], utc=True))
    grouped = working.groupby("timestamp")[["actual_ac_energy_kwh", "prediction"]].sum()
    plt.figure(figsize=(10, 4.5))
    plt.plot(grouped.index, grouped["actual_ac_energy_kwh"], label="Actual")
    plt.plot(grouped.index, grouped["prediction"], label="Predicted")
    plt.ylabel("Energy (kWh)")
    plt.title(title)
    plt.legend()


def _hist(frame: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.hist(frame["residual"], bins=min(30, max(5, len(frame) // 3)))
    plt.xlabel("Residual: actual - predicted (kWh)")


def _residual_scatter(frame: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.scatter(frame["prediction"], frame["residual"], s=12)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted (kWh)")
    plt.ylabel("Residual (kWh)")


def _hourly(frame: pd.DataFrame) -> None:
    working = frame.assign(hour=pd.to_datetime(frame["timestamp"], utc=True).dt.hour)
    profile = working.groupby("hour")["residual"].mean()
    plt.figure(figsize=(8, 4.5))
    plt.plot(profile.index.to_numpy(dtype=float), profile.to_numpy(dtype=float), marker="o")
    plt.xlabel("UTC hour")
    plt.ylabel("Mean residual (kWh)")


def _cumulative(frame: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4.5))
    plt.plot(frame["actual_ac_energy_kwh"].cumsum(), label="Actual")
    plt.plot(frame["prediction"].cumsum(), label="Predicted")
    plt.ylabel("Cumulative energy (kWh)")
    plt.legend()


def _importance(frame: pd.DataFrame, column: str) -> None:
    top = frame.sort_values(column).tail(20)
    plt.figure(figsize=(8, max(4.5, len(top) * 0.28)))
    plt.barh(top["feature"], top[column])
    plt.xlabel("XGBoost gain importance (not causal)")
