"""Preprocessing, estimator, persistence, metric, and selection tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from solarpulse_ai.training.config import MODEL_IDS, TrainingConfig
from solarpulse_ai.training.contracts import TrainingContract
from solarpulse_ai.training.estimators import PersistencePredictor, build_model
from solarpulse_ai.training.metrics import calculate_metrics, daylight_metrics, metrics_by_site
from solarpulse_ai.training.predictions import postprocess, prediction_frame
from solarpulse_ai.training.selection import common_cohort, rank_models


def _contract(*, categorical: bool = False) -> TrainingContract:
    predictors = ("numeric", "category") if categorical else ("numeric",)
    return TrainingContract(
        predictors=predictors,
        numerical=("numeric",),
        categorical=("category",) if categorical else (),
        boolean=(),
        metadata=("split",),
        forecast_horizon_hours=24,
    )


@pytest.mark.parametrize("model_id", MODEL_IDS[1:])
def test_trainable_models_are_deterministic_and_finite(model_id: str) -> None:
    """All learned baselines fit train-only preprocessing and reproduce predictions."""
    features = pd.DataFrame({"numeric": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]})
    target = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    config = TrainingConfig(
        minimum_train_rows=1,
        minimum_validation_rows=1,
        minimum_test_rows=1,
        random_forest_estimators=10,
        histogram_boosting_max_iter=10,
    )
    first = build_model(model_id, _contract(), config).fit(features, target)
    second = build_model(model_id, _contract(), config).fit(features, target)
    assert isinstance(first, Pipeline)
    assert np.isfinite(first.predict(features)).all()
    np.testing.assert_allclose(first.predict(features), second.predict(features))


def test_preprocessing_statistics_use_training_only() -> None:
    """Validation/test mutation cannot affect fitted median or Ridge scaling."""
    train = pd.DataFrame({"numeric": [1.0, np.nan, 3.0]})
    config = TrainingConfig(minimum_train_rows=1, minimum_validation_rows=1, minimum_test_rows=1)
    model = build_model("ridge", _contract(), config).fit(train, [1.0, 2.0, 3.0])
    preprocessor = model.named_steps["preprocessor"]
    numeric = preprocessor.named_transformers_["numeric"]
    assert numeric.named_steps["imputer"].statistics_[0] == 2.0
    original = model.predict(pd.DataFrame({"numeric": [2.0]}))
    mutated = pd.DataFrame({"numeric": [2.0, 1_000_000.0]})
    assert model.predict(mutated)[0] == pytest.approx(original[0])


def test_unknown_categories_are_handled() -> None:
    """One-hot vocabularies are fitted on training and ignore future categories."""
    train = pd.DataFrame({"numeric": [1.0, 2.0], "category": ["clear", "clear"]})
    model = build_model(
        "ridge",
        _contract(categorical=True),
        TrainingConfig(minimum_train_rows=1, minimum_validation_rows=1, minimum_test_rows=1),
    ).fit(train, [1.0, 2.0])
    result = model.predict(pd.DataFrame({"numeric": [3.0], "category": ["storm"]}))
    assert np.isfinite(result).all()


def test_persistence_uses_exact_lag_not_target_or_row_position() -> None:
    """Persistence reads only the configured Phase 5 lag column."""
    frame = pd.DataFrame({"ac_energy_lag_24h": [9.0, 3.0], "ac_energy_kwh": [100.0, 200.0]})
    predictor = PersistencePredictor(24)
    np.testing.assert_array_equal(predictor.predict(frame), [9.0, 3.0])
    with pytest.raises(ValueError, match="requires"):
        predictor.predict(frame.drop(columns=["ac_energy_lag_24h"]))


def test_metrics_formulas_zero_wape_daylight_and_sites() -> None:
    """Core, capacity-normalised, daylight, per-site, and zero-energy cases are safe."""
    metrics = calculate_metrics(
        np.array([1.0, 3.0]),
        np.array([2.0, 1.0]),
        full_row_count=4,
        capacity_kwp=np.array([2.0, 2.0]),
    )
    assert metrics["mae_kwh"] == pytest.approx(1.5)
    assert metrics["rmse_kwh"] == pytest.approx((2.5) ** 0.5)
    assert metrics["median_absolute_error_kwh"] == pytest.approx(1.5)
    assert metrics["mean_bias_error_kwh"] == pytest.approx(-0.5)
    assert metrics["wape"] == pytest.approx(0.75)
    assert metrics["prediction_coverage_pct"] == 50.0
    assert metrics["capacity_normalised_mae_kwh_per_kwp"] == pytest.approx(0.75)
    assert calculate_metrics(np.zeros(2), np.ones(2))["wape"] is None
    rows = pd.DataFrame(
        {
            "site_id": ["a", "b"],
            "ac_energy_kwh": [1.0, 3.0],
            "is_daylight": [True, False],
        }
    )
    assert daylight_metrics(rows, np.array([2.0, 1.0]))["prediction_count"] == 1  # type: ignore[index]
    assert set(metrics_by_site(rows, np.array([2.0, 1.0]))["site_id"]) == {"a", "b"}


def test_common_cohort_and_selection_policy() -> None:
    """Every candidate shares a cohort and tie-breaking is deterministic."""
    rows = pd.DataFrame({"ac_energy_lag_24h": [1.0, np.nan, 3.0], "ac_energy_kwh": [1.0, 2.0, 3.0]})
    cohort, aligned, details = common_cohort(
        rows,
        {"ridge": np.array([1.0, 2.0, 3.0]), "dummy_mean": np.array([2.0, 2.0, 2.0])},
        "ac_energy_lag_24h",
    )
    assert len(cohort) == 2
    assert all(len(values) == 2 for values in aligned.values())
    assert details["excluded_reasons"] == {"persistence_unavailable": 1}
    ranked = rank_models(
        [
            {"model_identifier": "ridge", "mae_kwh": 1.0, "rmse_kwh": 2.0},
            {"model_identifier": "dummy_mean", "mae_kwh": 1.0, "rmse_kwh": 2.0},
            {"model_identifier": "random_forest", "mae_kwh": 0.5, "rmse_kwh": 5.0},
        ]
    )
    assert [item["model_identifier"] for item in ranked] == [
        "random_forest",
        "dummy_mean",
        "ridge",
    ]


def test_negative_postprocessing_and_prediction_schema() -> None:
    """Raw negatives stay counted, clipping is optional, and reports stay narrow."""
    clipped = postprocess(np.array([-1.0, 2.0]), clip_negative=True)
    assert clipped.raw_negative_count == 1
    np.testing.assert_array_equal(clipped.values, [0.0, 2.0])
    np.testing.assert_array_equal(postprocess(np.array([-1.0]), clip_negative=False).values, [-1.0])
    with pytest.raises(ValueError, match="non-finite"):
        postprocess(np.array([np.inf]), clip_negative=True)
    rows = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "site_id": ["a"],
            "split": ["test"],
            "ac_energy_kwh": [1.0],
            "predictor_secret": [999],
        }
    )
    report = prediction_frame(rows, np.array([2.0]), "ridge")
    assert "predictor_secret" not in report
    assert list(report) == [
        "timestamp",
        "site_id",
        "split",
        "actual_ac_energy_kwh",
        "prediction",
        "residual",
        "absolute_error",
        "squared_error",
        "model_identifier",
    ]
