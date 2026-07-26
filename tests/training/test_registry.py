"""Local model registry tests."""

from pathlib import Path

import pytest

from solarpulse_ai.training.persistence import sha256_file
from solarpulse_ai.training.registry import ModelRegistry


def _record(tmp_path: Path, version: str) -> dict[str, object]:
    artifact = tmp_path / f"{version}.joblib"
    artifact.write_bytes(version.encode())
    return {
        "model_version": version,
        "run_id": f"run-{version}",
        "model_family": "tuned_xgboost",
        "configuration_checksum": "a" * 64,
        "dataset_checksum": "b" * 64,
        "feature_manifest_checksum": "c" * 64,
        "training_timestamp": "2025-01-01T00:00:00+00:00",
        "forecast_horizon_hours": 24,
        "validation_metrics": {"mae_kwh": 1.0},
        "test_metrics": {"mae_kwh": 999.0},
        "artifact_path": str(artifact),
        "artifact_checksum": sha256_file(artifact),
    }


def test_candidate_promotion_archives_previous_champion(tmp_path: Path) -> None:
    """Validation-governed promotion keeps one champion and archives its predecessor."""
    registry = ModelRegistry(tmp_path / "registry.json")
    first = registry.register_candidate(_record(tmp_path, "model-one"))
    assert first["status"] == "candidate"
    assert registry.verify_artifact("model-one")
    registry.promote("model-one", selection_based_on_validation=True)
    registry.register_candidate(_record(tmp_path, "model-two"))
    registry.promote("model-two", selection_based_on_validation=True)
    assert registry.current_champion()["model_version"] == "model-two"  # type: ignore[index]
    statuses = {item["model_version"]: item["status"] for item in registry.list_models()}
    assert statuses == {"model-one": "archived", "model-two": "champion"}


def test_registry_rejects_test_driven_promotion_and_bad_artifact(tmp_path: Path) -> None:
    """Test metrics cannot drive promotion and checksum failures block registration."""
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register_candidate(_record(tmp_path, "model-one"))
    with pytest.raises(ValueError, match="validation"):
        registry.promote("model-one", selection_based_on_validation=False)
    bad = _record(tmp_path, "model-bad")
    bad["artifact_checksum"] = "0" * 64
    with pytest.raises(ValueError, match="verification failed"):
        registry.register_candidate(bad)
    assert not (tmp_path / ".registry.json.").exists()
