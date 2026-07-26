"""Atomic local champion/challenger model registry."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

from solarpulse_ai.training.experiment import SAFE_IDENTIFIER
from solarpulse_ai.training.persistence import sha256_file

STATUSES = ("candidate", "champion", "archived", "rejected")
OBJECTIVE = "ac_energy_kwh_24h"


class ModelRegistry:
    """Read and atomically mutate trusted local model metadata."""

    def __init__(self, path: str | Path) -> None:
        """Bind the registry to one generated JSON path."""
        self.path = Path(path)

    def list_models(self) -> list[dict[str, Any]]:
        """Return registered models in insertion order."""
        return list(self._load()["models"])

    def current_champion(self, objective: str = OBJECTIVE) -> dict[str, Any] | None:
        """Return the single champion for an objective."""
        champions = [
            model
            for model in self.list_models()
            if model["objective"] == objective and model["status"] == "champion"
        ]
        if len(champions) > 1:
            raise ValueError(f"registry contains multiple champions for {objective}")
        return champions[0] if champions else None

    def verify_artifact(self, version: str) -> bool:
        """Verify a registered artifact exists and matches its checksum."""
        model = self._find(version)
        artifact = Path(str(model["artifact_path"]))
        return artifact.is_file() and sha256_file(artifact) == model["artifact_checksum"]

    def register_candidate(self, record: dict[str, Any]) -> dict[str, Any]:
        """Register a complete verified training artifact as a candidate."""
        required = {
            "model_version",
            "run_id",
            "model_family",
            "configuration_checksum",
            "dataset_checksum",
            "feature_manifest_checksum",
            "training_timestamp",
            "forecast_horizon_hours",
            "validation_metrics",
            "test_metrics",
            "artifact_path",
            "artifact_checksum",
        }
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"registry record is missing fields: {missing}")
        version = str(record["model_version"])
        if not SAFE_IDENTIFIER.fullmatch(version):
            raise ValueError("model_version contains unsafe path characters")
        payload = self._load()
        if any(model["model_version"] == version for model in payload["models"]):
            raise ValueError(f"model version already registered: {version}")
        candidate = {
            **record,
            "objective": str(record.get("objective", OBJECTIVE)),
            "status": "candidate",
            "previous_champion_version": (
                self.current_champion(str(record.get("objective", OBJECTIVE))) or {}
            ).get("model_version"),
        }
        artifact = Path(str(candidate["artifact_path"]))
        if not artifact.is_file() or sha256_file(artifact) != candidate["artifact_checksum"]:
            raise ValueError("model artifact verification failed")
        payload["models"].append(candidate)
        self._write(payload)
        return candidate

    def promote(self, version: str, *, selection_based_on_validation: bool) -> dict[str, Any]:
        """Promote a candidate only under the declared validation selection policy."""
        if not selection_based_on_validation:
            raise ValueError("promotion must be based on validation selection, never test metrics")
        payload = self._load()
        target = cast(
            dict[str, Any] | None,
            next(
                (model for model in payload["models"] if model["model_version"] == version),
                None,
            ),
        )
        if target is None:
            raise ValueError(f"unknown model version: {version}")
        if target["status"] not in {"candidate", "champion"}:
            raise ValueError("only a complete candidate can be promoted")
        artifact = Path(str(target["artifact_path"]))
        if not artifact.is_file() or sha256_file(artifact) != target["artifact_checksum"]:
            raise ValueError("model artifact verification failed")
        for model in payload["models"]:
            if model["objective"] == target["objective"] and model["status"] == "champion":
                model["status"] = "archived"
        target["status"] = "champion"
        self._write(payload)
        return target

    def archive(self, version: str) -> dict[str, Any]:
        """Archive one candidate or champion."""
        payload = self._load()
        target = cast(
            dict[str, Any] | None,
            next(
                (model for model in payload["models"] if model["model_version"] == version),
                None,
            ),
        )
        if target is None:
            raise ValueError(f"unknown model version: {version}")
        target["status"] = "archived"
        self._write(payload)
        return target

    def _find(self, version: str) -> dict[str, Any]:
        target = next(
            (model for model in self.list_models() if model["model_version"] == version), None
        )
        if target is None:
            raise ValueError(f"unknown model version: {version}")
        return target

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "models": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid model registry: {error}") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ValueError("model registry must contain a models list")
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
