"""Checksums, software provenance, and verified model persistence."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
import sklearn


class Predictor(Protocol):
    """Common structural type for fitted sklearn and persistence predictors."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate one prediction per feature row."""
        ...


def sha256_file(path: str | Path) -> str:
    """Return the streaming SHA-256 checksum of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software_versions() -> dict[str, str]:
    """Record the model-runtime versions needed for reproducibility."""
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def git_commit() -> str | None:
    """Return the repository commit when available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_and_verify(
    model: Predictor,
    path: str | Path,
    verification_features: pd.DataFrame,
    expected_predictions: np.ndarray,
) -> str:
    """Persist, reload, and prove prediction equivalence before returning checksum."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    loaded = joblib.load(destination)
    actual = np.asarray(loaded.predict(verification_features), dtype=float)
    if not np.allclose(actual, expected_predictions, rtol=1e-12, atol=1e-12):
        raise ValueError("loaded-model predictions do not match pre-save predictions")
    return sha256_file(destination)
