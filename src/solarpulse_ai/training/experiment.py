"""Local reproducible experiment identifiers, checksums, and report layout."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from solarpulse_ai.training.reports import write_json

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def canonical_checksum(payload: object) -> str:
    """Hash a JSON-compatible value using canonical encoding."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_run_id(config: dict[str, Any], *, now: datetime | None = None) -> str:
    """Generate a readable UTC/config/model-family identifier."""
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{canonical_checksum(config)[:8]}-xgb"


def prepare_run_directory(root: str | Path, run_id: str, *, overwrite: bool = False) -> Path:
    """Validate the identifier and protect completed runs from silent overwrite."""
    if not SAFE_IDENTIFIER.fullmatch(run_id):
        raise ValueError("run_id contains unsafe path characters")
    directory = Path(root) / run_id
    if directory.exists() and not overwrite:
        raise ValueError(f"experiment run already exists: {run_id}")
    if directory.exists() and overwrite:
        marker = directory / ".solarpulse-experiment"
        if not marker.is_file():
            raise ValueError("refusing to overwrite a directory not created by SolarPulse AI")
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "charts").mkdir(exist_ok=True)
    (directory / ".solarpulse-experiment").write_text("generated\n", encoding="utf-8")
    return directory


def dataset_fingerprint(
    *, dataset_path: Path, dataset_checksum: str, frame_rows: int, split_counts: dict[str, int]
) -> dict[str, Any]:
    """Record a privacy-preserving dataset identity without input rows."""
    return {
        "file_name": dataset_path.name,
        "sha256": dataset_checksum,
        "row_count": frame_rows,
        "split_counts": split_counts,
    }


def write_artifact_checksums(directory: Path, summary: dict[str, Any]) -> dict[str, str]:
    """Hash generated regular files and add the mapping to the summary."""
    from solarpulse_ai.training.persistence import sha256_file

    checksums = {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {".solarpulse-experiment", "final_metrics.json"}
    }
    summary["artifact_checksums"] = checksums
    write_json(summary, directory / "final_metrics.json")
    return checksums
