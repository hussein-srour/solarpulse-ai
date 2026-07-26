"""Deterministic bounded XGBoost candidate generation and ranking."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from typing import Any

from solarpulse_ai.training.advanced_config import XGBoostSearchSpace


def generate_candidates(
    space: XGBoostSearchSpace, *, budget: int, random_seed: int
) -> list[dict[str, Any]]:
    """Return the same unique candidate sequence for the same inputs."""
    names = tuple(type(space).model_fields)
    combinations = list(itertools.product(*(getattr(space, name) for name in names)))
    random.Random(random_seed).shuffle(combinations)
    selected = combinations[: min(budget, len(combinations))]
    candidates: list[dict[str, Any]] = []
    for values in selected:
        parameters = dict(zip(names, values, strict=True))
        encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        candidates.append(
            {
                "candidate_id": f"xgb-{hashlib.sha256(encoded.encode()).hexdigest()[:10]}",
                "parameters": parameters,
            }
        )
    return candidates


def complexity_score(parameters: dict[str, Any]) -> float:
    """Provide a stable, documented tie-break proxy for tree capacity."""
    estimators = float(parameters["n_estimators"])
    depth = float(parameters["max_depth"])
    return float(estimators * (2.0**depth))


def rank_candidates(records: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    """Rank successful candidates and retain failed candidates after them."""
    metric_key = f"mean_{metric}"
    std_key = f"std_{metric}"
    successful = [record for record in records if record.get("status") == "succeeded"]
    if not successful:
        failures = "; ".join(str(record.get("failure_message")) for record in records)
        raise ValueError(f"all hyperparameter candidates failed: {failures}")
    ranked = sorted(
        successful,
        key=lambda record: (
            float(record[metric_key]),
            float(record[std_key]),
            complexity_score(dict(record["parameters"])),
            str(record["candidate_id"]),
        ),
    )
    output = [{"candidate_rank": rank, **record} for rank, record in enumerate(ranked, 1)]
    output.extend(
        {"candidate_rank": None, **record}
        for record in records
        if record.get("status") != "succeeded"
    )
    return output
