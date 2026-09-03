"""Process-wide cached handle to the latest uplift artifact."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.ml.uplift.models.artifact import (
    UpliftArtifact,
    UpliftModelUnavailable,
    load_uplift_artifact,
)


@lru_cache(maxsize=1)
def get_default_uplift_artifact() -> UpliftArtifact | None:
    try:
        return load_uplift_artifact()
    except UpliftModelUnavailable:
        return None


def try_load_uplift_artifact(path: Path | None = None) -> UpliftArtifact | None:
    try:
        return load_uplift_artifact(path)
    except UpliftModelUnavailable:
        return None


def reset_default_uplift_artifact() -> None:
    get_default_uplift_artifact.cache_clear()


__all__ = [
    "get_default_uplift_artifact",
    "try_load_uplift_artifact",
    "reset_default_uplift_artifact",
]
