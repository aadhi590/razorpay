"""Model artifact: everything inference needs, in one joblib file.

The artifact is self-contained -- ``load_artifact`` needs nothing from the rest
of the ML package except this module. Training NEVER happens at import/startup;
only :mod:`app.ml.training.train` (invoked by the CLI) writes artifacts.
"""
from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from app.ml.config import (
    ARTIFACT_DIR,
    FEATURE_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
    artifact_path,
    latest_artifact_path,
)

ARTIFACT_SCHEMA_VERSION = 1


class ModelUnavailable(RuntimeError):
    """Raised when no usable model artifact can be loaded."""


@dataclass
class ModelArtifact:
    # --- the fitted, calibrated sklearn estimator (Pipeline inside) -----
    estimator: object

    # --- identity / versioning ------------------------------------
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION
    feature_version: str = FEATURE_VERSION
    artifact_schema_version: int = ARTIFACT_SCHEMA_VERSION
    algorithm: str = ""
    selected_reason: str = ""

    # --- feature contract ----------------------------------------
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    all_features: list[str] = field(default_factory=list)
    label: str = "recovered"

    # --- provenance / evaluation --------------------------------
    dataset: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    data_quality: dict = field(default_factory=dict)
    action_coverage: dict = field(default_factory=dict)
    model_comparison: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    validation_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    interpretability: dict = field(default_factory=dict)
    feature_baseline: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sklearn_version: str = sklearn.__version__
    python_version: str = sys.version.split()[0]
    platform: str = platform.platform()

    @property
    def version(self) -> str:
        return self.model_version

    def summary(self) -> dict:
        """Metadata only (no fitted estimator) -- safe to log / return via API."""
        d = asdict(self)
        d.pop("estimator", None)
        return d


def save_artifact(
    artifact: ModelArtifact,
    path: Path | None = None,
    *,
    update_latest: bool | None = None,
) -> Path:
    path = Path(path) if path is not None else artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    # Maintain a stable "latest" pointer (a full copy -- symlinks are unreliable
    # on Windows). Only refresh it when writing into the real artifact dir, so a
    # test that trains into a tmp dir never clobbers the committed artifact.
    # ``update_latest`` overrides this heuristic.
    if update_latest is None:
        update_latest = path.parent.resolve() == ARTIFACT_DIR.resolve()
    if update_latest:
        joblib.dump(artifact, latest_artifact_path())
    return path


def load_artifact(path: Path | None = None) -> ModelArtifact:
    path = Path(path) if path is not None else latest_artifact_path()
    if not path.exists():
        raise ModelUnavailable(f"no model artifact at {path}")
    try:
        artifact = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - any load failure => unavailable
        raise ModelUnavailable(f"could not load artifact {path}: {exc}") from exc
    if not isinstance(artifact, ModelArtifact):
        raise ModelUnavailable(f"artifact at {path} is not a ModelArtifact")
    if artifact.artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ModelUnavailable(
            f"artifact schema {artifact.artifact_schema_version} != "
            f"{ARTIFACT_SCHEMA_VERSION}"
        )
    if not getattr(artifact, "all_features", None):
        raise ModelUnavailable("artifact is missing its feature contract")
    return artifact
