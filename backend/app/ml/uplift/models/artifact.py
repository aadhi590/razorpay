"""The versioned uplift artifact: one joblib file with everything inference needs.

Self-contained -- ``load_uplift_artifact`` needs nothing from the rest of the
uplift package except this module and the estimator classes it pickles.
Training only ever happens in ``app.ml.uplift.training`` (CLI).
"""
from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from app.ml.uplift.config import (
    CONTROL_ACTION,
    TREATMENT_ACTIONS,
    UPLIFT_ARTIFACT_DIR,
    UPLIFT_FEATURE_VERSION,
    UPLIFT_MODEL_NAME,
    UPLIFT_MODEL_VERSION,
    uplift_artifact_path,
    uplift_latest_artifact_path,
)
from app.ml.uplift.estimators.base import BaseUpliftEstimator
from app.ml.uplift.features import ALL_FEATURES, BASELINE_FEATURES

UPLIFT_ARTIFACT_SCHEMA_VERSION = 1


class UpliftModelUnavailable(RuntimeError):
    """No usable uplift artifact could be loaded -- callers must fall back."""


@dataclass
class UpliftArtifact:
    # -- the fitted champion estimator (S- or T-learner, calibrated inside) --
    estimator: BaseUpliftEstimator

    # -- identity / versioning --------------------------------
    model_name: str = UPLIFT_MODEL_NAME
    model_version: str = UPLIFT_MODEL_VERSION
    feature_version: str = UPLIFT_FEATURE_VERSION
    artifact_schema_version: int = UPLIFT_ARTIFACT_SCHEMA_VERSION
    learner_type: str = ""
    base_algorithm: str = ""
    champion_reason: str = ""

    # -- feature contract ------------------------------------
    all_features: list[str] = field(default_factory=lambda: list(ALL_FEATURES))
    baseline_features: list[str] = field(default_factory=lambda: list(BASELINE_FEATURES))
    control_action: str = CONTROL_ACTION
    treatment_actions: list[str] = field(default_factory=lambda: list(TREATMENT_ACTIONS))

    # -- provenance / evaluation ----------------------------
    dataset: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    propensity_diagnostics: dict = field(default_factory=dict)
    integrity: dict = field(default_factory=dict)
    model_comparison: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    estimator_metadata: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

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
        d = asdict(self)
        d.pop("estimator", None)
        return d


def save_uplift_artifact(
    artifact: UpliftArtifact,
    path: Path | None = None,
    *,
    update_latest: bool | None = None,
) -> Path:
    path = Path(path) if path is not None else uplift_artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    # Only refresh the shared "latest" pointer when writing into the real
    # artifact dir (so a test writing to a tmp dir never clobbers a committed
    # artifact). Explicit ``update_latest`` overrides.
    if update_latest is None:
        update_latest = path.parent.resolve() == UPLIFT_ARTIFACT_DIR.resolve()
    if update_latest:
        joblib.dump(artifact, uplift_latest_artifact_path())
    return path


def load_uplift_artifact(path: Path | None = None) -> UpliftArtifact:
    path = Path(path) if path is not None else uplift_latest_artifact_path()
    if not path.exists():
        raise UpliftModelUnavailable(f"no uplift artifact at {path}")
    try:
        artifact = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise UpliftModelUnavailable(f"could not load uplift artifact {path}: {exc}") from exc
    if not isinstance(artifact, UpliftArtifact):
        raise UpliftModelUnavailable(f"artifact at {path} is not an UpliftArtifact")
    if artifact.artifact_schema_version != UPLIFT_ARTIFACT_SCHEMA_VERSION:
        raise UpliftModelUnavailable(
            f"uplift artifact schema {artifact.artifact_schema_version} != "
            f"{UPLIFT_ARTIFACT_SCHEMA_VERSION}"
        )
    if not getattr(artifact, "all_features", None):
        raise UpliftModelUnavailable("uplift artifact is missing its feature contract")
    return artifact
