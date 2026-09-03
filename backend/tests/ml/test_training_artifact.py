from __future__ import annotations

import pytest

from app.ml.models.artifact import (
    ModelArtifact,
    ModelUnavailable,
    load_artifact,
    save_artifact,
)

pytestmark = pytest.mark.needs_data


def test_fast_training_produces_loadable_artifact(fast_training_run):
    run = fast_training_run
    assert run.artifact_path.exists()
    assert run.report_path.exists()

    artifact = load_artifact(run.artifact_path)
    assert isinstance(artifact, ModelArtifact)
    assert artifact.all_features
    assert artifact.algorithm in {
        "LogisticRegression", "HistGradientBoostingClassifier", "RandomForestClassifier"
    }
    # provenance present
    for key in ("dataset", "split", "data_quality", "test_metrics", "calibration"):
        assert getattr(artifact, key)
    tm = artifact.test_metrics["classification"]
    assert 0.0 <= tm["roc_auc"] <= 1.0
    assert tm["n"] > 0
    # comparison covered >= 2 model families
    assert len(artifact.model_comparison) >= 2


def test_missing_artifact_raises_model_unavailable(tmp_path):
    with pytest.raises(ModelUnavailable):
        load_artifact(tmp_path / "nope.joblib")


def test_corrupt_artifact_raises_model_unavailable(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a joblib file")
    with pytest.raises(ModelUnavailable):
        load_artifact(bad)


def test_report_flags_synthetic_benchmark(fast_training_run):
    run = fast_training_run
    assert run.report["synthetic_benchmark"] is True
    assert "SYNTHETIC" in run.report["synthetic_warning"].upper()
