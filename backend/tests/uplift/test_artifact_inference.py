"""Phase 9: artifact save/load + inference (model object and HTTP endpoint)."""
from __future__ import annotations

import asyncio
import json

import pytest

from app.ml.uplift.models.artifact import (
    UpliftArtifact,
    UpliftModelUnavailable,
    load_uplift_artifact,
    save_uplift_artifact,
)

pytestmark = pytest.mark.needs_data


def test_artifact_roundtrip(fitted_t_learner, tmp_path):
    art = UpliftArtifact(
        estimator=fitted_t_learner,
        learner_type="t_learner",
        base_algorithm="logreg",
        dataset={"n_rows": 1},
    )
    path = save_uplift_artifact(art, tmp_path / "uplift_recovery_uplift_v1.joblib")
    assert path.exists()
    loaded = load_uplift_artifact(path)
    assert isinstance(loaded, UpliftArtifact)
    assert loaded.learner_type == "t_learner"
    assert loaded.all_features
    # the estimator survived the pickle round-trip and still scores
    assert hasattr(loaded.estimator, "predict_baseline")


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(UpliftModelUnavailable):
        load_uplift_artifact(tmp_path / "nope.joblib")


def test_corrupt_artifact_raises(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not joblib")
    with pytest.raises(UpliftModelUnavailable):
        load_uplift_artifact(bad)


def test_predict_for_event_shape_and_economics(uplift_model, conn):
    from sqlalchemy import text

    reid = conn.execute(
        text("select recovery_event_id from interventions order by id limit 1")
    ).scalar()
    result = uplift_model.predict_for_event(conn, int(reid))
    d = result.as_dict()
    assert 0.0 <= d["baseline_probability"] <= 1.0
    assert len(d["actions"]) == 4
    amount = d["amount_paise"]
    for a in d["actions"]:
        # incremental revenue == uplift * amount ; net == incr - cost
        assert a["incremental_expected_revenue_paise"] == pytest.approx(
            a["uplift"] * amount, rel=1e-4, abs=1.0
        )
        assert a["net_incremental_value_paise"] == pytest.approx(
            a["incremental_expected_revenue_paise"] - a["cost_paise"], abs=1.0
        )
    # ranked by net incremental value, best first
    nets = [a["net_incremental_value_paise"] for a in d["actions"]]
    assert nets == sorted(nets, reverse=True)
    assert [a["rank"] for a in d["actions"]] == [1, 2, 3, 4]


def _call(method: str, path: str):
    from app.main import app

    p, _, q = path.partition("?")
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": p, "raw_path": p.encode(),
        "query_string": q.encode(), "root_path": "",
        "headers": [(b"host", b"t")], "client": ("t", 1), "server": ("t", 80),
    }
    inbox = [{"type": "http.request", "body": b"", "more_body": False}]
    out = {"body": b""}

    async def recv():
        return inbox.pop(0)

    async def send(m):
        if m["type"] == "http.response.start":
            out["status"] = m["status"]
        elif m["type"] == "http.response.body":
            out["body"] += m.get("body", b"")

    asyncio.run(app(scope, recv, send))
    return out["status"], json.loads(out["body"] or b"null")


def test_model_info_endpoint(uplift_model):
    sc, body = _call("GET", "/api/v1/uplift/model")
    assert sc == 200
    assert body["available"] is True
    assert body["learner_type"] in {"s_learner", "t_learner"}
    assert body["synthetic_benchmark"] is True


def test_uplift_scores_endpoint(uplift_model, open_treatment_event):
    tid = open_treatment_event["treatment_id"]
    sc, body = _call(
        "GET", f"/api/v1/uplift/recovery-events/{tid}/uplift-scores"
    )
    assert sc == 200
    assert body["available"] is True
    assert body["baseline_probability"] is not None
    assert len(body["actions"]) == 4
    assert "SYNTHETIC" in body["note"].upper()
    for a in body["actions"]:
        assert set(a) >= {
            "action", "treatment_probability", "uplift",
            "incremental_expected_revenue_paise", "net_incremental_value_paise", "rank",
        }


def test_uplift_scores_404_for_missing_event(uplift_model):
    sc, _ = _call("GET", "/api/v1/uplift/recovery-events/999999999/uplift-scores")
    assert sc == 404
