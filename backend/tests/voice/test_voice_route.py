"""GET /api/v1/voice/{audio_id} -- serves a generated audio file, 404 otherwise,
and rejects path traversal. Retrieval is NOT delivery to a customer."""
from __future__ import annotations

import asyncio

import pytest

from app.services.voice import VoiceConfig
from tests.voice.conftest import FakeSynthesizer, make_service


@pytest.fixture(autouse=True)
def _point_settings_at_tmp(monkeypatch, tts_dir):
    # the route reads the output dir from settings; redirect it without .env
    from app.services import voice as voice_mod
    monkeypatch.setattr(voice_mod.settings, "TTS_OUTPUT_DIR", str(tts_dir), raising=False)


def _raw_get(path: str):
    from app.main import app

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "", "headers": [(b"host", b"t")],
        "client": ("t", 1), "server": ("t", 80),
    }
    inbox = [{"type": "http.request", "body": b"", "more_body": False}]
    out = {"body": b"", "headers": []}

    async def recv():
        return inbox.pop(0)

    async def send(m):
        if m["type"] == "http.response.start":
            out["status"] = m["status"]
            out["headers"] = m.get("headers", [])
        elif m["type"] == "http.response.body":
            out["body"] += m.get("body", b"")

    asyncio.run(app(scope, recv, send))
    hdrs = {k.decode().lower(): v.decode() for k, v in out["headers"]}
    return out["status"], out["body"], hdrs


def test_serves_a_generated_audio_file(tts_dir):
    svc = make_service(tts_dir, synthesizer=FakeSynthesizer())
    r = svc.synthesize("Namaste, dobara try karein.", key="re100")
    assert r.generated

    sc, body, hdrs = _raw_get(r.audio_url)   # "/api/v1/voice/<id>"
    assert sc == 200
    assert hdrs["content-type"] == "audio/wav"
    assert body == (tts_dir / f"{r.audio_id}.wav").read_bytes()
    assert len(body) > 50


def test_unknown_audio_id_is_404(tts_dir):
    sc, _, _ = _raw_get("/api/v1/voice/re999-doesnotexist")
    assert sc == 404


def test_path_traversal_is_rejected(tts_dir):
    (tts_dir.parent / "secret.wav").write_bytes(b"top secret")
    # ".." can't appear in a single path segment via the router, but the
    # resolver guard is the real defence -- exercise an encoded attempt.
    sc, _, _ = _raw_get("/api/v1/voice/..%2F..%2Fsecret")
    assert sc in (400, 404)
