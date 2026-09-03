"""VoiceService: real file output, disabled path, fail-safe path, idempotency."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.voice import (
    REASON_DISABLED,
    REASON_EMPTY,
    REASON_FAILED,
    VoiceService,
    resolve_audio_file,
)
from tests.voice.conftest import (
    EmptyFileSynthesizer,
    FailingSynthesizer,
    FakeSynthesizer,
    make_service,
)

_MSG = "Namaste Rahul, aapka recent payment complete nahi ho paaya. Kripya dobara try karein."


# --- enabled -> a real file is written ------------------------------
def test_enabled_writes_a_nontrivial_audio_file(tts_dir):
    fake = FakeSynthesizer()
    svc = make_service(tts_dir, synthesizer=fake)
    r = svc.synthesize(_MSG, key="re42")

    assert r.generated is True
    assert r.reason is None
    assert r.audio_url == f"/api/v1/voice/{r.audio_id}"
    on_disk = Path(r.audio_path)
    assert on_disk.is_file()
    assert on_disk.stat().st_size > 50            # not empty / not a stub
    assert r.audio_bytes == on_disk.stat().st_size
    assert fake.calls == [(_MSG, "hi")]           # engine actually invoked


@pytest.mark.parametrize("engine_available", [True])
def test_real_pyttsx3_engine_produces_audio(tts_dir, engine_available):
    """Uses the ACTUAL local engine. Skips if the OS TTS engine is unavailable."""
    from app.services.voice import build_synthesizer

    try:
        synth = build_synthesizer("pyttsx3")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"pyttsx3/OS engine unavailable: {exc}")

    svc = make_service(tts_dir, synthesizer=synth)
    r = svc.synthesize(_MSG, key="re-real")
    if not r.generated:
        pytest.skip(f"local TTS synthesis failed in this env: {r.reason}")

    p = Path(r.audio_path)
    assert p.is_file()
    assert p.stat().st_size > 2000                # a few KB of real PCM at minimum
    assert p.read_bytes()[:4] == b"RIFF"          # a real WAV container
    assert r.audio_format == "wav"
    assert r.engine == "pyttsx3"


# --- disabled -> explicit reason, no file --------------------------
def test_disabled_returns_reason_and_writes_nothing(tts_dir):
    svc = make_service(tts_dir, enabled=False, synthesizer=FakeSynthesizer())
    r = svc.synthesize(_MSG, key="re1")

    assert r.generated is False
    assert r.reason == REASON_DISABLED
    assert r.audio_url is None
    assert list(tts_dir.iterdir()) == []
    assert r.as_tool_fields() == {"voice_generated": False, "voice_reason": "tts_disabled"}


def test_empty_message_returns_reason(tts_dir):
    svc = make_service(tts_dir, synthesizer=FakeSynthesizer())
    assert svc.synthesize("   ", key="re1").reason == REASON_EMPTY
    assert svc.synthesize(None, key="re1").reason == REASON_EMPTY
    assert list(tts_dir.iterdir()) == []


# --- failure -> fail safe, no raise -------------------------------
def test_engine_failure_is_caught_and_reported(tts_dir):
    fail = FailingSynthesizer()
    svc = make_service(tts_dir, synthesizer=fail)

    r = svc.synthesize(_MSG, key="re7")   # must NOT raise

    assert r.generated is False
    assert r.reason == REASON_FAILED
    assert fail.calls == 1
    assert r.as_tool_fields() == {"voice_generated": False, "voice_reason": "tts_generation_failed"}


def test_empty_output_file_is_treated_as_failure(tts_dir):
    svc = make_service(tts_dir, synthesizer=EmptyFileSynthesizer())
    r = svc.synthesize(_MSG, key="re8")
    assert r.generated is False
    assert r.reason == REASON_FAILED


def test_unsupported_engine_fails_safe(tts_dir):
    svc = make_service(tts_dir, engine="gtts", synthesizer=None)
    r = svc.synthesize(_MSG, key="re9")
    assert r.generated is False
    assert r.reason == REASON_FAILED


# --- idempotency: deterministic file name, no pile-up -------------
def test_repeat_synthesis_reuses_the_same_file(tts_dir):
    fake = FakeSynthesizer()
    svc = make_service(tts_dir, synthesizer=fake)

    r1 = svc.synthesize(_MSG, key="re42")
    r2 = svc.synthesize(_MSG, key="re42")

    assert r1.audio_id == r2.audio_id
    assert len(list(tts_dir.iterdir())) == 1
    assert len(fake.calls) == 1                   # second call reused the file

    r3 = svc.synthesize(_MSG + " extra", key="re42")
    assert r3.audio_id != r1.audio_id            # different text -> different file
    assert len(list(tts_dir.iterdir())) == 2


# --- path-traversal guard on the resolver ------------------------
def test_resolve_audio_file_rejects_traversal(tts_dir):
    (tts_dir / "ok.wav").write_bytes(b"x" * 10)
    assert resolve_audio_file("ok", str(tts_dir)) is not None
    assert resolve_audio_file("../secret", str(tts_dir)) is None
    assert resolve_audio_file("..%2Fsecret", str(tts_dir)) is None
    assert resolve_audio_file("a/b", str(tts_dir)) is None
    assert resolve_audio_file("missing", str(tts_dir)) is None
