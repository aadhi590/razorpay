"""The TTS layer plugged into the existing agent tool flow.

Gemini is mocked (ScriptedProvider); the audio engine is a FakeSynthesizer.
Asserts the recovery flow is unaffected by voice outcomes and that the tool
result / run result distinguish generated vs not-generated with a reason.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from app.agent.runner import run_recovery_agent
from app.models.agent_events import AgentEvent
from app.services.voice import VoiceService
from tests.agent.conftest import _cleanup, _make_event
from tests.agent.fakes import ScriptedProvider
from tests.voice.conftest import FailingSynthesizer, FakeSynthesizer, make_service

_HINGLISH = "Namaste, aapka payment complete nahi hua. Kripya secure link se dobara try karein."

_STOP_WITH_MSG = ("stop_recovery", {
    "stop_reason": "action_executed_awaiting_outcome",
    "reasoning_summary": "done",
    "customer_message": _HINGLISH,
})


@pytest.fixture
def voice_db():
    from app.database import SessionLocal

    s = SessionLocal()
    _cleanup(s)  # also removes AgentEvent/AuditLog for the tagged test events
    try:
        yield s
    finally:
        _cleanup(s)
        s.rollback()
        s.close()


def _run(db, tts_dir, provider, *, synthesizer=None, enabled=True, dry_run=True):
    event_id = _make_event(db)
    svc = make_service(tts_dir, enabled=enabled, synthesizer=synthesizer)
    res = run_recovery_agent(
        db, event_id, dry_run=dry_run, provider=provider,
        voice_service=svc, persist=True,
    )
    return event_id, res


# --- generated: tool result + run result carry the audio artifact ---
def test_agent_run_generates_voice_from_stop_message(voice_db, tts_dir):
    fake = FakeSynthesizer()
    prov = ScriptedProvider([
        ("get_recovery_event_context", {}),
        _STOP_WITH_MSG,
    ])
    event_id, res = _run(voice_db, tts_dir, prov, synthesizer=fake)

    expected_id = VoiceService.audio_id_for(f"re{event_id}", _HINGLISH)
    assert res.voice_generated is True
    assert res.voice_reason is None
    assert res.audio_url == f"/api/v1/voice/{expected_id}"
    assert res.voice_engine == "fake"
    assert fake.calls == [(_HINGLISH, "hi")]
    assert (tts_dir / f"{expected_id}.wav").is_file()

    # persisted trace records the voice block (file only -- never a call)
    ev = voice_db.scalars(
        select(AgentEvent).where(
            AgentEvent.recovery_event_id == event_id,
            AgentEvent.event_type == "agent_recovery_run",
        )
    ).first()
    assert ev.input_context["voice"]["voice_generated"] is True
    assert "no phone call" in ev.input_context["voice"]["note"]


def test_execute_action_dry_run_also_generates_voice(voice_db, tts_dir):
    fake = FakeSynthesizer()
    prov = ScriptedProvider([
        ("get_recovery_event_context", {}),
        ("execute_recovery_action", {
            "action_type": "whatsapp_nudge",
            "customer_message": _HINGLISH,
            "reason": "best eligible",
        }),
        ("stop_recovery", {"stop_reason": "action_executed_awaiting_outcome",
                           "reasoning_summary": "done"}),
    ])
    event_id, res = _run(voice_db, tts_dir, prov, synthesizer=fake)
    exec_entry = [t for t in res.tool_trace if t.tool == "execute_recovery_action"][0]
    assert exec_entry.ok is True
    assert res.voice_generated is True
    assert len(list(tts_dir.iterdir())) == 1


# --- disabled: recovery flow unaffected ---------------------------
def test_tts_disabled_does_not_break_the_run(voice_db, tts_dir):
    prov = ScriptedProvider([("get_recovery_event_context", {}), _STOP_WITH_MSG])
    event_id, res = _run(voice_db, tts_dir, prov, enabled=False)

    assert res.status == "completed"
    assert res.stop_reason == "action_executed_awaiting_outcome"
    assert res.voice_generated is False
    assert res.voice_reason == "tts_disabled"
    assert res.audio_url is None
    assert list(tts_dir.iterdir()) == []


# --- failure: fail safe, run still completes ---------------------
def test_tts_failure_falls_back_to_text_only(voice_db, tts_dir):
    fail = FailingSynthesizer()
    prov = ScriptedProvider([("get_recovery_event_context", {}), _STOP_WITH_MSG])
    event_id, res = _run(voice_db, tts_dir, prov, synthesizer=fail)

    assert res.status == "completed"              # recovery flow continued
    assert res.customer_message == _HINGLISH      # text message intact
    assert res.voice_generated is False
    assert res.voice_reason == "tts_generation_failed"
    assert fail.calls == 1
