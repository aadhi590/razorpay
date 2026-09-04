"""GroqProvider + the LLM_PROVIDER selection factory.

No live Groq call here -- the HTTP layer is monkeypatched. The single real
live call is done separately (see the stage report), not in the suite.
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from app.agent.config import AgentConfig
from app.agent.providers import make_provider
from app.agent.providers.base import (
    AuthError,
    MalformedResponseError,
    ProviderUnavailable,
    RateLimitedError,
    ToolSpec,
    TransientError,
)
from app.agent.providers.gemini import GeminiProvider
from app.agent.providers.groq import GroqProvider


def _cfg(**kw) -> AgentConfig:
    base = dict(
        api_key="gsk_TEST_do_not_log", model="llama-3.3-70b-versatile",
        provider="groq", max_turns=6, timeout_seconds=5.0,
        max_transient_retries=2, max_rate_limit_retries=0,
    )
    base.update(kw)
    return AgentConfig(**base)


def _httperror(code: int, body: dict, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.groq.com/openai/v1/chat/completions",
        code=code, msg="err", hdrs=headers or {},
        fp=io.BytesIO(json.dumps(body).encode()),
    )


# --- factory: the fallback switch ------------------------------------

def test_factory_returns_gemini_by_default(monkeypatch):
    from app.agent import config as agent_config

    monkeypatch.setattr(agent_config.settings, "LLM_PROVIDER", "gemini", raising=False)
    monkeypatch.setattr(agent_config.settings, "GEMINI_API_KEY", "AIza_test", raising=False)
    prov = make_provider()
    assert isinstance(prov, GeminiProvider)


def test_factory_returns_groq_when_selected(monkeypatch):
    from app.agent import config as agent_config

    monkeypatch.setattr(agent_config.settings, "LLM_PROVIDER", "groq", raising=False)
    monkeypatch.setattr(agent_config.settings, "GROQ_API_KEY", "gsk_test", raising=False)
    prov = make_provider()
    assert isinstance(prov, GroqProvider)
    assert prov.model == agent_config.settings.GROQ_MODEL


def test_factory_unknown_provider_falls_back_to_gemini(monkeypatch):
    from app.agent import config as agent_config

    monkeypatch.setattr(agent_config.settings, "LLM_PROVIDER", "bananas", raising=False)
    monkeypatch.setattr(agent_config.settings, "GEMINI_API_KEY", "AIza_test", raising=False)
    assert isinstance(make_provider(), GeminiProvider)


def test_groq_provider_unavailable_without_key():
    with pytest.raises(ProviderUnavailable):
        GroqProvider(_cfg(api_key=None))


# --- request translation -------------------------------------------

def test_to_messages_translates_the_internal_conversation():
    conv = [
        {"type": "user_text", "text": "start"},
        {"type": "tool_call", "name": "get_recovery_event_context", "arguments": {}},
        {"type": "tool_result", "name": "get_recovery_event_context",
         "payload": {"amount_paise": 49900}},
        {"type": "tool_call", "name": "get_action_scores", "arguments": {"k": 1}},
        {"type": "tool_result", "name": "get_action_scores", "payload": {"scores": []}},
    ]
    msgs = GroqProvider._to_messages("SYS", conv)
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "start"}
    # assistant tool_call carries an id that the following tool message echoes
    assert msgs[2]["role"] == "assistant"
    call_id = msgs[2]["tool_calls"][0]["id"]
    assert msgs[3] == {"role": "tool", "tool_call_id": call_id,
                       "content": json.dumps({"amount_paise": 49900})}
    # second pair uses a distinct id
    assert msgs[4]["tool_calls"][0]["id"] != call_id
    assert msgs[5]["tool_call_id"] == msgs[4]["tool_calls"][0]["id"]
    assert json.loads(msgs[4]["tool_calls"][0]["function"]["arguments"]) == {"k": 1}


def test_generate_builds_openai_tools_and_forces_a_call(monkeypatch):
    prov = GroqProvider(_cfg())
    captured = {}

    def fake_post_with_retry(payload: bytes):
        captured["body"] = json.loads(payload)
        return {
            "choices": [{"message": {"tool_calls": [
                {"id": "call_abc", "type": "function", "function": {
                    "name": "get_recovery_event_context", "arguments": "{}"}}
            ]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }

    monkeypatch.setattr(prov, "_post_with_retry", fake_post_with_retry)
    turn = prov.generate(
        system_prompt="SYS", conversation=[{"type": "user_text", "text": "go"}],
        tools=[ToolSpec(name="get_recovery_event_context", description="d",
                        parameters={"type": "object", "properties": {}})],
    )
    assert captured["body"]["tool_choice"] == "required"
    assert captured["body"]["tools"][0]["type"] == "function"
    assert captured["body"]["tools"][0]["function"]["name"] == "get_recovery_event_context"
    assert turn.tool_call is not None
    assert turn.tool_call.name == "get_recovery_event_context"
    assert turn.tool_call.arguments == {}
    assert turn.total_tokens == 14


# --- response parsing ---------------------------------------------

def test_parse_turn_reads_a_tool_call():
    data = {
        "choices": [{"message": {"tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "execute_recovery_action",
                "arguments": '{"action_type": "whatsapp_nudge"}'}}
        ]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    turn = GroqProvider._parse_turn(data, 12)
    assert turn.tool_call.name == "execute_recovery_action"
    assert turn.tool_call.arguments == {"action_type": "whatsapp_nudge"}
    assert turn.raw_text is None


def test_parse_turn_handles_text_only_response_as_protocol_violation():
    data = {"choices": [{"message": {"content": "I think we should nudge them."}}]}
    turn = GroqProvider._parse_turn(data, 3)
    assert turn.tool_call is None
    assert "nudge" in turn.raw_text


def test_parse_turn_tolerates_bad_argument_json():
    data = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "stop_recovery", "arguments": "not json{"}}
    ]}}]}
    turn = GroqProvider._parse_turn(data, 1)
    assert turn.tool_call.name == "stop_recovery"
    assert turn.tool_call.arguments == {}


# --- error mapping: every fail-safe path must keep working ---------

@pytest.mark.parametrize(
    "code,body,expected",
    [
        (401, {"error": {"message": "Invalid API Key", "code": "invalid_api_key"}}, AuthError),
        (403, {"error": {"message": "forbidden"}}, AuthError),
        (429, {"error": {"message": "Rate limit reached"}}, RateLimitedError),
        (500, {"error": {"message": "internal"}}, TransientError),
        (502, {"error": {"message": "bad gateway"}}, TransientError),
        (503, {"error": {"message": "overloaded"}}, TransientError),
        (400, {"error": {"message": "tool schema invalid"}}, MalformedResponseError),
        (404, {"error": {"message": "model not found"}}, MalformedResponseError),
    ],
)
def test_http_errors_map_to_existing_categories(code, body, expected):
    with pytest.raises(expected):
        GroqProvider._raise_for_http_error(_httperror(code, body))


def test_429_carries_retry_after():
    with pytest.raises(RateLimitedError) as ei:
        GroqProvider._raise_for_http_error(
            _httperror(429, {"error": {"message": "slow down"}}, {"Retry-After": "2"})
        )
    assert ei.value.retry_after_seconds == 2.0


def test_timeout_becomes_transient(monkeypatch):
    prov = GroqProvider(_cfg())

    def boom(req, timeout):  # noqa: ARG001
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(TransientError):
        prov._post_once(b"{}")


def test_malformed_json_body_becomes_malformed_error(monkeypatch):
    prov = GroqProvider(_cfg())

    class _Resp:
        def read(self):
            return b"<html>not json</html>"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _Resp())
    with pytest.raises(MalformedResponseError):
        prov._post_once(b"{}")


def test_auth_error_message_has_no_key():
    with pytest.raises(AuthError) as ei:
        GroqProvider._raise_for_http_error(
            _httperror(401, {"error": {"message": "bad key"}})
        )
    assert "gsk_" not in str(ei.value)


# --- generate_json for the scheduler judgment --------------------

def test_generate_json_parses_a_structured_response(monkeypatch):
    prov = GroqProvider(_cfg())
    captured = {}

    def fake_post_once(payload: bytes):
        captured["body"] = json.loads(payload)
        return {"choices": [{"message": {
            "content": '{"decision": "skip_this_cycle", "reason": "quiet"}'
        }}]}

    monkeypatch.setattr(prov, "_post_once", fake_post_once)
    out = prov.generate_json(
        system_prompt="sys", user_prompt="history here",
        response_schema={"type": "object"},
    )
    assert out == {"decision": "skip_this_cycle", "reason": "quiet"}
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert "tools" not in captured["body"]


def test_generate_json_rejects_non_json_text(monkeypatch):
    prov = GroqProvider(_cfg())
    monkeypatch.setattr(
        prov, "_post_once",
        lambda payload: {"choices": [{"message": {"content": "sorry, no"}}]},
    )
    with pytest.raises(MalformedResponseError):
        prov.generate_json(system_prompt="s", user_prompt="u")
