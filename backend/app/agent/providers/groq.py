"""Groq provider -- OpenAI-compatible chat/completions REST API, stdlib only.

Groq (https://console.groq.com) serves open models (Llama, Kimi, Qwen, ...) over
an OpenAI-compatible endpoint with native tool calling and very low latency.
This provider is a drop-in alternative to
:class:`~app.agent.providers.gemini.GeminiProvider` for the recovery agent's
tool-calling loop; selected via ``LLM_PROVIDER=groq``.

Same transport contract as the Gemini provider:
* one synchronous request per :meth:`generate` call -- no batching / parallel;
* ``tool_choice = "required"`` forces a tool call every turn, so each turn is a
  structured decision (mirrors Gemini's ``functionCallingConfig.mode = "ANY"``);
* bounded retry: a few retries for 5xx / network; for 429 (Groq's free tier caps
  tokens-per-minute) it waits out the stated ``Retry-After`` a few times, since
  those clear on their own -- unlike Gemini's daily-quota 429s; none for auth /
  malformed-response errors.

Security: the key is sent only in the ``Authorization`` header, never in the URL
or any log / exception message.
"""
from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from app.agent.config import AgentConfig
from app.agent.providers.base import (
    AuthError,
    MalformedResponseError,
    ProviderUnavailable,
    RateLimitedError,
    ToolSpec,
    TransientError,
)
from app.agent.schemas import ProviderTurn, ToolCall

_DEFAULT_BASE = "https://api.groq.com/openai/v1"
# Groq's edge (Cloudflare) 403s the default ``Python-urllib`` user-agent
# ("error code: 1010"). A named UA clears it.
_USER_AGENT = "reclaim-recovery-agent/1.0"


class GroqProvider:
    """Concrete :class:`~app.agent.providers.base.LLMProvider` for Groq."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_settings()
        if not self.config.has_key:
            raise ProviderUnavailable(
                "GROQ_API_KEY is not configured; the agent cannot call Groq"
            )
        self.model = self.config.model
        self._base = (self.config.base_url or _DEFAULT_BASE).rstrip("/")

    # -- public API --------------------------------------------------
    def generate(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> ProviderTurn:
        body = {
            "model": self.model,
            "messages": self._to_messages(system_prompt, conversation),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ],
            "tool_choice": "required",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "parallel_tool_calls": False,
        }
        payload = json.dumps(body).encode("utf-8")

        started = time.monotonic()
        data = self._post_with_retry(payload)
        latency_ms = int((time.monotonic() - started) * 1000)
        return self._parse_turn(data, latency_ms)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any] | None = None,  # noqa: ARG002 - json_object mode
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """ONE bounded, non-tool-calling request returning the model's single
        structured JSON object. Mirrors ``GeminiProvider.generate_json`` exactly:
        no tools, no retry (:meth:`_post_once`), same typed error hierarchy. Used
        only for the scheduler's AI cycle-judgment call.

        Groq JSON mode requires the literal word "json" somewhere in the prompt;
        the scheduler prompt already asks for "the JSON object", and a guard note
        is appended defensively.
        """
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{user_prompt}\n\nRespond with a single JSON object only.",
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": max_output_tokens or self.config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        data = self._post_once(json.dumps(body).encode("utf-8"))
        return self._parse_json_object(data)

    @staticmethod
    def _parse_json_object(data: dict[str, Any]) -> dict[str, Any]:
        choices = data.get("choices") or []
        if not choices:
            raise MalformedResponseError("Groq returned no choices")
        text = ((choices[0] or {}).get("message") or {}).get("content")
        if not text or not str(text).strip():
            raise MalformedResponseError("Groq structured response contained no text")
        try:
            obj = json.loads(str(text))
        except (json.JSONDecodeError, ValueError) as exc:
            raise MalformedResponseError(
                f"Groq structured response was not valid JSON: {exc}"
            ) from None
        if not isinstance(obj, dict):
            raise MalformedResponseError(
                "Groq structured response was not a JSON object"
            )
        return obj

    # -- request translation ---------------------------------------
    @staticmethod
    def _to_messages(
        system_prompt: str, conversation: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Translate the agent's internal conversation entries into OpenAI-style
        ``messages``. The loop always emits a ``tool_result`` immediately after
        its ``tool_call``, so pairing them by a running id is exact."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        n = 0
        last_call_id = "call_0"
        for entry in conversation:
            kind = entry["type"]
            if kind == "user_text":
                messages.append({"role": "user", "content": entry["text"]})
            elif kind == "model_text":
                messages.append({"role": "assistant", "content": entry["text"]})
            elif kind == "tool_call":
                n += 1
                last_call_id = f"call_{n}"
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": last_call_id,
                                "type": "function",
                                "function": {
                                    "name": entry["name"],
                                    "arguments": json.dumps(
                                        entry.get("arguments") or {}
                                    ),
                                },
                            }
                        ],
                    }
                )
            elif kind == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": last_call_id,
                        "content": json.dumps(entry["payload"], default=str),
                    }
                )
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown conversation entry type {kind!r}")
        return messages

    # Groq's free tier caps tokens-per-minute (~8k). Mid-run 429s are short and
    # self-clearing (Groq tells you exactly how long), unlike Gemini's
    # daily-quota 429s -- so wait out a bounded Retry-After and retry a few
    # times rather than failing the whole run.
    _RATE_LIMIT_RETRIES = 4
    _MAX_RETRY_WAIT_SECONDS = 25.0

    # -- transport -------------------------------------------------
    def _post_with_retry(self, payload: bytes) -> dict[str, Any]:
        cfg = self.config
        transient_left = cfg.max_transient_retries
        rate_left = max(cfg.max_rate_limit_retries, self._RATE_LIMIT_RETRIES)
        attempt = 0

        while True:
            attempt += 1
            try:
                return self._post_once(payload)
            except RateLimitedError as exc:
                if rate_left <= 0:
                    raise
                rate_left -= 1
                delay = exc.retry_after_seconds or 15.0
                # small safety margin, then a hard ceiling
                time.sleep(min(delay + 0.5, self._MAX_RETRY_WAIT_SECONDS))
            except TransientError:
                if transient_left <= 0:
                    raise
                transient_left -= 1
                time.sleep(
                    min(
                        cfg.backoff_base_seconds * (2 ** (attempt - 1)),
                        cfg.backoff_max_seconds,
                    )
                )

    def _post_once(self, payload: bytes) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self._base}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.config.api_key or ''}",
                "user-agent": _USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.config.timeout_seconds
            ) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            self._raise_for_http_error(exc)
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            raise TransientError(f"network error contacting Groq: {exc}") from None

        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MalformedResponseError(
                f"Groq response was not valid JSON: {exc}"
            ) from None

    @staticmethod
    def _raise_for_http_error(exc: urllib.error.HTTPError) -> None:
        code = exc.code
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            err = detail.get("error", {})
            message = str(
                err.get("message", "") if isinstance(err, dict) else err
            )[:300]
            err_code = str(err.get("code", "")) if isinstance(err, dict) else ""
        except Exception:  # noqa: BLE001
            message = ""
            err_code = ""

        retry_after = None
        headers = getattr(exc, "headers", None)
        if headers is not None:
            try:
                ra = headers.get("retry-after") or headers.get("Retry-After")
                if ra is not None:
                    retry_after = float(ra)
            except (TypeError, ValueError):
                retry_after = None
        if retry_after is None and message:
            # Groq states it in the body: "Please try again in 12.8175s."
            m = re.search(r"try again in ([\d.]+)\s*s", message)
            if m:
                try:
                    retry_after = float(m.group(1))
                except ValueError:
                    retry_after = None

        if code in (401, 403) or "invalid_api_key" in err_code or "api key" in message.lower():
            raise AuthError(f"Groq authentication failed (HTTP {code})") from None
        if code == 429:
            raise RateLimitedError(
                f"Groq rate limit / quota exceeded (HTTP 429): {message}",
                retry_after_seconds=retry_after,
            ) from None
        if code in (500, 502, 503, 504):
            raise TransientError(
                f"Groq transient server error (HTTP {code})"
            ) from None
        raise MalformedResponseError(
            f"Groq request failed (HTTP {code}): {message}"
        ) from None

    # -- response parsing -----------------------------------------
    @staticmethod
    def _parse_turn(data: dict[str, Any], latency_ms: int) -> ProviderTurn:
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        choices = data.get("choices") or []
        if not choices:
            raise MalformedResponseError("Groq returned no choices")
        message = (choices[0] or {}).get("message") or {}

        tool_calls = message.get("tool_calls") or []
        for tc in tool_calls:
            fn = (tc or {}).get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            raw_args = fn.get("arguments")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except (json.JSONDecodeError, ValueError):
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            return ProviderTurn(
                tool_call=ToolCall(name=str(name), arguments=dict(args)),
                model="groq",
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        # No tool call -> protocol violation; the loop handles it.
        text = message.get("content")
        return ProviderTurn(
            tool_call=None,
            raw_text=(str(text).strip() if text else None) or "(empty response)",
            model="groq",
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
