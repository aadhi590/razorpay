"""Gemini provider -- Google Generative Language REST API, stdlib only.

Why stdlib ``urllib`` and not ``google-genai``: the project pins
``websockets==17.0.1`` (uvicorn) and the SDK requires ``websockets<17.0`` plus
~8 other transitive dependencies. The models.list probe already proved the raw
REST path works on this Python; function calling is fully supported over REST.

Security: the API key is sent **only** in the ``x-goog-api-key`` header, never
in the URL / query string, and is never logged or included in any exception
message.

Transport contract:
* one synchronous request per :meth:`generate` call -- no batching, no parallel,
  no background calls;
* ``toolConfig.functionCallingConfig.mode = "ANY"`` forces the model to answer
  every turn with a function call, so each turn is a structured decision;
* bounded retry: a small number of retries for 5xx / network, **at most one**
  retry for 429 (honouring ``Retry-After`` when small), and no retry at all for
  auth or malformed-response errors.
"""
from __future__ import annotations

import json
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

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider:
    """Concrete :class:`~app.agent.providers.base.LLMProvider` for Gemini."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_settings()
        if not self.config.has_key:
            raise ProviderUnavailable(
                "GEMINI_API_KEY is not configured; the agent cannot call Gemini"
            )
        self.model = self.config.model

    # -- public API ----------------------------------------------------
    def generate(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> ProviderTurn:
        body = self._build_body(system_prompt, conversation, tools)
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
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """ONE bounded, non-tool-calling request that returns the model's single
        structured JSON object.

        This is deliberately *not* :meth:`generate`: no ``tools``, no
        ``functionCallingConfig``, no multi-turn loop. It reuses this provider's
        config, endpoint, auth and typed error hierarchy, and makes exactly one
        HTTP attempt (:meth:`_post_once` -- no retry), so a caller that must fail
        safe just catches
        :class:`~app.agent.providers.base.ProviderError`. Used for a single small
        judgment (see the recovery scheduler's AI cycle judgment), never for the
        agent loop.
        """
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": (
                    max_output_tokens or self.config.max_output_tokens
                ),
                "candidateCount": 1,
                "responseMimeType": "application/json",
            },
        }
        if response_schema is not None:
            body["generationConfig"]["responseSchema"] = response_schema
        data = self._post_once(json.dumps(body).encode("utf-8"))
        return self._parse_json_object(data)

    @staticmethod
    def _parse_json_object(data: dict[str, Any]) -> dict[str, Any]:
        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise MalformedResponseError(
                "Gemini returned no candidates"
                + (f" (blocked: {block})" if block else "")
            )
        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
        text = "".join(
            str(p["text"]) for p in parts if isinstance(p, dict) and "text" in p
        ).strip()
        if not text:
            raise MalformedResponseError(
                "Gemini structured response contained no text"
            )
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise MalformedResponseError(
                f"Gemini structured response was not valid JSON: {exc}"
            ) from None
        if not isinstance(obj, dict):
            raise MalformedResponseError(
                "Gemini structured response was not a JSON object"
            )
        return obj

    # -- request construction --------------------------------------
    def _build_body(
        self,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        for entry in conversation:
            kind = entry["type"]
            if kind == "user_text":
                contents.append(
                    {"role": "user", "parts": [{"text": entry["text"]}]}
                )
            elif kind == "tool_call":
                part: dict[str, Any] = {
                    "functionCall": {
                        "name": entry["name"],
                        "args": entry.get("arguments", {}),
                    }
                }
                if entry.get("thought_signature"):
                    part["thoughtSignature"] = entry["thought_signature"]
                contents.append({"role": "model", "parts": [part]})
            elif kind == "tool_result":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": entry["name"],
                                    "response": entry["payload"],
                                }
                            }
                        ],
                    }
                )
            elif kind == "model_text":
                contents.append(
                    {"role": "model", "parts": [{"text": entry["text"]}]}
                )
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown conversation entry type {kind!r}")

        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                        for t in tools
                    ]
                }
            ],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
                "candidateCount": 1,
            },
        }

    # -- transport -------------------------------------------------
    def _post_with_retry(self, payload: bytes) -> dict[str, Any]:
        cfg = self.config
        transient_left = cfg.max_transient_retries
        rate_left = cfg.max_rate_limit_retries
        attempt = 0

        while True:
            attempt += 1
            try:
                return self._post_once(payload)
            except RateLimitedError as exc:
                if rate_left <= 0:
                    raise
                rate_left -= 1
                delay = exc.retry_after_seconds
                if delay is None or delay > cfg.backoff_max_seconds:
                    delay = min(
                        cfg.backoff_base_seconds * (2 ** (attempt - 1)),
                        cfg.backoff_max_seconds,
                    )
                time.sleep(delay)
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
        url = f"{_BASE}/{self.model}:generateContent"
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "content-type": "application/json",
                "x-goog-api-key": self.config.api_key or "",
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
            raise TransientError(f"network error contacting Gemini: {exc}") from None

        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MalformedResponseError(
                f"Gemini response was not valid JSON: {exc}"
            ) from None

    @staticmethod
    def _raise_for_http_error(exc: urllib.error.HTTPError) -> None:
        code = exc.code
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            message = str(detail.get("error", {}).get("message", ""))[:300]
        except Exception:  # noqa: BLE001
            message = ""
        retry_after = None
        headers = getattr(exc, "headers", None)
        if headers is not None:
            try:
                ra = headers.get("Retry-After")
                if ra is not None:
                    retry_after = float(ra)
            except (TypeError, ValueError):
                retry_after = None

        if code in (401, 403) or "API key" in message or "API_KEY" in message:
            raise AuthError(f"Gemini authentication failed (HTTP {code})") from None
        if code == 429:
            raise RateLimitedError(
                f"Gemini rate limit / quota exceeded (HTTP 429): {message}",
                retry_after_seconds=retry_after,
            ) from None
        if code in (500, 502, 503, 504):
            raise TransientError(
                f"Gemini transient server error (HTTP {code})"
            ) from None
        if code == 400 and (
            "API key" in message or "API_KEY_INVALID" in message
        ):
            raise AuthError("Gemini rejected the API key (HTTP 400)") from None
        raise MalformedResponseError(
            f"Gemini request failed (HTTP {code}): {message}"
        ) from None

    # -- response parsing -----------------------------------------
    @staticmethod
    def _parse_turn(data: dict[str, Any], latency_ms: int) -> ProviderTurn:
        usage = data.get("usageMetadata") or {}
        prompt_tokens = usage.get("promptTokenCount")
        output_tokens = usage.get("candidatesTokenCount")
        total_tokens = usage.get("totalTokenCount")

        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise MalformedResponseError(
                f"Gemini returned no candidates"
                + (f" (blocked: {block})" if block else "")
            )

        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
        texts: list[str] = []
        for part in parts:
            fc = part.get("functionCall")
            if fc and fc.get("name"):
                return ProviderTurn(
                    tool_call=ToolCall(
                        name=str(fc["name"]),
                        arguments=dict(fc.get("args") or {}),
                        thought_signature=part.get("thoughtSignature"),
                    ),
                    model="gemini",
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )
            if "text" in part:
                texts.append(str(part["text"]))

        # No function call -> protocol violation; the loop handles it.
        return ProviderTurn(
            tool_call=None,
            raw_text=" ".join(texts).strip() or "(empty response)",
            model="gemini",
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
