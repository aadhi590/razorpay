"""Tool contract.

Each tool has a name, a description, a strict input schema and a documented
output schema. The model can *request* a tool; the application validates the
request (schema + guardrails), executes it, and returns the structured result
to the model. Tools are the only way the agent touches the world.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.agent.providers.base import ToolSpec
from app.agent.state import RecoveryAgentState


@dataclass
class ToolContext:
    db: Session
    state: RecoveryAgentState
    dry_run: bool
    # Optional injected Razorpay client (tests / a pre-built adapter). When
    # None, the recovery execution service builds one from config after its
    # own test-mode guard passes. Never used on the dry-run path.
    razorpay_client: Any = None
    # Hinglish TTS. When None, the tools treat voice as disabled. Never
    # placed a call / delivered to a customer -- it produces an audio file only.
    voice_service: Any = None


class Tool(abc.ABC):
    name: str
    description: str
    parameters: dict[str, Any]          # JSON schema for the arguments
    output_schema: dict[str, Any] = {}  # documented shape of the return payload
    mutating: bool = False              # guardrails run before a mutating tool
    terminal: bool = False              # calling it ends the agent loop

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    @abc.abstractmethod
    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        """Execute and return a compact JSON-serialisable payload."""


_OBJECT = "object"


def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": _OBJECT,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema
