"""Gemini-backed autonomous recovery agent.

A bounded, tool-driven agent loop layered *on top of* the existing recovery
system. Gemini decides, turn by turn, which tool to call next given only what it
has observed so far; the application validates and executes every tool. The
quantitative ML / uplift layer is exposed only as a tool -- Gemini never
replaces it, and never touches the database directly.

Nothing in this package changes the behaviour of the rules / ML / uplift
policies, the orchestrator, or their endpoints.
"""
from __future__ import annotations

from app.agent.agent import RecoveryAgent
from app.agent.config import AgentConfig
from app.agent.runner import AgentRunError, run_recovery_agent
from app.agent.schemas import AgentRunResult

__all__ = [
    "RecoveryAgent",
    "AgentConfig",
    "AgentRunResult",
    "AgentRunError",
    "run_recovery_agent",
]
