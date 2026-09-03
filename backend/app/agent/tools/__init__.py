"""Tool registry + dispatch.

The registry is the single source of truth for what Gemini may request. The
loop looks a tool up by the name Gemini returned; there is no code path that
selects a tool on the agent's behalf.
"""
from __future__ import annotations

from app.agent.tools.action_tools import ACTION_TOOLS
from app.agent.tools.base import Tool, ToolContext
from app.agent.tools.context_tools import CONTEXT_TOOLS
from app.agent.tools.insight_tools import INSIGHT_TOOLS
from app.agent.tools.score_tools import SCORE_TOOLS

ALL_TOOLS: list[Tool] = [*CONTEXT_TOOLS, *SCORE_TOOLS, *INSIGHT_TOOLS, *ACTION_TOOLS]

TOOLS: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}

READ_ONLY_TOOL_NAMES = frozenset(
    t.name for t in ALL_TOOLS if not t.mutating and not t.terminal
)
TERMINAL_TOOL_NAMES = frozenset(t.name for t in ALL_TOOLS if t.terminal)

__all__ = [
    "Tool",
    "ToolContext",
    "TOOLS",
    "ALL_TOOLS",
    "READ_ONLY_TOOL_NAMES",
    "TERMINAL_TOOL_NAMES",
]
