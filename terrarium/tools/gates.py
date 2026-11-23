"""Per-phase tool gating driven by the ``environment.blocked_tools`` config.

``blocked_tools`` is a map of ``{phase_name: [tool names...]}``. The wildcard
``"*"`` blocks every environment tool in that phase, but never blackboard
tools; blackboard tools (e.g. ``post_message``) must be named explicitly.
"""

from collections.abc import Mapping
from typing import Any, Dict

WILDCARD = "*"

# Blackboard (communication) tools are never matched by the "*" wildcard.
BLACKBOARD_TOOL_NAMES = frozenset(
    {"post_message", "create_channel", "recall", "get_blackboard_events"}
)


def tool_blocked(env: Any, tool_name: str, phase: Any) -> bool:
    """Whether ``tool_name`` is blocked for ``phase`` on ``env``.

    Reads ``env.env_config.get("blocked_tools", {})``. Defaults to False when
    the environment or config is missing, so behavior is unchanged without
    ``blocked_tools`` configured.
    """
    if env is None or tool_name is None or phase is None:
        return False
    env_config = getattr(env, "env_config", {})
    if not isinstance(env_config, Mapping):
        return False
    blocked_tools = env_config.get("blocked_tools", {})
    if not isinstance(blocked_tools, Mapping):
        return False
    phase_blocked = blocked_tools.get(phase, [])
    if isinstance(phase_blocked, str):
        phase_blocked = [phase_blocked]
    if not isinstance(phase_blocked, (list, tuple, set, frozenset)):
        return False
    names = set(phase_blocked)
    if tool_name in names:
        return True
    return WILDCARD in names and tool_name not in BLACKBOARD_TOOL_NAMES


def blocked_response(tool_name: str, phase: Any) -> Dict[str, Any]:
    """The result returned when a tool call is blocked in a phase."""
    return {
        "blocked": True,
        "status": "blocked",
        "error": f"Tool '{tool_name}' is blocked in phase '{phase}'.",
    }
