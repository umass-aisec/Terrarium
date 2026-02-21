"""Public PersonalAssistant environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "PersonalAssistantEnvironment",
    "PersonalAssistantTools",
    "PersonalAssistantPrompts",
]

_LAZY_ATTRS = {
    "PersonalAssistantEnvironment": ".personal_assistant_env:PersonalAssistantEnvironment",
    "PersonalAssistantTools": ".personal_assistant_tools:PersonalAssistantTools",
    "PersonalAssistantPrompts": ".personal_assistant_prompts:PersonalAssistantPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
