"""Public PersonalAssistant environment namespace for Terrarium."""

from __future__ import annotations

from terrarium.environments._lazy import lazy_namespace

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


__getattr__, __dir__ = lazy_namespace(__name__, _LAZY_ATTRS, globals(), __all__)
