"""Public environment namespace for Terrarium.

Prefer importing environments from this package instead of the legacy `envs`
namespace.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AbstractEnvironment",
    "MeetingSchedulingEnvironment",
    "PersonalAssistantEnvironment",
    "SmartGridEnvironment",
    "JiraTicketEnvironment",
    "HospitalEnvironment",
]

_LAZY_ATTRS = {
    "AbstractEnvironment": ".abstract_environment:AbstractEnvironment",
    "MeetingSchedulingEnvironment": ".dcops:MeetingSchedulingEnvironment",
    "PersonalAssistantEnvironment": ".dcops:PersonalAssistantEnvironment",
    "SmartGridEnvironment": ".dcops:SmartGridEnvironment",
    "JiraTicketEnvironment": ".dcops:JiraTicketEnvironment",
    "HospitalEnvironment": ".dcops:HospitalEnvironment",
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
