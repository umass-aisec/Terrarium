"""DCOP environments module - contains MeetingScheduling, PersonalAssistant, SmartGrid, JiraTicket, and Hospital.

This package depends on the external CoLLAB submodule for instance generation
and scoring. CoLLAB is not a Python package at its root, so we add
`external/CoLLAB` to sys.path to make `problem_layer.*` imports available.
"""

from pathlib import Path
import sys
import importlib
from typing import Any

_COLLAB_ROOT = Path(__file__).resolve().parents[2] / "external" / "CoLLAB"
if _COLLAB_ROOT.exists():
    collab_str = str(_COLLAB_ROOT)
    if collab_str not in sys.path:
        sys.path.insert(0, collab_str)

__all__ = [
    "MeetingSchedulingEnvironment",
    "PersonalAssistantEnvironment",
    "SmartGridEnvironment",
    "JiraTicketEnvironment",
    "HospitalEnvironment",
]

_LAZY_ATTRS = {
    "MeetingSchedulingEnvironment": ".meeting_scheduling.meeting_scheduling_env:MeetingSchedulingEnvironment",
    "PersonalAssistantEnvironment": ".personal_assistant.personal_assistant_env:PersonalAssistantEnvironment",
    "SmartGridEnvironment": ".smart_grid.smart_grid_env:SmartGridEnvironment",
    "JiraTicketEnvironment": ".jira_ticket.jira_ticket_env:JiraTicketEnvironment",
    "HospitalEnvironment": ".hospital.hospital_env:HospitalEnvironment",
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
