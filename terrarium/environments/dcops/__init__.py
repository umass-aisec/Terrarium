"""DCOP environments module - MeetingScheduling, PersonalAssistant, SmartGrid, JiraTicket, and Hospital.

This package depends on an external CoLLAB checkout for instance generation
and scoring. CoLLAB is not a Python package at its root, so we add CoLLAB's
repo root to `sys.path` to make `problem_layer.*` imports available.

When running from a source checkout, the default location is `external/CoLLAB`.
When installed from PyPI, set `TERRARIUM_COLLAB_PATH=/path/to/CoLLAB`.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import os
import sys
from typing import Any

_COLLAB_ENV_VARS = ("TERRARIUM_COLLAB_PATH", "TERRARIUM_COLLAB_ROOT")


def _candidate_collab_roots() -> list[Path]:
    for var in _COLLAB_ENV_VARS:
        raw = os.getenv(var)
        if raw:
            return [Path(raw).expanduser()]

    return [
        # Source checkout default: repo-root/external/CoLLAB
        Path(__file__).resolve().parents[3] / "external" / "CoLLAB",
        # Common invocation pattern: run from a repo root with submodule
        Path.cwd() / "external" / "CoLLAB",
    ]


def _maybe_add_collab_to_syspath() -> None:
    for root in _candidate_collab_roots():
        try:
            root = root.resolve()
        except OSError:
            continue
        if not (root / "problem_layer").exists():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return


_maybe_add_collab_to_syspath()

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
