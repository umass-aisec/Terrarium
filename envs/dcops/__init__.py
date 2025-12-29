"""DCOP environments module - contains MeetingScheduling, PersonalAssistant, SmartGrid, and JiraTicket.

This package depends on the external CoLLAB submodule for instance generation
and scoring. CoLLAB is not a Python package at its root, so we add
`external/CoLLAB` to sys.path to make `problem_layer.*` imports available.
"""

from pathlib import Path
import sys

_COLLAB_ROOT = Path(__file__).resolve().parents[2] / "external" / "CoLLAB"
if _COLLAB_ROOT.exists():
    collab_str = str(_COLLAB_ROOT)
    if collab_str not in sys.path:
        sys.path.insert(0, collab_str)

from .meeting_scheduling import MeetingSchedulingEnvironment
from .personal_assistant import PersonalAssistantEnvironment
from .smart_grid import SmartGridEnvironment
from .jira_ticket import JiraTicketEnvironment

__all__ = [
    'MeetingSchedulingEnvironment',
    'PersonalAssistantEnvironment',
    'SmartGridEnvironment',
    'JiraTicketEnvironment',
]
