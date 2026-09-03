"""Public environment namespace for Terrarium.

Prefer importing environments from this package instead of the legacy `envs`
namespace.
"""

from __future__ import annotations

from terrarium.environments._lazy import lazy_namespace

__all__ = [
    "AbstractEnvironment",
    "MeetingSchedulingEnvironment",
    "PersonalAssistantEnvironment",
    "SmartGridEnvironment",
    "JiraTicketEnvironment",
    "HospitalEnvironment",
    "IsThisSeatTakenEnvironment",
]

_LAZY_ATTRS = {
    "AbstractEnvironment": ".abstract_environment:AbstractEnvironment",
    "MeetingSchedulingEnvironment": ".dcops:MeetingSchedulingEnvironment",
    "PersonalAssistantEnvironment": ".dcops:PersonalAssistantEnvironment",
    "SmartGridEnvironment": ".dcops:SmartGridEnvironment",
    "JiraTicketEnvironment": ".dcops:JiraTicketEnvironment",
    "HospitalEnvironment": ".dcops:HospitalEnvironment",
    "IsThisSeatTakenEnvironment": ".dcops:IsThisSeatTakenEnvironment",
}


__getattr__, __dir__ = lazy_namespace(__name__, _LAZY_ATTRS, globals(), __all__)
