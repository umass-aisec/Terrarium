"""Public MeetingScheduling environment namespace for Terrarium."""

from __future__ import annotations

from terrarium.environments._lazy import lazy_namespace

__all__ = [
    "MeetingSchedulingEnvironment",
    "MeetingSchedulingTools",
    "MeetingSchedulingPrompts",
]

_LAZY_ATTRS = {
    "MeetingSchedulingEnvironment": ".meeting_scheduling_env:MeetingSchedulingEnvironment",
    "MeetingSchedulingTools": ".meeting_scheduling_tools:MeetingSchedulingTools",
    "MeetingSchedulingPrompts": ".meeting_scheduling_prompts:MeetingSchedulingPrompts",
}


__getattr__, __dir__ = lazy_namespace(__name__, _LAZY_ATTRS, globals(), __all__)
