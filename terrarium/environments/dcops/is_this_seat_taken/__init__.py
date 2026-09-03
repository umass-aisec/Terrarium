"""Public IsThisSeatTaken environment namespace for Terrarium."""

from __future__ import annotations

from terrarium.environments._lazy import lazy_namespace

__all__ = [
    "IsThisSeatTakenEnvironment",
    "IsThisSeatTakenTools",
    "IsThisSeatTakenPrompts",
]

_LAZY_ATTRS = {
    "IsThisSeatTakenEnvironment": ".is_this_seat_taken_env:IsThisSeatTakenEnvironment",
    "IsThisSeatTakenTools": ".is_this_seat_taken_tools:IsThisSeatTakenTools",
    "IsThisSeatTakenPrompts": ".is_this_seat_taken_prompts:IsThisSeatTakenPrompts",
}


__getattr__, __dir__ = lazy_namespace(__name__, _LAZY_ATTRS, globals(), __all__)
