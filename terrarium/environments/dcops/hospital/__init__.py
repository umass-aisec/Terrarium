"""Public Hospital environment namespace for Terrarium."""

from __future__ import annotations

from terrarium.environments._lazy import lazy_namespace

__all__ = ["HospitalEnvironment", "HospitalTools", "HospitalPrompts"]

_LAZY_ATTRS = {
    "HospitalEnvironment": ".hospital_env:HospitalEnvironment",
    "HospitalTools": ".hospital_tools:HospitalTools",
    "HospitalPrompts": ".hospital_prompts:HospitalPrompts",
}


__getattr__, __dir__ = lazy_namespace(__name__, _LAZY_ATTRS, globals(), __all__)
