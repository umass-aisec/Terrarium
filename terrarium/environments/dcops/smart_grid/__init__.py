"""Public SmartGrid environment namespace for Terrarium."""

from __future__ import annotations

from terrarium.environments._lazy import lazy_namespace

__all__ = ["SmartGridEnvironment", "SmartGridTools", "SmartGridPrompts"]

_LAZY_ATTRS = {
    "SmartGridEnvironment": ".smart_grid_env:SmartGridEnvironment",
    "SmartGridTools": ".smartgrid_tools:SmartGridTools",
    "SmartGridPrompts": ".smartgrid_prompts:SmartGridPrompts",
}


__getattr__, __dir__ = lazy_namespace(__name__, _LAZY_ATTRS, globals(), __all__)
