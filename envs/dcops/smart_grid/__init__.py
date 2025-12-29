"""SmartGrid environment module."""
import importlib
from typing import Any

__all__ = ["SmartGridEnvironment", "SmartGridTools", "SmartGridPrompts"]

_LAZY_ATTRS = {
    "SmartGridEnvironment": ".smart_grid_env:SmartGridEnvironment",
    "SmartGridTools": ".smartgrid_tools:SmartGridTools",
    "SmartGridPrompts": ".smartgrid_prompts:SmartGridPrompts",
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
