from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Type


class EnvironmentToolsNotFoundError(LookupError):
    pass


_DISCOVERED = False
_TOOL_CLASS_BY_ENV_NAME: Dict[str, Type[Any]] = {}
_TOOL_CLASS_BY_NAME: Dict[str, Type[Any]] = {}


def _project_root() -> Path:
    # terrarium/tools/environment.py -> terrarium/tools -> terrarium -> project root
    return Path(__file__).resolve().parents[2]


def _iter_tools_modules(*, root: Path) -> list[str]:
    modules: list[str] = []
    for path in root.glob("terrarium/environments/**/*_tools.py"):
        if path.name.startswith("."):
            continue
        rel = path.relative_to(root).with_suffix("")
        modules.append(".".join(rel.parts))
    return sorted(set(modules))


def _discover_tools() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return

    root = _project_root()
    for module_path in _iter_tools_modules(root=root):
        try:
            module = importlib.import_module(module_path)
        except Exception:
            # Best-effort discovery: ignore modules that fail to import so one
            # optional dependency doesn't break unrelated environments.
            continue

        for name, obj in vars(module).items():
            if not inspect.isclass(obj):
                continue
            # Only include classes defined in this module (skip re-exports).
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            if not name.endswith("Tools"):
                continue

            _TOOL_CLASS_BY_NAME[name] = obj
            env_name = f"{name[:-5]}Environment"
            _TOOL_CLASS_BY_ENV_NAME.setdefault(env_name, obj)

    _DISCOVERED = True


def get_environment_tools_class(environment_name: str) -> Type[Any]:
    """Resolve the Tools class for a given environment name.

    Convention: `{Foo}Environment` -> `{Foo}Tools` in a `terrarium.environments`
    `*_tools.py` module.
    """
    _discover_tools()
    if not environment_name:
        raise EnvironmentToolsNotFoundError("environment_name is required")

    direct = _TOOL_CLASS_BY_ENV_NAME.get(environment_name)
    if direct is not None:
        return direct

    if environment_name.endswith("Environment"):
        tools_name = f"{environment_name[:-11]}Tools"
        by_name = _TOOL_CLASS_BY_NAME.get(tools_name)
        if by_name is not None:
            return by_name

    raise EnvironmentToolsNotFoundError(
        f"No Tools class found for environment {environment_name!r}. "
        f"Known environments: {', '.join(sorted(_TOOL_CLASS_BY_ENV_NAME)) or '(none)'}"
    )


def instantiate_environment_tools(
    environment_name: str, blackboard_manager: Any, environment: Any = None
) -> Any:
    """Instantiate the tools handler for an environment."""
    cls = get_environment_tools_class(environment_name)
    try:
        return cls(blackboard_manager, environment=environment)
    except TypeError:
        # Backward compatibility for tool classes that still use the old ctor.
        return cls(blackboard_manager)
