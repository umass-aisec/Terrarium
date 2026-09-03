"""Shared lazy-attribute machinery for environment namespace packages.

Each environment package exposes its classes lazily so that importing the
package does not pull in heavy optional dependencies until an attribute is
actually used. The mechanism was copy-pasted into every package; this is the
single copy they all delegate to.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Tuple


def lazy_namespace(
    package: str,
    lazy_attrs: Dict[str, str],
    namespace: Dict[str, Any],
    all_names: List[str],
) -> Tuple[Callable[[str], Any], Callable[[], List[str]]]:
    """Build the ``__getattr__``/``__dir__`` pair for a lazy namespace package.

    ``lazy_attrs`` maps an exported name to ``"<relative.module>:<attribute>"``.
    Resolved values are cached into ``namespace`` (the caller's ``globals()``).
    """

    def __getattr__(name: str) -> Any:
        target = lazy_attrs.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")

        module_path, _, attr = target.partition(":")
        module = importlib.import_module(module_path, package)
        value = getattr(module, attr)
        namespace[name] = value
        return value

    def __dir__() -> List[str]:
        return sorted(set(list(namespace.keys()) + all_names))

    return __getattr__, __dir__
