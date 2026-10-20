"""Lazy attribute loading for package ``__init__`` modules (PEP 562)."""

import importlib
import sys
from typing import Callable, Dict, Iterable, List, Tuple


def attach(
    package_name: str,
    exports: Dict[str, Iterable[str]],
    submodules: Iterable[str] = (),
) -> Tuple[Callable[[str], object], Callable[[], List[str]], List[str]]:
    """Build ``__getattr__``, ``__dir__`` and ``__all__`` for a package.

    Args:
        package_name: The package's ``__name__``.
        exports: Maps a defining module to the public names it provides.
            A name is imported from its module the first time it is accessed.
        submodules: Subpackages reachable as attributes, for example
            ``gigaspatial.processing``, imported on first access.

    Returns:
        The package's ``__getattr__``, ``__dir__`` and ``__all__``.
    """
    name_to_module = {
        name: module for module, names in exports.items() for name in names
    }
    submodules = set(submodules)

    def __getattr__(name: str) -> object:
        if name in name_to_module:
            value = getattr(importlib.import_module(name_to_module[name]), name)
        elif name in submodules:
            value = importlib.import_module(f"{package_name}.{name}")
        else:
            raise AttributeError(
                f"module {package_name!r} has no attribute {name!r}"
            )
        setattr(sys.modules[package_name], name, value)
        return value

    def __dir__() -> List[str]:
        return sorted(
            set(vars(sys.modules[package_name])) | set(name_to_module) | submodules
        )

    return __getattr__, __dir__, list(name_to_module)
