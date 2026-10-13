"""
Typed options for dataset handlers, sorted once at the public entry points.

Each component declares the options it understands through its
``options_type``: the config takes `ResolveOptions`, the downloader
`DownloadOptions` and the reader `ReadOptions`, or subclasses of them. A public
entry point (``load_data``, ``reader.load``, ``downloader.download``, ...)
sorts its keyword arguments into one options object per component with
`sort_options` and raises ``TypeError`` for keys no component declares. Below
the entry point only options objects are passed around.

A component whose ``options_type`` is None has not been migrated yet. It gets
the raw keyword arguments as a `LegacyKwargs` mapping, and the handler accepts
any key. This compatibility path is removed once every handler is migrated.
"""
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class ResolveOptions:
    """
    Options that change which data units and stored paths a source resolves to.

    They are part of the unit cache key, so every value must be hashable.

    Attributes:
        crs: CRS of a source that carries none (a bare geometry, points or a
            DataFrame). Defaults to EPSG:4326.
    """

    crs: Any = None


@dataclass(frozen=True)
class DownloadOptions:
    """
    Options for fetching data units.

    Attributes:
        show_progress: If True, shows a progress bar over the units.
    """

    show_progress: bool = True


@dataclass(frozen=True)
class ReadOptions:
    """Options for loading stored files into memory."""


class LegacyKwargs(dict):
    """Raw keyword arguments for a component that declares no ``options_type``."""


def option_keys(component) -> Optional[FrozenSet[str]]:
    """Return the option names a component accepts, or None for any key."""
    options_type = getattr(component, "options_type", None)
    if options_type is None:
        return None
    return frozenset(f.name for f in fields(options_type))


def select_kwargs(kwargs: Dict[str, Any], *components) -> Dict[str, Any]:
    """Return the keyword arguments that at least one of the components accepts."""
    keysets = [option_keys(c) for c in components]
    if any(keys is None for keys in keysets):
        return dict(kwargs)
    accepted = frozenset().union(*keysets)
    return {k: v for k, v in kwargs.items() if k in accepted}


def sort_options(kwargs: Dict[str, Any], *components, entry: str) -> Tuple[Any, ...]:
    """
    Sort keyword arguments into one options object per component.

    A key goes to every component that declares it.

    Args:
        kwargs: Keyword arguments received by a public entry point.
        *components: The components the entry point drives.
        entry: Name of the entry point, used in the error message.

    Returns:
        A tuple with one options object per component, in order.

    Raises:
        TypeError: If a key is accepted by none of the components.
    """
    unknown = set(kwargs) - set(select_kwargs(kwargs, *components))
    if unknown:
        names = ", ".join(repr(k) for k in sorted(unknown))
        raise TypeError(f"{entry}() got unexpected keyword argument(s): {names}")
    return tuple(_build(c, kwargs) for c in components)


def _build(component, kwargs: Dict[str, Any]):
    options_type = getattr(component, "options_type", None)
    if options_type is None:
        return LegacyKwargs(kwargs)
    keys = option_keys(component)
    return options_type(**{k: v for k, v in kwargs.items() if k in keys})


def option_value(options, name: str, default: Any = None) -> Any:
    """Read one option from an options object or a `LegacyKwargs` mapping."""
    if isinstance(options, LegacyKwargs):
        return options.get(name, default)
    return getattr(options, name, default)


def call_with_options(fn: Callable, *args, options):
    """Call a component hook with its options, or with raw kwargs if unmigrated."""
    if isinstance(options, LegacyKwargs):
        return fn(*args, **options)
    return fn(*args, options)
