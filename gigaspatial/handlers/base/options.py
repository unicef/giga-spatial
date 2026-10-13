"""
Typed options for dataset handlers, sorted once at the public entry points.

Each component declares the options it understands through its
``options_type``: the config takes `ResolveOptions`, the downloader
`DownloadOptions` and the reader `ReadOptions`, or subclasses of them. A public
entry point (``load_data``, ``reader.load``, ``downloader.download``, ...)
sorts its keyword arguments into one options object per component with
`sort_options` and raises ``TypeError`` for keys no component declares. Below
the entry point only options objects are passed around.
"""
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple, Union


@dataclass(frozen=True)
class ResolveOptions:
    """
    Options that change which data units and stored paths a source resolves to.

    They are part of the unit cache key, so every value must be hashable.

    Attributes:
        crs: CRS of a source that carries none (a bare geometry, points or a
            DataFrame). Defaults to EPSG:4326.
        country_geom_path: File holding the boundary of a country source,
            used instead of the default admin boundaries.
        country_data_store: Data store the country boundary is read from.
    """

    crs: Any = None
    country_geom_path: Optional[Union[str, Path]] = None
    country_data_store: Any = None


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


@dataclass(frozen=True)
class RasterReadOptions(ReadOptions):
    """
    Read options of raster datasets.

    Attributes:
        merge_rasters: If True, all rasters are loaded into one `TifProcessor`.
    """

    merge_rasters: bool = False


def option_keys(component) -> FrozenSet[str]:
    """Return the option names a component accepts."""
    return frozenset(f.name for f in fields(component.options_type))


def select_kwargs(kwargs: Dict[str, Any], *components) -> Dict[str, Any]:
    """Return the keyword arguments that at least one of the components accepts."""
    accepted = frozenset().union(*(option_keys(c) for c in components))
    return {k: v for k, v in kwargs.items() if k in accepted}


def split_kwargs(
    kwargs: Dict[str, Any], *components
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split keyword arguments into those the components accept and the rest."""
    accepted = select_kwargs(kwargs, *components)
    return accepted, {k: v for k, v in kwargs.items() if k not in accepted}


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
    keys = option_keys(component)
    return component.options_type(**{k: v for k, v in kwargs.items() if k in keys})
