"""Resolved data units and the private cache that holds them."""
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional

from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class UnitResolution:
    """The data units a source resolved to, and the search key used."""

    units: List[Any]
    search_key: Any


class UnitCache:
    """
    Resolutions per source key, plus a lazily filled search geometry slot.

    The geometry slot is separate because most loads never crop: it is
    filled only when a search geometry is asked for, or when the search key
    already is one.
    """

    def __init__(self):
        self._resolutions: Dict[Hashable, UnitResolution] = {}
        self._geometries: Dict[Hashable, BaseGeometry] = {}

    def get(self, key: Hashable) -> Optional[UnitResolution]:
        return self._resolutions.get(key)

    def put(self, key: Hashable, resolution: UnitResolution) -> None:
        self._resolutions[key] = resolution
        self._geometries.pop(key, None)
        if isinstance(resolution.search_key, BaseGeometry):
            self._geometries[key] = resolution.search_key

    def get_geometry(self, key: Hashable) -> Optional[BaseGeometry]:
        return self._geometries.get(key)

    def put_geometry(self, key: Hashable, geometry: BaseGeometry) -> None:
        self._geometries[key] = geometry

    def clear(self) -> None:
        self._resolutions.clear()
        self._geometries.clear()
