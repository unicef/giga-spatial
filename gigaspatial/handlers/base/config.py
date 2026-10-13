"""Abstract configuration for dataset handlers (`BaseHandlerConfig`)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Type, Union, Tuple, Iterable
import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Point, MultiPoint
from shapely.geometry.base import BaseGeometry
import multiprocessing
import logging

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.base.options import (
    LegacyKwargs,
    ResolveOptions,
    call_with_options,
    sort_options,
)
from gigaspatial.handlers.base.resolution import UnitCache, UnitResolution
from gigaspatial.handlers.base.source import SourceKind, classify_source


@dataclass
class BaseHandlerConfig(ABC):
    """
    Abstract base class for handler configurations.

    Standardizes common fields used across all handlers, such as file paths,
    worker counts, and data store abstractions.

    A source is resolved in two steps. ``get_search_key`` turns it into the
    value the dataset's own lookup understands (by default the search
    geometry), and ``get_relevant_data_units_by_key`` turns that key into data
    units. The search geometry, the area used for cropping, is always built
    by ``extract_search_geometry`` in ``crs``.

    Attributes:
        base_path: Root directory for the dataset's files.
        n_workers: Number of parallel workers for processing or downloading.
        data_store: Abstraction for file-system access (local or cloud).
        logger: Logger instance for tracking handler operations.
        units_match_country: True when the dataset is published per country,
            so data loaded for a country source already covers exactly that
            country and is not cropped.
        options_type: The `ResolveOptions` subclass this config's hooks take,
            or None for a config whose hooks still take raw keyword arguments.
    """

    units_match_country: ClassVar[bool] = False
    options_type: ClassVar[Optional[Type[ResolveOptions]]] = None

    base_path: Path = None
    n_workers: int = multiprocessing.cpu_count()
    data_store: DataStore = field(default_factory=LocalDataStore)
    logger: logging.Logger = field(default=None, repr=False)

    def __post_init__(self):
        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)

        self._units = UnitCache()

    @property
    def crs(self) -> str:
        """The default CRS for this configuration's geometries (default: EPSG:4326)."""
        return "EPSG:4326"

    def _cache_key(self, source, options):
        """Create a canonical cache key from the source and its resolve options."""
        if isinstance(options, LegacyKwargs):
            crs = options.get("crs")
            return (*self._source_key(source), None if crs is None else str(crs))
        return (*self._source_key(source), options)

    @staticmethod
    def _source_key(source):
        kind = classify_source(source)
        if kind in (SourceKind.COUNTRY, SourceKind.PATHS):
            return (kind.value, str(source))
        if kind is SourceKind.GEOMETRY:
            return ("geometry", source.wkt)
        if kind is SourceKind.GEODATAFRAME:
            return ("geometry", str(source.geometry.union_all().wkt))
        if kind is SourceKind.DATAFRAME:
            from gigaspatial.processing.geo import detect_coordinate_columns

            try:
                lat_col, lon_col = detect_coordinate_columns(source)
                pt_str = tuple(
                    zip(source[lat_col].tolist(), source[lon_col].tolist())
                )
                return ("dataframe_points", pt_str)
            except (ValueError, TypeError):
                pass
            return ("dataframe", str(id(source)))
        if kind is SourceKind.POINTS:
            pt_str = tuple(
                (p.x, p.y) if isinstance(p, Point) else tuple(p) for p in source
            )
            return ("points", pt_str)
        return ("other", str(source))

    def get_relevant_data_units(
        self,
        source: Union[
            str,
            List[Union[Tuple[float, float], Point]],
            BaseGeometry,
            gpd.GeoDataFrame,
        ],
        force_recompute: bool = False,
        **kwargs,
    ):
        """
        Retrieve data unit identifiers relevant to a specific source.

        Standardizes the resolution of a source (e.g., country code or geometry)
        into a list of data units (e.g., specific files or tiles needing to be
        loaded).

        Args:
            source: Input source identifier (ISO country code, geometry, point list, or GDF).
            force_recompute: If True, ignores cached units and recomputes.
            **kwargs: Resolve options (see ``options_type``).

        Returns:
            A list of relevant data unit identifiers.

        Raises:
            TypeError: If a keyword is not a resolve option.

        Note:
            Handlers customise resolution through ``get_search_key`` and
            ``get_relevant_data_units_by_key``; this method and its cache are
            not meant to be overridden.
        """
        (options,) = sort_options(kwargs, self, entry="get_relevant_data_units")
        return self._resolve_units(source, options, force_recompute)

    def _resolve_units(self, source, options, force_recompute: bool = False):
        key = self._cache_key(source, options)

        cached = None if force_recompute else self._units.get(key)
        if cached is not None:
            self.logger.debug(f"Using cached units for {key[0]}: {str(key[1])[:50]}...")
            return cached.units

        search_key = call_with_options(self.get_search_key, source, options=options)
        units = call_with_options(
            self.get_relevant_data_units_by_key, search_key, options=options
        )

        self._units.put(key, UnitResolution(units=units, search_key=search_key))
        return units

    def get_search_key(self, source, options=None, **kwargs) -> Any:
        """
        Turn a source into the value the dataset's unit lookup understands.

        Override for datasets indexed by something other than area, such as a
        country code, or return ``None`` for a single global dataset. The
        default is the search geometry; a key that is a geometry must be the
        search geometry, since the cache reuses it for cropping.

        Args:
            source: Input source (country, geometry, points, DataFrame or GDF).
            options: The resolve options.
            **kwargs: Raw options, for configs without an ``options_type``.

        Returns:
            The search key passed to ``get_relevant_data_units_by_key``.
        """
        return self.extract_search_geometry(source, options, **kwargs)

    @abstractmethod
    def get_relevant_data_units_by_key(self, search_key: Any, options) -> Any:
        """
        Identify data units associated with a search key.

        Configs without an ``options_type`` receive raw keyword arguments
        instead of ``options``.

        Args:
            search_key: The value returned by ``get_search_key``.
            options: The resolve options.

        Returns:
            A collection of relevant data unit identifiers.
        """
        pass

    def get_data_unit_path(self, unit: Any, options=None, **kwargs) -> Path:
        """
        Resolve a data unit to its single stored file.

        Implement this for datasets that store one file per unit, or override
        ``get_stored_paths`` instead.

        Args:
            unit: The data unit identifier (e.g., a filename or tile ID).
            options: The resolve options.
            **kwargs: Raw options, for configs without an ``options_type``.

        Returns:
            The path of the unit's stored file in the data store.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_data_unit_path or get_stored_paths"
        )

    def get_stored_paths(self, unit: Any, options=None, **kwargs) -> List[Path]:
        """
        Return the files a data unit is stored as in the data store.

        These are the files that end up in storage after download (e.g. the
        ``.tif`` extracted from a downloaded zip), not the remote resource.
        The default is the single path from ``get_data_unit_path``.

        Args:
            unit: The data unit identifier.
            options: The resolve options.
            **kwargs: Raw options, for configs without an ``options_type``.

        Returns:
            The unit's stored file paths.
        """
        if options is None:
            options = LegacyKwargs(kwargs)
        return [call_with_options(self.get_data_unit_path, unit, options=options)]

    def is_unit_available(self, unit: Any, paths: List[Path]) -> bool:
        """
        Return True if a data unit is already stored and needs no download.

        Args:
            unit: The data unit identifier.
            paths: The unit's stored paths, from ``get_stored_paths``.

        Returns:
            True if every stored path exists in the data store.
        """
        return all(self.data_store.file_exists(str(path)) for path in paths)

    def get_data_unit_paths(self, units: Union[Iterable[Any]], **kwargs) -> list:
        """
        Return the stored file paths of several data units, flattened.

        Args:
            units: Iterable (or DataFrame) of data unit identifiers.
            **kwargs: Resolve options (see ``options_type``).

        Returns:
            A flat list of all stored file paths.

        Raises:
            TypeError: If a keyword is not a resolve option.
        """
        (options,) = sort_options(kwargs, self, entry="get_data_unit_paths")
        return self._stored_paths_of(units, options)

    def _stored_paths_of(self, units, options) -> list:
        if isinstance(units, pd.DataFrame):
            units = units.to_dict("records")
        elif isinstance(units, (str, dict)) or not isinstance(units, Iterable):
            units = [units]

        return [path for unit in units for path in self._stored_paths(unit, options)]

    def _stored_paths(self, unit, options) -> List[Path]:
        return call_with_options(self.get_stored_paths, unit, options=options)

    def extract_search_geometry(self, source, options=None, **kwargs) -> BaseGeometry:
        """
        Build the search geometry of a source, in ``self.crs``.

        The input coordinates are read in the GeoDataFrame's own CRS, else in
        the ``crs`` option, else in EPSG:4326, and reprojected to ``self.crs``.
        Country sources use the country's admin boundary.

        Args:
            source: Country code or name, GeoDataFrame, shapely geometry,
                DataFrame with coordinate columns, or iterable of
                ``(lat, lon)`` pairs or ``Point`` objects.
            options: The resolve options; ``crs`` is the CRS of a source that
                carries none.
            **kwargs: Raw options, for configs without an ``options_type``.
                Country sources forward the keys other than ``crs`` to
                ``AdminBoundaries.create``.

        Returns:
            A single shapely geometry in ``self.crs``.

        Raises:
            ValueError: If the source has no geometry (e.g. a path or a filter
                dict).
        """
        if options is None or isinstance(options, LegacyKwargs):
            boundary_kwargs = {**(options or {}), **kwargs}
            source_crs = boundary_kwargs.pop("crs", None)
        else:
            boundary_kwargs = {}
            source_crs = options.crs
        source_crs = source_crs or "EPSG:4326"
        kind = classify_source(source)

        if kind is SourceKind.COUNTRY:
            from gigaspatial.handlers.boundaries import AdminBoundaries

            geometry = AdminBoundaries.create(
                country_code=source, **boundary_kwargs
            ).to_geoms()[0]
            source_crs = "EPSG:4326"
        elif kind is SourceKind.GEODATAFRAME:
            if source.crs is not None:
                source_crs = source.crs
            geometry = source.geometry.union_all()
        elif kind is SourceKind.GEOMETRY:
            geometry = source
        elif kind is SourceKind.DATAFRAME:
            from gigaspatial.processing.geo import convert_to_geodataframe

            geometry = convert_to_geodataframe(source, crs=source_crs).geometry.union_all()
        elif kind is SourceKind.POINTS:
            geometry = MultiPoint(
                [p if isinstance(p, Point) else Point(p[1], p[0]) for p in source]
            )
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        if CRS.from_user_input(source_crs) != CRS.from_user_input(self.crs):
            geometry = (
                gpd.GeoSeries([geometry], crs=source_crs).to_crs(self.crs).iloc[0]
            )
        return geometry

    def get_search_geometry(self, source, **kwargs) -> BaseGeometry:
        """
        Return the search geometry of a source, computing it at most once per
        source and caching it.

        Args:
            source: Input source.
            **kwargs: Resolve options (see ``options_type``).
        """
        (options,) = sort_options(kwargs, self, entry="get_search_geometry")
        return self._search_geometry(source, options)

    def _search_geometry(self, source, options) -> BaseGeometry:
        key = self._cache_key(source, options)
        geometry = self._units.get_geometry(key)
        if geometry is None:
            geometry = self.extract_search_geometry(source, options)
            self._units.put_geometry(key, geometry)
        return geometry

    def get_cached_search_geometry(self, source, **kwargs) -> Optional[BaseGeometry]:
        """Return the cached search geometry of a source, or None."""
        (options,) = sort_options(kwargs, self, entry="get_cached_search_geometry")
        return self._units.get_geometry(self._cache_key(source, options))

    def clear_unit_cache(self):
        """Clear cached units."""
        self._units.clear()
        self.logger.debug("Unit cache cleared")
