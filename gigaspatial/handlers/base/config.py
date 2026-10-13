"""Abstract configuration for dataset handlers (`BaseHandlerConfig`)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Union, Tuple, Iterable
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point, MultiPoint
from shapely.geometry.base import BaseGeometry
import multiprocessing
import logging

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
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
    """

    units_match_country: ClassVar[bool] = False

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

    def _cache_key(self, source, **kwargs):
        """Create a canonical cache key from the source and its ``crs`` option."""
        crs = kwargs.get("crs")
        return (*self._source_key(source), None if crs is None else str(crs))

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
            **kwargs: Additional parameters for source parsing.

        Returns:
            A list of relevant data unit identifiers.

        Note:
            Handlers customise resolution through ``get_search_key`` and
            ``get_relevant_data_units_by_key``; this method and its cache are
            not meant to be overridden.
        """
        key = self._cache_key(source, **kwargs)

        cached = None if force_recompute else self._units.get(key)
        if cached is not None:
            self.logger.debug(f"Using cached units for {key[0]}: {str(key[1])[:50]}...")
            return cached.units

        search_key = self.get_search_key(source, **kwargs)
        units = self.get_relevant_data_units_by_key(search_key, **kwargs)

        self._units.put(key, UnitResolution(units=units, search_key=search_key))
        return units

    def get_search_key(self, source, **kwargs) -> Any:
        """
        Turn a source into the value the dataset's unit lookup understands.

        Override for datasets indexed by something other than area, such as a
        country code, or return ``None`` for a single global dataset. The
        default is the search geometry; a key that is a geometry must be the
        search geometry, since the cache reuses it for cropping.

        Args:
            source: Input source (country, geometry, points, DataFrame or GDF).
            **kwargs: Additional parameters (e.g., ``crs`` of the source).

        Returns:
            The search key passed to ``get_relevant_data_units_by_key``.
        """
        return self.extract_search_geometry(source, **kwargs)

    @abstractmethod
    def get_relevant_data_units_by_key(self, search_key: Any, **kwargs) -> Any:
        """
        Identify data units associated with a search key.

        Args:
            search_key: The value returned by ``get_search_key``.
            **kwargs: Additional filtering parameters.

        Returns:
            A collection of relevant data unit identifiers.
        """
        pass

    @abstractmethod
    def get_data_unit_path(self, unit: Any, **kwargs) -> list:
        """
        Resolve a data unit identifier to its file path(s).

        Args:
            unit: The data unit identifier (e.g., a filename or tile ID).
            **kwargs: Additional resolution context.

        Returns:
            A list of paths corresponding to the unit.
        """
        pass

    def get_data_unit_paths(self, units: Union[Iterable[Any]], **kwargs) -> list:
        """
        Resolve multiple data unit identifiers to their corresponding file paths.

        Args:
            units: Iterable of data unit identifiers.
            **kwargs: Additional resolution context.

        Returns:
            A flat list of all resolved file paths.
        """
        if not isinstance(units, Iterable):
            units = [units]

        if not units:
            return []

        return [self.get_data_unit_path(unit=unit, **kwargs) for unit in units]

    def extract_search_geometry(self, source, **kwargs) -> BaseGeometry:
        """
        Build the search geometry of a source, in ``self.crs``.

        The input coordinates are read in the GeoDataFrame's own CRS, else in
        the ``crs`` option, else in EPSG:4326, and reprojected to ``self.crs``.
        Country sources use the country's admin boundary.

        Args:
            source: Country code or name, GeoDataFrame, shapely geometry,
                DataFrame with coordinate columns, or iterable of
                ``(lat, lon)`` pairs or ``Point`` objects.
            **kwargs: ``crs`` is the CRS of a source that carries none.
                Country sources forward the remaining keys to
                ``AdminBoundaries.create``.

        Returns:
            A single shapely geometry in ``self.crs``.

        Raises:
            ValueError: If the source has no geometry (e.g. a path or a filter
                dict).
        """
        source_crs = kwargs.pop("crs", None) or "EPSG:4326"
        kind = classify_source(source)

        if kind is SourceKind.COUNTRY:
            from gigaspatial.handlers.boundaries import AdminBoundaries

            geometry = AdminBoundaries.create(
                country_code=source, **kwargs
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
        """
        key = self._cache_key(source, **kwargs)
        geometry = self._units.get_geometry(key)
        if geometry is None:
            geometry = self.extract_search_geometry(source, **kwargs)
            self._units.put_geometry(key, geometry)
        return geometry

    def get_cached_search_geometry(self, source, **kwargs) -> Optional[BaseGeometry]:
        """Return the cached search geometry of a source, or None."""
        return self._units.get_geometry(self._cache_key(source, **kwargs))

    def clear_unit_cache(self):
        """Clear cached units."""
        self._units.clear()
        self.logger.debug("Unit cache cleared")
