"""Abstract configuration for dataset handlers (`BaseHandlerConfig`)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Union, Tuple, Iterable
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.geometry.base import BaseGeometry
import multiprocessing
import logging

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore


@dataclass
class BaseHandlerConfig(ABC):
    """
    Abstract base class for handler configurations.

    Standardizes common fields used across all handlers, such as file paths,
    worker counts, and data store abstractions.

    Attributes:
        base_path: Root directory for the dataset's files.
        n_workers: Number of parallel workers for processing or downloading.
        data_store: Abstraction for file-system access (local or cloud).
        logger: Logger instance for tracking handler operations.
    """

    base_path: Path = None
    n_workers: int = multiprocessing.cpu_count()
    data_store: DataStore = field(default_factory=LocalDataStore)
    logger: logging.Logger = field(default=None, repr=False)

    def __post_init__(self):
        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)

        self._unit_cache = {}

    @property
    def crs(self) -> str:
        """The default CRS for this configuration's geometries (default: EPSG:4326)."""
        return "EPSG:4326"

    def _cache_key(self, source, **kwargs):
        """Create a canonical cache key from source."""
        if isinstance(source, str):
            return ("country", source)
        if isinstance(source, BaseGeometry):
            return ("geometry", source.wkt)
        if isinstance(source, gpd.GeoDataFrame):
            return ("geometry", str(source.geometry.unary_union.wkt))
        if isinstance(source, pd.DataFrame) and not isinstance(source, gpd.GeoDataFrame):
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
        if isinstance(source, Iterable) and all(
            isinstance(p, (Point, tuple)) for p in source
        ):
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
        """
        key = self._cache_key(source, **kwargs)

        # Check cache unless forced recompute
        if not force_recompute and key in self._unit_cache:
            self.logger.debug(f"Using cached units for {key[0]}: {key[1][:50]}...")
            units, _ = self._unit_cache[key]  # Unpack tuple, only return units
            return units

        # Convert source to geometry and compute units
        geometry = self.extract_search_geometry(source, **kwargs)
        units = self.get_relevant_data_units_by_geometry(geometry, **kwargs)

        # Cache both units and geometry as tuple
        self._unit_cache[key] = (units, geometry)
        return units

    @abstractmethod
    def get_relevant_data_units_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> Any:
        """
        Identify data units associated with a specific geometry.

        Args:
            geometry: The spatial filter to find relevant units for.
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

    def extract_search_geometry(self, source, **kwargs):
        """
        Extract a canonical geometry representation from various input types.

        Supports ISO-2 country codes, GeoDataFrames, Shapely geometries,
        and lists of point coordinates.

        Args:
            source: Input source to convert to geometry.
            **kwargs: Additional parameters (e.g., crs for reprojection).

        Returns:
            A unified Shapely geometry object representing the source.

        Raises:
            ValueError: If the source type is unsupported or missing CRS.
        """
        if isinstance(source, str):
            # Use the admin boundary as geometry
            from gigaspatial.handlers.boundaries import AdminBoundaries

            return AdminBoundaries.create(country_code=source, **kwargs).to_geoms()[0]
        elif isinstance(source, gpd.GeoDataFrame):
            if crs := kwargs.get("crs", None):

                if not source.crs:
                    raise ValueError(
                        "Cannot extract search geometry. Please set a crs on the source object first."
                    )

                if source.crs != crs:
                    source = source.to_crs(crs)

            return source.geometry.union_all()
        elif isinstance(
            source,
            BaseGeometry,
        ):
            return source
        elif isinstance(source, pd.DataFrame) and not isinstance(source, gpd.GeoDataFrame):
            from gigaspatial.processing.geo import convert_to_geodataframe

            crs = kwargs.get("crs", "EPSG:4326")
            gdf = convert_to_geodataframe(source, crs=crs)
            return gdf.geometry.union_all()
        elif (
            not isinstance(source, (str, pd.DataFrame))
            and isinstance(source, Iterable)
            and all(isinstance(p, (Point, Iterable)) for p in source)
        ):
            points = [p if isinstance(p, Point) else Point(p[1], p[0]) for p in source]
            return MultiPoint(points)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

    def get_cached_search_geometry(self, source):
        key = self._cache_key(source)
        result = self._unit_cache.get(key)
        if result:
            _, geometry = result
            return geometry
        return None

    def clear_unit_cache(self):
        """Clear cached units."""
        self._unit_cache.clear()
        self.logger.debug("Unit cache cleared")
