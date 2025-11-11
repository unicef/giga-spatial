from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union, Tuple, Callable, Iterable
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.geometry.base import BaseGeometry
import multiprocessing
import logging

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.processing.tif_processor import TifProcessor


@dataclass
class BaseHandlerConfig(ABC):
    """
    Abstract base class for handler configuration objects.
    Provides standard fields for path, parallelism, data store, and logger.
    Extend this class for dataset-specific configuration.
    """

    base_path: Path = None
    n_workers: int = multiprocessing.cpu_count()
    data_store: DataStore = field(default_factory=LocalDataStore)
    logger: logging.Logger = field(default=None, repr=False)

    def __post_init__(self):
        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)

        self._unit_cache = {}

    def _cache_key(self, source, **kwargs):
        """Create a canonical cache key from source."""
        if isinstance(source, str):
            return ("country", source)
        if isinstance(source, BaseGeometry):
            return ("geometry", source.wkt)
        if isinstance(source, gpd.GeoDataFrame):
            return ("geometry", str(source.geometry.unary_union.wkt))
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
            str,  # country
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
        ],
        force_recompute: bool = False,
        **kwargs,
    ):
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
        Given a geometry, return a list of relevant data unit identifiers (e.g., tiles, files, resources).
        """
        pass

    @abstractmethod
    def get_data_unit_path(self, unit: Any, **kwargs) -> list:
        """
        Given a data unit identifier, return the corresponding file path.
        """
        pass

    def get_data_unit_paths(self, units: Union[Iterable[Any]], **kwargs) -> list:
        """
        Given data unit identifiers, return the corresponding file paths.
        """
        if not isinstance(units, Iterable):
            units = [units]

        if not units:
            return []

        return [self.get_data_unit_path(unit=unit, **kwargs) for unit in units]

    def extract_search_geometry(self, source, **kwargs):
        """General method to extract a canonical geometry from supported source types."""
        if isinstance(source, str):
            # Use the admin boundary as geometry
            from gigaspatial.handlers.boundaries import AdminBoundaries

            return (
                AdminBoundaries.create(country_code=source, **kwargs)
                .boundaries[0]
                .geometry
            )
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
        elif isinstance(source, Iterable) and all(
            isinstance(p, (Point, Iterable)) for p in source
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


class BaseHandlerDownloader(ABC):
    """
    Abstract base class for handler downloader classes.
    Standardizes config, data_store, and logger initialization.
    Extend this class for dataset-specific downloaders.
    """

    def __init__(
        self,
        config: Optional[BaseHandlerConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        if data_store:
            self.data_store = data_store
        elif config and hasattr(config, "data_store"):
            self.data_store = config.data_store
        else:
            self.data_store = LocalDataStore()

        self.logger = (
            logger
            or (getattr(config, "logger", None) if config else None)
            or global_config.get_logger(self.__class__.__name__)
        )

    @abstractmethod
    def download_data_unit(self, *args, **kwargs):
        """
        Abstract method to download data. Implement in subclasses.
        """
        pass

    @abstractmethod
    def download_data_units(self, *args, **kwargs):
        """
        Abstract method to download data. Implement in subclasses.
        """
        pass

    def download(self, source, **kwargs):
        """
        Given source download the data.
        """
        units = self.config.get_relevant_data_units(source, **kwargs)
        return self.download_data_units(units, **kwargs)


class BaseHandlerReader(ABC):
    """
    Abstract base class for handler reader classes.
    Provides common methods for resolving source paths and loading data.
    Supports resolving by country, points, geometry, GeoDataFrame, or explicit paths.
    Includes generic loader functions for raster and tabular data.
    """

    def __init__(
        self,
        config: Optional[BaseHandlerConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        if data_store:
            self.data_store = data_store
        elif config and hasattr(config, "data_store"):
            self.data_store = config.data_store
        else:
            self.data_store = LocalDataStore()

        self.logger = (
            logger
            or (getattr(config, "logger", None) if config else None)
            or global_config.get_logger(self.__class__.__name__)
        )

    def resolve_source_paths(
        self,
        source: Union[
            str,  # country code
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,  # path
            str,  # path
            List[Union[str, Path]],
        ],
        **kwargs,
    ) -> List[Union[str, Path]]:
        """
        Resolve source data paths based on the type of source input.

        Args:
            source: Can be a country code or name (str), list of points, geometry, GeoDataFrame, or explicit path(s)
            **kwargs: Additional parameters for path resolution

        Returns:
            List of resolved source paths
        """
        if (
            isinstance(source, Path)
            or (
                isinstance(source, (list, tuple, set))
                and all(isinstance(p, (str, Path)) for p in source)
            )
            or (isinstance(source, str) and "." in source)
        ):
            return self.resolve_by_paths(source)

        data_units = self.config.get_relevant_data_units(source, **kwargs)
        data_paths = self.config.get_data_unit_paths(data_units, **kwargs)

        self.logger.info(f"Resolved {len(data_paths)} paths!")
        return data_paths

    def resolve_by_paths(
        self, paths: Union[Path, str, List[Union[str, Path]]], **kwargs
    ) -> List[Union[str, Path]]:
        """
        Return explicit paths as a list.
        """
        if isinstance(paths, (str, Path)):
            return [paths]
        return list(paths)

    def _pre_load_hook(self, source_data_path, **kwargs) -> Any:
        """Hook called before loading data."""
        if isinstance(source_data_path, (Path, str)):
            source_data_path = [source_data_path]

        if not source_data_path:
            self.logger.warning("No paths found!")
            return []

        source_data_paths = [str(file_path) for file_path in source_data_path]

        self.logger.info(
            f"Pre-loading validation complete for {len(source_data_path)} files"
        )
        return source_data_paths

    def _post_load_hook(self, data, **kwargs) -> Any:
        """Hook called after loading data."""
        if isinstance(data, Iterable):
            if len(data) == 0:
                self.logger.warning("No data was loaded from the source files")
                return data

            self.logger.info(f"{len(data)} valid data records.")

        self.logger.info(f"Post-load processing complete.")

        return data

    def _check_file_exists(self, file_paths: List[Union[str, Path]]):
        """
        Check that all specified files exist in the data store.

        Args:
            file_paths (List[Union[str, Path]]): List of file paths to check.

        Raises:
            RuntimeError: If any file does not exist in the data store.
        """
        for file_path in file_paths:
            if not self.data_store.file_exists(str(file_path)):
                raise RuntimeError(
                    f"Source file does not exist in the data store: {file_path}"
                )

    def _load_raster_data(
        self,
        raster_paths: List[Union[str, Path]],
        merge_rasters: bool = False,
        **kwargs,
    ) -> Union[List[TifProcessor], TifProcessor]:
        """
        Load raster data from file paths.

        Args:
            raster_paths (List[Union[str, Path]]): List of file paths to raster files.
            merge_rasters (bool): If True, all rasters will be merged into a single TifProcessor.
                                  Defaults to False.

        Returns:
            Union[List[TifProcessor], TifProcessor]: List of TifProcessor objects or a single
                                                    TifProcessor if merge_rasters is True.
        """
        if merge_rasters and len(raster_paths) > 1:
            self.logger.info(
                f"Merging {len(raster_paths)} rasters into a single TifProcessor."
            )
            return TifProcessor(raster_paths, self.data_store, **kwargs)
        else:
            return [
                TifProcessor(data_path, self.data_store, **kwargs)
                for data_path in raster_paths
            ]

    def _load_tabular_data(
        self, file_paths: List[Union[str, Path]], read_function: Callable = read_dataset
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Load and concatenate tabular data from multiple files.

        Args:
            file_paths (List[Union[str, Path]]): List of file paths to load data from.
            read_function (Callable): Function to use for reading individual files.
                Defaults to read_dataset. Should accept (data_store, file_path) arguments.

        Returns:
            Union[pd.DataFrame, gpd.GeoDataFrame]: Concatenated data from all files.
                Returns empty DataFrame if no data is loaded.
        """
        all_data = []
        for file_path in file_paths:
            all_data.append(read_function(self.data_store, file_path))
        if not all_data:
            return pd.DataFrame()
        result = pd.concat(all_data, ignore_index=True)
        return result

    def crop_to_geometry(self, data, geometry, predicate="intersects", **kwargs):

        # Project geometry to the projection of the data if data has projection
        geom_crs = kwargs.get("crs", "EPSG:4326")
        if hasattr(data, "crs") and data.crs != geom_crs:
            geometry = (
                gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
                .to_crs(data.crs)
                .geometry[0]
            )

        # Tabular (GeoDataFrame) case
        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            if isinstance(data, pd.DataFrame):
                from gigaspatial.processing.geo import convert_to_geodataframe

                try:
                    data = convert_to_geodataframe(data, **kwargs)
                except:
                    return data

            # Clip to geometry
            return data[getattr(data.geometry, predicate)(geometry)]

        # Raster case
        if isinstance(data, TifProcessor):
            return data.clip_to_geometry(geometry=geometry, **kwargs)

        return data

    @abstractmethod
    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> Any:
        """
        Abstract method to load source data from paths.

        Args:
            source_data_path: List of source paths
            **kwargs: Additional parameters for data loading

        Returns:
            Loaded data (DataFrame, GeoDataFrame, etc.)
        """
        pass

    def load(
        self,
        source: Union[
            str,  # country
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,
            str,
            List[Union[str, Path]],
        ],
        crop_to_source: bool = False,
        **kwargs,
    ) -> Any:
        """
        Load data from the given source.

        Args:
            source: The data source (country code/name, points, geometry, paths, etc.).
            crop_to_source : bool, default False
                If True, crop loaded data to the exact source geometry
            **kwargs: Additional parameters to pass to the loading process.

        Returns:
            The loaded data. The type depends on the subclass implementation.
        """
        source_data_paths = self.resolve_source_paths(source, **kwargs)
        if not source_data_paths:
            self.logger.warning(
                "No source data paths resolved. There's no matching data to load!"
            )
            return None
        processed_paths = self._pre_load_hook(source_data_paths, **kwargs)
        if not processed_paths:
            self.logger.warning("No valid paths to load data from.")
            return None

        loaded_data = self.load_from_paths(processed_paths, **kwargs)
        loaded_data = self._post_load_hook(loaded_data, **kwargs)

        # Apply cropping if requested
        if crop_to_source and loaded_data is not None:
            search_geometry = self.config.get_cached_search_geometry(source)
            if search_geometry is not None and isinstance(
                search_geometry, BaseGeometry
            ):
                loaded_data = self.crop_to_geometry(loaded_data, search_geometry)
            else:
                # If no cached geometry, compute it
                search_geometry = self.config.extract_search_geometry(source, **kwargs)
                if isinstance(search_geometry, BaseGeometry):
                    loaded_data = self.crop_to_geometry(loaded_data, search_geometry)

        return loaded_data


class BaseHandler(ABC):
    """
    Abstract base class that orchestrates configuration, downloading, and reading functionality.

    This class serves as the main entry point for dataset handlers, providing a unified
    interface for data acquisition and loading. It manages the lifecycle of config,
    downloader, and reader components.

    Subclasses should implement the abstract methods to provide specific handler types
    and define how components are created and interact.
    """

    def __init__(
        self,
        config: Optional[BaseHandlerConfig] = None,
        downloader: Optional[BaseHandlerDownloader] = None,
        reader: Optional[BaseHandlerReader] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the BaseHandler with optional components.

        Args:
            config: Configuration object. If None, will be created via create_config()
            downloader: Downloader instance. If None, will be created via create_downloader()
            reader: Reader instance. If None, will be created via create_reader()
            data_store: Data store instance. Defaults to LocalDataStore if not provided
            logger: Logger instance. If not provided, creates one based on class name
        """
        # Initialize data store first as it's used by other components
        self.data_store = data_store or LocalDataStore()

        # Initialize logger
        self.logger = logger or global_config.get_logger(self.__class__.__name__)

        # Initialize or create config
        self._config = config
        if self._config is None:
            self._config = self.create_config(
                data_store=self.data_store, logger=self.logger
            )

        # Initialize or create downloader
        self._downloader = downloader
        if self._downloader is None:
            self._downloader = self.create_downloader(
                config=self._config, data_store=self.data_store, logger=self.logger
            )

        # Initialize or create reader
        self._reader = reader
        if self._reader is None:
            self._reader = self.create_reader(
                config=self._config, data_store=self.data_store, logger=self.logger
            )

    @property
    def config(self) -> BaseHandlerConfig:
        """Get the configuration object."""
        return self._config

    @property
    def downloader(self) -> BaseHandlerDownloader:
        """Get the downloader object."""
        return self._downloader

    @property
    def reader(self) -> BaseHandlerReader:
        """Get the reader object."""
        return self._reader

    # Abstract factory methods for creating components
    @abstractmethod
    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> BaseHandlerConfig:
        """
        Create and return a configuration object for this handler.

        Args:
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional configuration parameters

        Returns:
            Configured BaseHandlerConfig instance
        """
        pass

    @abstractmethod
    def create_downloader(
        self,
        config: BaseHandlerConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> BaseHandlerDownloader:
        """
        Create and return a downloader object for this handler.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional downloader parameters

        Returns:
            Configured BaseHandlerDownloader instance
        """
        pass

    @abstractmethod
    def create_reader(
        self,
        config: BaseHandlerConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> BaseHandlerReader:
        """
        Create and return a reader object for this handler.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional reader parameters

        Returns:
            Configured BaseHandlerReader instance
        """
        pass

    # High-level interface methods
    def ensure_data_available(
        self,
        source: Union[
            str,  # country
            List[Union[tuple, Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
            Path,  # path
            List[Union[str, Path]],  # list of paths
        ],
        force_download: bool = False,
        **kwargs,
    ) -> bool:
        """
        Ensure that data is available for the given source.

        This method checks if the required data exists locally, and if not (or if
        force_download is True), downloads it using the downloader.

        Args:
            source: The data source specification
            force_download: If True, download even if data exists locally
            **kwargs: Additional parameters passed to download methods

        Returns:
            bool: True if data is available after this operation
        """
        try:
            # Get relevant units (cached if already computed for this source)
            data_units = self.config.get_relevant_data_units(
                source, force_recompute=force_download, **kwargs
            )
            data_paths = self.config.get_data_unit_paths(data_units, **kwargs)

            # Check if data exists (unless force download)
            if not force_download:
                missing_paths = [
                    path
                    for path in data_paths
                    if not self.data_store.file_exists(str(path))
                ]
            else:
                # If force_download, treat all as missing
                missing_paths = data_paths

            if not missing_paths:
                self.logger.info("All required data is already available")
                return True

            # Map units to paths (assumes correspondence order; adapt if needed)
            path_to_unit = dict(zip(data_paths, data_units))
            if force_download:
                units_to_download = data_units
            else:
                units_to_download = [
                    path_to_unit[p] for p in missing_paths if p in path_to_unit
                ]

            if units_to_download:
                self.downloader.download_data_units(units_to_download, **kwargs)
            else:
                # Fallback - download by source if unit mapping isn't available
                self.downloader.download(source, **kwargs)

            # After attempted download, check again
            remaining_missing = [
                path
                for path in data_paths
                if not self.data_store.file_exists(str(path))
            ]
            if remaining_missing:
                self.logger.error(
                    f"Some data still missing after download: {remaining_missing}"
                )
                return False

            return True
        except Exception as e:
            self.logger.error(f"Failed to ensure data availability: {e}")
            return False

    def load_data(
        self,
        source: Union[
            str,  # country
            List[Union[tuple, Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
            Path,  # path
            List[Union[str, Path]],  # list of paths
        ],
        crop_to_source: bool = False,
        ensure_available: bool = True,
        **kwargs,
    ) -> Any:
        """
        Load data from the given source.

        Args:
            source: The data source specification
            ensure_available: If True, ensure data is downloaded before loading
            **kwargs: Additional parameters passed to load methods

        Returns:
            Loaded data (type depends on specific handler implementation)
        """
        if ensure_available:
            if not self.ensure_data_available(source, **kwargs):
                raise RuntimeError("Could not ensure data availability for loading")

        return self.reader.load(source, crop_to_source=crop_to_source, **kwargs)

    def download_and_load(
        self,
        source: Union[
            str,  # country
            List[Union[tuple, Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
            Path,  # path
            List[Union[str, Path]],  # list of paths
        ],
        crop_to_source: bool = False,
        force_download: bool = False,
        **kwargs,
    ) -> Any:
        """
        Convenience method to download (if needed) and load data in one call.

        Args:
            source: The data source specification
            force_download: If True, download even if data exists locally
            **kwargs: Additional parameters

        Returns:
            Loaded data
        """
        self.ensure_data_available(source, force_download=force_download, **kwargs)
        return self.reader.load(source, crop_to_source=crop_to_source, **kwargs)

    def get_available_data_info(
        self,
        source: Union[
            str,  # country
            List[Union[tuple, Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
        ],
        **kwargs,
    ) -> dict:
        """
        Get information about available data for the given source.

        Args:
            source: The data source specification
            **kwargs: Additional parameters

        Returns:
            dict: Information about data availability, paths, etc.
        """
        try:
            if hasattr(self.config, "get_relevant_data_units"):
                data_units = self.config.get_relevant_data_units(source, **kwargs)
                data_paths = self.config.get_data_unit_paths(data_units, **kwargs)
            else:
                data_paths = self.reader.resolve_source_paths(source, **kwargs)

            existing_paths = [
                path for path in data_paths if self.data_store.file_exists(str(path))
            ]
            missing_paths = [
                path
                for path in data_paths
                if not self.data_store.file_exists(str(path))
            ]

            return {
                "total_data_units": len(data_paths),
                "available_data_units": len(existing_paths),
                "missing_data_units": len(missing_paths),
                "available_paths": existing_paths,
                "missing_paths": missing_paths,
                "all_available": len(missing_paths) == 0,
            }

        except Exception as e:
            self.logger.error(f"Failed to get data info: {e}")
            return {
                "error": str(e),
                "total_data_units": 0,
                "available_data_units": 0,
                "missing_data_units": 0,
                "available_paths": [],
                "missing_paths": [],
                "all_available": False,
            }

    def cleanup(self):
        """
        Cleanup resources used by the handler.

        Override in subclasses if specific cleanup is needed.
        """
        self.logger.info(f"Cleaning up {self.__class__.__name__}")
        # Subclasses can override to add specific cleanup logic

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __repr__(self) -> str:
        """String representation of the handler."""
        return (
            f"{self.__class__.__name__}("
            f"config={self.config.__class__.__name__}, "
            f"downloader={self.downloader.__class__.__name__}, "
            f"reader={self.reader.__class__.__name__})"
        )
