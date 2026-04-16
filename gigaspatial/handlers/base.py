"""
Base classes and abstractions for dataset handlers.

This module defines the foundational architecture for all data handlers in
GigaSpatial. It provides abstract base classes for:
- Configuration (`BaseHandlerConfig`)
- Data downloaders (`BaseHandlerDownloader`)
- Data readers (`BaseHandlerReader`)
- Orchestrating handlers (`BaseHandler`)

Standardizing these interfaces ensures consistency across various data sources
(e.g., GHSL, WorldPop, OSM, Google Buildings).
"""
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
from tqdm import tqdm

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.processing.tif_processor import TifProcessor


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
    Abstract base class for data downloaders.

    Handles the logic for acquiring dataset files from remote sources and
    saving them to a `DataStore`. Supports parallel downloads and progress
    tracking.

    Attributes:
        config: Handler configuration object.
        data_store: Abstraction for file-system access.
        logger: Logger instance for tracking download progress.
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
        Download a single data unit.

        Must be implemented by subclasses to define the specific download
        logic for the dataset.

        Args:
            *args: Data unit identifier and other positional arguments.
            **kwargs: Additional parameters for the download.
        """
        pass

    def download_data_units(self, units: Iterable[Any], **kwargs) -> List[Any]:
        """
        Download multiple data units in parallel.

        Iterates through the provided units and downloads them using
        `download_data_unit`. Parallelism is controlled by `n_workers` in
         the configuration.

        Args:
            units: Iterable of data unit identifiers.
            **kwargs: Additional parameters for the download.

        Returns:
            A flat list of all downloaded file paths or data records.
        """
        if units is None or (hasattr(units, "__len__") and len(units) == 0):
            self.logger.warning("There is no matching data to download.")
            return []

        # Handle pandas DataFrame by converting rows to a list of dicts
        if isinstance(units, pd.DataFrame):
            units_list = [row for _, row in units.iterrows()]
        else:
            units_list = list(units) if not isinstance(units, list) else units

        desc = kwargs.get("desc", "Downloading data units")
        n_workers = getattr(self.config, "n_workers", 1)

        if n_workers > 1:
            import multiprocessing
            import functools

            with multiprocessing.Pool(n_workers) as pool:
                download_func = functools.partial(self.download_data_unit, **kwargs)
                results = list(
                    tqdm(
                        pool.imap(download_func, units_list),
                        total=len(units_list),
                        desc=desc,
                    )
                )
        else:
            results = [
                self.download_data_unit(unit, **kwargs)
                for unit in tqdm(units_list, desc=desc)
            ]

        # Filter out None and flatten any list results (e.g. from extracted archives)
        flattened = []
        for r in results:
            if r is None:
                continue
            if isinstance(r, (list, tuple)):
                flattened.extend(r)
            else:
                flattened.append(r)

        return flattened

    def download(self, source, **kwargs):
        """
        Acquire all data units relevant to a specific source.

        Resolves the source (e.g., country code) into units and triggers
        the download process.

        Args:
            source: Input source identifier.
            **kwargs: Additional parameters for unit resolution or download.

        Returns:
            A list of all downloaded items.
        """
        units = self.config.get_relevant_data_units(source, **kwargs)
        return self.download_data_units(units, **kwargs)


class BaseHandlerReader(ABC):
    """
    Abstract base class for data readers.

    Provides high-level methods for resolving data paths and loading various
    data types (raster, tabular) into memory. Standardizes the loading
    interface across disparate datasets.

    Attributes:
        config: Handler configuration object.
        data_store: Abstraction for file-system access.
        logger: Logger instance for tracking data loading.
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
            str,
            List[Union[Tuple[float, float], Point]],
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,
            str,
            List[Union[str, Path]],
        ],
        **kwargs,
    ) -> List[Union[str, Path]]:
        """
        Identify data file paths corresponding to a specific source input.

        Parses the input source (e.g., country code, geometry, or explicit
        paths) and resolves them into a list of absolute paths within the
        `DataStore`.

        Args:
            source: Source specification (country, geometry, points, or paths).
            **kwargs: Additional parameters for resolving paths.

        Returns:
            A list of resolved file paths.
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
        Validate that all specified files exist in the data store.

        Args:
            file_paths: List of file paths to check.

        Raises:
            RuntimeError: If any file does not exist in the configured data store.
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
        Initialize TifProcessors for one or more raster files.

        Args:
            raster_paths: List of file paths to raster files.
            merge_rasters: If True, combines all rasters into a single `TifProcessor`.
            **kwargs: Additional parameters for TifProcessor initialization.

        Returns:
            Single TifProcessor (if merged) or list of TifProcessors.
        """
        if merge_rasters or len(raster_paths) == 1:
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
        self,
        file_paths: List[Union[str, Path]],
        read_function: Callable = read_dataset,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Load and consolidate tabular data from multiple files.

        Args:
            file_paths: List of file paths to load.
            read_function: Callable used to read individual files.
            show_progress: If True, displays a progress bar.
            progress_desc: Custom description for the progress bar.
            **kwargs: Additional parameters for the read function.

        Returns:
            The concatenated DataFrame or GeoDataFrame.
        """
        all_data = []
        iterator: Iterable = file_paths
        if show_progress and file_paths:
            iterator = tqdm(
                file_paths,
                desc=progress_desc or "Loading tabular data",
                total=len(file_paths),
            )

        for file_path in iterator:
            all_data.append(
                read_function(file_path, data_store=self.data_store, **kwargs)
            )

        if not all_data:
            return pd.DataFrame()
        result = pd.concat(all_data, ignore_index=True)
        return result

    def crop_to_geometry(self, data, geometry, predicate="intersects", **kwargs):
        """
        Crop loaded data (raster or tabular) to a specific geometry.

        Args:
            data: The data object to crop (DataFrame, GDF, or TifProcessor).
            geometry: The spatial filter geometry.
            predicate: Spatial predicate for tabular filtering ('intersects', 'within', etc.).
            **kwargs: Additional parameters for cropping (e.g., crs, crop).

        Returns:
            The cropped data object.
        """

        # Project geometry to the projection of the data if data has projection
        geom_crs = kwargs.pop("crs", "EPSG:4326")
        if hasattr(data, "crs") and data.crs != geom_crs:
            geometry = (
                gpd.GeoDataFrame(geometry=[geometry], crs=geom_crs)
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
            clip_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "crop",
                    "all_touched",
                    "invert",
                    "nodata",
                    "pad",
                    "pad_width",
                    "return_clipped_processor",
                )
            }
            return data.clip_to_geometry(geometry=geometry, **clip_kwargs)

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
            str,
            List[Union[Tuple[float, float], Point]],
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
        Orchestrate the resolution and loading of data from a source.

        Standardizes the workflow: Resolve Paths -> Pre-load Hook -> Load ->
        Post-load Hook -> Optional Crop.

        Args:
            source: specification of the data to load.
            crop_to_source: If True, the data is spatially cropped to the source.
            **kwargs: Additional parameters for loading.

        Returns:
            The loaded data object.
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
            geom_crs = getattr(self.config, "crs", "EPSG:4326")
            if search_geometry is not None and isinstance(
                search_geometry, BaseGeometry
            ):
                loaded_data = self.crop_to_geometry(
                    loaded_data, search_geometry, crs=geom_crs
                )
            else:
                # If no cached geometry, compute it
                search_geometry = self.config.extract_search_geometry(source, **kwargs)
                if isinstance(search_geometry, BaseGeometry):
                    loaded_data = self.crop_to_geometry(
                        loaded_data, search_geometry, crs=geom_crs
                    )

        return loaded_data


class BaseHandler(ABC):
    """
    Main entry point and orchestrator for dataset handlers.

    Encapsulates a configuration, downloader, and reader to provide a unified
    API for data acquisition and access. Handlers managed by this class are
    intended to be the primary interface for library users.

    Subclasses must implement the factory methods to provide specific
    component implementations.
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
            str,
            List[Union[tuple, Point]],
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,
            List[Union[str, Path]],
        ],
        force_download: bool = False,
        **kwargs,
    ) -> bool:
        """
        Guarantee that the required data for a source exists in the `DataStore`.

        Checks for local availability of data files. If any are missing or if
        `force_download` is True, it triggers the downloader to acquire them.

        Args:
            source: Specification of the required data.
            force_download: If True, downloads data regardless of local state.
            **kwargs: Additional parameters for unit resolution or download.

        Returns:
            True if all required data is available in the DataStore, False otherwise.
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

            # After attempted download, refresh data paths in case the mapping
            # from units to local files has changed (e.g. ZIP → extracted .tif files)
            data_paths = self.config.get_data_unit_paths(data_units, **kwargs)

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
            str,
            List[Union[tuple, Point]],
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,
            List[Union[str, Path]],
        ],
        crop_to_source: bool = False,
        ensure_available: bool = True,
        **kwargs,
    ) -> Any:
        """
        Load data from a source, ensuring its availability first.

        This is the primary high-level method for users to retrieve data
        from a handler.

        Args:
            source: Specification of the data to load.
            crop_to_source: If True, crops the result to the source geometry.
            ensure_available: If True, downloads missing data before loading.
            **kwargs: Additional parameters for loading or downloading.

        Returns:
            The loaded data object.

        Raises:
            RuntimeError: If data availability cannot be ensured.
        """
        if ensure_available:
            if not self.ensure_data_available(source, **kwargs):
                raise RuntimeError("Could not ensure data availability for loading")

        return self.reader.load(source, crop_to_source=crop_to_source, **kwargs)

    def download_and_load(
        self,
        source: Union[
            str,
            List[Union[tuple, Point]],
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,
            List[Union[str, Path]],
        ],
        crop_to_source: bool = False,
        force_download: bool = False,
        **kwargs,
    ) -> Any:
        """
        Download (if missing) and load data in a single call.

        Args:
            source: Specification of the data to acquire and load.
            crop_to_source: If True, crops the result to the source geometry.
            force_download: If True, re-downloads data even if present.
            **kwargs: Additional parameters for downloading or loading.

        Returns:
            The loaded data object.
        """
        self.ensure_data_available(source, force_download=force_download, **kwargs)
        return self.reader.load(source, crop_to_source=crop_to_source, **kwargs)

    def get_available_data_info(
        self,
        source: Union[
            str,
            List[Union[tuple, Point]],
            BaseGeometry,
            gpd.GeoDataFrame,
        ],
        **kwargs,
    ) -> dict:
        """
        Determine the availability status of data for a given source.

        Args:
            source: Specification of the data to check.
            **kwargs: Additional parameters for resolving paths.

        Returns:
            A dictionary containing counts and paths of available/missing data.
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
