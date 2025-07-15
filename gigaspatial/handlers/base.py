from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union, Tuple, Callable, Iterable
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
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

    def get_relevant_data_units(
        self,
        source: Union[
            str,  # country
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
        ],
        **kwargs,
    ):
        if isinstance(source, str):
            data_units = self.get_relevant_data_units_by_country(source, **kwargs)
        elif isinstance(source, (BaseGeometry, gpd.GeoDataFrame)):
            data_units = self.get_relevant_data_units_by_geometry(source, **kwargs)
        elif isinstance(source, Iterable):
            if all(isinstance(p, (Iterable, Point)) for p in source):
                data_units = self.get_relevant_data_units_by_points(source, **kwargs)
            else:
                raise ValueError(
                    "List input to get_relevant_data_units must be all points."
                )
        else:
            raise NotImplementedError(f"Unsupported source type: {type(source)}")

        return data_units

    @abstractmethod
    def get_relevant_data_units_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> Any:
        """
        Given a geometry, return a list of relevant data unit identifiers (e.g., tiles, files, resources).
        """
        pass

    @abstractmethod
    def get_relevant_data_units_by_points(
        self, points: Iterable[Union[Point, tuple]], **kwargs
    ) -> Any:
        """
        Given a list of points, return a list of relevant data unit identifiers.
        """
        pass

    def get_relevant_data_units_by_country(self, country: str, **kwargs) -> Any:
        """
        Given a country code or name, return a list of relevant data unit identifiers.
        """
        from gigaspatial.handlers.boundaries import AdminBoundaries

        country_geometry = (
            AdminBoundaries.create(country_code=country, **kwargs)
            .boundaries[0]
            .geometry
        )
        return self.get_relevant_data_units_by_geometry(
            geometry=country_geometry, **kwargs
        )

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

    @abstractmethod
    def download(self, *args, **kwargs):
        """
        Abstract method to download data. Implement in subclasses.
        """
        pass


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
        if isinstance(source, (str, Path)):
            # Could be a country code or a path
            if self.data_store.file_exists(str(source)) or str(source).endswith(
                (".csv", ".tif", ".json", ".parquet", ".gz", ".geojson", ".zip")
            ):
                source_data_paths = self.resolve_by_paths(source)
            else:
                source_data_paths = self.resolve_by_country(source, **kwargs)
        elif isinstance(source, (BaseGeometry, gpd.GeoDataFrame)):
            source_data_paths = self.resolve_by_geometry(source, **kwargs)
        elif isinstance(source, Iterable):
            # List of points or paths
            if all(isinstance(p, (Iterable, Point)) for p in source):
                source_data_paths = self.resolve_by_points(source, **kwargs)
            elif all(isinstance(p, (str, Path)) for p in source):
                source_data_paths = self.resolve_by_paths(source)
            else:
                raise ValueError(
                    "List input to resolve_source_paths must be all points or all paths."
                )
        else:
            raise NotImplementedError(f"Unsupported source type: {type(source)}")

        self.logger.info(f"Resolved {len(source_data_paths)} paths!")
        return source_data_paths

    def resolve_by_country(self, country: str, **kwargs) -> List[Union[str, Path]]:
        """
        Resolve source paths for a given country code/name.
        Uses the config's get_relevant_data_units_by_country method.
        """
        if not self.config:
            raise ValueError("Config is required for resolving by country")
        data_units = self.config.get_relevant_data_units_by_country(country, **kwargs)
        return self.config.get_data_unit_paths(data_units, **kwargs)

    def resolve_by_points(
        self, points: List[Union[Tuple[float, float], Point]], **kwargs
    ) -> List[Union[str, Path]]:
        """
        Resolve source paths for a list of points.
        Uses the config's get_relevant_data_units_by_points method.
        """
        if not self.config:
            raise ValueError("Config is required for resolving by points")
        data_units = self.config.get_relevant_data_units_by_points(points, **kwargs)
        return self.config.get_data_unit_paths(data_units, **kwargs)

    def resolve_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> List[Union[str, Path]]:
        """
        Resolve source paths for a geometry or GeoDataFrame.
        Uses the config's get_relevant_data_units_by_geometry method.
        """
        if not self.config:
            raise ValueError("Config is required for resolving by geometry")
        data_units = self.config.get_relevant_data_units_by_geometry(geometry, **kwargs)
        return self.config.get_data_unit_paths(data_units, **kwargs)

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
        self, raster_paths: List[Union[str, Path]]
    ) -> List[TifProcessor]:
        """
        Load raster data from file paths.

        Args:
            raster_paths (List[Union[str, Path]]): List of file paths to raster files.

        Returns:
            List[TifProcessor]: List of TifProcessor objects for accessing the raster data.
        """
        return [
            TifProcessor(data_path, self.data_store, mode="single")
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
        **kwargs,
    ) -> Any:
        """
        Load data from the given source.

        Args:
            source: The data source (country code/name, points, geometry, paths, etc.).
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
        return self._post_load_hook(loaded_data, **kwargs)


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
            data_units = None
            data_paths = None
            # Resolve what data units are needed
            if hasattr(self.config, "get_relevant_data_units"):
                data_units = self.config.get_relevant_data_units(source, **kwargs)
                data_paths = self.config.get_data_unit_paths(data_units, **kwargs)
            else:
                # Fallback: try to resolve paths directly
                if hasattr(self.reader, "resolve_source_paths"):
                    data_paths = self.reader.resolve_source_paths(source, **kwargs)
                else:
                    self.logger.warning("Cannot determine required data paths")
                    return False

            # Check if data exists (unless force download)
            if not force_download:
                missing_paths = [
                    path
                    for path in data_paths
                    if not self.data_store.file_exists(str(path))
                ]
                if not missing_paths:
                    self.logger.info("All required data is already available")
                    return True
            else:
                # If force_download, treat all as missing
                missing_paths = data_paths

            if not missing_paths:
                self.logger.info("No missing data to download.")
                return True

            # Download logic
            if data_units is not None:
                # Map data_units to their paths and select only those that are missing
                unit_to_path = dict(zip(data_units, data_paths))
                if force_download:
                    # Download all units if force_download
                    self.downloader.download_data_units(data_units, **kwargs)
                else:
                    missing_units = [
                        unit
                        for unit, path in unit_to_path.items()
                        if path in missing_paths
                    ]
                    if missing_units:
                        self.downloader.download_data_units(missing_units, **kwargs)
            else:
                self.downloader.download(source, **kwargs)

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

        return self.reader.load(source, **kwargs)

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
        return self.reader.load(source, **kwargs)

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
