"""Orchestrating base class for dataset handlers (`BaseHandler`)."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
import logging

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.base.config import BaseHandlerConfig
from gigaspatial.handlers.base.downloader import BaseHandlerDownloader
from gigaspatial.handlers.base.reader import BaseHandlerReader


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
            True if all required data is available in the DataStore, False if
            some files are still missing after the download attempt.

        Raises:
            Exception: Errors raised while resolving the source into data units
                or paths propagate unchanged, as do programming errors
                (``TypeError``, ``AttributeError``, ``NameError``) raised while
                downloading. Other download errors are logged, and the
                availability check that follows decides the return value.
        """
        is_path_source = (
            isinstance(source, Path)
            or (
                isinstance(source, (list, tuple, set))
                and all(isinstance(p, (str, Path)) for p in source)
            )
            or (isinstance(source, str) and "." in source)
        )

        if is_path_source:
            if isinstance(source, (str, Path)):
                data_paths = [Path(source)]
            else:
                data_paths = [Path(p) for p in source]
            data_units = data_paths
        else:
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

        if is_path_source:
            self.logger.warning(
                f"Required explicit file paths are missing: {missing_paths}. "
                "Cannot automatically download them without an abstract source."
            )
            return False

        # Map units to paths (assumes correspondence order; adapt if needed)
        path_to_unit = dict(zip(data_paths, data_units))
        if force_download:
            units_to_download = data_units
        else:
            units_to_download = [
                path_to_unit[p] for p in missing_paths if p in path_to_unit
            ]

        try:
            if units_to_download:
                self.downloader.download_data_units(units_to_download, **kwargs)
            else:
                # Fallback - download by source if unit mapping isn't available
                self.downloader.download(source, **kwargs)
        except (TypeError, AttributeError, NameError):
            raise
        except Exception:
            self.logger.error("Download failed", exc_info=True)

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
