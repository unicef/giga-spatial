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
from gigaspatial.handlers.base.source import is_path_source


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

        Resolves the source into data units and downloads those that are not
        stored yet (``config.is_unit_available``), or all of them if
        `force_download` is True. Path sources are only checked for existence.

        Args:
            source: Specification of the required data.
            force_download: If True, downloads every unit regardless of local state.
            **kwargs: Additional parameters for unit resolution or download.

        Returns:
            True if all required data is available in the DataStore, False if
            some units (or explicit paths) are still missing after the
            download attempt.

        Raises:
            Exception: Errors raised while resolving the source into data units
                or paths propagate unchanged, as do programming errors
                (``TypeError``, ``AttributeError``, ``NameError``) raised while
                downloading. Other download errors are logged, and the
                availability check that follows decides the return value.
        """
        if is_path_source(source):
            paths = self.reader.resolve_by_paths(source)
            missing = [p for p in paths if not self.data_store.file_exists(str(p))]
            if missing:
                self.logger.warning(
                    f"Required explicit file paths are missing: {missing}. "
                    "Cannot automatically download them without an abstract source."
                )
                return False
            if force_download:
                self.logger.warning(
                    "force_download has no effect on path sources: they name "
                    "stored files, not data units to fetch."
                )
            return True

        units = self.config.get_relevant_data_units(
            source, force_recompute=force_download, **kwargs
        )
        units_to_download = (
            list(units) if force_download else self._missing_units(units, **kwargs)
        )
        if not units_to_download:
            self.logger.info("All required data is already available")
            return True

        try:
            self.downloader.download_data_units(units_to_download, **kwargs)
        except (TypeError, AttributeError, NameError):
            raise
        except Exception:
            self.logger.error("Download failed", exc_info=True)

        still_missing = self._missing_units(units_to_download, **kwargs)
        if still_missing:
            self.logger.error(
                f"{len(still_missing)} data unit(s) still missing after download: "
                f"{still_missing}"
            )
            return False

        return True

    def _missing_units(self, units, **kwargs) -> list:
        return [
            unit
            for unit in units
            if not self.config.is_unit_available(
                unit, self.config.get_stored_paths(unit, **kwargs)
            )
        ]

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
            A dictionary with the number of available and missing data units,
            and the stored paths that exist and do not exist.
        """
        try:
            units = self.config.get_relevant_data_units(source, **kwargs)
            available_units = 0
            existing_paths, missing_paths = [], []
            for unit in units:
                paths = self.config.get_stored_paths(unit, **kwargs)
                available_units += self.config.is_unit_available(unit, paths)
                for path in paths:
                    if self.data_store.file_exists(str(path)):
                        existing_paths.append(path)
                    else:
                        missing_paths.append(path)

            return {
                "total_data_units": len(units),
                "available_data_units": available_units,
                "missing_data_units": len(units) - available_units,
                "available_paths": existing_paths,
                "missing_paths": missing_paths,
                "all_available": available_units == len(units),
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
