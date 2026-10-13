"""Orchestrating base class for dataset handlers (`BaseHandler`)."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union
import geopandas as gpd
from pydantic import ValidationError
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
import logging

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.base.config import BaseHandlerConfig
from gigaspatial.handlers.base.downloader import BaseHandlerDownloader
from gigaspatial.handlers.base.options import select_kwargs, sort_options
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
        **config_kwargs,
    ):
        """
        Initialize the BaseHandler with optional components.

        Args:
            config: Configuration object. If None, will be created via create_config()
            downloader: Downloader instance. If None, will be created via create_downloader()
            reader: Reader instance. If None, will be created via create_reader()
            data_store: Data store instance. Defaults to LocalDataStore if not provided
            logger: Logger instance. If not provided, creates one based on class name
            **config_kwargs: Config fields (e.g. ``base_path``, ``n_workers``)
                passed to create_config().

        Raises:
            TypeError: If config fields are given together with ``config``, or
                a field is not one the config defines.
        """
        if config is not None and config_kwargs:
            names = ", ".join(repr(k) for k in sorted(config_kwargs))
            raise TypeError(
                f"{type(self).__name__}() got config fields ({names}) together "
                "with config=; set them on the config instead"
            )

        # Initialize data store first as it's used by other components
        self.data_store = data_store or LocalDataStore()

        # Initialize logger
        self.logger = logger or global_config.get_logger(self.__class__.__name__)

        # Initialize or create config
        self._config = config
        if self._config is None:
            try:
                self._config = self.create_config(
                    data_store=self.data_store, logger=self.logger, **config_kwargs
                )
            except ValidationError as error:
                unknown = [
                    ".".join(map(str, e["loc"]))
                    for e in error.errors()
                    if e["type"] == "unexpected_keyword_argument"
                ]
                if not unknown:
                    raise
                names = ", ".join(repr(k) for k in unknown)
                raise TypeError(
                    f"{type(self).__name__}() got unexpected config field(s): {names}"
                ) from error

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
            **kwargs: Resolve and download options.

        Returns:
            True if all required data is available in the DataStore, False if
            some units (or explicit paths) are still missing after the
            download attempt.

        Raises:
            TypeError: If a keyword is neither a resolve nor a download option.
            Exception: Errors raised while resolving the source into data units
                or paths propagate unchanged, as do programming errors
                (``TypeError``, ``AttributeError``, ``NameError``) raised while
                downloading. Other download errors are logged per unit, and the
                availability check that follows decides the return value.
        """
        resolve, download = sort_options(
            kwargs, self.config, self.downloader, entry="ensure_data_available"
        )
        return self._ensure_available(source, force_download, resolve, download)

    def _ensure_available(self, source, force_download: bool, resolve, download) -> bool:
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

        units = self.config._resolve_units(source, resolve, force_recompute=force_download)
        units_to_download = (
            list(units) if force_download else self._missing_units(units, resolve)
        )
        if not units_to_download:
            self.logger.info("All required data is already available")
            return True

        self.downloader._download_units(units_to_download, download)

        still_missing = self._missing_units(units_to_download, resolve)
        if still_missing:
            self.logger.error(
                f"{len(still_missing)} data unit(s) still missing after download: "
                f"{still_missing}"
            )
            return False

        return True

    def _missing_units(self, units, resolve) -> list:
        return [
            unit
            for unit in units
            if not self.config.is_unit_available(
                unit, self.config._stored_paths(unit, resolve)
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
            **kwargs: Resolve, download and read options.

        Returns:
            The loaded data object.

        Raises:
            TypeError: If a keyword is not an option of any component.
            RuntimeError: If data availability cannot be ensured.
        """
        resolve, download, _ = sort_options(
            kwargs, self.config, self.downloader, self.reader, entry="load_data"
        )
        if ensure_available:
            if not self._ensure_available(source, False, resolve, download):
                raise RuntimeError("Could not ensure data availability for loading")

        return self._read(source, crop_to_source, kwargs)

    def _read(self, source, crop_to_source: bool, kwargs) -> Any:
        return self.reader.load(
            source,
            crop_to_source=crop_to_source,
            **select_kwargs(kwargs, self.config, self.reader),
        )

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
            **kwargs: Resolve, download and read options.

        Returns:
            The loaded data object.

        Raises:
            TypeError: If a keyword is not an option of any component.
        """
        resolve, download, _ = sort_options(
            kwargs, self.config, self.downloader, self.reader, entry="download_and_load"
        )
        self._ensure_available(source, force_download, resolve, download)
        return self._read(source, crop_to_source, kwargs)

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
            **kwargs: Resolve options.

        Returns:
            A dictionary with the number of available and missing data units,
            and the stored paths that exist and do not exist.

        Raises:
            TypeError: If a keyword is not a resolve option.
        """
        (resolve,) = sort_options(kwargs, self.config, entry="get_available_data_info")
        try:
            units = self.config._resolve_units(source, resolve)
            available_units = 0
            existing_paths, missing_paths = [], []
            for unit in units:
                paths = self.config._stored_paths(unit, resolve)
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

    def select_options(self, kwargs: dict, *components) -> dict:
        """
        Return the keyword arguments that are options of this handler.

        For callers that share one set of keyword arguments between the
        handler and other consumers.

        Args:
            kwargs: The shared keyword arguments.
            *components: Keep only the options of these components (e.g.
                ``handler.config`` for resolve options); defaults to all.
        """
        components = components or (self.config, self.downloader, self.reader)
        return select_kwargs(kwargs, *components)

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
