"""Abstract downloader for dataset handlers (`BaseHandlerDownloader`)."""
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, List, Optional, Iterable, Type
import pandas as pd
import logging
from tqdm import tqdm

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.base.config import BaseHandlerConfig
from gigaspatial.handlers.base.options import DownloadOptions, sort_options


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
        options_type: The `DownloadOptions` subclass ``download_data_unit``
            takes.
    """

    options_type: ClassVar[Type[DownloadOptions]] = DownloadOptions

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

    PROGRAMMING_ERRORS = (TypeError, AttributeError, NameError)

    @abstractmethod
    def download_data_unit(self, unit: Any, options):
        """
        Download a single data unit.

        Must be implemented by subclasses to define the specific download
        logic for the dataset. Units may be downloaded concurrently from
        threads, so implementations must not share mutable state unguarded.

        Args:
            unit: Data unit identifier.
            options: The download options.

        Returns:
            The stored path, or a list of paths for a unit stored as several
            files.

        Raises:
            Exception: Any failure to download the unit. The base logs it and
                carries on with the other units.
        """
        pass

    def download_data_units(self, units: Iterable[Any], **kwargs) -> List[Any]:
        """
        Download multiple data units, in ``config.n_workers`` threads.

        A unit that fails is logged and skipped; the other units still
        download.

        Args:
            units: Iterable of data unit identifiers.
            **kwargs: Download options (see ``options_type``).

        Returns:
            A flat list of the downloaded file paths.

        Raises:
            TypeError: If a keyword is not a download option, or a downloader
                raises a programming error (``TypeError``, ``AttributeError``
                or ``NameError``).
        """
        (options,) = sort_options(kwargs, self, entry="download_data_units")
        return self._download_units(units, options)

    def _download_units(self, units, options) -> List[Any]:
        if units is None or (hasattr(units, "__len__") and len(units) == 0):
            self.logger.warning("There is no matching data to download.")
            return []

        # Handle pandas DataFrame by converting rows to a list of dicts
        if isinstance(units, pd.DataFrame):
            units_list = [row for _, row in units.iterrows()]
        else:
            units_list = list(units) if not isinstance(units, list) else units

        progress = {
            "desc": "Downloading data units",
            "disable": not options.show_progress,
        }
        n_workers = max(1, min(getattr(self.config, "n_workers", 1), len(units_list)))

        def _attempt(unit):
            try:
                return self.download_data_unit(unit, options), None
            except self.PROGRAMMING_ERRORS:
                raise
            except Exception as error:
                return None, error

        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                outcomes = list(
                    tqdm(pool.map(_attempt, units_list), total=len(units_list), **progress)
                )
        else:
            outcomes = [_attempt(unit) for unit in tqdm(units_list, **progress)]

        flattened, failures = [], []
        for unit, (result, error) in zip(units_list, outcomes):
            if error is not None:
                failures.append(unit)
                self.logger.error(
                    f"Failed to download unit {unit}: {error}", exc_info=error
                )
            elif isinstance(result, (list, tuple)):
                flattened.extend(result)
            elif result is not None:
                flattened.append(result)

        if failures:
            self.logger.error(
                f"{len(failures)} of {len(units_list)} data unit(s) failed to download"
            )
        return flattened

    def download(self, source, **kwargs):
        """
        Acquire all data units relevant to a specific source.

        Resolves the source (e.g., country code) into units and triggers
        the download process.

        Args:
            source: Input source identifier.
            **kwargs: Resolve and download options.

        Returns:
            A list of all downloaded items.

        Raises:
            TypeError: If a keyword is neither a resolve nor a download option.
        """
        resolve, download = sort_options(kwargs, self.config, self, entry="download")
        units = self.config._resolve_units(source, resolve)
        return self._download_units(units, download)
