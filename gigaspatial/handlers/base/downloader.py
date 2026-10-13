"""Abstract downloader for dataset handlers (`BaseHandlerDownloader`)."""
from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Optional, Iterable, Type
import pandas as pd
import logging
from tqdm import tqdm

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.base.config import BaseHandlerConfig
from gigaspatial.handlers.base.options import (
    DownloadOptions,
    call_with_options,
    option_value,
    sort_options,
)


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
            takes, or None for a downloader that still takes raw keyword
            arguments.
    """

    options_type: ClassVar[Optional[Type[DownloadOptions]]] = None

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
    def download_data_unit(self, unit: Any, options):
        """
        Download a single data unit.

        Must be implemented by subclasses to define the specific download
        logic for the dataset. Downloaders without an ``options_type``
        receive raw keyword arguments instead of ``options``.

        Args:
            unit: Data unit identifier.
            options: The download options.
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
            **kwargs: Download options (see ``options_type``).

        Returns:
            A flat list of all downloaded file paths or data records.

        Raises:
            TypeError: If a keyword is not a download option.
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
            "disable": not option_value(options, "show_progress", True),
        }
        n_workers = getattr(self.config, "n_workers", 1)

        if n_workers > 1:
            import multiprocessing
            import functools

            with multiprocessing.Pool(n_workers) as pool:
                download_func = functools.partial(
                    call_with_options, self.download_data_unit, options=options
                )
                results = list(
                    tqdm(
                        pool.imap(download_func, units_list),
                        total=len(units_list),
                        **progress,
                    )
                )
        else:
            results = [
                call_with_options(self.download_data_unit, unit, options=options)
                for unit in tqdm(units_list, **progress)
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
            **kwargs: Resolve and download options.

        Returns:
            A list of all downloaded items.

        Raises:
            TypeError: If a keyword is neither a resolve nor a download option.
        """
        resolve, download = sort_options(kwargs, self.config, self, entry="download")
        units = self.config._resolve_units(source, resolve)
        return self._download_units(units, download)
