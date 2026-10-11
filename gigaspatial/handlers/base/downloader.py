"""Abstract downloader for dataset handlers (`BaseHandlerDownloader`)."""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Iterable
import pandas as pd
import logging
from tqdm import tqdm

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.base.config import BaseHandlerConfig


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

        desc = kwargs.pop("desc", "Downloading data units")
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
