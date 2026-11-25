"""
Global Power Plant Database handler.

World Resources Institute publishes one zip per release of the plant table
under CC BY 4.0. A data unit is that release: its version and the zip URL.
Every source resolves to the one configured release, and the reader returns
the plant CSV as a DataFrame. Cropping uses the base, on latitude and longitude.

https://datasets.wri.org/datasets/global-power-plant-database
"""

from pathlib import Path
from typing import ClassVar, Dict, List, Literal

import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from gigaspatial.config import config as global_config
from gigaspatial.handlers.base import (
    BaseHandler,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    DownloadOptions,
    ReadOptions,
    ResolveOptions,
)
from gigaspatial.handlers.base.fetch import extract_zip_to_store

Release = Literal["1.1.0", "1.2.0", "1.3.0"]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class GlobalPowerPlantConfig(BaseHandlerConfig):
    """
    Configuration for one release of the Global Power Plant Database.

    Attributes:
        base_path: Directory of the stored plant tables, above the version.
        version: Release kept for this handler's lifetime.
    """

    RELEASE_URLS: ClassVar[Dict[str, str]] = {
        "1.1.0": (
            "https://datasets.wri.org/private-admin/dataset/"
            "53623dfd-3df6-4f15-a091-67457cdb571f/resource/"
            "26669133-9493-4f38-a32f-ce35d5ea6ab9/download/"
            "globalpowerplantdatabasev110.zip"
        ),
        "1.2.0": (
            "https://datasets.wri.org/private-admin/dataset/"
            "53623dfd-3df6-4f15-a091-67457cdb571f/resource/"
            "9a7587a6-964e-4bfc-8c09-0ba1f611f18b/download/"
            "globalpowerplantdatabasev120.zip"
        ),
        "1.3.0": (
            "https://datasets.wri.org/private-admin/dataset/"
            "53623dfd-3df6-4f15-a091-67457cdb571f/resource/"
            "66bcdacc-3d0e-46ad-9271-a5a76b1853d2/download/"
            "globalpowerplantdatabasev130.zip"
        ),
    }
    CSV_NAME: ClassVar[str] = "global_power_plant_database.csv"

    options_type = ResolveOptions
    units_match_country = False

    base_path: Path = Field(
        default=global_config.get_path("wri", "bronze") / "global_power_plant_database"
    )
    version: Release = "1.3.0"

    def get_search_key(self, source, options: ResolveOptions) -> None:
        """One global file per release: no source narrows it."""
        return None

    def get_relevant_data_units_by_key(
        self, search_key, options: ResolveOptions
    ) -> List[dict]:
        """
        Return the configured release.

        Args:
            search_key: Always None. The dataset is one global file.
            options: The resolve options.

        Returns:
            One dict, with the version and the zip URL.
        """
        return [
            {
                "version": self.version,
                "url": self.RELEASE_URLS[self.version],
            }
        ]

    def get_data_unit_path(self, unit: dict, options=None) -> Path:
        """
        Return the stored plant CSV of a release.

        Args:
            unit: A release from ``get_relevant_data_units_by_key``.
            options: Unused. The unit carries the version.

        Returns:
            The path of the extracted CSV in the data store.
        """
        return self.base_path / unit["version"] / self.CSV_NAME


class GlobalPowerPlantDownloader(BaseHandlerDownloader):
    """Downloads one Global Power Plant Database release into the data store."""

    options_type = DownloadOptions
    config_type = GlobalPowerPlantConfig

    def download_data_unit(self, unit: dict, options: DownloadOptions) -> List[Path]:
        """
        Download a release zip and store its plant CSV.

        Args:
            unit: A release from the config, with ``version`` and ``url``.
            options: ``show_progress`` controls the byte progress bar.

        Returns:
            The stored CSV path, as a one-item list.

        Raises:
            requests.RequestException: If the download fails.
            zipfile.BadZipFile: If the download is not a zip archive.
            FileNotFoundError: If the archive has no plant CSV.
        """
        return extract_zip_to_store(
            unit["url"],
            self.config.get_data_unit_path(unit).parent,
            self.data_store,
            pattern=rf"{self.config.CSV_NAME}$",
            show_progress=options.show_progress,
        )


class GlobalPowerPlantReader(BaseHandlerReader):
    """Loads a stored Global Power Plant Database CSV."""

    options_type = ReadOptions
    config_type = GlobalPowerPlantConfig

    def load_from_paths(
        self, source_data_path: List[str], options: ReadOptions
    ) -> pd.DataFrame:
        """
        Load stored plant tables into one DataFrame.

        Args:
            source_data_path: Stored CSV paths.
            options: The read options.

        Returns:
            The plant rows, including ``latitude`` and ``longitude``.
        """
        return self._load_tabular_data(file_paths=source_data_path)


class GlobalPowerPlantHandler(BaseHandler):
    """
    Handler for the Global Power Plant Database.

    One instance is one release. Pass the release as ``version``, for example
    ``GlobalPowerPlantHandler(version="1.2.0")``. The default is ``"1.3.0"``.
    """

    def create_config(self, data_store, logger, **kwargs) -> GlobalPowerPlantConfig:
        return GlobalPowerPlantConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(
        self, config, data_store, logger
    ) -> GlobalPowerPlantDownloader:
        return GlobalPowerPlantDownloader(
            config=config, data_store=data_store, logger=logger
        )

    def create_reader(self, config, data_store, logger) -> GlobalPowerPlantReader:
        return GlobalPowerPlantReader(
            config=config, data_store=data_store, logger=logger
        )
