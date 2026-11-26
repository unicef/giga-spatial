"""
FLOPROS flood protection standards handler.

Scussolini et al. (2016) publish one global shapefile of subnational
flood protection standards in the supplement of Natural Hazards and Earth
System Sciences, 16, 1049–1061 (doi:10.5194/nhess-16-1049-2016), under
CC BY 3.0. A data unit is that shapefile. Every source resolves to it, and
the reader returns the polygons as a GeoDataFrame. Cropping uses the base,
on the polygon geometry.

Standards are the return period, in years, of the flood a place is protected
against. ``MerL_Riv`` is the merged riverine layer: the design layer where
that layer has information for the unit, otherwise the policy layer, otherwise
the model layer. The supplement's Excel table of design and policy records
at a finer scale is not stored; the shapefile keeps the records that match
the subnational polygons.

https://nhess.copernicus.org/articles/16/1049/2016/
"""

from pathlib import Path
from typing import ClassVar, List, Tuple

import geopandas as gpd
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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class FLOPROSConfig(BaseHandlerConfig):
    """
    Configuration for the FLOPROS subnational shapefile.

    Attributes:
        base_path: Directory of the stored shapefile and its sidecars.
    """

    SUPPLEMENT_URL: ClassVar[str] = (
        "https://nhess.copernicus.org/articles/16/1049/2016/"
        "nhess-16-1049-2016-supplement.zip"
    )
    SHAPEFILE_STEM: ClassVar[str] = "FLOPROS_shp_V1"
    SHAPEFILE_SUFFIXES: ClassVar[Tuple[str, ...]] = (".shp", ".shx", ".dbf", ".prj")
    MEMBER_PATTERN: ClassVar[str] = (
        r".*/"
        + SHAPEFILE_STEM
        + r"\.("
        + "|".join(suffix.lstrip(".") for suffix in SHAPEFILE_SUFFIXES)
        + r")$"
    )

    options_type = ResolveOptions
    units_match_country = False

    base_path: Path = Field(default=global_config.get_path("flopros", "bronze"))

    def get_search_key(self, source, options: ResolveOptions) -> None:
        """One global shapefile: no source narrows it."""
        return None

    def get_relevant_data_units_by_key(
        self, search_key, options: ResolveOptions
    ) -> List[dict]:
        """
        Return the supplement shapefile.

        Args:
            search_key: Always None. The dataset is one global file.
            options: The resolve options.

        Returns:
            One dict, with the supplement zip URL.
        """
        return [{"url": self.SUPPLEMENT_URL}]

    def get_data_unit_path(self, unit: dict, options=None) -> Path:
        """
        Return the stored shapefile of the supplement.

        Args:
            unit: The unit from ``get_relevant_data_units_by_key``.
            options: Unused. The dataset is one file.

        Returns:
            The path of the extracted ``.shp`` in the data store.
        """
        return self.base_path / f"{self.SHAPEFILE_STEM}.shp"

    def is_unit_available(self, unit: dict, paths: List[Path]) -> bool:
        """
        Return True when the shapefile and its required sidecars are stored.

        Args:
            unit: The unit from ``get_relevant_data_units_by_key``.
            paths: Unused. Availability is the ``.shp``, ``.shx``, ``.dbf``,
                and ``.prj`` beside it.

        Returns:
            True if every required file exists in the data store.
        """
        shapefile = self.get_data_unit_path(unit)
        return all(
            self.data_store.file_exists(str(shapefile.with_suffix(suffix)))
            for suffix in self.SHAPEFILE_SUFFIXES
        )


class FLOPROSDownloader(BaseHandlerDownloader):
    """Downloads the FLOPROS shapefile into the data store."""

    options_type = DownloadOptions
    config_type = FLOPROSConfig

    def download_data_unit(self, unit: dict, options: DownloadOptions) -> List[Path]:
        """
        Download the supplement zip and store the shapefile parts.

        Args:
            unit: The unit from the config, with ``url``.
            options: ``show_progress`` controls the byte progress bar.

        Returns:
            The stored shapefile paths.

        Raises:
            requests.RequestException: If the download fails.
            zipfile.BadZipFile: If the download is not a zip archive.
            FileNotFoundError: If the archive has no FLOPROS shapefile.
        """
        return extract_zip_to_store(
            unit["url"],
            self.config.get_data_unit_path(unit).parent,
            self.data_store,
            pattern=self.config.MEMBER_PATTERN,
            show_progress=options.show_progress,
        )


class FLOPROSReader(BaseHandlerReader):
    """Loads the stored FLOPROS shapefile."""

    options_type = ReadOptions
    config_type = FLOPROSConfig

    def load_from_paths(
        self, source_data_path: List[str], options: ReadOptions
    ) -> gpd.GeoDataFrame:
        """
        Load the stored shapefile into one GeoDataFrame.

        Args:
            source_data_path: Stored ``.shp`` paths.
            options: The read options.

        Returns:
            The subnational polygons and their protection standards.
        """
        return self._load_tabular_data(file_paths=source_data_path)


class FLOPROSHandler(BaseHandler):
    """
    Handler for the FLOPROS flood protection standards.

    ``load_data`` returns the subnational polygons. Pass a country, a
    geometry, or points, and ``crop_to_source=True`` to keep the polygons
    that intersect that source.
    """

    def create_config(self, data_store, logger, **kwargs) -> FLOPROSConfig:
        return FLOPROSConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(self, config, data_store, logger) -> FLOPROSDownloader:
        return FLOPROSDownloader(config=config, data_store=data_store, logger=logger)

    def create_reader(self, config, data_store, logger) -> FLOPROSReader:
        return FLOPROSReader(config=config, data_store=data_store, logger=logger)
