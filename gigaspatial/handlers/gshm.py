"""
GEM Global Seismic Hazard Map handler.

The GEM Foundation publishes GSHM v2026.1 as one global point GeoPackage in a
zip archive on Zenodo (CC BY-NC-SA 4.0). Peak ground acceleration is in g, for
a 10% probability of exceedance in 50 years on reference rock (Vs30 800 m/s).

A source resolves to that single archive. The stored file is the extracted
GeoPackage. ``crop_to_source=True`` reads the points in the search geometry's
bounds, then keeps the points that intersect the geometry.
"""

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from gigaspatial.config import config as global_config
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.readers import read_dataset
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
from gigaspatial.handlers.base.options import sort_options
from gigaspatial.handlers.base.source import SourceKind, classify_source


ARCHIVE_URL = (
    "https://zenodo.org/api/records/20735384/files/gshm_v2026_1_vector.zip/content"
)
VERSION = "v2026.1"
LAYER = "v2026"
MEMBER_NAME = "gem_gshm_v2026.1.gpkg"
MEMBER_PATTERN = r"gshm_v2026_1_vector/gem_gshm_v2026\.1\.gpkg"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class GSHMConfig(BaseHandlerConfig):
    """
    Configuration for the GEM Global Seismic Hazard Map.

    Attributes:
        base_path: Directory of the stored GeoPackage.
    """

    ARCHIVE_URL = ARCHIVE_URL
    VERSION = VERSION
    MEMBER_NAME = MEMBER_NAME

    options_type = ResolveOptions
    units_match_country = False

    base_path: Path = Field(default=global_config.get_path("gshm", "bronze"))

    def get_search_key(self, source, options: ResolveOptions) -> None:
        """The map is one global file, so the source does not select a unit."""
        return None

    def get_relevant_data_units_by_key(
        self, search_key, options: ResolveOptions
    ) -> List[dict]:
        """
        Return the single Zenodo archive.

        Args:
            search_key: Always None.
            options: Resolve options. None of them select a different file.

        Returns:
            A one-element list with the archive URL.
        """
        return [{"url": self.ARCHIVE_URL}]

    def get_data_unit_path(self, unit: dict, options=None) -> Path:
        """
        Return the stored GeoPackage for the archive.

        Args:
            unit: The archive unit. The URL is fixed for this handler.
            options: Unused.

        Returns:
            ``{base_path}/v2026.1/gem_gshm_v2026.1.gpkg``.
        """
        return self.base_path / self.VERSION / self.MEMBER_NAME


class GSHMDownloader(BaseHandlerDownloader):
    """Downloads the GSHM archive and stores its GeoPackage."""

    options_type = DownloadOptions
    config_type = GSHMConfig

    def download_data_unit(self, unit: dict, options: DownloadOptions) -> Path:
        """
        Download the archive and store the point GeoPackage.

        Args:
            unit: The archive unit from the config.
            options: ``show_progress`` controls the byte progress bar.

        Returns:
            The stored GeoPackage path.

        Raises:
            requests.RequestException: If the download fails.
            zipfile.BadZipFile: If the download is not a zip archive.
            FileNotFoundError: If the GeoPackage member is missing.
        """
        stored = extract_zip_to_store(
            unit["url"],
            self.config.get_data_unit_path(unit).parent,
            self.data_store,
            MEMBER_PATTERN,
            show_progress=options.show_progress,
        )
        return stored[0]


class GSHMReader(BaseHandlerReader):
    """Loads the GSHM point layer."""

    options_type = ReadOptions
    config_type = GSHMConfig

    def load_from_paths(
        self, source_data_path: List[str], options: ReadOptions
    ) -> gpd.GeoDataFrame:
        """
        Load the stored GeoPackage.

        Args:
            source_data_path: Stored GeoPackage paths.
            options: The read options. The layer is fixed.

        Returns:
            The point layer, with ``lon``, ``lat``, ``pga``, and point geometry.
        """
        return self._read_layer(source_data_path, bbox=None)

    def load(self, source, crop_to_source: bool = False, **kwargs):
        """
        Load the point layer, optionally limited to the source geometry.

        A cropped load reads the points inside the search geometry's bounds,
        then applies the base geometry crop. The archive has no spatial index,
        so the bounds read still scans the file, and it does not build the
        global table first. A path source is returned whole.

        Args:
            source: Country, geometry, points, DataFrame, GeoDataFrame, or path.
            crop_to_source: If True, keep points that intersect the source.
            **kwargs: Resolve and read options.

        Returns:
            The point GeoDataFrame.

        Raises:
            TypeError: If a keyword is neither a resolve nor a read option.
            ValueError: If cropping is requested for a source with no geometry.
        """
        resolve, _read = sort_options(kwargs, self.config, self, entry="load")
        paths = self._resolve_paths(source, resolve)
        if not paths:
            self.logger.warning(
                "No source data paths resolved. There's no matching data to load!"
            )
            return None

        geometry = None
        if crop_to_source and self._crops(source):
            geometry = self.config.get_search_geometry(source, **asdict(resolve))

        loaded = self._read_layer(
            [str(path) for path in paths],
            bbox=None if geometry is None else geometry.bounds,
        )
        if geometry is not None and loaded is not None:
            loaded = self.crop_to_geometry(loaded, geometry, crs=self.config.crs)
        return loaded

    def _crops(self, source) -> bool:
        kind = classify_source(source)
        if kind is SourceKind.PATHS:
            return False
        if kind is SourceKind.COUNTRY and self.config.units_match_country:
            return False
        if kind is SourceKind.OTHER:
            raise ValueError(
                f"crop_to_source=True needs a source with a geometry, got {type(source)}"
            )
        return True

    def _read_layer(
        self, paths: List[str], bbox: Optional[tuple]
    ) -> gpd.GeoDataFrame:
        frames = [self._read_gpkg(path, bbox) for path in paths]
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True)

    def _read_gpkg(self, path: str, bbox: Optional[tuple]) -> gpd.GeoDataFrame:
        kwargs = {"layer": LAYER}
        if bbox is not None:
            kwargs["bbox"] = tuple(bbox)
        local = self._local_path(path)
        if local is not None:
            return gpd.read_file(local, **kwargs)
        return read_dataset(path, data_store=self.data_store, **kwargs)

    def _local_path(self, path: str):
        if isinstance(self.data_store, LocalDataStore):
            return self.data_store._resolve_path(path)
        return None


class GSHMHandler(BaseHandler):
    """
    Handler for the GEM Global Seismic Hazard Map v2026.1.

    ``GSHMHandler(base_path=...)`` stores the GeoPackage under that directory.
    """

    def create_config(self, data_store, logger, **kwargs) -> GSHMConfig:
        return GSHMConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(self, config, data_store, logger) -> GSHMDownloader:
        return GSHMDownloader(config=config, data_store=data_store, logger=logger)

    def create_reader(self, config, data_store, logger) -> GSHMReader:
        return GSHMReader(config=config, data_store=data_store, logger=logger)
