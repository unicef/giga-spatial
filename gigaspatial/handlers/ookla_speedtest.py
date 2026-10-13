"""
Ookla Speedtest Open Data handler.

This module provides handlers for interacting with Ookla's open datasets, which
are hosted on AWS S3 in Parquet format. It supports:
- Automatic dataset discovery by year and quarter.
- Filtering by network type (fixed vs mobile).
- Spatial filtering using Bing Maps Quadkeys (Zoom 16).
- Efficient Parquet loading into tabular or geospatial formats.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from datetime import datetime
import requests
from pathlib import Path
from pydantic import ConfigDict, Field
from typing import List, Literal, Optional, Tuple, Union
from pydantic.dataclasses import dataclass
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point

from gigaspatial.grid.mercator_tiles import MercatorTiles
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.config import config
from gigaspatial.handlers.base import BaseHandlerConfig
from gigaspatial.handlers.base import BaseHandlerDownloader
from gigaspatial.handlers.base import BaseHandlerReader
from gigaspatial.handlers.base import BaseHandler
from gigaspatial.handlers.base import DownloadOptions, ReadOptions, ResolveOptions
from gigaspatial.handlers.base.fetch import stream_to_store

import logging


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class OoklaSpeedtestConfig(BaseHandlerConfig):
    """
    Configuration for Ookla Speedtest dataset access.

    Manages dataset identification (type, year, quarter) and resolves remote
    S3 URLs to local storage paths.

    Attributes:
        type: Network connection type ('fixed' or 'mobile').
        year: Targeted data year (defaults to latest available).
        quarter: Targeted data quarter (1-4, defaults to latest available).
        base_path: Root directory for local parquet storage.
    """

    MIN_YEAR = 2019
    MAX_YEAR = datetime.today().year
    MAX_QUARTER = int(np.floor((datetime.today().month - 1) / 3))
    if MAX_QUARTER == 0:
        MAX_YEAR -= 1
        MAX_QUARTER = 4

    BASE_URL = "https://ookla-open-data.s3.amazonaws.com/parquet/performance"

    base_path: Path = Field(default=config.get_path("ookla_speedtest", "bronze"))

    type: Literal["fixed", "mobile"] = Field(...)
    year: Optional[int] = Field(default=None, ge=MIN_YEAR, le=MAX_YEAR)
    quarter: Optional[int] = Field(default=None, ge=0, le=4)

    options_type = ResolveOptions

    def __post_init__(self):
        if self.year is None:
            self.year = self.MAX_YEAR
            self.logger.warning(
                "Year not provided. Using the latest available data year: %s", self.year
            )
        if self.quarter is None:
            self.quarter = self.MAX_QUARTER
            self.logger.warning(
                "Quarter not provided. Using the latest available data quarter for year %s: %s",
                self.year,
                self.quarter,
            )

        super().__post_init__()
        self.DATASET_URL = self._get_dataset_url(self.type, self.year, self.quarter)

    def _get_dataset_url(self, type: str, year: int, quarter: int) -> str:
        """Constructs the remote S3 URL for a specific Ookla dataset.

        Args:
            type: Network type ('fixed' or 'mobile').
            year: Data year.
            quarter: Data quarter (1-4).

        Returns:
            The full remote S3 URL to the Parquet file.
        """
        month = [1, 4, 7, 10]
        quarter_start = datetime(year, month[self.quarter - 1], 1)
        return f"{self.BASE_URL}/type={type}/year={quarter_start:%Y}/quarter={quarter}/{quarter_start:%Y-%m-%d}_performance_{type}_tiles.parquet"

    @staticmethod
    def get_available_datasets():
        """
        Enumerate all potentially available Ookla Speedtest datasets on S3.

        Returns:
            A list of dictionaries containing 'service_type', 'year', and 'quarter'.
        """
        start_year = 2019  # first data year
        max_year = datetime.today().year
        max_quarter = np.floor((datetime.today().month - 1) / 3)
        if max_quarter == 0:
            max_year -= 1
            max_quarter = 4

        ookla_tiles = []
        for year in range(start_year, max_year + 1):
            for quarter in range(1, 5):
                if year == max_year and quarter > max_quarter:
                    continue
                for type in ["fixed", "mobile"]:
                    ookla_tiles.append(
                        {"service_type": type, "year": year, "quarter": quarter}
                    )

        return ookla_tiles

    def get_search_key(self, source, options: ResolveOptions) -> None:
        """Ookla data is one global file per quarter: no source narrows it."""
        return None

    def get_relevant_data_units_by_key(
        self, search_key, options: ResolveOptions
    ) -> List[str]:
        """
        Return the Parquet URL for the configured year and quarter.

        Args:
            search_key: Always None; the dataset is a single global file.
            options: The resolve options (unused).

        Returns:
            A list of dataset URLs.
        """
        return [self.DATASET_URL]

    def get_data_unit_path(self, unit: str, options=None) -> Path:
        """Resolves a remote Ookla dataset URL to its local storage path.

        Args:
            unit: The remote Parquet URL.
            options: The resolve options (unused).

        Returns:
            Absolute path to the local data file.
        """
        return self.base_path / unit.split("/")[-1]


class OoklaSpeedtestDownloader(BaseHandlerDownloader):
    """
    Downloader for Ookla Speedtest Parquet files.

    Acquires performance tiles from AWS S3 and persists them to local storage.
    """

    options_type = DownloadOptions

    def __init__(
        self,
        config: Union[OoklaSpeedtestConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = (
            config
            if isinstance(config, OoklaSpeedtestConfig)
            else OoklaSpeedtestConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(self, url: str, options: DownloadOptions) -> Path:
        """
        Download a single Ookla Parquet file.

        Args:
            url: Remote S3 URL.
            options: ``show_progress`` controls the progress bar.

        Returns:
            The stored file path.

        Raises:
            requests.RequestException: If the download fails.
        """
        return stream_to_store(
            url,
            self.config.get_data_unit_path(url),
            self.data_store,
            show_progress=options.show_progress,
        )


class OoklaSpeedtestReader(BaseHandlerReader):
    """
    Reader for Ookla Speedtest Parquet files.

    Loads and processes performance tiles into tabular formats.
    """

    QUADKEY_ZOOM_LEVEL = 16
    options_type = ReadOptions

    def __init__(
        self,
        config: Union[OoklaSpeedtestConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = (
            config
            if isinstance(config, OoklaSpeedtestConfig)
            else OoklaSpeedtestConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], options: ReadOptions
    ) -> pd.DataFrame:
        """
        Load Ookla data from local Parquet files.

        Args:
            source_data_path: List of local file paths.
            options: The read options.

        Returns:
            A pandas DataFrame of speedtest results.
        """
        result = self._load_tabular_data(file_paths=source_data_path)
        return result

    def crop_to_geometry(self, data, geometry, **kwargs) -> pd.DataFrame:
        """
        Keep the rows whose zoom 16 quadkey tile intersects the geometry.

        Args:
            data: Ookla rows with a ``quadkey`` column.
            geometry: Search geometry in EPSG:4326.
            **kwargs: Ignored.

        Returns:
            The matching rows, re-indexed from zero.
        """
        quadkeys = MercatorTiles.from_spatial(
            geometry, zoom_level=self.QUADKEY_ZOOM_LEVEL
        ).quadkeys
        return data[data["quadkey"].isin(quadkeys)].reset_index(drop=True)

    def load(
        self,
        source: Optional[
            Union[
                str,  # country
                List[Union[Tuple[float, float], Point]],  # points
                BaseGeometry,  # geometry
                gpd.GeoDataFrame,  # geodataframe
                Path,  # path
                str,  # path
                List[Union[str, Path]],
            ]
        ] = None,
        **kwargs,
    ) -> pd.DataFrame:
        return super().load(source=source, **kwargs)


class OoklaSpeedtestHandler(BaseHandler):
    """
    Unified handler for Ookla Speedtest data.

    Coordinates configuration, downloading, and spatial filtering of Ookla
    performance tiles using Mercator tile indices.
    """

    def __init__(
        self,
        type: Literal["fixed", "mobile"],
        year: Optional[int] = None,
        quarter: Optional[int] = None,
        config: Optional[OoklaSpeedtestConfig] = None,
        downloader: Optional[OoklaSpeedtestDownloader] = None,
        reader: Optional[OoklaSpeedtestReader] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        """
        Initialize the Ookla Speedtest handler.

        Args:
            type: Network type ('fixed' or 'mobile').
            year: Data year.
            quarter: Data quarter (1-4).
            config: Optional configuration override.
            downloader: Optional downloader instance.
            reader: Optional reader instance.
            data_store: Storage interface for persistence.
            logger: Component logger instance.
            **kwargs: Additional configuration parameters.
        """
        self._type = type
        self._year = year
        self._quarter = quarter

        super().__init__(
            config=config,
            downloader=downloader,
            reader=reader,
            data_store=data_store,
            logger=logger,
        )

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> OoklaSpeedtestConfig:
        return OoklaSpeedtestConfig(
            type=self._type,
            year=self._year,
            quarter=self._quarter,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_downloader(
        self,
        config: OoklaSpeedtestConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> OoklaSpeedtestDownloader:
        return OoklaSpeedtestDownloader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def create_reader(
        self,
        config: OoklaSpeedtestConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> OoklaSpeedtestReader:
        return OoklaSpeedtestReader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def load_data(
        self,
        source: Union[
            str,  # country
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
            Path,  # path
            str,  # path
            List[Union[str, Path]],
        ] = None,
        crop_to_source: bool = True,
        process_geospatial: bool = False,
        ensure_available: bool = True,
        **kwargs,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Acquire and filter Ookla Speedtest data.

        Args:
            source: Geographic filter (country, geometry, points) or direct
                file paths. None loads the whole quarter.
            crop_to_source: If True, keeps only the tiles intersecting the
                source. Path sources and None are never cropped.
            process_geospatial: If True, returns a GeoDataFrame with tile geometries.
            ensure_available: If True, executes download if data is missing locally.
            **kwargs: Handler options (e.g. ``show_progress``).

        Returns:
            Tabular or geospatial data for the requested area.
        """
        result = super().load_data(
            source,
            crop_to_source=crop_to_source and source is not None,
            ensure_available=ensure_available,
            **kwargs,
        )

        if process_geospatial:
            # Convert 'tile' column from WKT to geometry
            result["geometry"] = result["tile"].apply(wkt.loads)
            return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")

        return result
