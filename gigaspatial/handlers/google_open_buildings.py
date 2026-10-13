"""
Google Open Buildings dataset handler.

This module provides specialized handlers for the Google Open Buildings dataset,
supporting both point and polygon geometries. It handles S2-tile-based resolution,
multithreaded downloads from Google's public buckets, and spatial filtering.
"""
from dataclasses import dataclass
from pathlib import Path
import functools
import multiprocessing
from typing import List, Optional, Union, Literal, Tuple
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
import requests
from tqdm import tqdm
import logging

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.handlers.base import (
    BaseHandlerReader,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandler,
    DownloadOptions,
    ReadOptions,
    ResolveOptions,
)
from gigaspatial.config import config as global_config


@dataclass(frozen=True)
class GoogleOpenBuildingsResolveOptions(ResolveOptions):
    """
    Resolve options of Google Open Buildings.

    Attributes:
        data_type: Geometry format of the tiles ('polygons' or 'points').
            It is stored on each resolved unit, so the unit names its file.
    """

    data_type: Literal["polygons", "points"] = "polygons"


@dataclass
class GoogleOpenBuildingsConfig(BaseHandlerConfig):
    """
    Configuration for Google Open Buildings dataset files.

    Maintains settings for S2 tile resolution and local storage paths.

    Attributes:
        base_path: Root directory for storing building data.
        data_types: Supported geometry formats ('polygons', 'points').
    """

    TILES_URL: str = (
        "https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson"
    )
    options_type = GoogleOpenBuildingsResolveOptions

    base_path: Path = global_config.get_path("google_open_buildings", "bronze")
    data_types: tuple = ("polygons", "points")

    def __post_init__(self):
        super().__post_init__()
        self._load_s2_tiles()

    def _load_s2_tiles(self):
        """Loads S2 tiles from the remote GeoJSON file into a GeoDataFrame.

        Raises:
            requests.exceptions.RequestException: If the remote file cannot be fetched.
        """
        response = requests.get(self.TILES_URL)
        response.raise_for_status()
        self.tiles_gdf = gpd.GeoDataFrame.from_features(
            response.json()["features"], crs="EPSG:4326"
        )

    def get_relevant_data_units_by_key(
        self, search_key: BaseGeometry, options: GoogleOpenBuildingsResolveOptions
    ) -> List[dict]:
        """
        Identify S2 tiles intersecting a search geometry.

        Args:
            search_key: The search geometry, in EPSG:4326.
            options: ``data_type`` is stored on each unit.

        Returns:
            A list of tile records (ID, URL, size and data type).
        """
        mask = self.tiles_gdf.geometry.intersects(search_key)
        records = self.tiles_gdf.loc[mask, ["tile_id", "tile_url", "size_mb"]].to_dict(
            "records"
        )
        return [{**record, "data_type": options.data_type} for record in records]

    def get_data_unit_path(self, unit: Union[pd.Series, dict, str], options=None) -> Path:
        """
        Resolve a building tile to its local compressed CSV path.

        Args:
            unit: Tile record, or a bare tile ID for polygons.
            options: The resolve options (unused; the unit holds its data type).

        Returns:
            Absolute path to the local data file.
        """
        if isinstance(unit, (pd.Series, dict)):
            tile_id, data_type = unit["tile_id"], unit.get("data_type", "polygons")
        else:
            tile_id, data_type = unit, "polygons"
        return self.base_path / f"{data_type}_s2_level_4_{tile_id}_buildings.csv.gz"


class GoogleOpenBuildingsDownloader(BaseHandlerDownloader):
    """
    Downloader for Google's Open Buildings dataset.

    Manages the acquisition of building footprint data (CSV files) from Google's
    public dataset buckets. Supports point and polygon geometries.
    """

    options_type = DownloadOptions

    def __init__(
        self,
        config: Optional[GoogleOpenBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the downloader.

        Args:
            config: Optional configuration for file paths and download settings.
                    If None, a default `GoogleOpenBuildingsConfig` is used.
            data_store: Optional instance of a `DataStore` for managing data
                        storage. If None, a `LocalDataStore` is used.
            logger: Optional custom logger instance. If None, a default logger
                    named after the module is created and used.
        """
        config = config or GoogleOpenBuildingsConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(
        self, tile_info: Union[pd.Series, dict], options: DownloadOptions
    ) -> Optional[str]:
        """
        Download a single building data file for a specific tile.

        Args:
            tile_info: Tile record with 'tile_url', 'tile_id' and 'data_type'.
            options: The download options.

        Returns:
            The local file path on success, or None if the download failed.
        """

        tile_url = tile_info["tile_url"]
        if tile_info.get("data_type", "polygons") == "points":
            tile_url = tile_url.replace("polygons", "points")

        try:
            response = requests.get(tile_url, stream=True)
            response.raise_for_status()

            file_path = str(self.config.get_data_unit_path(tile_info))

            with self.data_store.open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

                self.logger.debug(
                    f"Successfully downloaded tile: {tile_info['tile_id']}"
                )
                return file_path

        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Failed to download tile {tile_info['tile_id']}: {str(e)}"
            )
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading dataset: {str(e)}")
            return None


    def download_by_country(
        self,
        country: str,
        data_type: Literal["polygons", "points"] = "polygons",
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """
        Download Google Open Buildings data for a specific country.

        This is a convenience method to download data for an entire country
        using its code or name.

        Args:
            country: The country code (e.g., 'USA', 'GBR') or name.
            data_type: The type of building data to download ('polygons' or 'points').
                       Defaults to 'polygons'.
            data_store: Optional instance of a `DataStore` to be used by
                        `AdminBoundaries` for loading country boundaries. If None,
                        `AdminBoundaries` will use its default data loading.
            country_geom_path: Optional path to a GeoJSON file containing the
                               country boundary. If provided, this boundary is used
                               instead of the default from `AdminBoundaries`.

        Returns:
            A list of local file paths for the successfully downloaded tiles
            for the specified country.
        """
        return self.download(
            country,
            data_type=data_type,
            country_data_store=data_store,
            country_geom_path=country_geom_path,
        )


class GoogleOpenBuildingsReader(BaseHandlerReader):
    """
    Reader for Google Open Buildings data.

    Coordinates the loading of building footprint data (CSV) from the local
    data store into GeoDataFrames.
    """

    options_type = ReadOptions

    def __init__(
        self,
        config: Optional[GoogleOpenBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config or GoogleOpenBuildingsConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], options: ReadOptions
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Load building data from specific file paths.

        Args:
            source_data_path: List of local paths to building data files.
            options: The read options.

        Returns:
            A GeoDataFrame containing the aggregated building data.
        """
        result = self._load_tabular_data(file_paths=source_data_path)
        return result

    def load_points(self, source, crop_to_source: bool = False, **kwargs) -> gpd.GeoDataFrame:
        """Convenience method to load building point data.

        Args:
            source: Geographic filter (country, geometry, or points).
            crop_to_source: If True, crops the result to the source geometry.
            **kwargs: Additional parameters for `load`.

        Returns:
            GeoDataFrame containing building points.
        """
        return self.load(
            source=source, crop_to_source=crop_to_source, data_type="points", **kwargs
        )

    def load_polygons(self, source, crop_to_source: bool = False, **kwargs) -> gpd.GeoDataFrame:
        """Convenience method to load building polygon data.

        Args:
            source: Geographic filter (country, geometry, or points).
            crop_to_source: If True, crops the result to the source geometry.
            **kwargs: Additional parameters for `load`.

        Returns:
            GeoDataFrame containing building polygons.
        """
        return self.load(
            source=source, crop_to_source=crop_to_source, data_type="polygons", **kwargs
        )


class GoogleOpenBuildingsHandler(BaseHandler):
    """
    Unified handler for the Google Open Buildings dataset.

    Provides a top-level API for acquiring and accessing building footprints
    across multiple geometry formats (points and polygons).
    """

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> GoogleOpenBuildingsConfig:
        """
        Create a configuration instance for Google Open Buildings.

        Args:
            data_store: DataStore for tile indexing.
            logger: Component logger.
            **kwargs: Configuration overrides.

        Returns:
            A configured GoogleOpenBuildingsConfig.
        """
        return GoogleOpenBuildingsConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(
        self,
        config: GoogleOpenBuildingsConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> GoogleOpenBuildingsDownloader:
        """
        Create and return a GoogleOpenBuildingsDownloader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional downloader parameters

        Returns:
            Configured GoogleOpenBuildingsDownloader instance
        """
        return GoogleOpenBuildingsDownloader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def create_reader(
        self,
        config: GoogleOpenBuildingsConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> GoogleOpenBuildingsReader:
        """
        Create a reader instance for Google Open Buildings.

        Args:
            config: Handler configuration.
            data_store: DataStore for reading files.
            logger: Component logger.
            **kwargs: Reader parameters.

        Returns:
            A configured GoogleOpenBuildingsReader.
        """
        return GoogleOpenBuildingsReader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def load_points(
        self,
        source: Union[
            str,  # country
            List[Union[tuple, Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
            Path,  # path
            List[Union[str, Path]],  # list of paths
        ],
        crop_to_source: bool = False,
        ensure_available: bool = True,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Load point data from Google Open Buildings dataset.

        Args:
            source: The data source specification
            ensure_available: If True, ensure data is downloaded before loading
            **kwargs: Additional parameters passed to load methods

        Returns:
            GeoDataFrame containing building point data
        """
        return self.load_data(
            source=source,
            crop_to_source=crop_to_source,
            ensure_available=ensure_available,
            data_type="points",
            **kwargs,
        )

    def load_polygons(
        self,
        source: Union[
            str,  # country
            List[Union[tuple, Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
            Path,  # path
            List[Union[str, Path]],  # list of paths
        ],
        crop_to_source: bool = False,
        ensure_available: bool = True,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Load polygon data from Google Open Buildings dataset.

        Args:
            source: The data source specification
            ensure_available: If True, ensure data is downloaded before loading
            **kwargs: Additional parameters passed to load methods

        Returns:
            GeoDataFrame containing building polygon data
        """
        return self.load_data(
            source=source,
            crop_to_source=crop_to_source,
            ensure_available=ensure_available,
            data_type="polygons",
            **kwargs,
        )
