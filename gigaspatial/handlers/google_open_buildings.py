from dataclasses import dataclass
from pathlib import Path
import functools
import multiprocessing
from typing import List, Optional, Union, Literal, Tuple, Iterable
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint, Point
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
)
from gigaspatial.config import config as global_config


@dataclass
class GoogleOpenBuildingsConfig(BaseHandlerConfig):
    """
    Configuration for Google Open Buildings dataset files.
    Implements the BaseHandlerConfig interface for data unit resolution.
    """

    TILES_URL: str = (
        "https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson"
    )
    base_path: Path = global_config.get_path("google_open_buildings", "bronze")
    data_types: tuple = ("polygons", "points")

    def __post_init__(self):
        super().__post_init__()
        self._load_s2_tiles()

    def _load_s2_tiles(self):
        """Load S2 tiles from GeoJSON file."""
        response = requests.get(self.TILES_URL)
        response.raise_for_status()
        self.tiles_gdf = gpd.GeoDataFrame.from_features(
            response.json()["features"], crs="EPSG:4326"
        )

    def get_relevant_data_units_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> List[dict]:
        """
        Return intersecting tiles for a given geometry or GeoDataFrame.
        """
        return self._get_relevant_tiles(geometry)

    def get_relevant_data_units_by_points(
        self, points: Iterable[Union[Point, tuple]], **kwargs
    ) -> List[dict]:
        """
        Return intersecting tiles for a list of points.
        """
        return self._get_relevant_tiles(points)

    def get_data_unit_path(
        self,
        unit: Union[pd.Series, dict, str],
        data_type: str = "polygons",
        **kwargs,
    ) -> Path:
        """
        Given a tile row or tile_id, return the corresponding file path.
        """
        tile_id = (
            unit["tile_id"]
            if isinstance(unit, pd.Series) or isinstance(unit, dict)
            else unit
        )
        return self.base_path / f"{data_type}_s2_level_4_{tile_id}_buildings.csv.gz"

    def get_data_unit_paths(
        self,
        units: Union[pd.DataFrame, Iterable[Union[dict, str]]],
        data_type: str = "polygons",
        **kwargs,
    ) -> list:
        """
        Given data unit identifiers, return the corresponding file paths.
        """
        if isinstance(units, pd.DataFrame):
            return [
                self.get_data_unit_path(row, data_type=data_type, **kwargs)
                for _, row in units.iterrows()
            ]
        return super().get_data_unit_paths(units, data_type=data_type)

    def _get_relevant_tiles(
        self,
        source: Union[
            BaseGeometry,
            gpd.GeoDataFrame,
            Iterable[Union[Point, tuple]],
        ],
    ) -> List[dict]:
        """
        Identify and return the S2 tiles that spatially intersect with the given geometry.
        """
        if isinstance(source, gpd.GeoDataFrame):
            if source.crs != "EPSG:4326":
                source = source.to_crs("EPSG:4326")
            search_geom = source.geometry.unary_union
        elif isinstance(source, BaseGeometry):
            search_geom = source
        elif isinstance(source, Iterable) and all(
            len(pt) == 2 or isinstance(pt, Point) for pt in source
        ):
            points = [
                pt if isinstance(pt, Point) else Point(pt[1], pt[0]) for pt in source
            ]
            search_geom = MultiPoint(points)
        else:
            raise ValueError(
                f"Expected Geometry, GeoDataFrame or iterable object of Points got {source.__class__}"
            )
        mask = (
            tile_geom.intersects(search_geom) for tile_geom in self.tiles_gdf.geometry
        )
        return self.tiles_gdf.loc[mask, ["tile_id", "tile_url", "size_mb"]].to_dict(
            "records"
        )


class GoogleOpenBuildingsDownloader(BaseHandlerDownloader):
    """A class to handle downloads of Google's Open Buildings dataset."""

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
        self,
        tile_info: Union[pd.Series, dict],
        data_type: Literal["polygons", "points"] = "polygons",
    ) -> Optional[str]:
        """Download data file for a single tile."""

        tile_url = tile_info["tile_url"]
        if data_type == "points":
            tile_url = tile_url.replace("polygons", "points")

        try:
            response = requests.get(tile_url, stream=True)
            response.raise_for_status()

            file_path = str(
                self.config.get_data_unit_path(
                    tile_info["tile_id"], data_type=data_type
                )
            )

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

    def download_data_units(
        self,
        tiles: Union[pd.DataFrame, List[dict]],
        data_type: Literal["polygons", "points"] = "polygons",
    ) -> List[str]:
        """Download data files for multiple tiles."""

        if len(tiles) == 0:
            self.logger.warning(f"There is no matching data")
            return []

        with multiprocessing.Pool(self.config.n_workers) as pool:
            download_func = functools.partial(
                self.download_data_unit, data_type=data_type
            )
            file_paths = list(
                tqdm(
                    pool.imap(
                        download_func,
                        (
                            [row for _, row in tiles.iterrows()]
                            if isinstance(tiles, pd.DataFrame)
                            else tiles
                        ),
                    ),
                    total=len(tiles),
                    desc=f"Downloading {data_type} data",
                )
            )

        return [path for path in file_paths if path is not None]

    def download(
        self,
        source: Union[
            str,  # country
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,  # shapely geoms
            gpd.GeoDataFrame,
        ],
        data_type: Literal["polygons", "points"] = "polygons",
        **kwargs,
    ) -> List[str]:
        """Download Google Open Buildings data for a specified geographic region.

        The region can be defined by a country code/name, a list of points,
        a Shapely geometry, or a GeoDataFrame. This method identifies the
        relevant S2 tiles intersecting the region and downloads the
        specified type of data (polygons or points) for those tiles in parallel.

        Args:
            source: Defines the geographic area for which to download data.
                    Can be:
                      - A string representing a country code or name.
                      - A list of (latitude, longitude) tuples or Shapely Point objects.
                      - A Shapely BaseGeometry object (e.g., Polygon, MultiPolygon).
                      - A GeoDataFrame with geometry column in EPSG:4326.
            data_type: The type of building data to download ('polygons' or 'points').
                       Defaults to 'polygons'.
            **kwargs: Additional keyword arguments that are passed to
                      `AdminBoundaries.create()` if `source` is a country code.
                      For example, `path` to a custom boundaries file.

        Returns:
            A list of local file paths for the successfully downloaded tiles.
            Returns an empty list if no data is found for the region or if
            all downloads fail.
        """

        tiles = self.config.get_relevant_data_units(source, **kwargs)
        return self.download_data_units(tiles, data_type)

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
            source=country,
            data_type=data_type,
            data_store=data_store,
            path=country_geom_path,
        )


class GoogleOpenBuildingsReader(BaseHandlerReader):
    """
    Reader for Google Open Buildings data, supporting country, points, and geometry-based resolution.
    """

    def __init__(
        self,
        config: Optional[GoogleOpenBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config or GoogleOpenBuildingsConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> gpd.GeoDataFrame:
        """
        Load building data from Google Open Buildings dataset.
        Args:
            source_data_path: List of file paths to load
        Returns:
            GeoDataFrame containing building data
        """
        result = self._load_tabular_data(file_paths=source_data_path)
        return result

    def load(self, source, data_type="polygons", **kwargs):
        return super().load(source=source, data_type=data_type, **kwargs)

    def load_points(self, source, **kwargs):
        """This is a convenience method to load points data"""
        return self.load(source=source, data_type="points", **kwargs)

    def load_polygons(self, source, **kwargs):
        """This is a convenience method to load polygons data"""
        return self.load(source=source, data_type="polygons", **kwargs)


class GoogleOpenBuildingsHandler(BaseHandler):
    """
    Handler for Google Open Buildings dataset.

    This class provides a unified interface for downloading and loading Google Open Buildings data.
    It manages the lifecycle of configuration, downloading, and reading components.
    """

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> GoogleOpenBuildingsConfig:
        """
        Create and return a GoogleOpenBuildingsConfig instance.

        Args:
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional configuration parameters

        Returns:
            Configured GoogleOpenBuildingsConfig instance
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
        Create and return a GoogleOpenBuildingsReader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional reader parameters

        Returns:
            Configured GoogleOpenBuildingsReader instance
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
            ensure_available=ensure_available,
            data_type="polygons",
            **kwargs,
        )
