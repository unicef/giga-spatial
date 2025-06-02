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
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.boundaries import AdminBoundaries
from gigaspatial.handlers.base_reader import BaseHandlerReader
from gigaspatial.config import config as global_config


@dataclass
class GoogleOpenBuildingsConfig:
    """Configuration for Google Open Buildings dataset files."""

    TILES_URL = "https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson"

    base_path: Path = global_config.get_path("google_open_buildings", "bronze")
    data_types: tuple = ("polygons", "points")
    n_workers: int = 4  # number of workers for parallel processing

    def __post_init__(self):
        self._load_s2_tiles()

    def _load_s2_tiles(self):
        """Load S2 tiles from GeoJSON file."""
        response = requests.get(self.TILES_URL)
        response.raise_for_status()

        # Convert to GeoDataFrame
        self.tiles_gdf = gpd.GeoDataFrame.from_features(
            response.json()["features"], crs="EPSG:4326"
        )

    def get_intersecting_tiles(
        self,
        source: Union[
            BaseGeometry,
            gpd.GeoDataFrame,
            List[Union[Point, tuple]],
        ],
    ) -> pd.DataFrame:
        """
        Identify and return the S2 tiles that spatially intersect with the given geometry.

        The input geometry can be a Shapely geometry object, a GeoDataFrame,
        or a list of Point objects or (lon, lat) tuples. The method ensures
        the input geometry is in EPSG:4326 for the spatial intersection.

        Args:
            source: A Shapely geometry, a GeoDataFrame, or a list of Point
                      objects or (lat, lon) tuples representing the area of interest.

        Returns:
            A pandas DataFrame containing the 'tile_id', 'tile_url', and
            'size_mb' for the intersecting tiles.

        Raises:
            ValueError: If the input `source` is not one of the supported types.
        """

        if isinstance(source, gpd.GeoDataFrame):
            if source.crs != "EPSG:4326":
                source = source.to_crs("EPSG:4326")
            search_geom = source.geometry.unary_union
        elif isinstance(
            source,
            BaseGeometry,
        ):
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

        # Find intersecting tiles
        mask = (
            tile_geom.intersects(search_geom) for tile_geom in self.tiles_gdf.geometry
        )

        return self.tiles_gdf.loc[mask, ["tile_id", "tile_url", "size_mb"]]

    def get_download_size_estimate(
        self,
        geometry: Union[BaseGeometry, gpd.GeoDataFrame, List[Union[Point, tuple]]],
    ) -> float:
        """
        Estimate the total download size of the tiles intersecting a given geometry.

        This method takes a geographic area (as a Shapely geometry,
        GeoDataFrame, or list of points/tuples), finds the corresponding
        S2 tiles, and sums up their reported sizes.

        Args:
            geometry: A Shapely geometry, a GeoDataFrame, or a list of Point
                      objects or (lat, lon) tuples representing the area of interest.

        Returns:
            The estimated total download size in megabytes (MB) for the
            intersecting tiles.
        """
        gdf_tiles = self.get_intersecting_tiles(geometry)

        return gdf_tiles["size_mb"].sum()

    def get_tile_path(
        self, tile_id: str, data_type: Literal["polygons", "points"]
    ) -> Path:
        """
        Construct the full path for a tile file.

        Args:
            tile_id: S2 tile identifier
            data_type: Type of building data ('polygons' or 'points')

        Returns:
            Full path to the tile file
        """
        if data_type not in self.data_types:
            raise ValueError(f"data_type must be one of {self.data_types}")

        return self.base_path / f"{data_type}_s2_level_4_{tile_id}_buildings.csv.gz"

    def get_tile_paths(
        self, tiles: pd.DataFrame, data_type: Literal["polygons", "points"]
    ) -> List:
        if tiles.empty:
            return []

        return [
            self.get_tile_path(tile_id=tile, data_type=data_type)
            for tile in tiles.tile_id
        ]


class GoogleOpenBuildingsDownloader:
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
        self.data_store = data_store or LocalDataStore()
        self.config = config or GoogleOpenBuildingsConfig()
        self.logger = logger or global_config.get_logger(self.__class__.__name__)

    def _download_tile(
        self,
        tile_info: Union[pd.Series, dict],
        data_type: Literal["polygons", "points"],
    ) -> Optional[str]:
        """Download data file for a single tile."""

        tile_url = tile_info["tile_url"]
        if data_type == "points":
            tile_url = tile_url.replace("polygons", "points")

        try:
            response = requests.get(tile_url, stream=True)
            response.raise_for_status()

            file_path = str(self.config.get_tile_path(tile_info["tile_id"], data_type))

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

    def _download_tiles(self, tiles: gpd.GeoDataFrame, data_type="polygons"):
        """Download data file for multiple tiles."""

        if tiles.empty:
            self.logger.warning(f"There is no matching data for the region")
            return []

        with multiprocessing.Pool(self.config.n_workers) as pool:
            download_func = functools.partial(self._download_tile, data_type=data_type)
            file_paths = list(
                tqdm(
                    pool.imap(download_func, [row for _, row in tiles.iterrows()]),
                    total=len(tiles),
                    desc=f"Downloading {data_type} data",
                )
            )

        return [path for path in file_paths if path is not None]

    def download_data(
        self,
        source: Union[
            str,  # country code
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,  # shapely geoms
            gpd.GeoDataFrame,
        ],
        data_type: Literal["polygons", "points"] = "polygons",
        **kwargs,
    ):
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
        if isinstance(source, str):
            region = (
                AdminBoundaries.create(country_code=source, **kwargs)
                .boundaries[0]
                .geometry
            )
        elif isinstance(source, (BaseGeometry, Iterable)):
            region = source
        else:
            raise ValueError(
                f"Data downloads supported for Country, Geometry, GeoDataFrame or iterable object of Points got {region.__class__}"
            )

        # Get intersecting tiles
        gdf_tiles = self.config.get_intersecting_tiles(region)

        return self._download_tiles(gdf_tiles, data_type)

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

        return self.download_data(
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
    ):
        super().__init__(data_store=data_store)
        self.config = config or GoogleOpenBuildingsConfig()

    def _post_load_hook(self, data, **kwargs) -> gpd.GeoDataFrame:
        """Post-processing after loading data files."""
        if data.empty:
            self.logger.warning("No data was loaded from the source files")
            return data

        self.logger.info(
            f"Post-load processing complete. {len(data)} valid building records."
        )
        return data

    def resolve_by_country(self, country: str, **kwargs) -> List[Union[str, Path]]:
        area = (
            AdminBoundaries.create(country_code=country, **kwargs)
            .boundaries[0]
            .geometry
        )
        return self.resolve_by_geometry(geometry=area, **kwargs)

    def resolve_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> List[Union[str, Path]]:
        tiles = self.config.get_intersecting_tiles(geometry)
        return self.config.get_tile_paths(tiles, **kwargs)

    def resolve_by_points(
        self, points: List[Union[Point, Tuple[float, float]]], **kwargs
    ) -> List[Union[str, Path]]:
        return self.resolve_by_geometry(points, **kwargs)

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

    def load(self, source, data_type="polygons"):
        return super().load(source=source, data_type=data_type)

    def load_points(self, source):
        return self.load(source=source, data_type="points")

    def load_polygons(self, source):
        return self.load(source=source, data_type="polygons")
