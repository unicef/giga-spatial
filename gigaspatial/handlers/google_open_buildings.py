from dataclasses import dataclass
from pathlib import Path
import functools
import multiprocessing
from typing import List, Optional, Union, Literal
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
import requests
from tqdm import tqdm
import logging

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.handlers.boundaries import AdminBoundaries
from gigaspatial.processing.geo import calculate_distance, convert_to_geodataframe
from gigaspatial.config import config


@dataclass
class GoogleOpenBuildingsConfig:
    """Configuration for Google Open Buildings dataset files."""

    base_path: Path = config.get_path("google_open_buildings", "bronze")
    data_types: tuple = ("polygons", "points")
    n_workers: int = 4  # number of workers for parallel processing

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


class GoogleOpenBuildingsDownloader:
    """A class to handle downloads of Google's Open Buildings dataset."""

    TILES_URL = "https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson"

    def __init__(
        self,
        config: Optional[GoogleOpenBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the downloader.

        Args:
            config: Optional configuration for file paths
            data_store: Instance of DataStore for accessing data storage
            logger: Optional custom logger. If not provided, uses default logger.
        """
        self.logger = logger or config.get_logger(__name__)
        self.data_store = data_store or LocalDataStore()
        self.config = config or GoogleOpenBuildingsConfig()

        # Load and cache S2 tiles
        self._load_s2_tiles()

    def _load_s2_tiles(self):
        """Load S2 tiles from GeoJSON file."""
        response = requests.get(self.TILES_URL)
        response.raise_for_status()

        # Convert to GeoDataFrame
        self.tiles_gdf = gpd.GeoDataFrame.from_features(
            response.json()["features"], crs="EPSG:4326"
        )

    def _get_intersecting_tiles(
        self, geometry: Union[Polygon, MultiPolygon, gpd.GeoDataFrame]
    ) -> pd.DataFrame:
        """Get tiles that intersect with the given geometry."""

        if isinstance(geometry, gpd.GeoDataFrame):
            if geometry.crs != "EPSG:4326":
                geometry = geometry.to_crs("EPSG:4326")
            search_geom = geometry.geometry.unary_union
        else:
            search_geom = geometry

        # Find intersecting tiles
        mask = (
            tile_geom.intersects(search_geom) for tile_geom in self.tiles_gdf.geometry
        )

        return self.tiles_gdf.loc[mask, ["tile_id", "tile_url", "size_mb"]]

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

    def get_download_size_estimate(
        self, geometry: Union[Polygon, MultiPolygon, gpd.GeoDataFrame]
    ) -> float:
        """
        Estimate the download size in MB for a given geometry or GeoDataFrame.

        Args:
            geometry: Shapely Polygon/MultiPolygon or GeoDataFrame with geometries

        Returns:
            Estimated size in megabytes
        """
        gdf_tiles = self._get_intersecting_tiles(geometry)

        return gdf_tiles["size_mb"].sum()

    def download_by_country(
        self,
        country_code: str,
        data_type: Literal["polygons", "points"] = "polygons",
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """
        Download Google Open Buildings data for a specific country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code
            data_type: Type of data to download ('polygons' or 'points')

        Returns:
            List of paths to downloaded files
        """

        gdf_admin0 = AdminBoundaries.create(
            country_code=country_code,
            admin_level=0,
            data_store=data_store,
            path=country_geom_path,
        ).to_geodataframe()

        # Get intersecting tiles
        gdf_tiles = self._get_intersecting_tiles(gdf_admin0)

        if gdf_tiles.empty:
            self.logger.warning(f"There is no matching data for {country_code}")
            return []

        # Download tiles in parallel
        with multiprocessing.Pool(self.config.n_workers) as pool:
            download_func = functools.partial(self._download_tile, data_type=data_type)
            file_paths = list(
                tqdm(
                    pool.imap(download_func, [row for _, row in gdf_tiles.iterrows()]),
                    total=len(gdf_tiles),
                    desc=f"Downloading {data_type} for {country_code}",
                )
            )

        # Filter out None values (failed downloads)
        return [path for path in file_paths if path is not None]

    def download_by_points(
        self,
        points_gdf: gpd.GeoDataFrame,
        data_type: Literal["polygons", "points"] = "polygons",
    ) -> List[str]:
        """
        Download Google Open Buildings data for areas containing specific points.

        Args:
            points_gdf: GeoDataFrame containing points of interest
            data_type: Type of data to download ('polygons' or 'points')

        Returns:
            List of paths to downloaded files
        """
        # Get intersecting tiles
        gdf_tiles = self._get_intersecting_tiles(points_gdf)

        if gdf_tiles.empty:
            self.logger.warning(f"There is no matching data for the points")
            return []

        # Download tiles in parallel
        with multiprocessing.Pool(self.config.n_workers) as pool:
            download_func = functools.partial(self._download_tile, data_type=data_type)
            file_paths = list(
                tqdm(
                    pool.imap(download_func, [row for _, row in gdf_tiles.iterrows()]),
                    total=len(gdf_tiles),
                    desc=f"Downloading {data_type} for points dataset",
                )
            )

        # Filter out None values (failed downloads)
        return [path for path in file_paths if path is not None]


class GoogleOpenBuildingsMapper:
    """A class to map and analyze Google Open Buildings data in relation to points of interest."""

    def __init__(
        self,
        config: Optional[GoogleOpenBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
    ):
        """
        Initialize the GoogleOpenBuildingsMapper.

        Args:
            n_workers: Number of workers for parallel processing
            config: Optional configuration for file paths
        """
        self.config = config or GoogleOpenBuildingsConfig()
        self.handler = GoogleOpenBuildingsDownloader(data_store=data_store)

    def load_data(
        self, tiles: pd.DataFrame, data_type: Literal["polygons", "points"]
    ) -> pd.DataFrame:
        """
        Load building data from tiles, downloading if necessary.

        Args:
            tiles: DataFrame containing tile information
            data_type: Type of data to load ('polygons' or 'points')

        Returns:
            DataFrame containing building data
        """
        all_data = []

        for _, tile in tiles.iterrows():
            file_path = str(self.config.get_tile_path(tile.tile_id, data_type))

            if not self.handler.data_store.file_exists(file_path):
                downloaded_path = self.handler._download_tile(tile, data_type)
                if downloaded_path is None:
                    continue

                file_path = downloaded_path

            df = read_dataset(self.handler.data_store, file_path)

            # If data type is polygons, convert WKT strings to geometry objects
            if data_type == "polygons":
                df["geometry"] = df["geometry"].apply(wkt.loads)
                df = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

            all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        # Concatenate all tile data
        result = pd.concat(all_data, ignore_index=True)

        # If polygons, ensure result is GeoDataFrame
        if data_type == "polygons":
            result = gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")

        return result

    def map_nearest_building(
        self,
        df_points: Union[pd.DataFrame, gpd.GeoDataFrame],
        max_distance: Optional[float] = None,
    ):
        """
        Map each point to its nearest building.

        Args:
            df_points: DataFrame or GeoDataFrame containing points
            max_distance: Maximum distance in meters to consider (optional)

        Returns:
            GeoDataFrame with nearest building information
        """
        if isinstance(df_points, pd.DataFrame):
            gdf_points = convert_to_geodataframe(df_points)

        intersection_tiles = self.handler._get_intersecting_tiles(gdf_points)

        if intersection_tiles.empty:
            return df_points

        df_buildings = self.load_data(tiles=intersection_tiles, data_type="points")

        tree = cKDTree(df_buildings[["latitude", "longitude"]])

        _, idx = tree.query(df_points[["latitude", "longitude"]], k=1)

        df_nearest_buildings = df_buildings.iloc[idx]

        dist = calculate_distance(
            lat1=df_points.latitude,
            lon1=df_points.longitude,
            lat2=df_nearest_buildings.latitude,
            lon2=df_nearest_buildings.longitude,
        )

        df_points["nearest_building_full_plus_code"] = (
            df_nearest_buildings.full_plus_code.to_numpy()
        )
        df_points["nearest_building_distance"] = dist

        # Filter by max distance if specified
        if max_distance is not None:
            df_points = df_points[
                df_points["nearest_building_distance"] <= max_distance
            ]

        return df_points
