from typing import Iterable, Optional, Tuple, List, Union, Any
import requests
from pathlib import Path
import mercantile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import geopandas as gpd
import pandas as pd

from gigaspatial.grid.mercator_tiles import MercatorTiles
from gigaspatial.processing.geo import convert_to_geodataframe, buffer_geodataframe
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.config import config


class MapboxImageDownloader:
    """Class to download images from Mapbox Static Images API using a specific style"""

    BASE_URL = "https://api.mapbox.com/styles/v1"

    def __init__(
        self,
        access_token: str = config.MAPBOX_ACCESS_TOKEN,
        style_id: Optional[str] = None,
        data_store: Optional[DataStore] = None,
    ):
        """
        Initialize the downloader with Mapbox credentials

        Args:
            access_token: Mapbox access token
            style_id: Mapbox style ID to use for image download
            data_store: Instance of DataStore for accessing data storage
        """
        self.access_token = access_token
        self.style_id = style_id if style_id else "mapbox/satellite-v9"
        self.data_store = data_store or LocalDataStore()
        self.logger = config.get_logger(self.__class__.__name__)

    def _construct_url(self, bounds: Iterable[float], image_size: str) -> str:
        """Construct the Mapbox Static Images API URL"""
        bounds_str = f"[{','.join(map(str, bounds))}]"

        return (
            f"{self.BASE_URL}/{self.style_id}/static/{bounds_str}/{image_size}"
            f"?access_token={self.access_token}&attribution=false&logo=false"
        )

    def _download_single_image(self, url: str, output_path: Path) -> bool:
        """Download a single image from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()

            with self.data_store.open(str(output_path), "wb") as f:
                f.write(response.content)
            return True
        except Exception as e:
            self.logger.warning(f"Error downloading {output_path.name}: {str(e)}")
            return False

    def download_images_by_tiles(
        self,
        mercator_tiles: "MercatorTiles",
        output_dir: Union[str, Path],
        image_size: Tuple[int, int] = (512, 512),
        max_workers: int = 4,
        image_prefix: str = "image_",
    ) -> None:
        """
        Download images for given mercator tiles using the specified style

        Args:
            mercator_tiles: MercatorTiles instance containing quadkeys
            output_dir: Directory to save images
            image_size: Tuple of (width, height) for output images
            max_workers: Maximum number of concurrent downloads
            image_prefix: Prefix for output image names
        """
        output_dir = Path(output_dir)
        # self.data_store.makedirs(str(output_dir), exist_ok=True)

        image_size_str = f"{image_size[0]}x{image_size[1]}"
        total_tiles = len(mercator_tiles.quadkeys)

        self.logger.info(
            f"Downloading {total_tiles} tiles with size {image_size_str}..."
        )

        def _get_tile_bounds(quadkey: str) -> List[float]:
            """Get tile bounds from quadkey"""
            tile = mercantile.quadkey_to_tile(quadkey)
            bounds = mercantile.bounds(tile)
            return [bounds.west, bounds.south, bounds.east, bounds.north]

        def download_image(quadkey: str) -> bool:
            bounds = _get_tile_bounds(quadkey)
            file_name = f"{image_prefix}{quadkey}.png"

            url = self._construct_url(bounds, image_size_str)
            success = self._download_single_image(url, output_dir / file_name)

            return success

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(download_image, quadkey)
                for quadkey in mercator_tiles.quadkeys
            ]

            successful_downloads = 0
            with tqdm(total=total_tiles) as pbar:
                for future in as_completed(futures):
                    if future.result():
                        successful_downloads += 1
                    pbar.update(1)

        self.logger.info(
            f"Successfully downloaded {successful_downloads}/{total_tiles} images!"
        )

    def download_images_by_bounds(
        self,
        gdf: gpd.GeoDataFrame,
        output_dir: Union[str, Path],
        image_size: Tuple[int, int] = (512, 512),
        max_workers: int = 4,
        image_prefix: str = "image_",
    ) -> None:
        """
        Download images for given points using the specified style

        Args:
            gdf_points: GeoDataFrame containing bounding box polygons
            output_dir: Directory to save images
            image_size: Tuple of (width, height) for output images
            max_workers: Maximum number of concurrent downloads
            image_prefix: Prefix for output image names
        """
        output_dir = Path(output_dir)
        # self.data_store.makedirs(str(output_dir), exist_ok=True)

        image_size_str = f"{image_size[0]}x{image_size[1]}"
        total_images = len(gdf)

        self.logger.info(
            f"Downloading {total_images} images with size {image_size_str}..."
        )

        def download_image(idx: Any, bounds: Tuple[float, float, float, float]) -> bool:
            file_name = f"{image_prefix}{idx}.png"
            url = self._construct_url(bounds, image_size_str)
            success = self._download_single_image(url, output_dir / file_name)
            return success

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(download_image, row.Index, row.geometry.bounds)
                for row in gdf.itertuples()
            ]

            successful_downloads = 0
            with tqdm(total=total_images) as pbar:
                for future in as_completed(futures):
                    if future.result():
                        successful_downloads += 1
                    pbar.update(1)

        self.logger.info(
            f"Successfully downloaded {successful_downloads}/{total_images} images!"
        )

    def download_images_by_coordinates(
        self,
        data: Union[pd.DataFrame, List[Tuple[float, float]]],
        res_meters_pixel: float,
        output_dir: Union[str, Path],
        image_size: Tuple[int, int] = (512, 512),
        max_workers: int = 4,
        image_prefix: str = "image_",
    ) -> None:
        """
        Download images for given coordinates by creating bounded boxes around points

        Args:
            data: Either a DataFrame with either latitude/longitude columns or a geometry column or a list of (lat, lon) tuples
            res_meters_pixel: Size of the bounding box in meters (creates a square)
            output_dir: Directory to save images
            image_size: Tuple of (width, height) for output images
            max_workers: Maximum number of concurrent downloads
            image_prefix: Prefix for output image names
        """

        if isinstance(data, pd.DataFrame):
            coordinates_df = data
        else:
            coordinates_df = pd.DataFrame(data, columns=["latitude", "longitude"])

        gdf = convert_to_geodataframe(coordinates_df)

        buffered_gdf = buffer_geodataframe(
            gdf, res_meters_pixel / 2, cap_style="square"
        )

        self.download_images_by_bounds(
            buffered_gdf, output_dir, image_size, max_workers, image_prefix
        )
