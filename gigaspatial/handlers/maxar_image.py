from typing import Optional, Tuple, List, Union, Any, Literal
from pydantic import BaseModel, Field, HttpUrl, field_validator
from pathlib import Path
import mercantile
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from owslib.wms import WebMapService
from time import sleep
from gigaspatial.grid.mercator_tiles import MercatorTiles
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.processing.geo import (
    convert_to_geodataframe,
    buffer_geodataframe,
)
from gigaspatial.config import config as global_config


class MaxarConfig(BaseModel):
    """Configuration for Maxar Image Downloader using Pydantic"""

    username: str = Field(
        default=global_config.MAXAR_USERNAME, description="Maxar API username"
    )
    password: str = Field(
        default=global_config.MAXAR_PASSWORD, description="Maxar API password"
    )
    connection_string: str = Field(
        default=global_config.MAXAR_CONNECTION_STRING,
        description="Maxar WMS connection string",
    )

    base_url: HttpUrl = Field(
        default="https://evwhs.digitalglobe.com/mapservice/wmsaccess?",
        description="Base URL for Maxar WMS service",
    )

    layers: List[Literal["DigitalGlobe:ImageryFootprint", "DigitalGlobe:Imagery"]] = (
        Field(
            default=["DigitalGlobe:Imagery"],
            description="List of layers to request from WMS",
        )
    )

    feature_profile: str = Field(
        default="Most_Aesthetic_Mosaic_Profile",
        description="Feature profile to use for WMS requests",
    )

    coverage_cql_filter: str = Field(
        default="", description="CQL filter for coverage selection"
    )

    exceptions: str = Field(
        default="application/vnd.ogc.se_xml",
        description="Exception handling format for WMS",
    )

    transparent: bool = Field(
        default=True,
        description="Whether the requested images should have transparency",
    )

    image_format: Literal["image/png", "image/jpeg", "image/geotiff"] = Field(
        default="image/png",
    )

    data_crs: Literal["EPSG:4326", "EPSG:3395", "EPSG:3857", "CAR:42004"] = Field(
        default="EPSG:4326"
    )

    max_retries: int = Field(
        default=3, description="Number of retries for failed image downloads"
    )

    retry_delay: int = Field(default=5, description="Delay in seconds between retries")

    @field_validator("username", "password", "connection_string")
    @classmethod
    def validate_non_empty(cls, value: str, field) -> str:
        """Ensure required credentials are provided"""
        if not value or value.strip() == "":
            raise ValueError(
                f"{field.name} cannot be empty. Please provide a valid {field.name}."
            )
        return value

    @property
    def wms_url(self) -> str:
        """Generate the full WMS URL with connection string"""
        return f"{self.base_url}connectid={self.connection_string}"

    @property
    def suffix(self) -> str:
        return f".{self.image_format.split('/')[1]}"


class MaxarImageDownloader:
    """Class to download images from Maxar"""

    def __init__(
        self,
        config: Optional[MaxarConfig] = None,
        data_store: Optional[DataStore] = None,
    ):
        """
        Initialize the downloader with Maxar config.

        Args:
            config: MaxarConfig instance containing credentials and settings
            data_store: Instance of DataStore for accessing data storage
        """
        self.config = config or MaxarConfig()
        self.wms = WebMapService(
            self.config.wms_url,
            username=self.config.username,
            password=self.config.password,
        )
        self.data_store = data_store or LocalDataStore()
        self.logger = global_config.get_logger(self.__class__.__name__)

    def _download_single_image(self, bbox, output_path: Union[Path, str], size) -> bool:
        """Download a single image from bbox and pixel size"""
        for attempt in range(self.config.max_retries):
            try:
                img_data = self.wms.getmap(
                    bbox=bbox,
                    layers=self.config.layers,
                    srs=self.config.data_crs,
                    size=size,
                    featureProfile=self.config.feature_profile,
                    coverage_cql_filter=self.config.coverage_cql_filter,
                    exceptions=self.config.exceptions,
                    transparent=self.config.transparent,
                    format=self.config.image_format,
                )
                self.data_store.write_file(str(output_path), img_data.read())
                return True
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} of downloading {output_path.name} failed: {str(e)}"
                )
                if attempt < self.config.max_retries - 1:
                    sleep(self.config.retry_delay)
                else:
                    self.logger.warning(
                        f"Failed to download {output_path.name} after {self.config.max_retries} attemps: {str(e)}"
                    )
                    return False

    def download_images_by_tiles(
        self,
        mercator_tiles: "MercatorTiles",
        output_dir: Union[str, Path],
        image_size: Tuple[int, int] = (512, 512),
        image_prefix: str = "maxar_image_",
    ) -> None:
        """
        Download images for given mercator tiles using the specified style

        Args:
            mercator_tiles: MercatorTiles instance containing quadkeys
            output_dir: Directory to save images
            image_size: Tuple of (width, height) for output images
            image_prefix: Prefix for output image names
        """
        output_dir = Path(output_dir)

        image_size_str = f"{image_size[0]}x{image_size[1]}"
        total_tiles = len(mercator_tiles.quadkeys)

        self.logger.info(
            f"Downloading {total_tiles} tiles with size {image_size_str}..."
        )

        def _get_tile_bounds(quadkey: str) -> Tuple[float]:
            """Get tile bounds from quadkey"""
            tile = mercantile.quadkey_to_tile(quadkey)
            bounds = mercantile.bounds(tile)
            return (bounds.west, bounds.south, bounds.east, bounds.north)

        def download_image(
            quadkey: str, image_size: Tuple[int, int], suffix: str = self.config.suffix
        ) -> bool:
            bounds = _get_tile_bounds(quadkey)
            file_name = f"{image_prefix}{quadkey}{suffix}"

            success = self._download_single_image(
                bounds, output_dir / file_name, image_size
            )

            return success

        successful_downloads = 0
        with tqdm(total=total_tiles) as pbar:
            for quadkey in mercator_tiles.quadkeys:
                if download_image(quadkey, image_size):
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
        image_prefix: str = "maxar_image_",
    ) -> None:
        """
        Download images for given points using the specified style

        Args:
            gdf_points: GeoDataFrame containing bounding box polygons
            output_dir: Directory to save images
            image_size: Tuple of (width, height) for output images
            image_prefix: Prefix for output image names
        """
        output_dir = Path(output_dir)

        image_size_str = f"{image_size[0]}x{image_size[1]}"
        total_images = len(gdf)

        self.logger.info(
            f"Downloading {total_images} images with size {image_size_str}..."
        )

        def download_image(
            idx: Any,
            bounds: Tuple[float, float, float, float],
            image_size,
            suffix: str = self.config.suffix,
        ) -> bool:
            file_name = f"{image_prefix}{idx}{suffix}"
            success = self._download_single_image(
                bounds, output_dir / file_name, image_size
            )
            return success

        gdf = gdf.to_crs(self.config.data_crs)

        successful_downloads = 0
        with tqdm(total=total_images) as pbar:
            for row in gdf.itertuples():
                if download_image(row.Index, tuple(row.geometry.bounds), image_size):
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
        image_prefix: str = "maxar_image_",
    ) -> None:
        """
        Download images for given coordinates by creating bounded boxes around points

        Args:
            data: Either a DataFrame with either latitude/longitude columns or a geometry column or a list of (lat, lon) tuples
            res_meters_pixel: resolution in meters per pixel
            output_dir: Directory to save images
            image_size: Tuple of (width, height) for output images
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

        buffered_gdf = buffered_gdf.to_crs(self.config.data_crs)

        self.download_images_by_bounds(
            buffered_gdf, output_dir, image_size, image_prefix
        )
