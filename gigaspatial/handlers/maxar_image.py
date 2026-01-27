from typing import Optional, Tuple, List, Union, Any, Literal, Dict
from pydantic import BaseModel, Field, HttpUrl, field_validator
from datetime import datetime, date
from pathlib import Path
from tqdm import tqdm
from time import sleep
import json

import geopandas as gpd
import pandas as pd
import mercantile

from owslib.wms import WebMapService
from owslib.wfs import WebFeatureService

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

    # API Key authentication
    api_key: str = Field(
        default=global_config.MAXAR_API_KEY,
        description="Maxar API key for authentication",
    )

    # Optional OAuth token support
    bearer_token: Optional[str] = Field(
        default=None, description="OAuth bearer token (alternative to API key)"
    )

    # Updated base URL
    base_url: HttpUrl = Field(
        default="https://pro.gegd.com/streaming/v1/ogc/wms",
        description="Base URL for Maxar WMS service",
    )

    # Layers
    layers: List[Literal["Maxar:Imagery", "Maxar:FinishedFeature"]] = Field(
        default=["Maxar:Imagery"], description="List of layers to request from WMS"
    )

    # Profile
    profile: Optional[str] = Field(
        default=None, description="Stacking profile for imagery order"
    )

    # Filter
    cql_filter: str = Field(default="", description="CQL filter for feature selection")

    # Styles parameter
    styles: Optional[Literal["raster", "footprints"]] = Field(
        default=None,
        description="Render style: 'raster' for imagery (default), 'footprints' for Maxar:FinishedFeature layer",
    )

    # WMS version
    wms_version: str = Field(default="1.3.0", description="WMS version to use")

    transparent: bool = Field(
        default=True, description="Removes blank background space from returned images"
    )
    image_format: Literal["image/png", "image/jpeg", "image/geotiff"] = Field(
        default="image/png"
    )
    data_crs: Literal["EPSG:4326", "EPSG:3395", "EPSG:3857"] = Field(
        default="EPSG:4326"
    )
    max_retries: int = Field(default=3)
    retry_delay: int = Field(default=5)

    # Authentication method preference
    auth_method: Literal["api_key", "bearer_token", "header"] = Field(
        default="api_key",
        description="Authentication method: 'api_key' (query param), 'header' (custom header), 'bearer_token'",
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str, info) -> str:
        # Only validate if bearer_token is not provided
        if not value and not info.data.get("bearer_token"):
            raise ValueError("Either api_key or bearer_token must be provided")
        return value

    @property
    def suffix(self) -> str:
        return f".{self.image_format.split('/')[1]}"


class MaxarImageDownloader:

    def __init__(
        self,
        config: Optional[Union[MaxarConfig, dict]] = None,
        data_store: Optional[DataStore] = None,
        **kwargs,
    ):
        """
        Initialize the downloader with Maxar config.

        Args:
            config: MaxarConfig instance, dict with config values, or None
            data_store: Instance of DataStore for accessing data storage
            **kwargs: Individual config parameters (overrides config dict/object)

        Examples:
            # Using config object
            downloader = MaxarImageDownloader(config=MaxarConfig(api_key="..."))

            # Using dict
            downloader = MaxarImageDownloader(config={"api_key": "...", "transparent": False})

            # Using kwargs
            downloader = MaxarImageDownloader(api_key="...", image_format="image/jpeg")

            # Mixing dict and kwargs (kwargs take precedence)
            downloader = MaxarImageDownloader(
                config={"api_key": "..."},
                transparent=False
            )
        """
        # Handle different input types
        if config is None:
            # No config provided, use defaults with any kwargs
            self.config = MaxarConfig(**kwargs)
        elif isinstance(config, dict):
            # Dict provided, merge with kwargs (kwargs take precedence)
            merged_config = {**config, **kwargs}
            self.config = MaxarConfig(**merged_config)
        elif isinstance(config, MaxarConfig):
            # MaxarConfig object provided
            if kwargs:
                # If kwargs provided, create new config with overrides
                config_dict = config.model_dump()
                merged_config = {**config_dict, **kwargs}
                self.config = MaxarConfig(**merged_config)
            else:
                # Use config as-is
                self.config = config
        else:
            raise TypeError(
                f"config must be MaxarConfig, dict, or None, got {type(config)}"
            )

        self.data_store = data_store or LocalDataStore()
        self.logger = global_config.get_logger(self.__class__.__name__)

        # Build headers for authentication
        self.headers = self._build_auth_headers()

        # Initialize WMS with new authentication
        self.wms = self._initialize_wms()

    def _build_auth_headers(self) -> dict:
        """Build authentication headers based on config"""
        headers = {}

        if self.config.auth_method == "bearer_token" and self.config.bearer_token:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"
        elif self.config.auth_method == "header" and self.config.api_key:
            headers["maxar-api-key"] = self.config.api_key

        return headers

    def _initialize_wms(self):
        """Initialize WebMapService with updated authentication"""
        url = str(self.config.base_url)

        # Add API key as query param if using that method
        if self.config.auth_method == "api_key" and self.config.api_key:
            url = f"{url}?maxar_api_key={self.config.api_key}"

        # OWSLib may need headers passed differently
        return WebMapService(
            url,
            version=self.config.wms_version,
            headers=self.headers if self.headers else None,
        )

    def _initialize_wfs(self):
        """Initialize WebFeatureService with authentication"""
        # WFS uses same base URL pattern as WMS
        wfs_url = str(self.config.base_url).replace("/ogc/wms", "/ogc/wfs")

        # Add API key as query param if using that method
        if self.config.auth_method == "api_key" and self.config.api_key:
            wfs_url = f"{wfs_url}?maxar_api_key={self.config.api_key}"

        return WebFeatureService(
            wfs_url,
            version="2.0.0",  # WFS 2.0.0 is the default and recommended
            headers=self.headers if self.headers else None,
        )

    def _download_single_image(self, bbox, output_path: Union[Path, str], size) -> bool:
        """Download a single image from bbox and pixel size"""
        for attempt in range(self.config.max_retries):
            try:
                # Build parameters for getmap request
                params = {
                    "bbox": bbox,
                    "layers": self.config.layers,
                    "srs": self.config.data_crs,
                    "size": size,
                    "transparent": self.config.transparent,
                    "format": self.config.image_format,
                }

                # Add optional parameters
                if self.config.cql_filter:
                    params["cql_filter"] = self.config.cql_filter

                if self.config.profile:
                    params["profile"] = self.config.profile

                if self.config.styles:
                    params["styles"] = self.config.styles

                # Make request
                img_data = self.wms.getmap(**params)

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
                        f"Failed to download {output_path.name} after {self.config.max_retries} attempts: {str(e)}"
                    )
                    return False

    def _download_single_image_with_metadata(
        self,
        bbox: Tuple[float, float, float, float],
        output_path: Union[Path, str],
        size: Tuple[int, int],
        save_metadata: bool = True,
        feature_count: int = 10,
    ) -> bool:
        """
        Download image and optionally save associated metadata

        Args:
            bbox: Bounding box for image
            output_path: Path to save image
            size: Image dimensions
            save_meta If True, saves metadata as JSON alongside image
            count: Number of features to return with image (1-1000, default 10)
        """
        import datetime

        def _convert_timestamp(item_date_object):
            """Convert datetime/date objects to string format for JSON serialization"""
            if isinstance(item_date_object, (date, datetime, pd.Timestamp)):
                return item_date_object.strftime("%Y-%m-%d %H:%M:%S.%f")
            return item_date_object

        # Download the image
        success = self._download_single_image(bbox, output_path, size)

        if success and save_metadata:
            try:
                # Get metadata for this bbox
                # Use CQL filter with bbox to get precise features
                cql = f"BBOX(featureGeometry,{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"

                if self.config.cql_filter:
                    cql = f"{self.config.cql_filter} AND {cql}"

                metadata_gdf = self.get_imagery_metadata(
                    cql_filter=cql, count=feature_count
                )

                if len(metadata_gdf) > 0:
                    # Save metadata as JSON
                    metadata_path = Path(output_path).with_suffix(".json")

                    # Drop geometry column to avoid serialization issues
                    # (geometry is not needed in JSON metadata)
                    metadata_df = metadata_gdf.drop(
                        columns=["geometry"], errors="ignore"
                    )

                    # Convert to dict and serialize with timestamp handling
                    dict_ = metadata_df.to_dict(orient="records")
                    metadata_json = json.dumps(
                        dict_, default=_convert_timestamp, indent=2
                    )

                    # Save to file
                    with self.data_store.open(metadata_path, "w") as f:
                        f.write(metadata_json)

                    self.logger.info(f"Saved metadata to {metadata_path}")
                else:
                    self.logger.warning(f"No metadata found for bbox {bbox}")

            except Exception as e:
                self.logger.warning(f"Failed to save meta {str(e)}")

        return success

    def download_images_by_tiles(
        self,
        mercator_tiles: "MercatorTiles",
        output_dir: Union[str, Path],
        image_size: Tuple[int, int] = (512, 512),
        image_prefix: str = "maxar_image_",
        save_metadata: bool = False,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
    ) -> None:
        """
        Download images for given mercator tiles

        Args:
            mercator_tiles: MercatorTiles instance containing quadkeys
            output_dir: Directory to save images
            image_size: Tuple of (width, height) for output images
            image_prefix: Prefix for output image names
            save_metadata: If True, saves WFS metadata as JSON alongside each image
            start_date: Filter images acquired on or after this date (YYYY-MM-DD)
            end_date: Filter images acquired on or before this date (YYYY-MM-DD)

        Examples:
            # Download only recent imagery
            downloader.download_images_by_tiles(
                mercator_tiles=tiles,
                output_dir="output/",
                start_date="2024-01-01"
            )

            # Download imagery from specific date range
            downloader.download_images_by_tiles(
                mercator_tiles=tiles,
                output_dir="output/",
                start_date="2023-06-01",
                end_date="2023-12-31"
            )
        """
        # Build date filter if dates provided
        date_filter = self.build_date_filter(start_date, end_date)

        # Combine with existing config filter
        if date_filter:
            if self.config.cql_filter:
                # Temporarily add date filter to config
                original_filter = self.config.cql_filter
                self.config.cql_filter = (
                    f"({self.config.cql_filter}) AND ({date_filter})"
                )
            else:
                original_filter = ""
                self.config.cql_filter = date_filter
        else:
            original_filter = None

        try:
            output_dir = Path(output_dir)
            image_size_str = f"{image_size[0]}x{image_size[1]}"
            total_tiles = len(mercator_tiles.quadkeys)

            self.logger.info(
                f"Downloading {total_tiles} tiles with size {image_size_str}..."
            )

            if date_filter:
                self.logger.info(f"Date filter applied: {date_filter}")

            def _get_tile_bounds(quadkey: str) -> Tuple[float]:
                """Get tile bounds from quadkey"""
                tile = mercantile.quadkey_to_tile(quadkey)
                bounds = mercantile.bounds(tile)
                return (bounds.west, bounds.south, bounds.east, bounds.north)

            def download_image(
                quadkey: str,
                image_size: Tuple[int, int],
                suffix: str = self.config.suffix,
            ) -> bool:
                bounds = _get_tile_bounds(quadkey)
                file_name = f"{image_prefix}{quadkey}{suffix}"
                output_path = output_dir / file_name

                if save_metadata:
                    success = self._download_single_image_with_metadata(
                        bounds, output_path, image_size, save_metadata=True
                    )
                else:
                    success = self._download_single_image(
                        bounds, output_path, image_size
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
        finally:
            # Restore original filter
            if original_filter is not None:
                self.config.cql_filter = original_filter

    def download_images_by_bounds(
        self,
        gdf: gpd.GeoDataFrame,
        output_dir: Union[str, Path],
        image_size: Tuple[int, int] = (512, 512),
        image_prefix: str = "maxar_image_",
        save_metadata: bool = False,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
    ) -> None:
        """
        Download images for given bounding box polygons

        Args:
            gdf: GeoDataFrame containing bounding box polygons
            output_dir: Directory to save images
            image_size: Tuple of (width, height) for output images
            image_prefix: Prefix for output image names
            save_metadata: If True, saves WFS metadata as JSON alongside each image
            start_date: Filter images acquired on or after this date (YYYY-MM-DD)
            end_date: Filter images acquired on or before this date (YYYY-MM-DD)
        """
        # Build date filter if dates provided
        date_filter = self.build_date_filter(start_date, end_date)

        # Combine with existing config filter
        if date_filter:
            if self.config.cql_filter:
                original_filter = self.config.cql_filter
                self.config.cql_filter = (
                    f"({self.config.cql_filter}) AND ({date_filter})"
                )
            else:
                original_filter = ""
                self.config.cql_filter = date_filter
        else:
            original_filter = None

        try:

            output_dir = Path(output_dir)
            image_size_str = f"{image_size[0]}x{image_size[1]}"
            total_images = len(gdf)

            self.logger.info(
                f"Downloading {total_images} images with size {image_size_str}..."
            )
            if date_filter:
                self.logger.info(f"Date filter applied: {date_filter}")

            def download_image(
                idx: Any,
                bounds: Tuple[float, float, float, float],
                image_size: Tuple[int, int],
                suffix: str = self.config.suffix,
            ) -> bool:
                file_name = f"{image_prefix}{idx}{suffix}"
                output_path = output_dir / file_name

                if save_metadata:
                    success = self._download_single_image_with_metadata(
                        bounds, output_path, image_size, save_metadata=True
                    )
                else:
                    success = self._download_single_image(
                        bounds, output_path, image_size
                    )
                return success

            # Ensure GeoDataFrame is in the correct CRS
            gdf = gdf.to_crs(self.config.data_crs)

            successful_downloads = 0
            with tqdm(total=total_images) as pbar:
                for row in gdf.itertuples():
                    if download_image(
                        row.Index, tuple(row.geometry.bounds), image_size
                    ):
                        successful_downloads += 1
                    pbar.update(1)

            self.logger.info(
                f"Successfully downloaded {successful_downloads}/{total_images} images!"
            )
        finally:
            if original_filter is not None:
                self.config.cql_filter = original_filter

    def download_images_by_coordinates(
        self,
        data: Union[pd.DataFrame, List[Tuple[float, float]]],
        res_meters_pixel: float,
        output_dir: Union[str, Path],
        image_size: Tuple[int, int] = (512, 512),
        image_prefix: str = "maxar_image_",
        save_metadata: bool = False,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
    ) -> None:
        """
        Download images for given coordinates by creating bounding boxes around points

        Args:
            Either a DataFrame with latitude/longitude columns or geometry column,
                or a list of (lat, lon) tuples
            res_meters_pixel: Resolution in meters per pixel
            output_dir: Directory to save images
            image_size: Tuple of (width, height) for output images
            image_prefix: Prefix for output image names
            save_metadata: If True, saves WFS metadata as JSON alongside each image
            start_date: Filter images acquired on or after this date (YYYY-MM-DD)
            end_date: Filter images acquired on or before this date (YYYY-MM-DD)
        """
        # Convert input to DataFrame if needed
        if isinstance(data, pd.DataFrame):
            coordinates_df = data
        else:
            coordinates_df = pd.DataFrame(data, columns=["latitude", "longitude"])

        # Convert to GeoDataFrame and create square buffers around points
        gdf = convert_to_geodataframe(coordinates_df)
        buffered_gdf = buffer_geodataframe(
            gdf, res_meters_pixel / 2, cap_style="square"
        )

        # Ensure correct CRS and delegate to download_images_by_bounds
        buffered_gdf = buffered_gdf.to_crs(self.config.data_crs)

        self.download_images_by_bounds(
            buffered_gdf,
            output_dir,
            image_size,
            image_prefix,
            save_metadata=save_metadata,
            start_date=start_date,
            end_date=end_date,
        )

    def get_imagery_metadata(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        cql_filter: Optional[str] = None,
        count: int = 100,
        output_format: str = "application/json",
        sort_by: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Get metadata for imagery features using WFS GetFeature

        Args:
            bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326
                  Note: Cannot be used together with cql_filter parameter
            cql_filter: CQL filter string for querying by attributes
                       If you need both bbox and filter, include bbox in CQL:
                       "source='WV02' AND BBOX(featureGeometry,x1,y1,x2,y2)"
            count: Number of features to return (1-1000, default 100)
            output_format: Response format (default: application/json)
            sort_by: Sort order, e.g., "acquisitionDate+A" for ascending

        Returns:
            GeoDataFrame with feature metadata

        Available metadata fields:
            - acquisitionDate: Image acquisition timestamp
            - source: Sensor (WV02, WV03, blacksky, capella, etc.)
            - sensorType: 'Optical' or 'Radar'
            - cloudCover: Cloud coverage percentage (0-90)
            - groundSampleDistance: Spatial resolution in meters
            - sunElevation/sunAzimuth: Sun angle information
            - offNadirAngle: Sensor viewing angle
            - productName: DAILY_TAKE, RDOG_BASE, etc.
            - bandDescription: Panchromatic, Pan Sharpened Natural Color, etc.
            - niirs: Image quality rating (0-9)
            - And more (see WFS documentation)

        Example:
            # Query by bounding box
            metadata = downloader.get_imagery_metadata(
                bbox=(-105.0, 39.7, -104.9, 39.8)
            )

            # Query by CQL filter with bbox
            metadata = downloader.get_imagery_metadata(
                cql_filter="source='WV02' AND cloudCover<0.20 AND "
                          "BBOX(featureGeometry,-105.0,39.7,-104.9,39.8)"
            )

            # Query by date and sensor
            metadata = downloader.get_imagery_metadata(
                cql_filter="(acquisitionDate>='2024-01-01') AND (source='WV03')"
            )
        """
        # Initialize WFS if not already done
        if not hasattr(self, "wfs_service"):
            self.wfs_service = self._initialize_wfs()

        try:
            # Build request parameters
            params = {
                "typename": "Maxar:FinishedFeature",  # Feature type for imagery
                "outputFormat": output_format,
                "maxfeatures": min(count, 1000),  # Cap at 1000
            }

            # Add optional parameters
            # IMPORTANT: bbox and cql_filter cannot be used together
            if bbox and not cql_filter:
                params["bbox"] = bbox
            elif cql_filter:
                params["filter"] = cql_filter
            elif not bbox and not cql_filter:
                self.logger.warning(
                    "Neither bbox nor cql_filter provided. "
                    "Will return first {} features.".format(count)
                )

            if sort_by:
                params["sortBy"] = sort_by

            # Add profile if configured
            if self.config.profile:
                params["profile"] = self.config.profile

            # Make WFS GetFeature request
            response = self.wfs_service.getfeature(**params)

            # Parse response to GeoDataFrame
            if output_format == "application/json":
                gdf = gpd.read_file(response)
            else:
                # For XML/GML formats
                gdf = gpd.read_file(response)

            self.logger.info(f"Retrieved {len(gdf)} features from WFS")
            return gdf

        except Exception as e:
            self.logger.error(f"Failed to get imagery meta {str(e)}")
            raise

    def get_metadata_for_bbox(
        self, bbox: Tuple[float, float, float, float], **kwargs
    ) -> Dict[str, Any]:
        """
        Convenience method to get metadata summary for a bounding box

        Returns a dict with summary statistics and the full GeoDataFrame
        """
        gdf = self.get_imagery_metadata(bbox=bbox, **kwargs)

        if len(gdf) == 0:
            return {"feature_count": 0, "features": gdf, "summary": "No features found"}

        summary = {
            "feature_count": len(gdf),
            "features": gdf,
            "sensors": gdf["source"].unique().tolist() if "source" in gdf else [],
            "date_range": {
                "earliest": (
                    gdf["acquisitionDate"].min() if "acquisitionDate" in gdf else None
                ),
                "latest": (
                    gdf["acquisitionDate"].max() if "acquisitionDate" in gdf else None
                ),
            },
            "cloud_cover": {
                "min": float(gdf["cloudCover"].min()) if "cloudCover" in gdf else None,
                "max": float(gdf["cloudCover"].max()) if "cloudCover" in gdf else None,
                "mean": (
                    float(gdf["cloudCover"].mean()) if "cloudCover" in gdf else None
                ),
            },
            "resolution_range": {
                "best": (
                    float(gdf["groundSampleDistance"].min())
                    if "groundSampleDistance" in gdf
                    else None
                ),
                "worst": (
                    float(gdf["groundSampleDistance"].max())
                    if "groundSampleDistance" in gdf
                    else None
                ),
            },
        }

        return summary

    def build_date_filter(
        self,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
        date_field: str = "acquisitionDate",
    ) -> str:
        """
        Build a CQL date filter string

        Args:
            start_date: Start date (inclusive). Accepts:
                    - String: "2020-01-01", "2020-01-01T10:30:00"
                    - datetime/date object
            end_date: End date (inclusive). Same formats as start_date
            date_field: Field name to filter on (default: "acquisitionDate")
                    Other options: "createdDate", "latestAcquisitionTime", etc.

        Returns:
            CQL filter string for date range

        Examples:
            # Last 30 days
            filter = downloader.build_date_filter(
                start_date="2024-01-01",
                end_date="2024-01-31"
            )

            # Only start date (everything after)
            filter = downloader.build_date_filter(start_date="2024-01-01")

            # Only end date (everything before)
            filter = downloader.build_date_filter(end_date="2024-01-31")
        """

        def format_date(d: Union[str, datetime, date]) -> str:
            """Convert various date formats to YYYY-MM-DD string"""
            if isinstance(d, str):
                # Already a string, validate and return
                # Handle both YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS formats
                return d
            elif isinstance(d, datetime):
                return d.strftime("%Y-%m-%d")
            elif isinstance(d, date):
                return d.strftime("%Y-%m-%d")
            else:
                raise TypeError(f"Invalid date type: {type(d)}")

        # Build filter based on provided dates
        if start_date and end_date:
            start = format_date(start_date)
            end = format_date(end_date)
            # Use BETWEEN syntax for cleaner query
            return f"{date_field} BETWEEN '{start}' AND '{end}'"
        elif start_date:
            start = format_date(start_date)
            return f"{date_field}>='{start}'"
        elif end_date:
            end = format_date(end_date)
            return f"{date_field}<='{end}'"
        else:
            return ""
