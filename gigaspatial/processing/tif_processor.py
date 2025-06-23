import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Optional, Tuple, Union, Literal
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from contextlib import contextmanager
from shapely.geometry import box, Polygon, MultiPolygon
from pathlib import Path
import rasterio
from rasterio.mask import mask

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.config import config


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TifProcessor:
    """
    A class to handle tif data processing, supporting single-band, RGB, RGBA, and multi-band data.
    """

    dataset_path: Union[Path, str]
    data_store: Optional[DataStore] = None
    mode: Literal["single", "rgb", "rgba", "multi"] = "single"

    def __post_init__(self):
        """Validate inputs and set up logging."""
        self.data_store = self.data_store or LocalDataStore()
        self.logger = config.get_logger(self.__class__.__name__)
        self._cache = {}

        if not self.data_store.file_exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        self._load_metadata()

        # Validate mode and band count
        if self.mode == "rgba" and self.count != 4:
            raise ValueError("RGBA mode requires a 4-band TIF file")
        if self.mode == "rgb" and self.count != 3:
            raise ValueError("RGB mode requires a 3-band TIF file")
        if self.mode == "single" and self.count != 1:
            raise ValueError("Single mode requires a 1-band TIF file")
        if self.mode == "multi" and self.count < 2:
            raise ValueError("Multi mode requires a TIF file with 2 or more bands")

    @contextmanager
    def open_dataset(self):
        """Context manager for accessing the dataset"""
        with self.data_store.open(self.dataset_path, "rb") as f:
            with rasterio.MemoryFile(f.read()) as memfile:
                with memfile.open() as src:
                    yield src

    def _load_metadata(self):
        """Load metadata from the TIF file if not already cached"""
        if not self._cache:
            with self.open_dataset() as src:
                self._cache["transform"] = src.transform
                self._cache["crs"] = src.crs.to_string()
                self._cache["bounds"] = src.bounds
                self._cache["width"] = src.width
                self._cache["height"] = src.height
                self._cache["resolution"] = (abs(src.transform.a), abs(src.transform.e))
                self._cache["x_transform"] = src.transform.a
                self._cache["y_transform"] = src.transform.e
                self._cache["nodata"] = src.nodata
                self._cache["count"] = src.count
                self._cache["dtype"] = src.dtypes[0]

    @property
    def transform(self):
        """Get the transform from the TIF file"""
        return self._cache["transform"]

    @property
    def crs(self):
        """Get the coordinate reference system from the TIF file"""
        return self._cache["crs"]

    @property
    def bounds(self):
        """Get the bounds of the TIF file"""
        return self._cache["bounds"]

    @property
    def resolution(self) -> Tuple[float, float]:
        """Get the x and y resolution (pixel width and height or pixel size) from the TIF file"""
        return self._cache["resolution"]

    @property
    def x_transform(self) -> float:
        """Get the x transform from the TIF file"""
        return self._cache["x_transform"]

    @property
    def y_transform(self) -> float:
        """Get the y transform from the TIF file"""
        return self._cache["y_transform"]

    @property
    def count(self) -> int:
        """Get the band count from the TIF file"""
        return self._cache["count"]

    @property
    def nodata(self) -> int:
        """Get the value representing no data in the rasters"""
        return self._cache["nodata"]

    @property
    def tabular(self) -> pd.DataFrame:
        """Get the data from the TIF file"""
        if not hasattr(self, "_tabular"):
            try:
                if self.mode == "single":
                    self._tabular = self._to_band_dataframe(
                        drop_nodata=True, drop_values=[]
                    )
                elif self.mode == "rgb":
                    self._tabular = self._to_rgb_dataframe(drop_nodata=True)
                elif self.mode == "rgba":
                    self._tabular = self._to_rgba_dataframe(drop_transparent=True)
                elif self.mode == "multi":
                    self._tabular = self._to_multi_band_dataframe(
                        drop_nodata=True,
                        drop_values=[],
                        band_names=None,  # Use default band naming
                    )
                else:
                    raise ValueError(
                        f"Invalid mode: {self.mode}. Must be one of: single, rgb, rgba, multi"
                    )
            except Exception as e:
                raise ValueError(
                    f"Failed to process TIF file in mode '{self.mode}'. "
                    f"Please ensure the file is valid and matches the selected mode. "
                    f"Original error: {str(e)}"
                )

        return self._tabular

    def to_dataframe(self) -> pd.DataFrame:
        return self.tabular

    def get_zoned_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Convert the processed TIF data into a GeoDataFrame, where each row represents a pixel zone.
        Each zone is defined by its bounding box, based on pixel resolution and coordinates.
        """
        self.logger.info("Converting data to GeoDataFrame with zones...")

        df = self.tabular

        x_res, y_res = self.resolution

        # create bounding box for each pixel
        geometries = [
            box(lon - x_res / 2, lat - y_res / 2, lon + x_res / 2, lat + y_res / 2)
            for lon, lat in zip(df["lon"], df["lat"])
        ]

        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=self.crs)

        self.logger.info("Conversion to GeoDataFrame complete!")
        return gdf

    def sample_by_coordinates(
        self, coordinate_list: List[Tuple[float, float]]
    ) -> Union[np.ndarray, dict]:
        self.logger.info("Sampling raster values at the coordinates...")

        with self.open_dataset() as src:
            if self.mode == "rgba":
                if self.count != 4:
                    raise ValueError("RGBA mode requires a 4-band TIF file")

                rgba_values = {"red": [], "green": [], "blue": [], "alpha": []}

                for band_idx, color in enumerate(["red", "green", "blue", "alpha"], 1):
                    rgba_values[color] = [
                        vals[0]
                        for vals in src.sample(coordinate_list, indexes=band_idx)
                    ]

                return rgba_values

            elif self.mode == "rgb":
                if self.count != 3:
                    raise ValueError("RGB mode requires a 3-band TIF file")

                rgb_values = {"red": [], "green": [], "blue": []}

                for band_idx, color in enumerate(["red", "green", "blue"], 1):
                    rgb_values[color] = [
                        vals[0]
                        for vals in src.sample(coordinate_list, indexes=band_idx)
                    ]

                return rgb_values
            else:
                if src.count != 1:
                    raise ValueError("Single band mode requires a 1-band TIF file")
                return np.array([vals[0] for vals in src.sample(coordinate_list)])

    def _sample_polygon(self, polygon: Union[Polygon, MultiPolygon]) -> np.ndarray:
        """
        Sample raster values within a single polygon or multipolygon.

        Args:
            polygon: A shapely Polygon or MultiPolygon.

        Returns:
            np.ndarray: Flattened array of valid raster values within the polygon.
        """
        with self.open_dataset() as src:
            out_image, _ = mask(src, [polygon], crop=True)
            nodata = self.nodata
            # out_image shape: (bands, rows, cols)
            if nodata is not None:
                values = out_image[out_image != nodata].flatten()
            else:
                values = out_image.flatten()
            return values

    def sample_by_polygons(
        self,
        polygon_list: List[Union[Polygon, MultiPolygon]],
        stat: str = "mean",
        warn_on_error: bool = True,
    ) -> np.ndarray:
        """
        Sample raster values by polygons and compute a statistic for each polygon.

        Args:
            polygon_list: List of shapely Polygon or MultiPolygon objects.
            stat: Statistic to compute ('mean', 'median', 'sum', 'min', 'max').
            warn_on_error: If True, log a warning when a polygon cannot be processed. Default is True.

        Returns:
            np.ndarray: Array of computed statistics for each polygon (np.nan if error).
        """
        results = []
        for polygon in polygon_list:
            try:
                values = self._sample_polygon(polygon)
                if len(values) == 0:
                    results.append(np.nan)
                else:
                    if stat == "mean":
                        results.append(np.mean(values))
                    elif stat == "median":
                        results.append(np.median(values))
                    elif stat == "sum":
                        results.append(np.sum(values))
                    elif stat == "min":
                        results.append(np.min(values))
                    elif stat == "max":
                        results.append(np.max(values))
                    else:
                        raise ValueError(f"Unknown statistic: {stat}")

            except Exception as e:
                if warn_on_error:
                    self.logger.warning(f"Error processing polygon: {e}")
                results.append(np.nan)

        return np.array(results)

    def _to_rgba_dataframe(self, drop_transparent: bool = False) -> pd.DataFrame:
        """
        Convert RGBA TIF to DataFrame with separate columns for R, G, B, A values.
        """
        self.logger.info("Processing RGBA dataset...")

        with self.open_dataset() as src:
            if self.count != 4:
                raise ValueError("RGBA mode requires a 4-band TIF file")

            # Read all four bands
            red, green, blue, alpha = src.read()

            x_coords, y_coords = self._get_pixel_coordinates()

            if drop_transparent:
                mask = alpha > 0
                red = np.extract(mask, red)
                green = np.extract(mask, green)
                blue = np.extract(mask, blue)
                alpha = np.extract(mask, alpha)
                lons = np.extract(mask, x_coords)
                lats = np.extract(mask, y_coords)
            else:
                lons = x_coords.flatten()
                lats = y_coords.flatten()
                red = red.flatten()
                green = green.flatten()
                blue = blue.flatten()
                alpha = alpha.flatten()

            # Create DataFrame with RGBA values
            data = pd.DataFrame(
                {
                    "lon": lons,
                    "lat": lats,
                    "red": red,
                    "green": green,
                    "blue": blue,
                    "alpha": alpha,
                }
            )

            # Normalize alpha values if they're not in [0, 1] range
            if data["alpha"].max() > 1:
                data["alpha"] = data["alpha"] / data["alpha"].max()

        self.logger.info("RGBA dataset is processed!")
        return data

    def _to_rgb_dataframe(self, drop_nodata: bool = True) -> pd.DataFrame:
        """Convert RGB TIF to DataFrame with separate columns for R, G, B values."""
        if self.mode != "rgb":
            raise ValueError("Use appropriate method for current mode")

        self.logger.info("Processing RGB dataset...")

        with self.open_dataset() as src:
            if self.count != 3:
                raise ValueError("RGB mode requires a 3-band TIF file")

            # Read all three bands
            red, green, blue = src.read()

            x_coords, y_coords = self._get_pixel_coordinates()

            if drop_nodata:
                nodata_value = src.nodata
                if nodata_value is not None:
                    mask = ~(
                        (red == nodata_value)
                        | (green == nodata_value)
                        | (blue == nodata_value)
                    )
                    red = np.extract(mask, red)
                    green = np.extract(mask, green)
                    blue = np.extract(mask, blue)
                    lons = np.extract(mask, x_coords)
                    lats = np.extract(mask, y_coords)
                else:
                    lons = x_coords.flatten()
                    lats = y_coords.flatten()
                    red = red.flatten()
                    green = green.flatten()
                    blue = blue.flatten()
            else:
                lons = x_coords.flatten()
                lats = y_coords.flatten()
                red = red.flatten()
                green = green.flatten()
                blue = blue.flatten()

            data = pd.DataFrame(
                {
                    "lon": lons,
                    "lat": lats,
                    "red": red,
                    "green": green,
                    "blue": blue,
                }
            )

        self.logger.info("RGB dataset is processed!")
        return data

    def _to_band_dataframe(
        self, band_number: int = 1, drop_nodata: bool = True, drop_values: list = []
    ) -> pd.DataFrame:
        """Process single-band TIF to DataFrame."""
        if self.mode != "single":
            raise ValueError("Use appropriate method for current mode")

        self.logger.info("Processing single-band dataset...")

        if band_number <= 0 or band_number > self.count:
            self.logger.error(
                f"Error: Band number {band_number} is out of range. The file has {self.count} bands."
            )
            return None

        with self.open_dataset() as src:

            band = src.read(band_number)

            x_coords, y_coords = self._get_pixel_coordinates()

            values_to_mask = []
            if drop_nodata:
                nodata_value = src.nodata
                if nodata_value is not None:
                    values_to_mask.append(nodata_value)

            if drop_values:
                values_to_mask.extend(drop_values)

            if values_to_mask:
                data_mask = ~np.isin(band, values_to_mask)
                pixel_values = np.extract(data_mask, band)
                lons = np.extract(data_mask, x_coords)
                lats = np.extract(data_mask, y_coords)
            else:
                pixel_values = band.flatten()
                lons = x_coords.flatten()
                lats = y_coords.flatten()

            data = pd.DataFrame({"lon": lons, "lat": lats, "pixel_value": pixel_values})

        self.logger.info("Dataset is processed!")
        return data

    def _to_multi_band_dataframe(
        self,
        drop_nodata: bool = True,
        drop_values: list = [],
        band_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Process multi-band TIF to DataFrame with all bands included.

        Args:
            drop_nodata (bool): Whether to drop nodata values. Defaults to True.
            drop_values (list): Additional values to drop from the dataset. Defaults to empty list.
            band_names (Optional[List[str]]): Custom names for the bands. If None, bands will be named using
                                            the band descriptions from the GeoTIFF metadata if available,
                                            otherwise 'band_1', 'band_2', etc.

        Returns:
            pd.DataFrame: DataFrame containing coordinates and all band values
        """
        self.logger.info("Processing multi-band dataset...")

        with self.open_dataset() as src:
            # Read all bands
            stack = src.read()

            x_coords, y_coords = self._get_pixel_coordinates()

            # Initialize dictionary with coordinates
            data_dict = {"lon": x_coords.flatten(), "lat": y_coords.flatten()}

            # Get band descriptions from metadata if available
            if band_names is None and hasattr(src, "descriptions") and src.descriptions:
                band_names = [
                    desc if desc else f"band_{i+1}"
                    for i, desc in enumerate(src.descriptions)
                ]

            # Process each band
            for band_idx in range(self.count):
                band_data = stack[band_idx]

                # Handle nodata and other values to drop
                if drop_nodata or drop_values:
                    values_to_mask = []
                    if drop_nodata and src.nodata is not None:
                        values_to_mask.append(src.nodata)
                    if drop_values:
                        values_to_mask.extend(drop_values)

                    if values_to_mask:
                        data_mask = ~np.isin(band_data, values_to_mask)
                        band_values = np.extract(data_mask, band_data)
                        if band_idx == 0:  # Only need to mask coordinates once
                            data_dict["lon"] = np.extract(data_mask, x_coords)
                            data_dict["lat"] = np.extract(data_mask, y_coords)
                    else:
                        band_values = band_data.flatten()
                else:
                    band_values = band_data.flatten()

                # Use custom band names if provided, otherwise use descriptions or default naming
                band_name = (
                    band_names[band_idx]
                    if band_names and len(band_names) > band_idx
                    else f"band_{band_idx + 1}"
                )
                data_dict[band_name] = band_values

        self.logger.info("Multi-band dataset is processed!")
        return pd.DataFrame(data_dict)

    def _get_pixel_coordinates(self):
        """Helper method to generate coordinate arrays for all pixels"""
        if "pixel_coords" not in self._cache:
            # use cached values
            bounds = self._cache["bounds"]
            width = self._cache["width"]
            height = self._cache["height"]
            pixel_size_x = self._cache["x_transform"]
            pixel_size_y = self._cache["y_transform"]

            self._cache["pixel_coords"] = np.meshgrid(
                np.linspace(
                    bounds.left + pixel_size_x / 2,
                    bounds.right - pixel_size_x / 2,
                    width,
                ),
                np.linspace(
                    bounds.top + pixel_size_y / 2,
                    bounds.bottom - pixel_size_y / 2,
                    height,
                ),
            )

        return self._cache["pixel_coords"]


def sample_multiple_tifs_by_coordinates(
    tif_processors: List[TifProcessor], coordinate_list: List[Tuple[float, float]]
):
    """
    Sample raster values from multiple TIFF files for given coordinates.

    Parameters:
    - tif_processors: List of TifProcessor instances.
    - coordinate_list: List of (x, y) coordinates.

    Returns:
    - A NumPy array of sampled values, taking the first non-nodata value encountered.
    """
    sampled_values = np.full(len(coordinate_list), np.nan, dtype=np.float32)

    for tp in tif_processors:
        values = tp.sample_by_coordinates(coordinate_list=coordinate_list)

        if tp.nodata is not None:
            mask = (np.isnan(sampled_values)) & (
                values != tp.nodata
            )  # Replace only NaNs
        else:
            mask = np.isnan(sampled_values)  # No explicit nodata, replace all NaNs

        sampled_values[mask] = values[mask]  # Update only missing values

    return sampled_values


def sample_multiple_tifs_by_polygons(
    tif_processors: List[TifProcessor],
    polygon_list: List[Union[Polygon, MultiPolygon]],
    stat: str = "mean",
    warn_on_error=False,
) -> np.ndarray:
    """
    Sample raster values from multiple TIFF files for polygons in a list and join the results.

    Parameters:
    - tif_processors: List of TifProcessor instances.
    - polygon_list: List of polygon geometries (can include MultiPolygons).
    - stat: Aggregation statistic to compute within each polygon (mean, median, sum, min, max).
    - warn_on_error: If True, log a warning when polygon(s) cannot be processed. Default is False.

    Returns:
    - A NumPy array of sampled values, taking the first non-nodata value encountered.
    """
    sampled_values = np.full(len(polygon_list), np.nan, dtype=np.float32)

    for tp in tif_processors:
        values = tp.sample_by_polygons(
            polygon_list=polygon_list, stat=stat, warn_on_error=warn_on_error
        )

        mask = np.isnan(sampled_values)  # replace all NaNs

        sampled_values[mask] = values[mask]  # Update only values with samapled value

    return sampled_values
