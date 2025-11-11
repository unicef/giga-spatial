import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import scipy.sparse as sp
from typing import List, Optional, Tuple, Union, Literal, Callable, Dict, Any
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from contextlib import contextmanager
from shapely.geometry import box, Polygon, MultiPolygon, MultiPoint
from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from functools import partial
import multiprocessing
from tqdm import tqdm
import tempfile
import shutil
import os

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.config import config

# Global variables for multiprocessing workers
src_handle = None
memfile_handle = None


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TifProcessor:
    """
    A class to handle tif data processing, supporting single-band, RGB, RGBA, and multi-band data.
    Can merge multiple rasters into one during initialization.
    """

    dataset_path: Union[Path, str, List[Union[Path, str]]]
    data_store: Optional[DataStore] = None
    mode: Literal["single", "rgb", "rgba", "multi"] = "single"
    merge_method: Literal["first", "last", "min", "max", "mean"] = "first"
    target_crs: Optional[str] = None  # For reprojection if needed
    resampling_method: Resampling = Resampling.nearest
    reprojection_resolution: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        """Validate inputs, merge rasters if needed, and set up logging."""
        self.data_store = self.data_store or LocalDataStore()
        self.logger = config.get_logger(self.__class__.__name__)
        self._cache = {}
        self._temp_dir = tempfile.mkdtemp()
        self._merged_file_path = None
        self._reprojected_file_path = None
        self._clipped_file_path = None

        # Handle multiple dataset paths
        if isinstance(self.dataset_path, list):
            if len(self.dataset_path) > 1:
                self.dataset_paths = [Path(p) for p in self.dataset_path]
                self._validate_multiple_datasets()
                self._merge_rasters()
                self.dataset_path = self._merged_file_path
        else:
            self.dataset_paths = [Path(self.dataset_path)]
            # For absolute paths with LocalDataStore, check file existence directly
            # to avoid path resolution issues
            if isinstance(self.data_store, LocalDataStore) and os.path.isabs(
                str(self.dataset_path)
            ):
                if not os.path.exists(str(self.dataset_path)):
                    raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
            elif not self.data_store.file_exists(str(self.dataset_path)):
                raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

            # Reproject single raster during initialization if target_crs is set
            if self.target_crs:
                self.logger.info(f"Reprojecting single raster to {self.target_crs}...")
                with self.data_store.open(str(self.dataset_path), "rb") as f:
                    with rasterio.MemoryFile(f.read()) as memfile:
                        with memfile.open() as src:
                            self._reprojected_file_path = self._reproject_to_temp_file(
                                src, self.target_crs
                            )
                self.dataset_path = self._reprojected_file_path

        self._load_metadata()
        self._validate_mode_band_compatibility()

    @contextmanager
    def open_dataset(self):
        """Context manager for accessing the dataset, handling temporary reprojected files."""
        if self._merged_file_path:
            with rasterio.open(self._merged_file_path) as src:
                yield src
        elif self._reprojected_file_path:
            with rasterio.open(self._reprojected_file_path) as src:
                yield src
        elif self._clipped_file_path:
            with rasterio.open(self._clipped_file_path) as src:
                yield src
        elif isinstance(self.data_store, LocalDataStore):
            with rasterio.open(str(self.dataset_path)) as src:
                yield src
        else:
            with self.data_store.open(str(self.dataset_path), "rb") as f:
                with rasterio.MemoryFile(f.read()) as memfile:
                    with memfile.open() as src:
                        yield src

    def reproject_to(
        self,
        target_crs: str,
        output_path: Optional[Union[str, Path]] = None,
        resampling_method: Optional[Resampling] = None,
        resolution: Optional[Tuple[float, float]] = None,
    ):
        """
        Reprojects the current raster to a new CRS and optionally saves it.

        Args:
            target_crs: The CRS to reproject to (e.g., "EPSG:4326").
            output_path: The path to save the reprojected raster. If None,
                         it is saved to a temporary file.
            resampling_method: The resampling method to use.
            resolution: The target resolution (pixel size) in the new CRS.
        """
        self.logger.info(f"Reprojecting raster to {target_crs}...")

        # Use provided or default values
        resampling_method = resampling_method or self.resampling_method
        resolution = resolution or self.reprojection_resolution

        with self.open_dataset() as src:
            if src.crs.to_string() == target_crs:
                self.logger.info(
                    "Raster is already in the target CRS. No reprojection needed."
                )
                # If output_path is specified, copy the file
                if output_path:
                    self.data_store.copy_file(str(self.dataset_path), output_path)
                return self.dataset_path

            dst_path = output_path or os.path.join(
                self._temp_dir, f"reprojected_single_{os.urandom(8).hex()}.tif"
            )

            with rasterio.open(
                dst_path,
                "w",
                **self._get_reprojection_profile(src, target_crs, resolution),
            ) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst.transform,
                        dst_crs=dst.crs,
                        resampling=resampling_method,
                        num_threads=multiprocessing.cpu_count(),
                    )

            self.logger.info(f"Reprojection complete. Output saved to {dst_path}")
            return Path(dst_path)

    def get_raster_info(self) -> Dict[str, Any]:
        """Get comprehensive raster information."""
        return {
            "count": self.count,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "bounds": self.bounds,
            "transform": self.transform,
            "dtypes": self.dtype,
            "nodata": self.nodata,
            "mode": self.mode,
            "is_merged": self.is_merged,
            "source_count": self.source_count,
        }

    def _reproject_to_temp_file(
        self, src: rasterio.DatasetReader, target_crs: str
    ) -> str:
        """Helper to reproject a raster and save it to a temporary file."""
        dst_path = os.path.join(
            self._temp_dir, f"reprojected_temp_{os.urandom(8).hex()}.tif"
        )
        profile = self._get_reprojection_profile(
            src, target_crs, self.reprojection_resolution
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst.transform,
                    dst_crs=dst.crs,
                    resampling=self.resampling_method,
                )
        return dst_path

    def _validate_multiple_datasets(self):
        """Validate that all datasets exist and have compatible properties."""
        if len(self.dataset_paths) < 2:
            raise ValueError("Multiple dataset paths required for merging")

        with self.data_store.open(str(self.dataset_paths[0]), "rb") as f:
            with rasterio.MemoryFile(f.read()) as memfile:
                with memfile.open() as ref_src:
                    ref_count = ref_src.count
                    ref_dtype = ref_src.dtypes[0]
                    ref_crs = ref_src.crs
                    ref_transform = ref_src.transform
                    ref_nodata = ref_src.nodata

        for i, path in enumerate(self.dataset_paths[1:], 1):
            with self.data_store.open(str(path), "rb") as f:
                with rasterio.MemoryFile(f.read()) as memfile:
                    with memfile.open() as src:
                        if src.count != ref_count:
                            raise ValueError(
                                f"Dataset {i} has {src.count} bands, expected {ref_count}"
                            )
                        if src.dtypes[0] != ref_dtype:
                            raise ValueError(
                                f"Dataset {i} has dtype {src.dtypes[0]}, expected {ref_dtype}"
                            )
                        if not self.target_crs and src.crs != ref_crs:
                            self.logger.warning(
                                f"Dataset {i} has CRS {src.crs}, expected {ref_crs}. "
                                "Consider setting target_crs parameter for reprojection before merging."
                            )
                        if self.target_crs is None and not self._transforms_compatible(
                            src.transform, ref_transform
                        ):
                            self.logger.warning(
                                f"Dataset {i} has different resolution. Resampling may be needed."
                            )
                        if src.nodata != ref_nodata:
                            self.logger.warning(
                                f"Dataset {i} has different nodata value: {src.nodata} vs {ref_nodata}"
                            )

    def _get_reprojection_profile(
        self,
        src: rasterio.DatasetReader,
        target_crs: str,
        resolution: Optional[Tuple[float, float]],
        compression: str = "lzw",
    ):
        """Calculates and returns the profile for a reprojected raster."""
        if resolution:
            src_res = (abs(src.transform.a), abs(src.transform.e))
            self.logger.info(
                f"Using target resolution: {resolution}. Source resolution: {src_res}."
            )
            # Calculate transform and dimensions based on the new resolution
            dst_transform, width, height = calculate_default_transform(
                src.crs,
                target_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=resolution,
            )
        else:
            # Keep original resolution but reproject
            dst_transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

        profile = src.profile.copy()
        profile.update(
            {
                "crs": target_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
                "compress": compression,  # Add compression to save space
            }
        )
        return profile

    def _transforms_compatible(self, transform1, transform2, tolerance=1e-6):
        """Check if two transforms have compatible pixel sizes."""
        return (
            abs(transform1.a - transform2.a) < tolerance
            and abs(transform1.e - transform2.e) < tolerance
        )

    def _merge_rasters(self):
        """Merge multiple rasters into a single raster."""
        self.logger.info(f"Merging {len(self.dataset_paths)} rasters...")

        # Open all datasets and handle reprojection if needed
        datasets_to_merge = []
        temp_reprojected_files = []
        try:
            for path in self.dataset_paths:
                with self.data_store.open(str(path), "rb") as f:
                    with rasterio.MemoryFile(f.read()) as memfile:
                        with memfile.open() as src:
                            if self.target_crs and src.crs != self.target_crs:
                                self.logger.info(
                                    f"Reprojecting {path.name} to {self.target_crs} before merging."
                                )
                                reprojected_path = self._reproject_to_temp_file(
                                    src, self.target_crs
                                )
                                temp_reprojected_files.append(reprojected_path)
                                datasets_to_merge.append(
                                    rasterio.open(reprojected_path)
                                )
                            else:
                                temp_path = os.path.join(
                                    self._temp_dir,
                                    f"temp_{path.stem}_{os.urandom(4).hex()}.tif",
                                )
                                temp_reprojected_files.append(temp_path)

                                profile = src.profile
                                with rasterio.open(temp_path, "w", **profile) as dst:
                                    dst.write(src.read())
                                datasets_to_merge.append(rasterio.open(temp_path))

            self._merged_file_path = os.path.join(self._temp_dir, "merged_raster.tif")

            if self.merge_method == "mean":
                merged_array, merged_transform = self._merge_with_mean(
                    datasets_to_merge
                )
            else:
                merged_array, merged_transform = merge(
                    datasets_to_merge,
                    method=self.merge_method,
                    resampling=self.resampling_method,
                )

            # Get profile from the first file in the list (all should be compatible now)
            ref_src = datasets_to_merge[0]
            profile = ref_src.profile.copy()
            profile.update(
                {
                    "height": merged_array.shape[-2],
                    "width": merged_array.shape[-1],
                    "transform": merged_transform,
                    "crs": self.target_crs if self.target_crs else ref_src.crs,
                }
            )

            with rasterio.open(self._merged_file_path, "w", **profile) as dst:
                dst.write(merged_array)
        finally:
            for dataset in datasets_to_merge:
                if hasattr(dataset, "close"):
                    dataset.close()

            # Clean up temporary files immediately
            for temp_file in temp_reprojected_files:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

        self.logger.info("Raster merging completed!")

    def _merge_with_mean(self, src_files):
        """Merge rasters using mean aggregation."""
        # Get bounds and resolution for merged raster
        bounds = src_files[0].bounds
        transform = src_files[0].transform

        for src in src_files[1:]:
            bounds = rasterio.coords.BoundingBox(
                min(bounds.left, src.bounds.left),
                min(bounds.bottom, src.bounds.bottom),
                max(bounds.right, src.bounds.right),
                max(bounds.top, src.bounds.top),
            )

        # Calculate dimensions for merged raster
        width = int((bounds.right - bounds.left) / abs(transform.a))
        height = int((bounds.top - bounds.bottom) / abs(transform.e))

        # Create new transform for merged bounds
        merged_transform = rasterio.transform.from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top, width, height
        )

        estimated_memory = height * width * src_files[0].count * 8  # float64
        if estimated_memory > 1e9:  # 1GB threshold
            self.logger.warning(
                f"Large memory usage expected: {estimated_memory/1e9:.1f}GB"
            )

        # Initialize arrays for sum and count
        sum_array = np.zeros((src_files[0].count, height, width), dtype=np.float64)
        count_array = np.zeros((height, width), dtype=np.int32)

        # Process each source file
        for src in src_files:
            # Read data
            data = src.read()

            # Calculate offset in merged raster
            src_bounds = src.bounds
            col_off = int((src_bounds.left - bounds.left) / abs(transform.a))
            row_off = int((bounds.top - src_bounds.top) / abs(transform.e))

            # Get valid data mask
            if src.nodata is not None:
                valid_mask = data[0] != src.nodata
            else:
                valid_mask = np.ones(data[0].shape, dtype=bool)

            # Add to sum and count arrays
            end_row = row_off + data.shape[1]
            end_col = col_off + data.shape[2]

            sum_array[:, row_off:end_row, col_off:end_col] += np.where(
                valid_mask, data, 0
            )
            count_array[row_off:end_row, col_off:end_col] += valid_mask.astype(np.int32)

        # Calculate mean
        mean_array = np.divide(
            sum_array,
            count_array,
            out=np.full_like(
                sum_array, src_files[0].nodata or 0, dtype=sum_array.dtype
            ),
            where=count_array > 0,
        )

        return mean_array.astype(src_files[0].dtypes[0]), merged_transform

    def _load_metadata(self):
        """Load metadata from the TIF file if not already cached"""
        try:
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
        except (rasterio.errors.RasterioIOError, FileNotFoundError) as e:
            raise FileNotFoundError(f"Could not read raster metadata: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading metadata: {e}")

    @property
    def is_merged(self) -> bool:
        """Check if this processor was created from multiple rasters."""
        return len(self.dataset_paths) > 1

    @property
    def source_count(self) -> int:
        """Get the number of source rasters."""
        return len(self.dataset_paths)

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
    def dtype(self):
        """Get the data types from the TIF file"""
        return self._cache.get("dtype", [])

    @property
    def width(self):
        return self._cache["width"]

    @property
    def height(self):
        return self._cache["height"]

    def to_dataframe(
        self, drop_nodata=True, check_memory=True, **kwargs
    ) -> pd.DataFrame:
        """
        Convert raster to DataFrame.

        Args:
            drop_nodata: Whether to drop nodata values
            check_memory: Whether to check memory before operation (default True)
            **kwargs: Additional arguments

        Returns:
            pd.DataFrame with raster data
        """
        # Memory guard check
        if check_memory:
            self._memory_guard("conversion", threshold_percent=80.0)

        try:
            if self.mode == "single":
                return self._to_dataframe(
                    band_number=kwargs.get("band_number", 1),
                    drop_nodata=drop_nodata,
                    band_names=kwargs.get("band_names", None),
                )
            else:
                return self._to_dataframe(
                    band_number=None,  # All bands
                    drop_nodata=drop_nodata,
                    band_names=kwargs.get("band_names", None),
                )
        except Exception as e:
            raise ValueError(
                f"Failed to process TIF file in mode '{self.mode}'. "
                f"Please ensure the file is valid and matches the selected mode. "
                f"Original error: {str(e)}"
            )

        return df

    def to_geodataframe(self, check_memory=True, **kwargs) -> gpd.GeoDataFrame:
        """
        Convert the processed TIF data into a GeoDataFrame, where each row represents a pixel zone.
        Each zone is defined by its bounding box, based on pixel resolution and coordinates.

        Args:
            check_memory: Whether to check memory before operation
            **kwargs: Additional arguments passed to to_dataframe()

        Returns:
            gpd.GeoDataFrame with raster data
        """
        # Memory guard check
        if check_memory:
            self._memory_guard("conversion", threshold_percent=80.0)

        df = self.to_dataframe(check_memory=False, **kwargs)

        x_res, y_res = self.resolution

        # create bounding box for each pixel
        geometries = [
            box(lon - x_res / 2, lat - y_res / 2, lon + x_res / 2, lat + y_res / 2)
            for lon, lat in zip(df["lon"], df["lat"])
        ]

        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=self.crs)

        return gdf

    def to_dataframe_chunked(
        self, drop_nodata=True, chunk_size=None, target_memory_mb=500, **kwargs
    ):
        """
        Convert raster to DataFrame using chunked processing for memory efficiency.

        Automatically routes to the appropriate chunked method based on mode.
        Chunk size is automatically calculated based on target memory usage.

        Args:
            drop_nodata: Whether to drop nodata values
            chunk_size: Number of rows per chunk (auto-calculated if None)
            target_memory_mb: Target memory per chunk in MB (default 500)
            **kwargs: Additional arguments (band_number, band_names, etc.)
        """

        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunk_size(
                "conversion", target_memory_mb
            )

        windows = self._get_chunk_windows(chunk_size)

        # SIMPLE ROUTING
        if self.mode == "single":
            return self._to_dataframe_chunked(
                windows,
                band_number=kwargs.get("band_number", 1),
                drop_nodata=drop_nodata,
                band_names=kwargs.get("band_names", None),
            )
        else:  # rgb, rgba, multi
            return self._to_dataframe_chunked(
                windows,
                band_number=None,
                drop_nodata=drop_nodata,
                band_names=kwargs.get("band_names", None),
            )

    def clip_to_geometry(
        self,
        geometry: Union[
            Polygon, MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries, List[dict], dict
        ],
        crop: bool = True,
        all_touched: bool = True,
        invert: bool = False,
        nodata: Optional[Union[int, float]] = None,
        pad: bool = False,
        pad_width: float = 0.5,
        return_clipped_processor: bool = True,
    ) -> Union["TifProcessor", tuple]:
        """
        Clip raster to geometry boundaries.

        Parameters:
        -----------
        geometry : various
            Geometry to clip to. Can be:
            - Shapely Polygon or MultiPolygon
            - GeoDataFrame or GeoSeries
            - List of GeoJSON-like dicts
            - Single GeoJSON-like dict
        crop : bool, default True
            Whether to crop the raster to the extent of the geometry
        all_touched : bool, default True
            Include pixels that touch the geometry boundary
        invert : bool, default False
            If True, mask pixels inside geometry instead of outside
        nodata : int or float, optional
            Value to use for masked pixels. If None, uses raster's nodata value
        pad : bool, default False
            Pad geometry by half pixel before clipping
        pad_width : float, default 0.5
            Width of padding in pixels if pad=True
        return_clipped_processor : bool, default True
            If True, returns new TifProcessor with clipped data
            If False, returns (clipped_array, transform, metadata)

        Returns:
        --------
        TifProcessor or tuple
            Either new TifProcessor instance or (array, transform, metadata) tuple
        """
        # Handle different geometry input types
        shapes = self._prepare_geometry_for_clipping(geometry)

        # Validate CRS compatibility
        self._validate_geometry_crs(geometry)

        # Perform the clipping
        with self.open_dataset() as src:
            try:
                clipped_data, clipped_transform = mask(
                    dataset=src,
                    shapes=shapes,
                    crop=crop,
                    all_touched=all_touched,
                    invert=invert,
                    nodata=nodata,
                    pad=pad,
                    pad_width=pad_width,
                    filled=True,
                )

                # Update metadata for the clipped raster
                clipped_meta = src.meta.copy()
                clipped_meta.update(
                    {
                        "height": clipped_data.shape[1],
                        "width": clipped_data.shape[2],
                        "transform": clipped_transform,
                        "nodata": nodata if nodata is not None else src.nodata,
                    }
                )

            except ValueError as e:
                if "Input shapes do not overlap raster" in str(e):
                    raise ValueError(
                        "The geometry does not overlap with the raster. "
                        "Check that both are in the same coordinate reference system."
                    ) from e
                else:
                    raise e

        if return_clipped_processor:
            # Create a new TifProcessor with the clipped data
            return self._create_clipped_processor(clipped_data, clipped_meta)
        else:
            return clipped_data, clipped_transform, clipped_meta

    def clip_to_bounds(
        self,
        bounds: tuple,
        bounds_crs: Optional[str] = None,
        return_clipped_processor: bool = True,
    ) -> Union["TifProcessor", tuple]:
        """
        Clip raster to rectangular bounds.

        Parameters:
        -----------
        bounds : tuple
            Bounding box as (minx, miny, maxx, maxy)
        bounds_crs : str, optional
            CRS of the bounds. If None, assumes same as raster CRS
        return_clipped_processor : bool, default True
            If True, returns new TifProcessor, else returns (array, transform, metadata)

        Returns:
        --------
        TifProcessor or tuple
            Either new TifProcessor instance or (array, transform, metadata) tuple
        """
        # Create bounding box geometry
        bbox_geom = box(*bounds)

        # If bounds_crs is specified and different from raster CRS, create GeoDataFrame for reprojection
        if bounds_crs is not None:
            raster_crs = self.crs

            if not self.crs == bounds_crs:
                # Create GeoDataFrame with bounds CRS and reproject
                bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs=bounds_crs)
                bbox_gdf = bbox_gdf.to_crs(raster_crs)
                bbox_geom = bbox_gdf.geometry.iloc[0]

        return self.clip_to_geometry(
            geometry=bbox_geom,
            crop=True,
            return_clipped_processor=return_clipped_processor,
        )

    def to_graph(
        self,
        connectivity: Literal[4, 8] = 4,
        band: Optional[int] = None,
        include_coordinates: bool = False,
        graph_type: Literal["networkx", "sparse"] = "networkx",
        check_memory: bool = True,
    ) -> Union[nx.Graph, sp.csr_matrix]:
        """
        Convert raster to graph based on pixel adjacency.

        Args:
            connectivity: 4 or 8-connectivity
            band: Band number (1-indexed)
            include_coordinates: Include x,y coordinates in nodes
            graph_type: 'networkx' or 'sparse'
            check_memory: Whether to check memory before operation

        Returns:
            Graph representation of raster
        """

        # Memory guard check
        if check_memory:
            self._memory_guard("graph", threshold_percent=80.0)

        with self.open_dataset() as src:
            band_idx = band - 1 if band is not None else 0
            if band_idx < 0 or band_idx >= src.count:
                raise ValueError(
                    f"Band {band} not available. Raster has {src.count} bands"
                )

            data = src.read(band_idx + 1)
            nodata = src.nodata if src.nodata is not None else self.nodata
            valid_mask = (
                data != nodata if nodata is not None else np.ones_like(data, dtype=bool)
            )

            height, width = data.shape

            # Find all valid pixels
            valid_rows, valid_cols = np.where(valid_mask)
            num_valid_pixels = len(valid_rows)

            # Create a sequential mapping from (row, col) to a node ID
            node_map = np.full(data.shape, -1, dtype=int)
            node_map[valid_rows, valid_cols] = np.arange(num_valid_pixels)

            # Define neighborhood offsets
            if connectivity == 4:
                # von Neumann neighborhood (4-connectivity)
                offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            else:  # connectivity == 8
                # Moore neighborhood (8-connectivity)
                offsets = [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]

            # Collect nodes and edges
            nodes_to_add = []
            edges_to_add = []

            for i in range(num_valid_pixels):
                row, col = valid_rows[i], valid_cols[i]
                current_node_id = node_map[row, col]

                # Prepare node attributes
                node_attrs = {"value": float(data[row, col])}
                if include_coordinates:
                    x, y = src.xy(row, col)
                    node_attrs["x"] = x
                    node_attrs["y"] = y
                nodes_to_add.append((current_node_id, node_attrs))

                # Find neighbors and collect edges
                for dy, dx in offsets:
                    neighbor_row, neighbor_col = row + dy, col + dx

                    # Check if neighbor is within bounds and is a valid pixel
                    if (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_col < width
                        and valid_mask[neighbor_row, neighbor_col]
                    ):
                        neighbor_node_id = node_map[neighbor_row, neighbor_col]

                        # Ensure each edge is added only once
                        if current_node_id < neighbor_node_id:
                            neighbor_value = float(data[neighbor_row, neighbor_col])
                            edges_to_add.append(
                                (current_node_id, neighbor_node_id, neighbor_value)
                            )

            if graph_type == "networkx":
                G = nx.Graph()
                G.add_nodes_from(nodes_to_add)
                G.add_weighted_edges_from(edges_to_add)
                return G
            else:  # sparse matrix
                edges_array = np.array(edges_to_add)
                row_indices = edges_array[:, 0]
                col_indices = edges_array[:, 1]
                weights = edges_array[:, 2]

                # Add reverse edges for symmetric matrix
                from_idx = np.append(row_indices, col_indices)
                to_idx = np.append(col_indices, row_indices)
                weights = np.append(weights, weights)

                return sp.coo_matrix(
                    (weights, (from_idx, to_idx)),
                    shape=(num_valid_pixels, num_valid_pixels),
                ).tocsr()

    def sample_by_coordinates(
        self, coordinate_list: List[Tuple[float, float]], **kwargs
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
            elif self.count > 1:
                return np.array(
                    [vals for vals in src.sample(coordinate_list, **kwargs)]
                )
            else:
                return np.array([vals[0] for vals in src.sample(coordinate_list)])

    def sample_by_polygons(
        self,
        polygon_list,
        stat: Union[str, Callable, List[Union[str, Callable]]] = "mean",
    ):
        """
        Sample raster values by polygons and compute statistic(s) for each polygon.

        Args:
            polygon_list: List of shapely Polygon or MultiPolygon objects.
            stat: Statistic(s) to compute. Can be:
                - Single string: 'mean', 'median', 'sum', 'min', 'max', 'std', 'count'
                - Single callable: custom function that takes array and returns scalar
                - List of strings/callables: multiple statistics to compute

        Returns:
            If single stat: np.ndarray of computed statistics for each polygon
            If multiple stats: List of dictionaries with stat names as keys
        """
        # Determine if single or multiple stats
        single_stat = not isinstance(stat, list)
        stats_list = [stat] if single_stat else stat

        # Prepare stat functions
        stat_funcs = []
        stat_names = []

        for s in stats_list:
            if callable(s):
                stat_funcs.append(s)
                stat_names.append(
                    s.__name__
                    if hasattr(s, "__name__")
                    else f"custom_{len(stat_names)}"
                )
            else:
                # Handle string statistics
                if s == "count":
                    stat_funcs.append(len)
                else:
                    stat_funcs.append(getattr(np, s))
                stat_names.append(s)

        results = []

        with self.open_dataset() as src:
            for polygon in tqdm(polygon_list):
                try:
                    out_image, _ = mask(src, [polygon], crop=True, filled=False)

                    # Use masked arrays for more efficient nodata handling
                    if hasattr(out_image, "mask"):
                        valid_data = out_image.compressed()
                    else:
                        valid_data = (
                            out_image[out_image != self.nodata]
                            if self.nodata
                            else out_image.flatten()
                        )

                    if len(valid_data) == 0:
                        if single_stat:
                            results.append(np.nan)
                        else:
                            results.append({name: np.nan for name in stat_names})
                    else:
                        if single_stat:
                            results.append(stat_funcs[0](valid_data))
                        else:
                            # Compute all statistics for this polygon
                            polygon_stats = {}
                            for func, name in zip(stat_funcs, stat_names):
                                try:
                                    polygon_stats[name] = func(valid_data)
                                except Exception:
                                    polygon_stats[name] = np.nan
                            results.append(polygon_stats)

                except Exception:
                    if single_stat:
                        results.append(np.nan)
                    else:
                        results.append({name: np.nan for name in stat_names})

        return np.array(results) if single_stat else results

    def sample_by_polygons_batched(
        self,
        polygon_list: List[Union[Polygon, MultiPolygon]],
        stat: Union[str, Callable] = "mean",
        batch_size: int = 100,
        n_workers: int = 4,
        show_progress: bool = True,
        check_memory: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Sample raster values by polygons in parallel using batching.

        Args:
            polygon_list: List of Shapely Polygon or MultiPolygon objects
            stat: Statistic to compute
            batch_size: Number of polygons per batch
            n_workers: Number of worker processes
            show_progress: Whether to display progress bar
            check_memory: Whether to check memory before operation
            **kwargs: Additional arguments

        Returns:
            np.ndarray of statistics for each polygon
        """
        import sys

        # Memory guard check with n_workers consideration
        if check_memory:
            is_safe = self._memory_guard(
                "batched_sampling",
                threshold_percent=85.0,
                n_workers=n_workers,
                raise_error=False,
            )

            if not is_safe:
                # Suggest reducing n_workers
                memory_info = self._check_available_memory()
                estimates = self._estimate_memory_usage("batched_sampling", n_workers=1)

                # Calculate optimal workers
                suggested_workers = max(
                    1, int(memory_info["available"] * 0.7 / estimates["per_worker"])
                )

                warnings.warn(
                    f"Consider reducing n_workers from {n_workers} to {suggested_workers} "
                    f"to reduce memory pressure.",
                    ResourceWarning,
                )

        # Platform check
        if sys.platform in ["win32", "darwin"]:
            import warnings
            import multiprocessing as mp

            if mp.get_start_method(allow_none=True) != "fork":
                warnings.warn(
                    "Batched sampling may not work on Windows/macOS. "
                    "Use sample_by_polygons() if you encounter errors.",
                    RuntimeWarning,
                )

        def _chunk_list(data_list, chunk_size):
            """Yield successive chunks from data_list."""
            for i in range(0, len(data_list), chunk_size):
                yield data_list[i : i + chunk_size]

        if len(polygon_list) == 0:
            return np.array([])

        stat_func = stat if callable(stat) else getattr(np, stat)
        polygon_chunks = list(_chunk_list(polygon_list, batch_size))

        with multiprocessing.Pool(
            initializer=self._initializer_worker, processes=n_workers
        ) as pool:
            process_func = partial(self._process_polygon_batch, stat_func=stat_func)
            if show_progress:
                batched_results = list(
                    tqdm(
                        pool.imap(process_func, polygon_chunks),
                        total=len(polygon_chunks),
                        desc=f"Sampling polygons",
                    )
                )
            else:
                batched_results = list(pool.imap(process_func, polygon_chunks))

            results = [item for sublist in batched_results for item in sublist]

        return np.array(results)

    def _initializer_worker(self):
        """
        Initializer function for each worker process.
        Opens the raster dataset and stores it in a process-local variable.
        This function runs once per worker, not for every task.
        """
        global src_handle, memfile_handle

        # Priority: merged > reprojected > original (same as open_dataset)
        local_file_path = None
        if self._merged_file_path:
            # Merged file is a local temp file
            local_file_path = self._merged_file_path
        elif self._reprojected_file_path:
            # Reprojected file is a local temp file
            local_file_path = self._reprojected_file_path
        elif isinstance(self.data_store, LocalDataStore):
            # Local file - can open directly
            local_file_path = str(self.dataset_path)

        if local_file_path:
            # Open local file directly
            with open(local_file_path, "rb") as f:
                memfile_handle = rasterio.MemoryFile(f.read())
                src_handle = memfile_handle.open()
        else:
            # Custom DataStore
            with self.data_store.open(str(self.dataset_path), "rb") as f:
                memfile_handle = rasterio.MemoryFile(f.read())
                src_handle = memfile_handle.open()

    def _get_worker_dataset(self):
        """Get dataset handle for worker process."""
        global src_handle
        if src_handle is None:
            raise RuntimeError("Raster dataset not initialized in this process.")
        return src_handle

    def _process_single_polygon(self, polygon, stat_func):
        """
        Helper function to process a single polygon.
        This will be run in a separate process.
        """
        try:
            src = self._get_worker_dataset()
            out_image, _ = mask(src, [polygon], crop=True, filled=False)

            if hasattr(out_image, "mask"):
                valid_data = out_image.compressed()
            else:
                valid_data = (
                    out_image[out_image != self.nodata]
                    if self.nodata
                    else out_image.flatten()
                )

            return stat_func(valid_data) if len(valid_data) > 0 else np.nan
        except RuntimeError as e:
            self.logger.error(f"Worker not initialized: {e}")
            return np.nan
        except Exception as e:
            self.logger.debug(f"Error processing polygon: {e}")
            return np.nan

    def _process_polygon_batch(self, polygon_batch, stat_func):
        """
        Processes a batch of polygons.
        """
        return [
            self._process_single_polygon(polygon, stat_func)
            for polygon in polygon_batch
        ]

    def _to_dataframe(
        self,
        band_number: Optional[int] = None,
        drop_nodata: bool = True,
        band_names: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Process TIF to DataFrame - handles both single-band and multi-band.

        Args:
            band_number: Specific band to read (1-indexed). If None, reads all bands.
            drop_no Whether to drop nodata values
            band_names: Custom names for bands (multi-band only)

        Returns:
            pd.DataFrame with lon, lat, and band value(s)
        """
        with self.open_dataset() as src:
            if band_number is not None:
                # SINGLE BAND MODE
                band = src.read(band_number)
                mask = self._build_data_mask(band, drop_nodata, src.nodata)
                lons, lats = self._extract_coordinates_with_mask(mask)
                pixel_values = (
                    np.extract(mask, band) if mask is not None else band.flatten()
                )
                band_name = band_names if isinstance(band_names, str) else "pixel_value"

                return pd.DataFrame({"lon": lons, "lat": lats, band_name: pixel_values})
            else:
                # MULTI-BAND MODE (all bands)
                stack = src.read()

                # Auto-detect band names by mode
                if band_names is None:
                    if self.mode == "rgb":
                        band_names = ["red", "green", "blue"]
                    elif self.mode == "rgba":
                        band_names = ["red", "green", "blue", "alpha"]
                    else:
                        band_names = [
                            src.descriptions[i] or f"band_{i+1}"
                            for i in range(self.count)
                        ]

                # Build mask (checks ALL bands!)
                mask = self._build_multi_band_mask(stack, drop_nodata, src.nodata)

                # Create DataFrame
                data_dict = self._bands_to_dict(stack, self.count, band_names, mask)
                df = pd.DataFrame(data_dict)

                # RGBA: normalize alpha if needed
                if (
                    self.mode == "rgba"
                    and "alpha" in df.columns
                    and df["alpha"].max() > 1
                ):
                    df["alpha"] = df["alpha"] / 255.0

            return df

    def _to_dataframe_chunked(
        self,
        windows: List[rasterio.windows.Window],
        band_number: Optional[int] = None,
        drop_nodata: bool = True,
        band_names: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Universal chunked converter for ALL modes."""

        chunks = []
        iterator = tqdm(windows, desc="Processing chunks") if show_progress else windows

        with self.open_dataset() as src:
            # Auto-detect band names ONCE (before loop)
            if band_number is None and band_names is None:
                if self.mode == "rgb":
                    band_names = ["red", "green", "blue"]
                elif self.mode == "rgba":
                    band_names = ["red", "green", "blue", "alpha"]
                else:  # multi
                    band_names = [
                        src.descriptions[i] or f"band_{i+1}" for i in range(self.count)
                    ]

            for window in iterator:
                if band_number is not None:
                    # SINGLE BAND
                    band_chunk = src.read(band_number, window=window)
                    mask = self._build_data_mask(band_chunk, drop_nodata, src.nodata)
                    lons, lats = self._get_chunk_coordinates(window, src)
                    band_name = (
                        band_names if isinstance(band_names, str) else "pixel_value"
                    )

                    # Build chunk DataFrame (could use helper but simple enough)
                    if mask is not None:
                        mask_flat = mask.flatten()
                        chunk_df = pd.DataFrame(
                            {
                                "lon": lons[mask_flat],
                                "lat": lats[mask_flat],
                                band_name: band_chunk.flatten()[mask_flat],
                            }
                        )
                    else:
                        chunk_df = pd.DataFrame(
                            {"lon": lons, "lat": lats, band_name: band_chunk.flatten()}
                        )
                else:
                    # MULTI-BAND (includes RGB/RGBA)
                    stack_chunk = src.read(window=window)
                    mask = self._build_multi_band_mask(
                        stack_chunk, drop_nodata, src.nodata
                    )
                    lons, lats = self._get_chunk_coordinates(window, src)

                    # Build DataFrame using helper
                    band_dict = {
                        band_names[i]: stack_chunk[i] for i in range(self.count)
                    }
                    chunk_df = self._build_chunk_dataframe(lons, lats, band_dict, mask)

                    # RGBA: normalize alpha
                    if self.mode == "rgba" and "alpha" in chunk_df.columns:
                        if chunk_df["alpha"].max() > 1:
                            chunk_df["alpha"] = chunk_df["alpha"] / 255.0

                chunks.append(chunk_df)

        result = pd.concat(chunks, ignore_index=True)
        return result

    def _prepare_geometry_for_clipping(
        self,
        geometry: Union[
            Polygon,
            MultiPolygon,
            MultiPoint,
            gpd.GeoDataFrame,
            gpd.GeoSeries,
            List[dict],
            dict,
        ],
    ) -> List[dict]:
        """Convert various geometry formats to list of GeoJSON-like dicts for rasterio.mask"""

        if isinstance(geometry, MultiPoint):
            # Use bounding box of MultiPoint
            minx, miny, maxx, maxy = geometry.bounds
            bbox = box(minx, miny, maxx, maxy)
            return [bbox.__geo_interface__]

        if isinstance(geometry, (Polygon, MultiPolygon)):
            # Shapely geometry
            return [geometry.__geo_interface__]

        elif isinstance(geometry, gpd.GeoDataFrame):
            # GeoDataFrame - use all geometries
            return [
                geom.__geo_interface__ for geom in geometry.geometry if geom is not None
            ]

        elif isinstance(geometry, gpd.GeoSeries):
            # GeoSeries
            return [geom.__geo_interface__ for geom in geometry if geom is not None]

        elif isinstance(geometry, dict):
            # Single GeoJSON-like dict
            return [geometry]

        elif isinstance(geometry, list):
            # List of GeoJSON-like dicts
            return geometry

        else:
            raise TypeError(
                f"Unsupported geometry type: {type(geometry)}. "
                "Supported types: Shapely geometries, GeoDataFrame, GeoSeries, "
                "GeoJSON-like dict, or list of GeoJSON-like dicts."
            )

    def _validate_geometry_crs(
        self,
        original_geometry: Any,
    ) -> None:
        """Validate that geometry CRS matches raster CRS"""

        # Get raster CRS
        raster_crs = self.crs

        # Try to get geometry CRS
        geometry_crs = None

        if isinstance(original_geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
            geometry_crs = original_geometry.crs
        elif hasattr(original_geometry, "crs"):
            geometry_crs = original_geometry.crs

        # Warn if CRS mismatch detected
        if geometry_crs is not None and raster_crs is not None:
            if not raster_crs == geometry_crs:
                self.logger.warning(
                    f"CRS mismatch detected! Raster CRS: {raster_crs}, "
                    f"Geometry CRS: {geometry_crs}. "
                    "Consider reprojecting geometry to match raster CRS for accurate clipping."
                )

    def _create_clipped_processor(
        self, clipped_data: np.ndarray, clipped_meta: dict
    ) -> "TifProcessor":
        """
        Helper to create a new TifProcessor instance from clipped data.
        Saves the clipped data to a temporary file and initializes a new TifProcessor.
        """
        # Create a temporary placeholder file to initialize the processor
        # This allows us to get the processor's temp_dir
        placeholder_dir = tempfile.mkdtemp()
        placeholder_path = os.path.join(
            placeholder_dir, f"placeholder_{os.urandom(8).hex()}.tif"
        )

        # Create a minimal valid TIF file as placeholder
        placeholder_transform = rasterio.transform.from_bounds(0, 0, 1, 1, 1, 1)
        with rasterio.open(
            placeholder_path,
            "w",
            driver="GTiff",
            width=1,
            height=1,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=placeholder_transform,
        ) as dst:
            dst.write(np.zeros((1, 1, 1), dtype="uint8"))

        # Create a new TifProcessor instance with the placeholder
        # Use LocalDataStore() since the temp file is always a local absolute path
        new_processor = TifProcessor(
            dataset_path=placeholder_path,
            data_store=LocalDataStore(),  # Always use LocalDataStore for temp files
            mode=self.mode,
        )

        # Now save the clipped file directly to the new processor's temp directory
        # Similar to how _reproject_to_temp_file works
        clipped_file_path = os.path.join(
            new_processor._temp_dir, f"clipped_{os.urandom(8).hex()}.tif"
        )

        with rasterio.open(clipped_file_path, "w", **clipped_meta) as dst:
            dst.write(clipped_data)

        # Verify file was created successfully
        if not os.path.exists(clipped_file_path):
            raise RuntimeError(f"Failed to create clipped file at {clipped_file_path}")

        # Set the clipped file path and update processor attributes
        new_processor._clipped_file_path = clipped_file_path
        new_processor.dataset_path = clipped_file_path
        new_processor.dataset_paths = [Path(clipped_file_path)]

        # Clean up placeholder file and directory
        try:
            os.remove(placeholder_path)
            os.rmdir(placeholder_dir)
        except OSError:
            pass

        # Reload metadata since the path changed
        new_processor._load_metadata()

        self.logger.info(f"Clipped raster saved to temporary file: {clipped_file_path}")

        return new_processor

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

    def _get_chunk_coordinates(self, window, src):
        """Get coordinates for a specific window chunk."""
        transform = src.window_transform(window)
        rows, cols = np.meshgrid(
            np.arange(window.height), np.arange(window.width), indexing="ij"
        )
        xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
        return np.array(xs), np.array(ys)

    def _extract_coordinates_with_mask(self, mask=None):
        """Extract flattened coordinates, optionally applying a mask."""
        x_coords, y_coords = self._get_pixel_coordinates()

        if mask is not None:
            return np.extract(mask, x_coords), np.extract(mask, y_coords)

        return x_coords.flatten(), y_coords.flatten()

    def _build_data_mask(self, data, drop_nodata=True, nodata_value=None):
        """Build a boolean mask for filtering data based on nodata values."""
        if not drop_nodata or nodata_value is None:
            return None

        return data != nodata_value

    def _build_multi_band_mask(
        self,
        bands: np.ndarray,
        drop_nodata: bool = True,
        nodata_value: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Build mask for multi-band data - drops pixels where ANY band has nodata.

        Args:
            bands: 3D array of shape (n_bands, height, width)
            drop_nodata Whether to drop nodata values
            nodata_value: The nodata value to check

        Returns:
            Boolean mask or None if no masking needed
        """
        if not drop_nodata or nodata_value is None:
            return None

        # Check if ANY band has nodata at each pixel location
        has_nodata = np.any(bands == nodata_value, axis=0)

        # Return True where ALL bands are valid
        valid_mask = ~has_nodata

        return valid_mask if not valid_mask.all() else None

    def _bands_to_dict(self, bands, band_count, band_names, mask=None):
        """Read specified bands and return as a dictionary with optional masking."""

        lons, lats = self._extract_coordinates_with_mask(mask)
        data_dict = {"lon": lons, "lat": lats}

        for idx, name in enumerate(band_names[:band_count]):
            band_data = bands[idx]
            data_dict[name] = (
                np.extract(mask, band_data) if mask is not None else band_data.flatten()
            )

        return data_dict

    def _calculate_optimal_chunk_size(
        self, operation: str = "conversion", target_memory_mb: int = 500
    ) -> int:
        """
        Calculate optimal chunk size (number of rows) based on target memory usage.

        Args:
            operation: Type of operation ('conversion', 'graph')
            target_memory_mb: Target memory per chunk in megabytes

        Returns:
            Number of rows per chunk
        """
        bytes_per_element = np.dtype(self.dtype).itemsize
        n_bands = self.count
        width = self.width

        # Adjust for operation type
        if operation == "conversion":
            # DataFrame overhead is roughly 2x
            bytes_per_row = width * n_bands * bytes_per_element * 2
        elif operation == "graph":
            # Graph needs additional space for edges
            bytes_per_row = width * bytes_per_element * 4  # Estimate
        else:
            bytes_per_row = width * n_bands * bytes_per_element

        target_bytes = target_memory_mb * 1024 * 1024
        chunk_rows = max(1, int(target_bytes / bytes_per_row))

        # Ensure chunk size doesn't exceed total height
        chunk_rows = min(chunk_rows, self.height)

        self.logger.info(
            f"Calculated chunk size: {chunk_rows} rows "
            f"(~{self._format_bytes(chunk_rows * bytes_per_row)} per chunk)"
        )

        return chunk_rows

    def _get_chunk_windows(self, chunk_size: int) -> List[rasterio.windows.Window]:
        """
        Generate window objects for chunked reading.

        Args:
            chunk_size: Number of rows per chunk

        Returns:
            List of rasterio.windows.Window objects
        """
        windows = []
        for row_start in range(0, self.height, chunk_size):
            row_end = min(row_start + chunk_size, self.height)
            window = rasterio.windows.Window(
                col_off=0,
                row_off=row_start,
                width=self.width,
                height=row_end - row_start,
            )
            windows.append(window)

        return windows

    def _format_bytes(self, bytes_value: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} PB"

    def _check_available_memory(self) -> dict:
        """
        Check available system memory.

        Returns:
            Dict with total, available, and used memory info
        """
        import psutil

        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
            "available_human": self._format_bytes(memory.available),
        }

    def _estimate_memory_usage(
        self, operation: str = "conversion", n_workers: int = 1
    ) -> dict:
        """
        Estimate memory usage for various operations.

        Args:
            operation: Type of operation ('conversion', 'batched_sampling', 'merge', 'graph')
            n_workers: Number of workers (for batched_sampling)

        Returns:
            Dict with estimated memory usage in bytes and human-readable format
        """
        bytes_per_element = np.dtype(self.dtype).itemsize
        n_pixels = self.width * self.height
        n_bands = self.count

        estimates = {}

        if operation == "conversion":
            # to_dataframe/to_geodataframe: full raster + DataFrame overhead
            raster_memory = n_pixels * n_bands * bytes_per_element
            # DataFrame overhead (roughly 2x for storage + processing)
            dataframe_memory = (
                n_pixels * n_bands * 16
            )  # 16 bytes per value in DataFrame
            total = raster_memory + dataframe_memory
            estimates["raster"] = raster_memory
            estimates["dataframe"] = dataframe_memory
            estimates["total"] = total

        elif operation == "batched_sampling":
            # Each worker loads full raster into MemoryFile
            # Need to get file size
            if self._merged_file_path:
                file_path = self._merged_file_path
            elif self._reprojected_file_path:
                file_path = self._reprojected_file_path
            else:
                file_path = str(self.dataset_path)

            try:
                import os

                file_size = os.path.getsize(file_path)
            except:
                # Estimate if can't get file size
                file_size = n_pixels * n_bands * bytes_per_element * 1.2  # Add overhead

            estimates["per_worker"] = file_size
            estimates["total"] = file_size * n_workers

        elif operation == "merge":
            # _merge_with_mean uses float64 arrays
            raster_memory = n_pixels * n_bands * 8  # float64
            estimates["sum_array"] = raster_memory
            estimates["count_array"] = n_pixels * 4  # int32
            estimates["total"] = raster_memory + n_pixels * 4

        elif operation == "graph":
            # to_graph: data + node_map + edges
            data_memory = n_pixels * bytes_per_element
            node_map_memory = n_pixels * 4  # int32
            # Estimate edges (rough: 4-connectivity = 4 edges per pixel)
            edges_memory = n_pixels * 4 * 3 * 8  # 3 values per edge, float64
            total = data_memory + node_map_memory + edges_memory
            estimates["data"] = data_memory
            estimates["node_map"] = node_map_memory
            estimates["edges"] = edges_memory
            estimates["total"] = total

        # Add human-readable format
        estimates["human_readable"] = self._format_bytes(estimates["total"])

        return estimates

    def _memory_guard(
        self,
        operation: str,
        threshold_percent: float = 80.0,
        n_workers: Optional[int] = None,
        raise_error: bool = False,
    ) -> bool:
        """
        Check if operation is safe to perform given memory constraints.

        Args:
            operation: Type of operation to check
            threshold_percent: Maximum % of available memory to use (default 80%)
            n_workers: Number of workers (for batched operations)
            raise_error: If True, raise MemoryError instead of warning

        Returns:
            True if operation is safe, False otherwise

        Raises:
            MemoryError: If raise_error=True and memory insufficient
        """
        import warnings

        estimates = self._estimate_memory_usage(operation, n_workers=n_workers or 1)
        memory_info = self._check_available_memory()

        estimated_usage = estimates["total"]
        available = memory_info["available"]
        threshold = available * (threshold_percent / 100.0)

        is_safe = estimated_usage <= threshold

        if not is_safe:
            usage_str = self._format_bytes(estimated_usage)
            available_str = memory_info["available_human"]

            message = (
                f"Memory warning: {operation} operation may require {usage_str} "
                f"but only {available_str} is available. "
                f"Current memory usage: {memory_info['percent']:.1f}%"
            )

            if raise_error:
                raise MemoryError(message)
            else:
                warnings.warn(message, ResourceWarning)
                if hasattr(self, "logger"):
                    self.logger.warning(message)

        return is_safe

    def _validate_mode_band_compatibility(self):
        """Validate that mode matches band count."""
        mode_requirements = {
            "single": (1, "1-band"),
            "rgb": (3, "3-band"),
            "rgba": (4, "4-band"),
        }

        if self.mode in mode_requirements:
            required_count, description = mode_requirements[self.mode]
            if self.count != required_count:
                raise ValueError(
                    f"{self.mode.upper()} mode requires a {description} TIF file"
                )
        elif self.mode == "multi" and self.count < 2:
            raise ValueError("Multi mode requires a TIF file with 2 or more bands")

    def __enter__(self):
        return self

    def __del__(self):
        """Clean up temporary files and directories."""
        if hasattr(self, "_temp_dir") and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def cleanup(self):
        """Explicit cleanup method for better control."""
        if hasattr(self, "_temp_dir") and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
            self.logger.info("Cleaned up temporary files")

    def __exit__(self, exc_type, exc_value, traceback):
        """Proper context manager exit with cleanup."""
        self.cleanup()
        return False
