import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import scipy.sparse as sp
from typing import List, Optional, Tuple, Union, Literal, Callable, Dict, Any
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from contextlib import contextmanager
from shapely.geometry import box, Polygon, MultiPolygon
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

        # Handle multiple dataset paths
        if isinstance(self.dataset_path, list):
            if len(self.dataset_path) > 1:
                self.dataset_paths = [Path(p) for p in self.dataset_path]
                self._validate_multiple_datasets()
                self._merge_rasters()
                self.dataset_path = self._merged_file_path
        else:
            self.dataset_paths = [Path(self.dataset_path)]
            if not self.data_store.file_exists(str(self.dataset_path)):
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
        """Context manager for accessing the dataset, handling temporary reprojected files."""
        if self._merged_file_path:
            with rasterio.open(self._merged_file_path) as src:
                yield src
        elif self._reprojected_file_path:
            with rasterio.open(self._reprojected_file_path) as src:
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

    def to_dataframe(self, drop_nodata=True, **kwargs) -> pd.DataFrame:
        try:
            if self.mode == "single":
                df = self._to_band_dataframe(drop_nodata=drop_nodata, **kwargs)
            elif self.mode == "rgb":
                df = self._to_rgb_dataframe(drop_nodata=drop_nodata)
            elif self.mode == "rgba":
                df = self._to_rgba_dataframe(drop_transparent=drop_nodata)
            elif self.mode == "multi":
                df = self._to_multi_band_dataframe(drop_nodata=drop_nodata, **kwargs)
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

        return df

    def to_geodataframe(self, **kwargs) -> gpd.GeoDataFrame:
        """
        Convert the processed TIF data into a GeoDataFrame, where each row represents a pixel zone.
        Each zone is defined by its bounding box, based on pixel resolution and coordinates.
        """
        df = self.to_dataframe(**kwargs)

        x_res, y_res = self.resolution

        # create bounding box for each pixel
        geometries = [
            box(lon - x_res / 2, lat - y_res / 2, lon + x_res / 2, lat + y_res / 2)
            for lon, lat in zip(df["lon"], df["lat"])
        ]

        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=self.crs)

        return gdf

    def to_graph(
        self,
        connectivity: Literal[4, 8] = 4,
        band: Optional[int] = None,
        include_coordinates: bool = False,
        graph_type: Literal["networkx", "sparse"] = "networkx",
        chunk_size: Optional[int] = None,
    ) -> Union[nx.Graph, sp.csr_matrix]:
        """
        Convert raster to graph based on pixel adjacency.
        """
        if chunk_size is not None:
            raise NotImplementedError("Chunked processing is not yet implemented.")

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
                row_indices.extend(col_indices)
                col_indices.extend(row_indices)
                weights.extend(weights)

                return sp.coo_matrix(
                    (weights, (row_indices, col_indices)),
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
        **kwargs,
    ) -> np.ndarray:
        """
        Sample raster values by polygons in parallel using batching.
        """

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
            batched_results = list(
                tqdm(
                    pool.imap(process_func, polygon_chunks),
                    total=len(polygon_chunks),
                    desc=f"Sampling polygons",
                )
            )

            results = [item for sublist in batched_results for item in sublist]

        return np.array(results)

    def _initializer_worker(self):
        """
        Initializer function for each worker process.
        Opens the raster dataset and stores it in a process-local variable.
        This function runs once per worker, not for every task.
        """
        global src_handle, memfile_handle
        with self.data_store.open(str(self.dataset_path), "rb") as f:
            memfile_handle = rasterio.MemoryFile(f.read())
            src_handle = memfile_handle.open()

    def _process_single_polygon(self, polygon, stat_func):
        """
        Helper function to process a single polygon.
        This will be run in a separate process.
        """
        global src_handle
        if src_handle is None:
            # This should not happen if the initializer is set up correctly,
            # but it's a good defensive check.
            raise RuntimeError("Raster dataset not initialized in this process.")

        try:
            out_image, _ = mask(src_handle, [polygon], crop=True, filled=False)

            if hasattr(out_image, "mask"):
                valid_data = out_image.compressed()
            else:
                valid_data = (
                    out_image[out_image != self.nodata]
                    if self.nodata
                    else out_image.flatten()
                )

            if len(valid_data) == 0:
                return np.nan
            else:
                return stat_func(valid_data)
        except Exception:
            return np.nan

    def _process_polygon_batch(self, polygon_batch, stat_func):
        """
        Processes a batch of polygons.
        """
        return [
            self._process_single_polygon(polygon, stat_func)
            for polygon in polygon_batch
        ]

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
) -> np.ndarray:
    """
    Sample raster values from multiple TIFF files for polygons in a list and join the results.

    Parameters:
    - tif_processors: List of TifProcessor instances.
    - polygon_list: List of polygon geometries (can include MultiPolygons).
    - stat: Aggregation statistic to compute within each polygon (mean, median, sum, min, max).

    Returns:
    - A NumPy array of sampled values, taking the first non-nodata value encountered.
    """
    sampled_values = np.full(len(polygon_list), np.nan, dtype=np.float32)

    for tp in tif_processors:
        values = tp.sample_by_polygons(polygon_list=polygon_list, stat=stat)

        mask = np.isnan(sampled_values)  # replace all NaNs

        sampled_values[mask] = values[mask]  # Update only values with samapled value

    return sampled_values
