"""
High-performance Raster (TIF) processing engine.
Provides comprehensive tools for TIF data handling, including merging,
reprojection, clipping, graph conversion, and memory-efficient sampling.
Supports single-band, RGB, RGBA, and multi-band rasters.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import scipy.sparse as sp
from typing import List, Optional, Tuple, Union, Literal, Callable, Dict, Any
from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass
from contextlib import contextmanager
from shapely.geometry import box, Polygon, MultiPolygon, MultiPoint
from pathlib import Path
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from contextlib import ExitStack
from functools import partial
import copy
import multiprocessing
import tqdm
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


class _WorkingFile:
    """
    Owns a processor's working file: a stored file in its data store, or a
    derived file the processor wrote.

    Every read goes through `local_path`, which returns a local file. A stored
    file in a non-local data store is copied once into a temporary directory.
    Only the instance that created a temporary directory, in the process that
    created it, deletes it: copies and forked children own nothing, and make
    their own temporary directory if they need a local copy and can't find one.
    """

    def __init__(
        self, data_store: DataStore, stored_path: Optional[Union[Path, str]] = None
    ):
        self._data_store = data_store
        self._stored_path = stored_path
        self._derived_path: Optional[str] = None
        self._local_copy: Optional[str] = None
        self._temp_dir = tempfile.mkdtemp()
        self._owner_pid: Optional[int] = os.getpid()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._owner_pid = None

    @property
    def temp_dir(self) -> str:
        return self._temp_dir

    @property
    def path(self) -> Optional[Union[Path, str]]:
        """The working file: the derived file if there is one, else the stored file."""
        return self._derived_path or self._stored_path

    def owns_temp_dir(self) -> bool:
        return self._owner_pid == os.getpid()

    def derived_path(self, name: str) -> str:
        """A path for a new derived file in a temporary directory this instance owns."""
        return os.path.join(self._owned_temp_dir(), name)

    def adopt(self, derived_path: Union[Path, str]):
        """Make a derived file in this instance's temporary directory the working file."""
        self._derived_path = str(derived_path)

    def local_path(self) -> str:
        """
        A local path to the working file, copying a non-local stored file once.

        Raises:
            FileNotFoundError: If the derived file was deleted by `cleanup()`.
        """
        if self._derived_path:
            if not os.path.exists(self._derived_path):
                raise FileNotFoundError(
                    f"This TifProcessor was cleaned up: its working file "
                    f"{self._derived_path} was deleted. Create a new "
                    f"TifProcessor to read the raster again."
                )
            return self._derived_path
        if isinstance(self._data_store, LocalDataStore):
            return str(self._data_store._resolve_path(self._stored_path))
        if not self._has_local_copy():
            self._local_copy = self._copy_locally(
                self._data_store, self._stored_path, self._owned_temp_dir()
            )
        return self._local_copy

    @contextmanager
    def open(self):
        with rasterio.open(self.local_path()) as src:
            yield src

    @contextmanager
    def borrowed_local_path(self):
        """
        A local path to the working file for one read, such as a merge input.

        A non-local stored file without a local copy is copied for this read
        only, and the copy is deleted afterwards.
        """
        if (
            self._derived_path
            or isinstance(self._data_store, LocalDataStore)
            or self._has_local_copy()
        ):
            yield self.local_path()
            return
        with self.local_stored_files(self._data_store, [self._stored_path]) as paths:
            yield paths[0]

    @classmethod
    @contextmanager
    def local_stored_files(
        cls, data_store: DataStore, stored_paths: List[Union[Path, str]]
    ):
        """
        Local paths to stored files for one read.

        Files in a `LocalDataStore` are read in place. Files in any other data
        store are copied into a scratch directory that is deleted afterwards.
        """
        if isinstance(data_store, LocalDataStore):
            yield [str(data_store._resolve_path(p)) for p in stored_paths]
            return
        scratch_dir = tempfile.mkdtemp()
        try:
            yield [cls._copy_locally(data_store, p, scratch_dir) for p in stored_paths]
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def cleanup(self) -> bool:
        """Delete the temporary directory if this instance owns it; return whether it did."""
        if not self.owns_temp_dir():
            return False
        try:
            shutil.rmtree(self._temp_dir)
        except FileNotFoundError:
            pass
        self._owner_pid = None
        self._local_copy = None
        return True

    def _has_local_copy(self) -> bool:
        return self._local_copy is not None and os.path.exists(self._local_copy)

    def _owned_temp_dir(self) -> str:
        if not self.owns_temp_dir():
            self._temp_dir = tempfile.mkdtemp()
            self._owner_pid = os.getpid()
        return self._temp_dir

    @staticmethod
    def _copy_locally(
        data_store: DataStore, stored_path: Union[Path, str], directory: str
    ) -> str:
        local_path = os.path.join(
            directory, f"{os.urandom(4).hex()}_{Path(str(stored_path)).name}"
        )
        with data_store.open(str(stored_path), "rb") as src, open(
            local_path, "wb"
        ) as dst:
            shutil.copyfileobj(src, dst)
        return local_path


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TifProcessor:
    """
    Handler for TIF data processing and analysis.

    Supports advanced operations like merging multiple rasters, reprojection,
    clipping to geometries, and converting raster data to formats like
    DataFrames or Graphs.

    Attributes:
        dataset_path: Path(s) to the TIF file(s).
        data_store: DataStore instance for file access.
        mode: Processing mode ('single', 'rgb', 'rgba', 'multi').
        merge_method: Method for merging multiple rasters.
        target_crs: Optional CRS to reproject to. Required when merging files
            in different CRSs.
        resampling_method: Resampling algorithm to use.
        reprojection_resolution: Target pixel size for reprojection.
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
        # A processor from `_derive` arrives with its derived file written.
        if not hasattr(self, "_working_file"):
            self._set_up_working_file()
        self._load_metadata()
        self._validate_mode_band_compatibility()

    def _set_up_working_file(self):
        """Open the stored file, merging or reprojecting it into a derived file if asked."""
        # Handle multiple dataset paths
        if isinstance(self.dataset_path, list):
            self.dataset_paths = [Path(p) for p in self.dataset_path]
            self._working_file = _WorkingFile(self.data_store)
            with _WorkingFile.local_stored_files(
                self.data_store, self.dataset_paths
            ) as local_paths:
                self._write_derived(
                    "merged_raster.tif",
                    partial(self._merge_rasters, local_paths, self.dataset_paths),
                )
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
            self._working_file = _WorkingFile(self.data_store, self.dataset_path)

            # Reproject single raster during initialization if target_crs is set
            if self.target_crs:
                self.logger.info(f"Reprojecting single raster to {self.target_crs}...")
                with self._working_file.borrowed_local_path() as stored_path:
                    self._write_derived(
                        f"reprojected_temp_{os.urandom(8).hex()}.tif",
                        partial(
                            self._write_reprojection,
                            stored_path,
                            self.target_crs,
                            self.resampling_method,
                            self.reprojection_resolution,
                        ),
                    )

    def _settings(self) -> Dict[str, Any]:
        """The constructor fields a derived processor carries: all but `dataset_path`."""
        return {
            "mode": self.mode,
            "merge_method": self.merge_method,
            "target_crs": self.target_crs,
            "resampling_method": self.resampling_method,
            "reprojection_resolution": self.reprojection_resolution,
            "data_store": self.data_store,
        }

    def _write_derived(self, name: str, write: Callable[[str], None]):
        """
        Write a derived file into this processor's temporary directory with
        `write(path)`, and make it the working file.
        """
        derived_path = self._working_file.derived_path(name)
        write(derived_path)
        self._working_file.adopt(derived_path)
        self.dataset_path = self._working_file.path

    @classmethod
    def _derive(
        cls,
        settings: Dict[str, Any],
        dataset_paths: List[Path],
        name: str,
        write: Callable[["TifProcessor", str], None],
    ) -> "TifProcessor":
        """
        A new processor that owns the derived file `write(processor, path)`
        writes, without running construction-time merging or reprojection.

        `settings` are the constructor fields other than `dataset_path`;
        `target_crs` among them is metadata only. `dataset_paths` are the
        original stored files.
        """
        derived = object.__new__(cls)
        for field, value in settings.items():
            setattr(derived, field, value)
        derived.logger = config.get_logger(cls.__name__)
        derived.dataset_paths = list(dataset_paths)
        derived._working_file = _WorkingFile(derived.data_store)
        derived._write_derived(name, partial(write, derived))
        derived.__post_init__()
        return derived

    @field_validator("dataset_path")
    def validate_dataset_path(cls, value):
        """Validates that at least one dataset path is provided."""
        if isinstance(value, list):
            if path_len := len(value):
                if path_len == 1:
                    return value[0]
                return value

            raise ValueError("No dataset paths provided.")

        if isinstance(value, (Path, str)):
            return value

    @contextmanager
    def open_dataset(self):
        """
        Context manager for robustly accessing the TIF dataset.

        Opens the working file, a stored file or a merged, reprojected, or
        clipped file, from local disk. A stored file in a non-local data store
        is copied locally on first use and read from that copy afterwards.

        Yields:
            A rasterio.DatasetReader object.

        Raises:
            FileNotFoundError: If this processor's merged, reprojected, or
                clipped file was deleted by `cleanup()`.
        """
        with self._working_file.open() as src:
            yield src

    def to_crs(
        self,
        target_crs: str,
        resampling_method: Optional[Resampling] = None,
        resolution: Optional[Tuple[float, float]] = None,
    ) -> "TifProcessor":
        """
        Reproject the raster to a new CRS as a new processor.

        The new processor owns its reprojected file, so cleaning up either
        processor leaves the other readable. It carries this processor's
        settings and stored files, with `target_crs` set to the requested CRS.
        Save it with `save_to_file`.

        Args:
            target_crs: The destination CRS (e.g., "EPSG:4326").
            resampling_method: Resampling algorithm. Defaults to this
                processor's `resampling_method`.
            resolution: Target pixel resolution (x, y). Defaults to this
                processor's `reprojection_resolution` when the CRS differs.

        Returns:
            A new TifProcessor in `target_crs`. If the raster is already in an
            equal CRS and no `resolution` is passed, it reads a copy of this
            processor's raster.
        """
        settings = {**self._settings(), "target_crs": target_crs}
        working_path = self._working_file.local_path()

        with rasterio.open(working_path) as src:
            same_crs = src.crs == CRS.from_user_input(target_crs)
        if same_crs and resolution is None:
            self.logger.info("Raster is already in the target CRS. Copying it.")
            return self._derive(
                settings,
                self.dataset_paths,
                f"copied_{os.urandom(8).hex()}.tif",
                lambda _, dst_path: shutil.copyfile(working_path, dst_path),
            )

        if resampling_method is None:
            resampling_method = self.resampling_method
        if resolution is None:
            resolution = self.reprojection_resolution
        self.logger.info(f"Reprojecting raster to {target_crs}...")
        return self._derive(
            settings,
            self.dataset_paths,
            f"reprojected_{os.urandom(8).hex()}.tif",
            lambda _, dst_path: self._write_reprojection(
                working_path, target_crs, resampling_method, resolution, dst_path
            ),
        )

    def get_raster_info(
        self,
        include_statistics: bool = False,
        approx_ok: bool = False,
    ) -> Dict[str, Any]:
        """
        Get comprehensive metadata and statistics for the raster.

        Args:
            include_statistics: Whether to compute pixel statistics (mean, std, etc.).
            approx_ok: Whether to allow approximate statistics for speed.

        Returns:
            Dictionary containing metadata like dimensions, CRS, bounds, and optionally statistics.
        """
        info = {
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

        if include_statistics:
            info["statistics"] = self._get_basic_statistics(approx_ok=approx_ok)

        return info

    def _write_reprojection(
        self,
        src_path: str,
        target_crs: str,
        resampling_method: Resampling,
        resolution: Optional[Tuple[float, float]],
        dst_path: str,
    ):
        """Reproject the raster at `src_path` into a new file at `dst_path`."""
        with rasterio.open(src_path) as src:
            profile = self._get_reprojection_profile(src, target_crs, resolution)
            with rasterio.open(dst_path, "w", **profile) as dst:
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

    def _validate_multiple_datasets(
        self, local_paths: List[str], labels: List[Union[Path, str]]
    ):
        """Validate that all datasets exist and have compatible properties."""
        if len(local_paths) < 2:
            raise ValueError("Multiple dataset paths required for merging")

        with rasterio.open(local_paths[0]) as ref_src:
            ref_count = ref_src.count
            ref_dtype = ref_src.dtypes[0]
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform
            ref_nodata = ref_src.nodata

        for i, (path, local_path) in enumerate(zip(labels[1:], local_paths[1:]), 1):
            with rasterio.open(local_path) as src:
                if src.count != ref_count:
                    raise ValueError(
                        f"Dataset {i} has {src.count} bands, expected {ref_count}"
                    )
                if src.dtypes[0] != ref_dtype:
                    raise ValueError(
                        f"Dataset {i} has dtype {src.dtypes[0]}, expected {ref_dtype}"
                    )
                if not self.target_crs and src.crs != ref_crs:
                    raise ValueError(
                        f"Dataset {path} has CRS {src.crs}, but the first "
                        f"dataset {labels[0]} has CRS {ref_crs}. "
                        "Pass target_crs to reproject every file before merging."
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

    def _merge_rasters(
        self,
        local_paths: List[str],
        labels: List[Union[Path, str]],
        merged_file_path: str,
    ):
        """
        Merge the rasters at `local_paths` into a new file at `merged_file_path`.

        `labels` identify the rasters in log and error messages.

        Raises:
            ValueError: If the rasters are incompatible, including different
                CRSs without `target_crs`.
        """
        self._validate_multiple_datasets(local_paths, labels)
        self.logger.info(f"Merging {len(local_paths)} rasters...")

        # Open all datasets and handle reprojection if needed
        datasets_to_merge = []
        temp_reprojected_files = []
        try:
            for label, local_path in zip(labels, local_paths):
                with rasterio.open(local_path) as src:
                    needs_reprojection = self.target_crs and src.crs != self.target_crs
                if needs_reprojection:
                    self.logger.info(
                        f"Reprojecting {Path(str(label)).name} to {self.target_crs} before merging."
                    )
                    reprojected_path = self._working_file.derived_path(
                        f"reprojected_temp_{os.urandom(8).hex()}.tif"
                    )
                    self._write_reprojection(
                        local_path,
                        self.target_crs,
                        self.resampling_method,
                        self.reprojection_resolution,
                        reprojected_path,
                    )
                    temp_reprojected_files.append(reprojected_path)
                    datasets_to_merge.append(rasterio.open(reprojected_path))
                else:
                    datasets_to_merge.append(rasterio.open(local_path))

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
                    "dtype": merged_array.dtype.name,
                }
            )

            with rasterio.open(merged_file_path, "w", **profile) as dst:
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

        mean_dtype = np.promote_types(src_files[0].dtypes[0], np.float32)
        return mean_array.astype(mean_dtype), merged_transform

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
        self,
        drop_nodata=True,
        check_memory=True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Convert the raster data into a pandas DataFrame.

        Args:
            drop_nodata: If True, pixels with the nodata value are excluded.
            check_memory: If True, checks system memory availability before loading.
            min_value: Optional minimum threshold to filter pixels.
            max_value: Optional maximum threshold to filter pixels.
            **kwargs: Additional arguments like `band_number` or `band_names`.

        Returns:
            A DataFrame with 'lon', 'lat', and band values.

        Raises:
            ValueError: If processing fails due to mode mismatch or invalid data,
                or if a `band_names` list in single mode has no name for the
                selected band.
        """
        # Memory guard check
        if check_memory:
            self._memory_guard("conversion", threshold_percent=80.0)

        if self.mode == "single":
            band_number = kwargs.get("band_number", 1)
            band_name = self._single_band_name(kwargs.get("band_names"), band_number)

        try:
            if self.mode == "single":
                return self._to_dataframe(
                    band_number=band_number,
                    drop_nodata=drop_nodata,
                    band_names=band_name,
                    min_value=min_value,
                    max_value=max_value,
                )
            else:
                return self._to_dataframe(
                    band_number=None,  # All bands
                    drop_nodata=drop_nodata,
                    band_names=kwargs.get("band_names", None),
                    min_value=min_value,
                    max_value=max_value,
                )
        except Exception as e:
            raise ValueError(
                f"Failed to process TIF file in mode '{self.mode}'. "
                f"Please ensure the file is valid and matches the selected mode. "
                f"Original error: {str(e)}"
            )

    def to_geodataframe(
        self,
        check_memory=True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Convert the raster data into a GeoDataFrame.

        Each row represents a pixel, with a Point or Box geometry representing
        its spatial extent.

        Args:
            check_memory: If True, checks system memory availability.
            min_value: Optional minimum threshold for pixel values.
            max_value: Optional maximum threshold for pixel values.
            **kwargs: Additional arguments passed to `to_dataframe`.

        Returns:
            A GeoDataFrame containing pixel centroids or boxes and their values.
        """
        # Memory guard check
        if check_memory:
            self._memory_guard("conversion", threshold_percent=80.0)

        # Get filtered DataFrame - geometry creation happens AFTER filtering
        df = self.to_dataframe(
            check_memory=False, min_value=min_value, max_value=max_value, **kwargs
        )

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
        Convert raster to DataFrame using memory-efficient chunked processing.

        Args:
            drop_nodata: Whether to exclude pixels with the nodata value.
            chunk_size: Specific number of rows per chunk. If None, it is auto-calculated.
            target_memory_mb: Target memory limit per chunk in megabytes.
            **kwargs: Additional arguments like `band_number` or `band_names`.

        Returns:
            A consolidated DataFrame containing all processed chunks.
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
        Clip the raster to the boundaries of specific geometries.

        Args:
            geometry: The geometry to clip to (Polygon, GDF, GeoSeries, etc.).
            crop: If True, the raster's extent is reduced to the geometry's bounding box.
            all_touched: If True, includes all pixels touched by the geometry.
            invert: If True, masks pixels *inside* the geometry.
            nodata: Override for the nodata value in the output.
            pad: Whether to pad the geometry before clipping.
            pad_width: Width of the padding in pixels.
            return_clipped_processor: If True, returns a new TifProcessor instance.

        Returns:
            A new TifProcessor instance (if return_clipped_processor is True) or
            a tuple of (clipped_array, transform, metadata).

        Raises:
            ValueError: If the geometry does not overlap with the raster or CRS is incompatible.
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
        Clip the raster to a rectangular bounding box.

        Args:
            bounds: Bounding box as (minx, miny, maxx, maxy).
            bounds_crs: The CRS of the input bounds. Defaults to raster CRS.
            return_clipped_processor: If True, returns a new TifProcessor instance.

        Returns:
            The clipped TifProcessor or tuple of data/metadata.
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
        Convert the raster into a graph representation based on pixel adjacency.

        Args:
            connectivity: Neighborhood connectivity (4 for von Neumann, 8 for Moore).
            band: Band number to use for node values (1-indexed).
            include_coordinates: If True, adds 'x' and 'y' attributes to nodes.
            graph_type: Output type ('networkx' for Graph object, 'sparse' for CSR matrix).
            check_memory: If True, validates memory availability before processing.

        Returns:
            A NetworkX Graph or a SciPy sparse CSR matrix.
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
                edges_array = np.array(edges_to_add).reshape(-1, 3)
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
        """
        Extract raster values at specific point coordinates.

        Args:
            coordinate_list: List of (longitude, latitude) tuples.
            **kwargs: Additional arguments passed to rasterio.sample.

        Returns:
            Numpy array of values (single-band) or dict of band values (RGB/RGBA).
        """
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
        Sample raster values within polygons and compute aggregate statistics.

        Args:
            polygon_list: List of Shapely Polygon or MultiPolygon objects.
            stat: Statistic(s) to compute. Can be a string (e.g., 'mean'),
                  a callable, or a list of both.

        Returns:
            Numpy array of results (if single stat) or a list of dictionaries (if multi-stat).
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
        Sample raster values by polygons in parallel using batch processing.

        Efficiently distributes sampling tasks across multiple worker processes.

        Args:
            polygon_list: List of Shapely Polygon or MultiPolygon objects.
            stat: Statistic to compute for each polygon.
            batch_size: Number of polygons to process in each worker batch.
            n_workers: Number of parallel processes to use.
            show_progress: If True, displays a progress bar.
            check_memory: If True, validates memory availability before starting.
            **kwargs: Additional arguments.

        Returns:
            Numpy array of statistics for each polygon.
        """
        import sys
        import warnings

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

        # Workers open the working file by path, so it must be local before the
        # pool starts; a worker initializer that raises makes the pool respawn
        # workers forever.
        self._working_file.local_path()
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
        global src_handle

        src_handle = rasterio.open(self._working_file.local_path())

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

    @staticmethod
    def _single_band_name(
        band_names: Optional[Union[str, List[str]]], band_number: int
    ) -> str:
        """Value-column name for single mode; a list is indexed by the 1-based band."""
        if isinstance(band_names, str):
            return band_names
        if isinstance(band_names, list):
            if len(band_names) < band_number:
                raise ValueError(
                    f"band_names has {len(band_names)} item(s), so it has no name "
                    f"for band {band_number}. Pass at least {band_number} names "
                    f"or a single string."
                )
            return band_names[band_number - 1]
        return "pixel_value"

    def _to_dataframe(
        self,
        band_number: Optional[int] = None,
        drop_nodata: bool = True,
        band_names: Optional[Union[str, List[str]]] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Process TIF to DataFrame - handles both single-band and multi-band.

        Args:
            band_number: Specific band to read (1-indexed). If None, reads all bands.
            drop_nodata: Whether to drop nodata values
            band_names: Custom names for bands. In single-band mode, a string, or
                a list indexed by the 1-based `band_number`.
            min_value: Minimum threshold for pixel values (exclusive)
            max_value: Maximum threshold for pixel values (exclusive)

        Returns:
            pd.DataFrame with lon, lat, and band value(s) filtered according to drop_nodata, min_value and max_value
        """
        with self.open_dataset() as src:
            if band_number is not None:
                # SINGLE BAND MODE
                band = src.read(band_number)
                nodata_value = src.nodata if src.nodata is not None else self.nodata

                # Build mask combining nodata and value thresholds
                mask = self._build_data_mask(
                    band, drop_nodata, nodata_value, min_value, max_value
                )

                # Extract coordinates and values with mask
                lons, lats = self._extract_coordinates_with_mask(mask)
                values = np.extract(mask, band) if mask is not None else band.flatten()

                band_name = self._single_band_name(band_names, band_number)

                return pd.DataFrame({"lon": lons, "lat": lats, band_name: values})
            else:
                # MULTI-BAND MODE (all bands)
                stack = src.read()
                nodata_value = src.nodata if src.nodata is not None else self.nodata

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
                # Build mask combining nodata and value thresholds
                mask = self._build_multi_band_mask(
                    stack, drop_nodata, nodata_value, min_value, max_value
                )

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

                    keep = mask.flatten() if mask is not None else slice(None)
                    chunk_df = pd.DataFrame(
                        {
                            "lon": lons[keep],
                            "lat": lats[keep],
                            **{
                                band_names[i]: stack_chunk[i].flatten()[keep]
                                for i in range(self.count)
                            },
                        }
                    )

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
        """A new processor over the clipped data, with this processor's settings and stored files."""

        def write(_, dst_path: str):
            with rasterio.open(dst_path, "w", **clipped_meta) as dst:
                dst.write(clipped_data)

        return self._derive(
            self._settings(),
            self.dataset_paths,
            f"clipped_{os.urandom(8).hex()}.tif",
            write,
        )

    def _get_basic_statistics(self, approx_ok: bool = False) -> Dict[str, Any]:
        """
        Compute per-band statistics (min, max, mean, std, sum, count).

        Args:
            approx_ok: Whether to allow approximate statistics.

        Returns:
            Dictionary containing per-band and overall statistics.
        """
        cache_key = "statistics_exact"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if approx_ok:
            self.logger.debug(
                "approx_ok requested for statistics, but only exact statistics are supported."
            )

        band_stats: List[Dict[str, Union[int, float, None]]] = []
        overall = {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "sum": 0.0,
            "count": 0,
        }

        with self.open_dataset() as src:
            nodata_value = src.nodata if src.nodata is not None else self.nodata
            total_sum = 0.0
            total_sq_sum = 0.0
            total_count = 0

            for band_idx in range(1, src.count + 1):
                band_min = None
                band_max = None
                band_sum = 0.0
                band_sq_sum = 0.0
                band_count = 0

                for _, window in src.block_windows(bidx=band_idx):
                    block = src.read(band_idx, window=window, masked=False)

                    if nodata_value is not None:
                        valid_mask = block != nodata_value
                        if not np.any(valid_mask):
                            continue
                        valid = block[valid_mask]
                    else:
                        valid = block

                    valid = valid.astype(np.float64, copy=False)
                    if valid.size == 0:
                        continue

                    block_min = float(valid.min())
                    block_max = float(valid.max())
                    block_sum = float(valid.sum())
                    block_sq_sum = float(np.square(valid, dtype=np.float64).sum())
                    block_count = int(valid.size)

                    band_min = (
                        block_min if band_min is None else min(band_min, block_min)
                    )
                    band_max = (
                        block_max if band_max is None else max(band_max, block_max)
                    )
                    band_sum += block_sum
                    band_sq_sum += block_sq_sum
                    band_count += block_count

                if band_count == 0:
                    band_stats.append(
                        {
                            "band": band_idx,
                            "min": None,
                            "max": None,
                            "mean": None,
                            "std": None,
                            "sum": 0.0,
                            "count": 0,
                        }
                    )
                    continue

                band_mean = band_sum / band_count
                variance = max((band_sq_sum / band_count) - band_mean**2, 0.0)
                band_std = variance**0.5

                band_stats.append(
                    {
                        "band": band_idx,
                        "min": band_min,
                        "max": band_max,
                        "mean": band_mean,
                        "std": band_std,
                        "sum": band_sum,
                        "count": band_count,
                    }
                )

                overall["min"] = (
                    band_min
                    if overall["min"] is None
                    else min(overall["min"], band_min)
                )
                overall["max"] = (
                    band_max
                    if overall["max"] is None
                    else max(overall["max"], band_max)
                )
                total_sum += band_sum
                total_sq_sum += band_sq_sum
                total_count += band_count

            if total_count > 0:
                overall["sum"] = total_sum
                overall["count"] = total_count
                overall["mean"] = total_sum / total_count
                overall_variance = max(
                    (total_sq_sum / total_count) - overall["mean"] ** 2, 0.0
                )
                overall["std"] = overall_variance**0.5

        result = {
            "bands": band_stats,
            "overall": overall,
            "approximate": False,
        }

        self._cache[cache_key] = result
        return result

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

    def _build_data_mask(
        self,
        data: np.ndarray,
        drop_nodata: bool = True,
        nodata_value: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Build a boolean mask for filtering data based on nodata and value thresholds.

        Args:
            Data array to mask
            drop_no Whether to drop nodata values
            nodata_value: The nodata value to filter
            min_value: Minimum value threshold (exclusive)
            max_value: Maximum value threshold (exclusive)

        Returns:
            Boolean mask or None if no masking needed
        """
        masks = []

        # Nodata mask
        if drop_nodata and nodata_value is not None:
            masks.append(data != nodata_value)

        # Min value threshold
        if min_value is not None:
            masks.append(data > min_value)

        # Max value threshold
        if max_value is not None:
            masks.append(data < max_value)

        if not masks:
            return None

        # Combine all masks with AND logic
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask &= mask

        return combined_mask

    def _build_multi_band_mask(
        self,
        bands: np.ndarray,
        drop_nodata: bool = True,
        nodata_value: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Build mask for multi-band data.

        Drops pixels where ANY band has nodata or fails value thresholds.

        Args:
            bands: 3D array of shape (nbands, height, width)
            drop_no Whether to drop nodata values
            nodata_value: The nodata value to check
            min_value: Minimum value threshold (exclusive)
            max_value: Maximum value threshold (exclusive)

        Returns:
            Boolean mask or None if no masking needed
        """
        masks = []

        # Nodata mask - any band has nodata
        if drop_nodata and nodata_value is not None:
            has_nodata = np.any(bands == nodata_value, axis=0)
            masks.append(~has_nodata)

        # Value threshold masks - any band fails threshold
        if min_value is not None:
            below_min = np.any(bands <= min_value, axis=0)
            masks.append(~below_min)

        if max_value is not None:
            above_max = np.any(bands >= max_value, axis=0)
            masks.append(~above_max)

        if not masks:
            return None

        # Combine all masks with AND logic
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask &= mask

        return combined_mask

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
        Calculate optimal chunk size based on target memory usage.

        Args:
            operation: Type of operation ('conversion', 'graph').
            target_memory_mb: Target memory per chunk in megabytes.

        Returns:
            Optimal number of rows per chunk.
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
                0,  # col_off
                row_start,  # row_off
                self.width,  # width
                row_end - row_start,  # height
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
            try:
                file_size = os.path.getsize(self._working_file.local_path())
            except OSError:
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
        Check if an operation is safe to perform given memory constraints.

        Args:
            operation: Type of operation to check.
            threshold_percent: Maximum % of available memory to use.
            n_workers: Number of workers (for parallel operations).
            raise_error: If True, raises MemoryError if unsafe.

        Returns:
            True if the operation is deemed safe, False otherwise.

        Raises:
            MemoryError: If raise_error is True and memory is insufficient.
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

    def save_to_file(
        self,
        output_path: Union[str, Path],
        compress: Optional[str] = "LZW",
        tiled: bool = True,
        blocksize: int = 512,
        bigtiff: Optional[str] = None,
        predictor: Optional[int] = None,
        num_threads: Optional[int] = None,
        cog: bool = False,
        overviews: Optional[List[int]] = None,
        overview_resampling: str = "nearest",
        **kwargs,
    ) -> Path:
        """
        Export the raster to a file with optimized settings.

        Args:
            output_path: Output file path.
            compress: Compression method (e.g., 'LZW', 'ZSTD').
            tiled: If True, tiles the output for better performance.
            blocksize: Block size for tiled output.
            bigtiff: 'YES', 'NO', or 'IF_NEEDED' for large files.
            predictor: Compression predictor (2 for int, 3 for float).
            num_threads: Number of threads for compression.
            cog: If True, creates a Cloud-Optimized GeoTIFF.
            overviews: Overview levels for COG.
            overview_resampling: Resampling method for overviews.
            **kwargs: Additional creation options for rasterio.

        Returns:
            Path to the saved TIF file.
        """
        output_path = Path(output_path)

        # Build creation options
        creation_options = {}

        if compress and compress.upper() != "NONE":
            creation_options["compress"] = compress.upper()

        if tiled:
            creation_options["tiled"] = True
            creation_options["blockxsize"] = blocksize
            creation_options["blockysize"] = blocksize

        if bigtiff:
            creation_options["BIGTIFF"] = bigtiff

        if predictor is not None:
            creation_options["predictor"] = predictor

        if num_threads is not None:
            creation_options["NUM_THREADS"] = num_threads

        # Add compression-specific options
        if compress:
            if compress.upper() == "DEFLATE" and "ZLEVEL" not in kwargs:
                kwargs["ZLEVEL"] = 6  # Default compression level
            elif compress.upper() == "ZSTD" and "ZSTD_LEVEL" not in kwargs:
                kwargs["ZSTD_LEVEL"] = 9  # Default compression level
            elif compress.upper() == "JPEG" and "JPEG_QUALITY" not in kwargs:
                kwargs["JPEG_QUALITY"] = 85  # Default quality
            elif compress.upper() == "WEBP" and "WEBP_LEVEL" not in kwargs:
                kwargs["WEBP_LEVEL"] = 75  # Default quality

        # Merge additional kwargs
        creation_options.update(kwargs)

        # Write to temporary file first (rasterio requires local file)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Use open_dataset context manager - handles merged/reprojected/clipped files automatically
            with self.open_dataset() as src:
                profile = src.profile.copy()
                profile.update(**creation_options)

                with rasterio.open(tmp_path, "w", **profile) as dst:
                    # Write all bands
                    for band_idx in range(1, src.count + 1):
                        data = src.read(band_idx)
                        dst.write(data, band_idx)

                    # Add overviews if requested
                    if overviews or cog:
                        if overviews is None:
                            # Auto-generate overview levels for COG
                            overviews = [2, 4, 8, 16]
                        dst.build_overviews(
                            overviews, getattr(Resampling, overview_resampling)
                        )

                    # Update tags to indicate COG if requested
                    if cog:
                        dst.update_tags(LAYOUT="COG")

            # Write through data store
            with open(tmp_path, "rb") as f:
                file_content = f.read()

            self.data_store.write_file(str(output_path), file_content)

            self.logger.info(f"Raster saved to {output_path}")

        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return output_path

    def save_array_to_file(
        self,
        array: np.ndarray,
        output_path: Union[str, Path],
        compress: Optional[str] = "LZW",
        tiled: bool = True,
        blocksize: int = 512,
        crs: Optional[Any] = None,
        transform: Optional[Any] = None,
        nodata: Optional[float] = None,
        **kwargs,
    ) -> Path:
        """
        Save a numpy array to a raster file using metadata from this processor.

        Args:
            array: 2D or 3D array of data to save.
            output_path: Destination file path.
            compress: Compression method.
            tiled: If True, tiles the output.
            blocksize: Block size for tiled output.
            crs: Optional CRS override.
            transform: Optional Affine transform override.
            nodata: Optional nodata value override.
            **kwargs: Additional creation options.

        Returns:
            Path to the saved TIF file.
        """
        output_path = Path(output_path)

        # Ensure array is at least 3D
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        elif array.ndim != 3:
            raise ValueError(f"Array must be 2D or 3D, got shape {array.shape}")

        num_bands = array.shape[0]
        height = array.shape[1]
        width = array.shape[2]

        # Get metadata from source using open_dataset
        with self.open_dataset() as src:
            if crs is None:
                crs = src.crs
            if transform is None:
                transform = src.transform
            if nodata is None:
                nodata = src.nodata
            dtype = array.dtype

        # Build profile
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": num_bands,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "nodata": nodata,
        }

        # Add creation options
        if compress and compress.upper() != "NONE":
            profile["compress"] = compress.upper()
        if tiled:
            profile["tiled"] = True
            profile["blockxsize"] = blocksize
            profile["blockysize"] = blocksize

        profile.update(kwargs)

        # Write to temporary file first
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Write the file - write all bands
            with rasterio.open(tmp_path, "w", **profile) as dst:
                for band_idx in range(num_bands):
                    dst.write(array[band_idx], band_idx + 1)

            # Write through data store
            with open(tmp_path, "rb") as f:
                file_content = f.read()

            self.data_store.write_file(str(output_path), file_content)

            self.logger.info(f"Array saved to {output_path}")

        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return output_path

    def __enter__(self):
        return self

    def __setstate__(self, state):
        self.__dict__.update(state)
        # A copy of the owner owns nothing; copy.copy would otherwise share it.
        self._working_file = copy.copy(self._working_file)

    @property
    def _temp_dir(self) -> str:
        return self._working_file.temp_dir

    def __del__(self):
        """Clean up temporary files and directories."""
        try:
            self._working_file.cleanup()
        except Exception:
            pass

    def cleanup(self):
        """
        Delete this processor's temporary files, including any local copy of
        a stored file.

        Only the processor that created them deletes them; on a copy, or when
        called again, this does nothing. A merged, reprojected, or clipped
        processor can't be read after cleanup; one over an original stored
        file still can.
        """
        if self._working_file.cleanup():
            self.logger.info("Cleaned up temporary files")

    def __exit__(self, *args):
        """Proper context manager exit with cleanup."""
        self.cleanup()


def _merge_processors(
    processors: List[TifProcessor],
    mode: Optional[Literal["single", "rgb", "rgba", "multi"]] = None,
    merge_method: Optional[Literal["first", "last", "min", "max", "mean"]] = None,
    target_crs: Optional[str] = None,
    resampling_method: Optional[Resampling] = None,
    reprojection_resolution: Optional[Tuple[float, float]] = None,
) -> TifProcessor:
    """
    Merge processors' working files into a new processor that owns the result.

    Only this module and `BaseViewGenerator` may call this.

    Settings left as None are taken from the first processor, as is its data
    store. `target_crs` is not inherited: processors in different CRSs raise
    `ValueError` unless it is passed. The result's stored files are the
    inputs' stored files, without duplicates. Local copies made to read
    non-local stored files are deleted once the merged file is written.
    """
    overrides = {
        "mode": mode,
        "merge_method": merge_method,
        "resampling_method": resampling_method,
        "reprojection_resolution": reprojection_resolution,
    }
    settings = {
        **processors[0]._settings(),
        **{name: value for name, value in overrides.items() if value is not None},
        "target_crs": target_crs,
    }
    stored_files = list(
        dict.fromkeys(path for tp in processors for path in tp.dataset_paths)
    )
    labels = [tp._working_file.path for tp in processors]
    with ExitStack() as stack:
        local_paths = [
            stack.enter_context(tp._working_file.borrowed_local_path())
            for tp in processors
        ]
        return TifProcessor._derive(
            settings,
            stored_files,
            "merged_raster.tif",
            lambda merged, dst_path: merged._merge_rasters(
                local_paths, labels, dst_path
            ),
        )
