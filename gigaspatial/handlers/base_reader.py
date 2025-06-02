from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union, Tuple, Callable, Iterable
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from gigaspatial.config import config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.processing.tif_processor import TifProcessor


class BaseHandlerReader(ABC):
    """
    Abstract base class for handler reader classes.
    Provides common methods for resolving source paths and loading data.
    Supports resolving by country, points, geometry, GeoDataFrame, or explicit paths.
    Includes generic loader functions for raster and tabular data.
    """

    def __init__(self, data_store: Optional[DataStore] = None):
        self.logger = config.get_logger(self.__class__.__name__)
        self.data_store = data_store or LocalDataStore()

    def resolve_source_paths(
        self,
        source: Union[
            str,  # country code
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,  # path
            str,  # path
            List[Union[str, Path]],
        ],
        **kwargs,
    ) -> List[Union[str, Path]]:
        """
        Resolve source data paths based on the type of source input.

        Args:
            source: Can be a country code or name (str), list of points, geometry, GeoDataFrame, or explicit path(s)
            **kwargs: Additional parameters for path resolution

        Returns:
            List of resolved source paths
        """
        if isinstance(source, (str, Path)):
            # Could be a country code or a path
            if self.data_store.file_exists(str(source)) or str(source).endswith(
                (".csv", ".tif", ".json", ".parquet", ".gz", ".geojson", ".zip")
            ):
                source_data_paths = self.resolve_by_paths(source)
            else:
                source_data_paths = self.resolve_by_country(source, **kwargs)
        elif isinstance(source, (BaseGeometry, gpd.GeoDataFrame)):
            source_data_paths = self.resolve_by_geometry(source, **kwargs)
        elif isinstance(source, Iterable):
            # List of points or paths
            if all(isinstance(p, (Iterable, Point)) for p in source):
                source_data_paths = self.resolve_by_points(source, **kwargs)
            elif all(isinstance(p, (str, Path)) for p in source):
                source_data_paths = self.resolve_by_paths(source)
            else:
                raise ValueError(
                    "List input to resolve_source_paths must be all points or all paths."
                )
        else:
            raise NotImplementedError(f"Unsupported source type: {type(source)}")

        self.logger.info(f"Resolved {len(source_data_paths)} paths!")
        return source_data_paths

    def resolve_by_country(self, country: str, **kwargs) -> List[Union[str, Path]]:
        """
        Resolve source paths for a given country code/name.
        Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses must implement resolve_by_country.")

    def resolve_by_points(
        self, points: List[Union[Tuple[float, float], Point]], **kwargs
    ) -> List[Union[str, Path]]:
        """
        Resolve source paths for a list of points.
        Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses must implement resolve_by_points.")

    def resolve_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> List[Union[str, Path]]:
        """
        Resolve source paths for a geometry or GeoDataFrame.
        Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses must implement resolve_by_geometry.")

    def resolve_by_paths(
        self, paths: Union[Path, str, List[Union[str, Path]]], **kwargs
    ) -> List[Union[str, Path]]:
        """
        Return explicit paths as a list.
        """
        if isinstance(paths, (str, Path)):
            return [paths]
        return list(paths)

    def _pre_load_hook(self, source_data_path, **kwargs) -> Any:
        """Hook called before loading data."""
        if isinstance(source_data_path, (Path, str)):
            source_data_path = [source_data_path]

        if not source_data_path:
            self.logger.warning("No paths found!")
            return []

        source_data_paths = [str(file_path) for file_path in source_data_path]

        self._check_file_exists(file_paths=source_data_paths)

        self.logger.info(
            f"Pre-loading validation complete for {len(source_data_path)} files"
        )
        return source_data_paths

    def _post_load_hook(self, data, **kwargs) -> Any:
        """Hook called after loading data."""
        return data

    def _check_file_exists(self, file_paths: List[Union[str, Path]]):
        """
        Check that all specified files exist in the data store.

        Args:
            file_paths (List[Union[str, Path]]): List of file paths to check.

        Raises:
            RuntimeError: If any file does not exist in the data store.
        """
        for file_path in file_paths:
            if not self.data_store.file_exists(str(file_path)):
                raise RuntimeError(
                    f"Source file does not exist in the data store: {file_path}"
                )

    def _load_raster_data(
        self, raster_paths: List[Union[str, Path]]
    ) -> List[TifProcessor]:
        """
        Load raster data from file paths.

        Args:
            raster_paths (List[Union[str, Path]]): List of file paths to raster files.

        Returns:
            List[TifProcessor]: List of TifProcessor objects for accessing the raster data.
        """
        return [
            TifProcessor(data_path, self.data_store, mode="single")
            for data_path in raster_paths
        ]

    def _load_tabular_data(
        self, file_paths: List[Union[str, Path]], read_function: Callable = read_dataset
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Load and concatenate tabular data from multiple files.

        Args:
            file_paths (List[Union[str, Path]]): List of file paths to load data from.
            read_function (Callable): Function to use for reading individual files.
                Defaults to read_dataset. Should accept (data_store, file_path) arguments.

        Returns:
            Union[pd.DataFrame, gpd.GeoDataFrame]: Concatenated data from all files.
                Returns empty DataFrame if no data is loaded.
        """
        all_data = []
        for file_path in file_paths:
            all_data.append(read_function(self.data_store, file_path))
        if not all_data:
            return pd.DataFrame()
        result = pd.concat(all_data, ignore_index=True)
        return result

    @abstractmethod
    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> Any:
        """
        Abstract method to load source data from paths.

        Args:
            source_data_path: List of source paths
            **kwargs: Additional parameters for data loading

        Returns:
            Loaded data (DataFrame, GeoDataFrame, etc.)
        """
        pass

    def load(
        self,
        source: Union[
            str,  # country
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,
            str,
            List[Union[str, Path]],
        ],
        **kwargs,
    ) -> Any:
        """
        Load data from the given source.

        Args:
            source: The data source (country code/name, points, geometry, paths, etc.).
            **kwargs: Additional parameters to pass to the loading process.

        Returns:
            The loaded data. The type depends on the subclass implementation.
        """
        source_data_paths = self.resolve_source_paths(source, **kwargs)
        if not source_data_paths:
            self.logger.warning(
                "No source data paths resolved. There's no matching data to load!"
            )
            return None
        processed_paths = self._pre_load_hook(source_data_paths, **kwargs)
        if not processed_paths:
            self.logger.warning("No valid paths to load data from.")
            return None

        loaded_data = self.load_from_paths(processed_paths, **kwargs)
        return self._post_load_hook(loaded_data, **kwargs)
