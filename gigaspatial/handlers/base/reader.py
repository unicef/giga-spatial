"""Abstract reader for dataset handlers (`BaseHandlerReader`)."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Union, Tuple, Callable, Iterable, Type
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
import logging
from tqdm import tqdm

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.processing.tif_processor import TifProcessor
from gigaspatial.handlers.base.config import BaseHandlerConfig, build_component_config
from gigaspatial.handlers.base.options import ReadOptions, sort_options
from gigaspatial.handlers.base.source import SourceKind, classify_source, is_path_source


class BaseHandlerReader(ABC):
    """
    Abstract base class for data readers.

    Provides high-level methods for resolving data paths and loading various
    data types (raster, tabular) into memory. Standardizes the loading
    interface across disparate datasets.

    Attributes:
        config: Handler configuration object.
        data_store: Abstraction for file-system access.
        logger: Logger instance for tracking data loading.
        options_type: The `ReadOptions` subclass ``load_from_paths`` takes.
        config_type: The config class built when the reader is given a dict
            or no config.
    """

    options_type: ClassVar[Type[ReadOptions]] = ReadOptions
    config_type: ClassVar[Optional[Type[BaseHandlerConfig]]] = None

    def __init__(
        self,
        config: Union[BaseHandlerConfig, dict, None] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            config: The handler's config, a dict of ``config_type`` fields, or
                None for a default ``config_type``.
            data_store: Data store to read from; defaults to the config's.
            logger: Logger; defaults to the config's.
        """
        config = build_component_config(self.config_type, config)
        self.config = config
        if data_store:
            self.data_store = data_store
        elif config and hasattr(config, "data_store"):
            self.data_store = config.data_store
        else:
            self.data_store = LocalDataStore()

        self.logger = (
            logger
            or (getattr(config, "logger", None) if config else None)
            or global_config.get_logger(self.__class__.__name__)
        )

    def resolve_source_paths(
        self,
        source: Union[
            str,
            List[Union[Tuple[float, float], Point]],
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,
            str,
            List[Union[str, Path]],
        ],
        **kwargs,
    ) -> List[Union[str, Path]]:
        """
        Identify data file paths corresponding to a specific source input.

        Parses the input source (e.g., country code, geometry, or explicit
        paths) and resolves them into a list of absolute paths within the
        `DataStore`.

        Args:
            source: Source specification (country, geometry, points, or paths).
            **kwargs: Resolve options (see ``config.options_type``).

        Returns:
            A list of resolved file paths.

        Raises:
            TypeError: If a keyword is not a resolve option.
        """
        (resolve,) = sort_options(kwargs, self.config, entry="resolve_source_paths")
        return self._resolve_paths(source, resolve)

    def _resolve_paths(self, source, resolve) -> List[Union[str, Path]]:
        if is_path_source(source):
            return self.resolve_by_paths(source)

        data_units = self.config._resolve_units(source, resolve)
        data_paths = self.config._stored_paths_of(data_units, resolve)

        self.logger.info(f"Resolved {len(data_paths)} paths!")
        return data_paths

    def resolve_by_paths(
        self, paths: Union[Path, str, List[Union[str, Path]]]
    ) -> List[Union[str, Path]]:
        """
        Return explicit paths as a list.
        """
        if isinstance(paths, (str, Path)):
            return [paths]
        return list(paths)

    def _check_file_exists(self, file_paths: List[Union[str, Path]]):
        """
        Validate that all specified files exist in the data store.

        Args:
            file_paths: List of file paths to check.

        Raises:
            RuntimeError: If any file does not exist in the configured data store.
        """
        for file_path in file_paths:
            if not self.data_store.file_exists(str(file_path)):
                raise RuntimeError(
                    f"Source file does not exist in the data store: {file_path}"
                )

    def _load_raster_data(
        self,
        raster_paths: List[Union[str, Path]],
        merge_rasters: bool = False,
        **kwargs,
    ) -> Union[List[TifProcessor], TifProcessor]:
        """
        Initialize TifProcessors for one or more raster files.

        Args:
            raster_paths: List of file paths to raster files.
            merge_rasters: If True, combines all rasters into a single `TifProcessor`.
            **kwargs: Additional parameters for TifProcessor initialization.

        Returns:
            Single TifProcessor (if merged) or list of TifProcessors.
        """
        if merge_rasters or len(raster_paths) == 1:
            self.logger.info(
                f"Merging {len(raster_paths)} rasters into a single TifProcessor."
            )
            return TifProcessor(raster_paths, self.data_store, **kwargs)
        else:
            return [
                TifProcessor(data_path, self.data_store, **kwargs)
                for data_path in raster_paths
            ]

    def _load_tabular_data(
        self,
        file_paths: List[Union[str, Path]],
        read_function: Callable = read_dataset,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Load and consolidate tabular data from multiple files.

        Args:
            file_paths: List of file paths to load.
            read_function: Callable used to read individual files.
            show_progress: If True, displays a progress bar.
            progress_desc: Custom description for the progress bar.
            **kwargs: Additional parameters for the read function.

        Returns:
            The concatenated DataFrame or GeoDataFrame.
        """
        all_data = []
        iterator: Iterable = file_paths
        if show_progress and file_paths:
            iterator = tqdm(
                file_paths,
                desc=progress_desc or "Loading tabular data",
                total=len(file_paths),
            )

        for file_path in iterator:
            all_data.append(
                read_function(file_path, data_store=self.data_store, **kwargs)
            )

        if not all_data:
            return pd.DataFrame()
        result = pd.concat(all_data, ignore_index=True)
        return result

    def crop_to_geometry(self, data, geometry, predicate="intersects", **kwargs):
        """
        Crop loaded data (raster or tabular) to a specific geometry.

        Args:
            data: The data object to crop (DataFrame, GDF, or TifProcessor).
            geometry: The spatial filter geometry.
            predicate: Spatial predicate for tabular filtering ('intersects', 'within', etc.).
            **kwargs: Additional parameters for cropping (e.g., crs, crop).

        Returns:
            The cropped data object.
        """

        # Project geometry to the projection of the data if data has projection
        geom_crs = kwargs.pop("crs", "EPSG:4326")
        if hasattr(data, "crs") and data.crs != geom_crs:
            geometry = (
                gpd.GeoDataFrame(geometry=[geometry], crs=geom_crs)
                .to_crs(data.crs)
                .geometry[0]
            )

        # Tabular (GeoDataFrame) case
        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            if isinstance(data, pd.DataFrame):
                from gigaspatial.processing.geo import convert_to_geodataframe

                try:
                    data = convert_to_geodataframe(data, **kwargs)
                except:
                    return data

            # Clip to geometry
            return data[getattr(data.geometry, predicate)(geometry)]

        # Raster case
        if isinstance(data, TifProcessor):
            clip_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "crop",
                    "all_touched",
                    "invert",
                    "nodata",
                    "pad",
                    "pad_width",
                    "return_clipped_processor",
                )
            }
            return data.clip_to_geometry(geometry=geometry, **clip_kwargs)

        return data

    @abstractmethod
    def load_from_paths(self, source_data_path: List[Union[str, Path]], options) -> Any:
        """
        Abstract method to load source data from paths.

        Args:
            source_data_path: List of source paths
            options: The read options.

        Returns:
            Loaded data (DataFrame, GeoDataFrame, etc.)
        """
        pass

    def load(
        self,
        source: Union[
            str,
            List[Union[Tuple[float, float], Point]],
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,
            str,
            List[Union[str, Path]],
        ],
        crop_to_source: bool = False,
        **kwargs,
    ) -> Any:
        """
        Orchestrate the resolution and loading of data from a source.

        Resolves the source to stored paths, loads them with
        ``load_from_paths``, then optionally crops to the source.

        Args:
            source: specification of the data to load.
            crop_to_source: If True, the data is cropped to the search geometry
                of the source. Path sources are never cropped, and neither are
                country sources when ``config.units_match_country`` is True.
            **kwargs: Resolve and read options.

        Returns:
            The loaded data object.

        Raises:
            TypeError: If a keyword is neither a resolve nor a read option.
            ValueError: If cropping is requested for a source with no
                geometry, such as a filter dict.
        """
        resolve, read = sort_options(kwargs, self.config, self, entry="load")
        return self._load(source, crop_to_source, resolve, read)

    def _load(self, source, crop_to_source: bool, resolve, read) -> Any:
        source_data_paths = self._resolve_paths(source, resolve)
        if not source_data_paths:
            self.logger.warning(
                "No source data paths resolved. There's no matching data to load!"
            )
            return None

        loaded_data = self.load_from_paths([str(p) for p in source_data_paths], read)

        if crop_to_source and loaded_data is not None:
            loaded_data = self._crop_to_source(loaded_data, source, resolve)

        return loaded_data

    def _crop_to_source(self, data, source, resolve):
        kind = classify_source(source)
        if kind is SourceKind.PATHS:
            self.logger.debug("Path source: data is returned uncropped")
            return data
        if kind is SourceKind.COUNTRY and self.config.units_match_country:
            self.logger.debug("Data units already match the country: no crop")
            return data

        if kind is SourceKind.OTHER:
            raise ValueError(
                f"crop_to_source=True needs a source with a geometry, got {type(source)}"
            )

        geometry = self.config._search_geometry(source, resolve)
        return self.crop_to_geometry(data, geometry, crs=self.config.crs)
