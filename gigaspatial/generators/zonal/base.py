from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, TypeVar, Generic
from shapely.geometry import Polygon

import geopandas as gpd
import pandas as pd
import numpy as np

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.writers import write_dataset
from gigaspatial.config import config as global_config
from gigaspatial.processing.geo import (
    aggregate_polygons_to_zones,
    aggregate_points_to_zones,
)
from gigaspatial.processing.tif_processor import (
    TifProcessor,
    sample_multiple_tifs_by_polygons,
)
from functools import lru_cache
import logging


class ZonalViewGeneratorConfig(BaseModel):
    """Configuration for zonal view generation.

    Attributes:
        base_path (Path): Base directory path for storing zonal views. Defaults to
            configured zonal views path.
        output_format (str): Default output format for saved views. Defaults to "parquet".
    """

    base_path: Path = Field(default=global_config.get_path("zonal", "views"))
    output_format: str = "parquet"
    ensure_available: bool = True


T = TypeVar("T")  # For zone type


class ZonalViewGenerator(ABC, Generic[T]):
    """Base class for mapping data to zonal datasets.

    This class provides the framework for mapping various data sources (points, polygons, rasters)
    to zonal geometries like grid tiles or catchment areas. It serves as an abstract base class
    that must be subclassed to implement specific zonal systems.

    The class supports three main types of data mapping:
    - Point data aggregation to zones
    - Polygon data aggregation with optional area weighting
    - Raster data sampling and statistics

    Attributes:
        data_store (DataStore): The data store for accessing input data.
        generator_config (ZonalViewGeneratorConfig): Configuration for the generator.
        logger: Logger instance for this class.
    """

    def __init__(
        self,
        config: Optional[ZonalViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: logging.Logger = None,
    ):
        """Initialize the ZonalViewGenerator.

        Args:
            generator_config (ZonalViewGeneratorConfig, optional): Configuration for the generator.
                If None, uses default configuration.
            data_store (DataStore, optional): The data store for accessing input data.
                If None, uses LocalDataStore.
        """
        self.config = config or ZonalViewGeneratorConfig()
        self.data_store = data_store or LocalDataStore()
        self.logger = logger or global_config.get_logger(self.__class__.__name__)
        self._view: Optional[pd.DataFrame] = None

    @abstractmethod
    def get_zonal_geometries(self) -> List[Polygon]:
        """Get the geometries of the zones.

        This method must be implemented by subclasses to return the actual geometric
        shapes of the zones (e.g., grid tiles, catchment boundaries, administrative areas).

        Returns:
            List[Polygon]: A list of Shapely Polygon objects representing zone geometries.
        """
        pass

    @abstractmethod
    def get_zone_identifiers(self) -> List[T]:
        """Get unique identifiers for each zone.

        This method must be implemented by subclasses to return identifiers that
        correspond one-to-one with the geometries returned by get_zonal_geometries().

        Returns:
            List[T]: A list of zone identifiers (e.g., quadkeys, H3 indices, tile IDs).
                The type T is determined by the specific zonal system implementation.
        """
        pass

    def get_zone_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert zones to a GeoDataFrame.

        Creates a GeoDataFrame containing zone identifiers and their corresponding
        geometries in WGS84 (EPSG:4326) coordinate reference system.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame with 'zone_id' and 'geometry' columns,
                where zone_id contains the identifiers and geometry contains the
                corresponding Polygon objects.
        """
        return gpd.GeoDataFrame(
            {
                "zone_id": self.get_zone_identifiers(),
                "geometry": self.get_zonal_geometries(),
            },
            crs="EPSG:4326",
        )

    @property
    def zone_gdf(self) -> gpd.GeoDataFrame:
        """Cached GeoDataFrame of zones.

        Returns:
            gpd.GeoDataFrame: Lazily-computed and cached GeoDataFrame of zone geometries
                and identifiers.
        """
        if not hasattr(self, "_zone_gdf"):
            self._zone_gdf = self.get_zone_geodataframe()
        return self._zone_gdf

    @property
    def view(self) -> pd.DataFrame:
        """The DataFrame representing the current zonal view.

        Returns:
            pd.DataFrame: The DataFrame containing zone IDs, and
                              any added variables. If no variables have been added,
                              it returns the base `zone_gdf` without geometries.
        """
        if self._view is None:
            self._view = self.zone_gdf.drop(columns="geometry")
        return self._view

    def add_variable_to_view(self, data_dict: Dict, column_name: str) -> None:
        """
        Adds a new variable (column) to the zonal view GeoDataFrame.

        This method takes a dictionary (typically the result of map_points or map_polygons)
        and adds its values as a new column to the internal `_view` (or `zone_gdf` if not yet initialized).
        The dictionary keys are expected to be the `zone_id` values.

        Args:
            data_dict (Dict): A dictionary where keys are `zone_id`s and values are
                              the data to be added.
            column_name (str): The name of the new column to be added to the GeoDataFrame.
        Raises:
            ValueError: If the `data_dict` keys do not match the `zone_id`s in the zonal view.
                        If the `column_name` already exists in the zonal view.
        """
        if self._view is None:
            self._view = self.zone_gdf.drop(columns="geometry")

        if column_name in self._view.columns:
            raise ValueError(
                f"Column '{column_name}' already exists in the zonal view."
            )

        # Create a pandas Series from the dictionary, aligning by index (zone_id)
        new_series = pd.Series(data_dict, name=column_name)

        # Before merging, ensure the zone_ids in data_dict match those in _view
        missing_zones_in_data = set(self._view["zone_id"]) - set(new_series.index)
        extra_zones_in_data = set(new_series.index) - set(self._view["zone_id"])

        if missing_zones_in_data:
            self.logger.warning(
                f"Warning: {len(missing_zones_in_data)} zone(s) from the zonal view "
                f"are missing in the provided data_dict for column '{column_name}'. "
                f"These zones will have NaN values for '{column_name}'. Missing: {list(missing_zones_in_data)[:5]}..."
            )
        if extra_zones_in_data:
            self.logger.warning(
                f"Warning: {len(extra_zones_in_data)} zone(s) in the provided data_dict "
                f"are not present in the zonal view for column '{column_name}'. "
                f"These will be ignored. Extra: {list(extra_zones_in_data)[:5]}..."
            )

        # Merge the new series with the _view based on 'zone_id'
        # Using .set_index() for efficient alignment
        original_index_name = self._view.index.name
        self._view = self._view.set_index("zone_id").join(new_series).reset_index()
        if original_index_name:  # Restore original index name if it existed
            self._view.index.name = original_index_name
        else:  # If it was a default integer index, ensure it's not named 'index'
            self._view.index.name = None

        self.logger.info(f"Added variable '{column_name}' to the zonal view.")

    def map_points(
        self,
        points: Union[pd.DataFrame, gpd.GeoDataFrame],
        value_columns: Optional[Union[str, List[str]]] = None,
        aggregation: Union[str, Dict[str, str]] = "count",
        predicate: str = "within",
        output_suffix: str = "",
        mapping_function: Optional[Callable] = None,
        **mapping_kwargs,
    ) -> Dict:
        """Map point data to zones with spatial aggregation.

        Aggregates point data to zones using spatial relationships. Points can be
        counted or have their attribute values aggregated using various statistical methods.

        Args:
            points (Union[pd.DataFrame, gpd.GeoDataFrame]): The point data to map.
                Must contain geometry information if DataFrame.
            value_columns (Union[str, List[str]], optional): Column name(s) containing
                values to aggregate. If None, only point counts are performed.
            aggregation (Union[str, Dict[str, str]]): Aggregation method(s) to use.
                Can be a single string ("count", "mean", "sum", "min", "max", etc.)
                or a dictionary mapping column names to aggregation methods.
            predicate (str): Spatial predicate for point-to-zone relationship.
                Options include "within", "intersects", "contains". Defaults to "within".
            output_suffix (str): Suffix to add to output column names. Defaults to empty string.
            mapping_function (Callable, optional): Custom function for mapping points to zones.
                If provided, signature should be mapping_function(self, points, **mapping_kwargs).
                When used, all other parameters except mapping_kwargs are ignored.
            **mapping_kwargs: Additional keyword arguments passed to the mapping function.

        Returns:
            Dict: Dictionary with zone IDs as keys and aggregated values as values.
                If value_columns is None, returns point counts per zone.
                If value_columns is specified, returns aggregated values per zone.
        """
        if mapping_function is not None:
            return mapping_function(self, points, **mapping_kwargs)

        self.logger.warning(
            "Using default points mapping implementation. Consider creating a specialized mapping function."
        )
        result = aggregate_points_to_zones(
            points=points,
            zones=self.zone_gdf,
            value_columns=value_columns,
            aggregation=aggregation,
            point_zone_predicate=predicate,
            zone_id_column="zone_id",
            output_suffix=output_suffix,
        )

        if isinstance(value_columns, str):
            return result.set_index("zone_id")[value_columns].to_dict()
        elif isinstance(value_columns, list):
            # If multiple value columns, return a dictionary of dictionaries
            # Or, if preferred, a dictionary where values are lists/tuples of results
            # For now, let's return a dict of series, which is common.
            # The previous version implied a single dictionary result from map_points/polygons
            # but with multiple columns, it's usually {zone_id: {col1: val1, col2: val2}}
            # or {col_name: {zone_id: val}}
            # In this version, it'll return a dictionary for each column.
            return {
                col: result.set_index("zone_id")[col].to_dict() for col in value_columns
            }
        else:  # If value_columns is None, it should return point_count
            self.logger.warning(
                "No `value_columns` provided. Mapping point counts. Consider passing `value_columns` and `aggregation` or `mapping_function`."
            )
            return result.set_index("zone_id")["point_count"].to_dict()

    def map_polygons(
        self,
        polygons,
        value_columns: Optional[Union[str, List[str]]] = None,
        aggregation: Union[str, Dict[str, str]] = "count",
        predicate: str = "intersects",
        **kwargs,
    ) -> Dict:
        """
        Maps polygon data to the instance's zones and aggregates values.

        This method leverages `aggregate_polygons_to_zones` to perform a spatial
        aggregation of polygon data onto the zones stored within this object instance.
        It can count polygons, or aggregate their values, based on different spatial
        relationships defined by the `predicate`.

        Args:
            polygons (Union[pd.DataFrame, gpd.GeoDataFrame]):
                The polygon data to map. Must contain geometry information if a
                DataFrame.
            value_columns (Union[str, List[str]], optional):
                The column name(s) from the `polygons` data to aggregate. If `None`,
                the method will automatically count the number of polygons that
                match the given `predicate` for each zone.
            aggregation (Union[str, Dict[str, str]], optional):
                The aggregation method(s) to use. Can be a single string (e.g., "sum",
                "mean", "max") or a dictionary mapping column names to specific
                aggregation methods. This is ignored and set to "count" if
                `value_columns` is `None`. Defaults to "count".
            predicate (Literal["intersects", "within", "fractional"], optional):
                The spatial relationship to use for aggregation:
                - "intersects": Counts or aggregates values for any polygon that
                  intersects a zone.
                - "within": Counts or aggregates values for polygons that are
                  entirely contained within a zone.
                - "fractional": Performs area-weighted aggregation. The value of a
                  polygon is distributed proportionally to the area of its overlap
                  with each zone.
                Defaults to "intersects".
            **kwargs:
                Additional keyword arguments to be passed to the underlying
                `aggregate_polygons_to_zones_new` function.

        Returns:
            Dict:
                A dictionary or a nested dictionary containing the aggregated values,
                with zone IDs as keys. If `value_columns` is a single string, the
                return value is a dictionary mapping zone ID to the aggregated value.
                If `value_columns` is a list, the return value is a nested dictionary
                mapping each column name to its own dictionary of aggregated values.

        Raises:
            ValueError: If `value_columns` is of an unexpected type after processing.

        Example:
            >>> # Assuming 'self' is an object with a 'zone_gdf' attribute
            >>> # Count all land parcels that intersect each zone
            >>> parcel_counts = self.map_polygons(landuse_polygons)
            >>>
            >>> # Aggregate total population within zones using area weighting
            >>> population_by_zone = self.map_polygons(
            ...     landuse_polygons,
            ...     value_columns="population",
            ...     predicate="fractional",
            ...     aggregation="sum"
            ... )
            >>>
            >>> # Get the sum of residential area and count of buildings within each zone
            >>> residential_stats = self.map_polygons(
            ...     building_polygons,
            ...     value_columns=["residential_area_sqm", "building_id"],
            ...     aggregation={"residential_area_sqm": "sum", "building_id": "count"},
            ...     predicate="intersects"
            ... )
        """

        if value_columns is None:
            self.logger.warning(
                f"No value_columns specified. Defaulting to counting polygons with {predicate} predicate."
            )
            temp_value_col = "_temp_polygon_count_dummy"
            polygons[temp_value_col] = 1
            actual_value_columns = temp_value_col
            aggregation = "count"  # Force count if no value columns
        else:
            actual_value_columns = value_columns

        result = aggregate_polygons_to_zones(
            polygons=polygons,
            zones=self.zone_gdf,
            value_columns=actual_value_columns,
            aggregation=aggregation,
            predicate=predicate,
            zone_id_column="zone_id",
        )

        # Convert the result GeoDataFrame to the expected dictionary format
        if isinstance(actual_value_columns, str):
            return result.set_index("zone_id")[actual_value_columns].to_dict()
        elif isinstance(actual_value_columns, list):
            return {
                col: result.set_index("zone_id")[col].to_dict()
                for col in actual_value_columns
            }
        else:
            raise ValueError("Unexpected type for actual_value_columns.")

    def map_rasters(
        self,
        tif_processors: List[TifProcessor],
        mapping_function: Optional[Callable] = None,
        stat: str = "mean",
        **mapping_kwargs,
    ) -> Union[np.ndarray, Dict]:
        """Map raster data to zones using zonal statistics.

        Samples raster values within each zone and computes statistics. Automatically
        handles coordinate reference system transformations between raster and zone data.

        Args:
            tif_processors (List[TifProcessor]): List of TifProcessor objects for
                accessing raster data. All processors should have the same CRS.
            mapping_function (Callable, optional): Custom function for mapping rasters
                to zones. If provided, signature should be mapping_function(self, tif_processors, **mapping_kwargs).
                When used, stat and other parameters except mapping_kwargs are ignored.
            stat (str): Statistic to calculate when aggregating raster values within
                each zone. Options include "mean", "sum", "min", "max", "std", etc.
                Defaults to "mean".
            **mapping_kwargs: Additional keyword arguments passed to the mapping function.

        Returns:
            Union[np.ndarray, Dict]: By default, returns a NumPy array of sampled values
                with shape (n_zones, 1), taking the first non-nodata value encountered.
                Custom mapping functions may return different data structures.

        Note:
            If the coordinate reference system of the rasters differs from the zones,
            the zone geometries will be automatically transformed to match the raster CRS.
        """
        if mapping_function is not None:
            return mapping_function(self, tif_processors, **mapping_kwargs)

        raster_crs = tif_processors[0].crs

        if raster_crs != self.zone_gdf.crs:
            self.logger.info(f"Projecting zones to raster CRS: {raster_crs}")
            zone_geoms = self._get_transformed_geometries(raster_crs)
        else:
            zone_geoms = self.get_zonal_geometries()

        # Sample raster values
        sampled_values = sample_multiple_tifs_by_polygons(
            tif_processors=tif_processors, polygon_list=zone_geoms, stat=stat
        )

        zone_ids = self.get_zone_identifiers()

        return {zone_id: value for zone_id, value in zip(zone_ids, sampled_values)}

    @lru_cache(maxsize=32)
    def _get_transformed_geometries(self, target_crs):
        """Get zone geometries transformed to target coordinate reference system.

        This method is cached to avoid repeated coordinate transformations for
        the same target CRS.

        Args:
            target_crs: Target coordinate reference system for transformation.

        Returns:
            List[Polygon]: List of zone geometries transformed to the target CRS.
        """
        return self.zone_gdf.to_crs(target_crs).geometry.tolist()

    def save_view(
        self,
        name: str,
        output_format: Optional[str] = None,
    ) -> Path:
        """Save the generated zonal view to disk.

        Args:
            name (str): Base name for the output file (without extension).
            output_format (str, optional): File format to save in (e.g., "parquet",
                "geojson", "shp"). If None, uses the format specified in config.

        Returns:
            Path: The full path where the view was saved.

        Note:
            The output directory is determined by the config.base_path setting.
            The file extension is automatically added based on the output format.
            This method now saves the internal `self.view`.
        """
        if self._view is None:
            self.logger.warning(
                "No variables have been added to the zonal view. Saving the base zone_gdf."
            )
            view_to_save = self.zone_gdf
        else:
            view_to_save = self._view

        format_to_use = output_format or self.config.output_format
        output_path = self.config.base_path / f"{name}.{format_to_use}"

        self.logger.info(f"Saving zonal view to {output_path}")

        if format_to_use in ["geojson", "shp", "gpkg"]:
            self.logger.warning(
                f"Saving to {format_to_use} requires converting back to GeoDataFrame. Geometry column will be re-added."
            )
            # Re-add geometry for saving to geospatial formats
            view_to_save = self.view.merge(
                self.zone_gdf[["zone_id", "geometry"]], on="zone_id", how="left"
            )

        write_dataset(
            data=view_to_save,
            path=str(output_path),
            data_store=self.data_store,
        )

        return output_path

    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns the current zonal view as a DataFrame.

        This method combines all accumulated variables in the view

        Returns:
            pd.DataFrame: The current view.
        """
        return self.view

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Returns the current zonal view merged with zone geometries as a GeoDataFrame.

        This method combines all accumulated variables in the view with the corresponding
        zone geometries, providing a spatially-enabled DataFrame for further analysis or export.

        Returns:
            gpd.GeoDataFrame: The current view merged with zone geometries.
        """
        return gpd.GeoDataFrame(
            (self.view).merge(
                self.zone_gdf[["zone_id", "geometry"]], on="zone_id", how="left"
            ),
            crs=self.zone_gdf.crs,
        )
