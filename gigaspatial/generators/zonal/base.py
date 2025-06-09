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
    convert_to_geodataframe,
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

    def to_geodataframe(self) -> gpd.GeoDataFrame:
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
            self._zone_gdf = self.to_geodataframe()
        return self._zone_gdf

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

        else:
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

            if not value_columns:
                return result["point_count"].to_dict()

            return result[value_columns].to_dict()

    def map_polygons(
        self,
        polygons: Union[pd.DataFrame, gpd.GeoDataFrame],
        value_columns: Optional[Union[str, List[str]]] = None,
        aggregation: Union[str, Dict[str, str]] = "sum",
        area_weighted: bool = False,
        area_column: str = "area_in_meters",
        mapping_function: Optional[Callable] = None,
        **mapping_kwargs,
    ) -> Dict:
        """Map polygon data to zones with optional area weighting.

        Aggregates polygon data to zones based on spatial intersections. Values can be
        weighted by the fractional area of intersection between polygons and zones.

        Args:
            polygons (Union[pd.DataFrame, gpd.GeoDataFrame]): The polygon data to map.
                Must contain geometry information if DataFrame.
            value_columns (Union[str, List[str]], optional): Column name(s) to aggregate.
                If None, only intersection areas will be calculated.
            aggregation (Union[str, Dict[str, str]]): Aggregation method(s) to use.
                Can be a single string ("sum", "mean", "max", "min") or a dictionary
                mapping column names to specific aggregation methods. Defaults to "sum".
            area_weighted (bool): Whether to weight values by fractional area of
                intersection. Defaults to False.
            area_column (str): Name of column to store calculated areas. Only used
                if area calculation is needed. Defaults to "area_in_meters".
            mapping_function (Callable, optional): Custom function for mapping polygons
                to zones. If provided, signature should be mapping_function(self, polygons, **mapping_kwargs).
                When used, all other parameters except mapping_kwargs are ignored.
            **mapping_kwargs: Additional keyword arguments passed to the mapping function.

        Returns:
            Dict: Dictionary with zone IDs as keys and aggregated values as values.
                Returns aggregated values for the specified value_columns.

        Raises:
            TypeError: If polygons cannot be converted to a GeoDataFrame.
        """
        if mapping_function is not None:
            return mapping_function(self, polygons, **mapping_kwargs)

        if area_column not in polygons_gdf:
            if not isinstance(polygons, gpd.GeoDataFrame):
                try:
                    polygons_gdf = convert_to_geodataframe(polygons)
                except:
                    raise TypeError(
                        "polygons must be a GeoDataFrame or convertible to one"
                    )
            else:
                polygons_gdf = polygons.copy()

            polygons_gdf[area_column] = polygons_gdf.to_crs(
                polygons_gdf.estimate_utm_crs()
            ).geometry.area

        if value_columns is None:
            self.logger.warning(
                "Using default polygon mapping implementation. Consider providing value_columns."
            )
            value_columns = area_column

        result = aggregate_polygons_to_zones(
            polygons=polygons_gdf,
            zones=self.zone_gdf,
            value_columns=value_columns,
            aggregation=aggregation,
            area_weighted=area_weighted,
            zone_id_column="zone_id",
        )

        return result[value_columns].to_dict()

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
                with shape (n_zones, n_rasters), taking the first non-nodata value encountered.
                Custom mapping functions may return different data structures.

        Note:
            If the coordinate reference system of the rasters differs from the zones,
            the zone geometries will be automatically transformed to match the raster CRS.
        """
        if mapping_function is not None:
            return mapping_function(self, tif_processors, **mapping_kwargs)

        self.logger.warning(
            "Using default raster mapping implementation. Consider creating a specialized mapping function."
        )

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

        return sampled_values

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
        view_data: gpd.GeoDataFrame,
        name: str,
        output_format: Optional[str] = None,
    ) -> Path:
        """Save the generated zonal view to disk.

        Args:
            view_data (gpd.GeoDataFrame): The zonal view data to save.
            name (str): Base name for the output file (without extension).
            output_format (str, optional): File format to save in (e.g., "parquet",
                "geojson", "shp"). If None, uses the format specified in generator_config.

        Returns:
            Path: The full path where the view was saved.

        Note:
            The output directory is determined by the generator_config.base_path setting.
            The file extension is automatically added based on the output format.
        """
        format_to_use = output_format or self.config.output_format
        output_path = self.config.base_path / f"{name}.{format_to_use}"

        self.logger.info(f"Saving zonal view to {output_path}")
        write_dataset(
            df=view_data,
            path=str(output_path),
            data_store=self.data_store,
            format=format_to_use,
        )

        return output_path
