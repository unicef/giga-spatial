from functools import lru_cache
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from shapely.geometry import MultiPolygon, Polygon

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.writers import write_dataset
from gigaspatial.handlers.ghsl import GHSLDataHandler
from gigaspatial.handlers.google_open_buildings import GoogleOpenBuildingsHandler
from gigaspatial.handlers.microsoft_global_buildings import MSBuildingsHandler
from gigaspatial.handlers.rwi import RWIHandler
from gigaspatial.handlers.worldpop import WPPopulationHandler
from gigaspatial.processing.buildings_engine import GoogleMSBuildingsEngine
from gigaspatial.processing.geo import (
    add_area_in_meters,
    aggregate_points_to_zones,
    aggregate_polygons_to_zones,
    get_centroids,
)
from gigaspatial.processing.tif_processor import TifProcessor

if TYPE_CHECKING:
    from gigaspatial.handlers.google_ms_combined_buildings import (
        GoogleMSBuildingsHandler,
    )


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


class ZonalViewGenerator(Generic[T]):
    """Generates enriched zonal views over polygonal zones.

    Handles zones defined as either a mapping of zone identifiers to geometries
    or a GeoDataFrame with an identifier column, and supports point, polygon,
    raster, and dataset enrichment operations (`map_wp_pop`, `map_built_s`,
    `map_smod`, `map_ghsl_pop`, `map_ghsl`, `map_buildings`,
    `map_google_buildings`, `map_ms_buildings`, `map_rwi`).

    Attributes:
        zone_dict (Dict[T, Polygon]): Mapping of zone identifiers to geometries.
        zone_id_column (str): Name of the source column containing zone identifiers.
        zone_data_crs (str): Coordinate reference system of the zone data.
        _zone_gdf (gpd.GeoDataFrame): Cached GeoDataFrame representation of zones.
        data_store (DataStore): The data store for accessing input data.
        config (ZonalViewGeneratorConfig): Configuration for the generator.
        logger: Logger instance for this class.
    """

    def __init__(
        self,
        zone_data: Union[Dict[T, Polygon], gpd.GeoDataFrame],
        zone_id_column: str = "zone_id",
        zone_data_crs: str = "EPSG:4326",
        config: Optional[ZonalViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: logging.Logger = None,
    ):
        """Initialize ZonalViewGenerator with zone geometries and identifiers.

        Args:
            zone_data (Union[Dict[T, Polygon], gpd.GeoDataFrame]): Zone definitions.
                Either a dictionary mapping zone identifiers to Polygon/MultiPolygon
                geometries, or a GeoDataFrame with geometries and a zone identifier column.
            zone_id_column (str): Name of the column containing zone identifiers.
                Only used if ``zone_data`` is a GeoDataFrame. Defaults to "zone_id".
            zone_data_crs (str): Coordinate reference system of the zone data.
                Defaults to "EPSG:4326" (WGS84).
            config (ZonalViewGeneratorConfig, optional): Generator configuration.
                If None, uses default configuration.
            data_store (DataStore, optional): Data store for accessing input data.
                If None, uses LocalDataStore.
            logger (logging.Logger, optional): Custom logger instance. If None, uses
                a default logger.

        Raises:
            TypeError: If ``zone_data`` is not a dictionary or GeoDataFrame, or if
                dictionary values are not Polygon/MultiPolygon geometries.
            ValueError: If ``zone_id_column`` is missing, contains duplicate values,
                collides with an existing ``'zone_id'`` column, or if
                ``zone_data_crs`` does not match the GeoDataFrame's CRS.
        """
        self.config = config or ZonalViewGeneratorConfig()
        self.data_store = data_store or LocalDataStore()
        self.logger = logger or global_config.get_logger(self.__class__.__name__)
        self._view: Optional[pd.DataFrame] = None
        if not hasattr(self, "_country"):
            self._country: Optional[str] = None

        self.zone_id_column = zone_id_column
        self.zone_data_crs = zone_data_crs

        if isinstance(zone_data, dict):
            for zone_id, geom in zone_data.items():
                if not isinstance(geom, (Polygon, MultiPolygon)):
                    raise TypeError(
                        f"Zone {zone_id}: Expected (Multi)Polygon, got {type(geom).__name__}"
                    )

            self.zone_dict = zone_data
            self._zone_gdf = gpd.GeoDataFrame(
                {
                    "zone_id": list(zone_data.keys()),
                    "geometry": list(zone_data.values()),
                },
                crs=zone_data_crs,
            )
            self.zone_id_column = "zone_id"
        else:
            if not isinstance(zone_data, gpd.GeoDataFrame):
                raise TypeError(
                    "zone_data must be either a Dict[T, Polygon] or a GeoDataFrame"
                )

            if zone_id_column not in zone_data.columns:
                raise ValueError(
                    f"Zone ID column '{zone_id_column}' not found in GeoDataFrame"
                )

            if zone_id_column != "zone_id" and "zone_id" in zone_data.columns:
                raise ValueError(
                    f"Column 'zone_id' already exists in GeoDataFrame; cannot rename '{zone_id_column}' to 'zone_id'."
                )

            if zone_data[zone_id_column].duplicated().any():
                raise ValueError(
                    f"Column '{zone_id_column}' provided as 'zone_id_column' contains duplicate values."
                )

            if zone_data_crs != zone_data.crs:
                raise ValueError(
                    f"Provided data crs '{zone_data_crs}' does not match to the crs of the data '{zone_data.crs}'"
                )

            self._zone_gdf = zone_data.rename(columns={zone_id_column: "zone_id"}).copy()
            self.zone_dict = dict(zip(zone_data[zone_id_column], zone_data.geometry))

    def _resolve_country(
        self, country: Optional[Union[str, List[str]]]
    ) -> Union[str, List[str]]:
        """Resolve an explicit country or fall back to ``self._country`` when set."""
        resolved = getattr(self, "_country", None) if country is None else country
        if resolved is None:
            raise ValueError(
                "country must be provided when the generator was not initialized from a country."
            )
        return resolved

    def get_zonal_geometries(self) -> List[Polygon]:
        """Get the geometry of each zone.

        Returns:
            List[Polygon]: A list of zone geometries in the order they appear in the
                underlying GeoDataFrame.
        """
        return self._zone_gdf.geometry.tolist()

    def get_zone_identifiers(self) -> List[T]:
        """Get the identifier for each zone.

        Returns:
            List[T]: A list of zone identifiers in the order they appear in the
                underlying GeoDataFrame.
        """
        return self._zone_gdf.zone_id.tolist()

    def get_zone_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert zones to a GeoDataFrame with standardized column names.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame with 'zone_id' and 'geometry' columns.
        """
        return self._zone_gdf.copy()

    @property
    def zone_gdf(self) -> gpd.GeoDataFrame:
        """GeoDataFrame of zone identifiers and geometries.

        Returns:
            gpd.GeoDataFrame: A copy of the canonical GeoDataFrame with
                'zone_id' and 'geometry' columns.
        """
        return self._zone_gdf.copy()

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
        Adds a new variable (column) to the zonal view DataFrame.

        This method takes a dictionary (typically the result of map_points or map_polygons)
        and adds its values as a new column to the internal `_view` (or `zone_gdf` if not yet initialized).
        The dictionary keys are expected to be the `zone_id` values.

        Args:
            data_dict (Dict): A dictionary where keys are `zone_id`s and values are
                              the data to be added.
            column_name (str): The name of the new column to be added to the DataFrame.
        Raises:
            ValueError: If the `column_name` already exists in the zonal view.
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

        Returns:
            Dict: Dictionary with zone IDs as keys and aggregated values as values.
                If value_columns is None, returns point counts per zone.
                If value_columns is specified, returns aggregated values per zone.
        """
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
                `aggregate_polygons_to_zones` function.

        Returns:
            Dict:
                A dictionary or a nested dictionary containing the aggregated values,
                with zone IDs as keys. If `value_columns` is a single string, the
                return value is a dictionary mapping zone ID to the aggregated value.
                If `value_columns` is a list, the return value is a nested dictionary
                mapping each column name to its own dictionary of aggregated values.

        Raises:
            ValueError: If `value_columns` is of an unexpected type after processing.
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
        raster_data: Union[TifProcessor, List[TifProcessor]],
        stat: str = "mean",
        **kwargs,
    ) -> Dict:
        """Map raster data to zones using zonal statistics.

        Samples raster values within each zone and computes statistics. Automatically
        handles coordinate reference system transformations between raster and zone data.

        Args:
            raster_data (Union[TifProcessor, List[TifProcessor]]):
                Either a TifProcessor object or a list of TifProcessor objects (which will be merged
                into a single TifProcessor for processing).
            stat (str): Statistic to calculate when aggregating raster values within
                each zone. Options include "mean", "sum", "min", "max", "std", etc.
                Defaults to "mean".
            **kwargs: Additional keyword arguments for raster data.

        Returns:
            Dict: By default, returns a dictionary of sampled values
                with zone IDs as keys.

        Note:
            If the coordinate reference system of the rasters differs from the zones,
            the zone geometries will be automatically transformed to match the raster CRS.
        """
        raster_processor: Optional[TifProcessor] = None

        if isinstance(raster_data, TifProcessor):
            raster_processor = raster_data
        elif isinstance(raster_data, list) and all(
            isinstance(x, TifProcessor) for x in raster_data
        ):
            if not raster_data:
                self.logger.info("No valid raster data provided")
                return self.view

            if len(raster_data) > 1:
                all_source_paths = [tp.dataset_path for tp in raster_data]

                self.logger.info(
                    f"Merging {len(all_source_paths)} rasters into a single TifProcessor for zonal statistics."
                )
                raster_processor = TifProcessor(
                    dataset_path=all_source_paths, data_store=self.data_store, **kwargs
                )
            else:
                raster_processor = raster_data[0]
        else:
            raise ValueError(
                "raster_data must be a TifProcessor object or a list of TifProcessor objects."
            )

        raster_crs = raster_processor.crs

        if raster_crs != self.zone_gdf.crs:
            self.logger.info(f"Projecting zones to raster CRS: {raster_crs}")
            zone_geoms = self._get_transformed_geometries(raster_crs)
        else:
            zone_geoms = self.get_zonal_geometries()

        # Sample raster values
        sampled_values = raster_processor.sample_by_polygons(
            polygon_list=zone_geoms, stat=stat
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

    def map_built_s(
        self,
        year=2020,
        resolution=100,
        stat: str = "sum",
        output_column: str = "built_surface_m2",
        handler: Optional[GHSLDataHandler] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Map GHSL Built-up Surface data to zones.

        Convenience method for mapping Global Human Settlement Layer Built-up Surface
        data using appropriate default parameters for built surface analysis.

        Args:
            year: The year of the data (default: 2020)
            resolution: The resolution in meters (default: 100)
            stat (str): Statistic to calculate for built surface values within each zone.
                Defaults to "sum" which gives total built surface area.
            output_column (str): The output column name. Defaults to "built_surface_m2".
            handler (Optional[GHSLDataHandler]): Optional pre-configured handler.
        Returns:
            pd.DataFrame: Updated view DataFrame and settlement classification.
                Adds a column with `output_column` containing the aggregated values.
        """
        handler = handler or GHSLDataHandler(
            product="GHS_BUILT_S",
            year=year,
            resolution=resolution,
            data_store=self.data_store,
            **kwargs,
        )

        return self.map_ghsl(
            handler=handler, stat=stat, output_column=output_column, **kwargs
        )

    def map_smod(
        self,
        year=2020,
        resolution=1000,
        stat: str = "median",
        output_column: str = "smod_class",
        handler: Optional[GHSLDataHandler] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Map GHSL Settlement Model data to zones.

        Convenience method for mapping Global Human Settlement Layer Settlement Model
        data using appropriate default parameters for settlement classification analysis.

        Args:
            year: The year of the data (default: 2020)
            resolution: The resolution in meters (default: 1000)
            stat (str): Statistic to calculate for settlement class values within each zone.
                Defaults to "median" which gives the predominant settlement class.
            output_column (str): The output column name. Defaults to "smod_class".
            handler (Optional[GHSLDataHandler]): Optional pre-configured handler.
        Returns:
            pd.DataFrame: Updated view DataFrame and settlement classification.
                Adds a column with `output_column` containing the aggregated values.
        """
        handler = handler or GHSLDataHandler(
            product="GHS_SMOD",
            year=year,
            resolution=resolution,
            data_store=self.data_store,
            coord_system=54009,
            **kwargs,
        )

        return self.map_ghsl(
            handler=handler, stat=stat, output_column=output_column, **kwargs
        )

    def map_ghsl(
        self,
        handler: Optional[GHSLDataHandler] = None,
        stat: str = "mean",
        output_column: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Map Global Human Settlement Layer data to zones.

        Loads and processes GHSL raster data for the intersecting tiles, then samples
        the raster values within each zone using the specified statistic.

        Args:
            handler (Optional[GHSLDataHandler]): Optional pre-configured handler
                for GHSL data. Defaults to None.
            stat (str): Statistic to calculate for raster values within each zone.
                Common options: "mean", "sum", "median", "min", "max".
                Defaults to "mean".
            output_column (Optional[str]): The output column name.
                If None, uses the GHSL product name in lowercase followed by underscore.

        Returns:
            pd.DataFrame: Updated DataFrame with GHSL metrics.
                Adds a column named as `output_column` containing the sampled values.

        Note:
            The method automatically determines which GHSL tiles intersect with the zones
            and loads only the necessary data for efficient processing.
        """
        coord_system = kwargs.pop("coord_system", None)
        if not coord_system:
            coord_system = 4326 if self.zone_data_crs == "EPSG:4326" else 54009

        handler = handler or GHSLDataHandler(
            data_store=self.data_store, coord_system=coord_system, **kwargs
        )
        self.logger.info(
            f"Mapping {handler.config.product} data (year: {handler.config.year}, resolution: {handler.config.resolution}m)"
        )
        tif_processors = handler.load_data(
            self.zone_gdf,
            ensure_available=self.config.ensure_available,
            merge_rasters=True,
            **handler.select_options(kwargs),
        )

        self.logger.info(
            f"Sampling {handler.config.product} data using '{stat}' statistic"
        )
        sampled_values = self.map_rasters(raster_data=tif_processors, stat=stat)

        column_name = (
            output_column
            if output_column
            else f"{handler.config.product.lower()}_{stat}"
        )

        self.add_variable_to_view(sampled_values, column_name)

        return self.view

    def map_google_buildings(
        self,
        handler: Optional[GoogleOpenBuildingsHandler] = None,
        use_polygons: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Map Google Open Buildings data to zones.

        Processes Google Open Buildings dataset to calculate building counts and total
        building area within each zone. Can use either point centroids (faster) or
        polygon geometries (more accurate) for spatial operations.

        Args:
            handler (Optional[GoogleOpenBuildingsHandler]): Optional pre-configured
                handler for Google Open Buildings data. Defaults to None.
            use_polygons (bool): Whether to use polygon geometries for buildings.
                If True, uses actual building polygons for more accurate area calculations
                but with slower performance. If False, uses building centroids with
                area values from attributes for faster processing. Defaults to False.

        Returns:
            pd.DataFrame: Updated DataFrame with building metrics.
                Adds columns:
                - 'google_buildings_count': Number of buildings in each zone
                - 'google_buildings_area_in_meters': Total building area in square meters

        Note:
            If no Google Buildings data is found for the zones, returns the original
            GeoDataFrame unchanged with a warning logged.
        """
        self.logger.info(
            f"Mapping Google Open Buildings data (use_polygons={use_polygons})"
        )

        self.logger.info("Loading Google Buildings point data")
        handler = handler or GoogleOpenBuildingsHandler(data_store=self.data_store)
        buildings_df = handler.load_points(
            self.zone_gdf, ensure_available=self.config.ensure_available, **kwargs
        )

        if buildings_df.empty:
            self.logger.warning("No Google buildings data found for the provided zones")
            return self._zone_gdf.copy()

        if not use_polygons:
            self.logger.info("Aggregating building data using points with attributes")
            result = self.map_points(
                points=buildings_df,
                value_columns=["full_plus_code", "area_in_meters"],
                aggregation={"full_plus_code": "count", "area_in_meters": "sum"},
                predicate="within",
            )

            count_result = result["full_plus_code"]
            area_result = result["area_in_meters"]

        else:
            self.logger.info(
                "Loading Google Buildings polygon data for more accurate mapping"
            )
            buildings_gdf = handler.load_polygons(
                self.zone_gdf, ensure_available=self.config.ensure_available, **kwargs
            )

            self.logger.info(
                "Calculating building areas with area-weighted aggregation"
            )
            area_result = self.map_polygons(
                buildings_gdf,
                value_columns="area_in_meters",
                aggregation="sum",
                predicate="fractional",
            )

            self.logger.info("Counting buildings using points data")
            count_result = self.map_points(points=buildings_df, predicate="within")

        self.add_variable_to_view(count_result, "google_buildings_count")
        self.add_variable_to_view(area_result, "google_buildings_area_in_meters")

        return self.view

    def map_ms_buildings(
        self,
        handler: Optional[MSBuildingsHandler] = None,
        use_polygons: bool = False,
    ) -> gpd.GeoDataFrame:
        """Map Microsoft Global Buildings data to zones.

        Processes Microsoft Global Buildings dataset to calculate building counts and
        total building area within each zone. Can use either centroid points (faster)
        or polygon geometries (more accurate) for spatial operations.

        Args:
            handler (Optional[MSBuildingsHandler]): Optional pre-configured handler
                for Microsoft Global Buildings data. If None, uses default configuration.
            use_polygons (bool): Whether to use polygon geometries for buildings.
                If True, uses actual building polygons for more accurate area calculations
                but with slower performance. If False, uses building centroids with
                area values from attributes for faster processing. Defaults to False.

        Returns:
            gpd.GeoDataFrame: Updated view DataFrame with building metrics.
                Adds columns:
                - 'ms_buildings_count': Number of buildings in each zone
                - 'ms_buildings_area_in_meters': Total building area in square meters

        Note:
            If no Microsoft Buildings data is found for the zones, returns the original
            GeoDataFrame unchanged with a warning logged. Building areas are calculated
            in meters using appropriate UTM projections.
        """
        self.logger.info("Mapping Microsoft Global Buildings data")

        self.logger.info("Loading Microsoft Buildings polygon data")
        handler = handler or MSBuildingsHandler(data_store=self.data_store)
        buildings_gdf = handler.load_data(
            self.zone_gdf, ensure_available=self.config.ensure_available
        )

        # Check if we found any buildings
        if buildings_gdf.empty:
            self.logger.warning(
                "No Microsoft buildings data found for the provided zones"
            )
            return self._zone_gdf.copy()

        buildings_gdf = add_area_in_meters(
            buildings_gdf, area_column_name="area_in_meters"
        )

        building_centroids = get_centroids(buildings_gdf)

        if not use_polygons:
            self.logger.info("Aggregating building data using points with attributes")

            result = self.map_points(
                points=building_centroids,
                value_columns=["type", "area_in_meters"],
                aggregation={"type": "count", "area_in_meters": "sum"},
                predicate="within",
            )

            count_result = result["type"]
            area_result = result["area_in_meters"]
        else:

            self.logger.info(
                "Calculating building areas with area-weighted aggregation"
            )
            area_result = self.map_polygons(
                buildings_gdf,
                value_columns="area_in_meters",
                aggregation="sum",
                predicate="fractional",
            )

            self.logger.info("Counting Microsoft buildings per zone")

            count_result = self.map_points(
                points=building_centroids, predicate="within"
            )

        self.add_variable_to_view(count_result, "ms_buildings_count")
        self.add_variable_to_view(area_result, "ms_buildings_area_in_meters")

        return self.view

    def map_buildings(
        self,
        country: Optional[str] = None,
        output_column: str = "building_count",
        handler: Optional["GoogleMSBuildingsHandler"] = None,
        **kwargs,
    ):
        """Map Google-Microsoft combined buildings data to zones.

        Efficiently counts buildings within each zone by leveraging spatial indexing
        and S2 grid partitioning. For partitioned datasets, only loads building tiles
        that intersect with zones, significantly improving performance for large countries.

        Parameters
        ----------
        country : Optional[str], default=None
            ISO 3166-1 alpha-3 country code (e.g., 'USA', 'BRA', 'IND').
            Defaults to the generator's country when initialized from a country string.
        output_column : str, default='building_count'
            Name of the column to add to the view containing building counts.
            Must be a valid pandas column name.
        handler : Optional[GoogleMSBuildingsHandler], default=None
            Optional pre-configured handler for combined building data.
        **kwargs : dict
            Additional keyword arguments passed to GoogleMSBuildingsHandler.

        Returns
        -------
        pd.DataFrame
            Updated view DataFrame with building counts added.
        """
        country = self._resolve_country(country)
        self.logger.info("Mapping Google-Microsoft Buildings data to zones")

        if handler is None:
            from gigaspatial.handlers.google_ms_combined_buildings import (
                GoogleMSBuildingsHandler,
            )

            handler = GoogleMSBuildingsHandler(
                partition_strategy="s2_grid", data_store=self.data_store
            )

        if self.config.ensure_available:
            if not handler.ensure_data_available(country, **kwargs):
                raise RuntimeError("Could not ensure data availability for loading")

        building_files = handler.reader.resolve_source_paths(
            country, **handler.select_options(kwargs, handler.config)
        )

        result = GoogleMSBuildingsEngine.count_buildings_in_zones(
            handler=handler,
            building_files=building_files,
            zones_gdf=self.zone_gdf,
            logger=self.logger,
        )

        # Log summary
        total_buildings = result.counts.sum()
        zones_with_buildings = (result.counts > 0).sum()
        self.logger.info(
            f"Mapping complete: {total_buildings:,.0f} buildings across "
            f"{zones_with_buildings}/{len(result.counts)} zones"
        )

        # Update the view and return
        self.add_variable_to_view(result.counts, output_column)
        return self.view

    def map_ghsl_pop(
        self,
        resolution=100,
        stat: str = "sum",
        output_column: str = "ghsl_pop",
        predicate: Literal["intersects", "fractional"] = "intersects",
        handler: Optional[GHSLDataHandler] = None,
        **kwargs,
    ):
        """Map GHSL Population (GHS_POP) data to zones.

        Args:
            resolution: Dataset resolution in meters (100 or 1000). Defaults to 100.
            stat: Aggregation statistic. Defaults to "sum".
            output_column: Destination column name. Defaults to "ghsl_pop".
            predicate: Spatial predicate ("intersects" or "fractional").
                Defaults to "intersects".
            handler: Optional pre-configured ``GHSLDataHandler``. Defaults to None.
            **kwargs: Extra keyword arguments forwarded to ``GHSLDataHandler`` or
                data loading.

        Returns:
            pd.DataFrame: Updated view DataFrame with population values.
        """
        handler = handler or GHSLDataHandler(
            product="GHS_POP",
            resolution=resolution,
            data_store=self.data_store,
            **kwargs,
        )

        if predicate == "fractional":
            if handler.config.resolution == 100:
                self.logger.warning(
                    "Fractional aggregations only supported for datasets with 1000m resolution. Using `intersects` as predicate"
                )
                predicate = "intersects"
            else:
                gdf_pop = handler.load_into_geodataframe(self.zone_gdf, **kwargs)

                result = self.map_polygons(
                    gdf_pop,
                    value_columns="pixel_value",
                    aggregation="sum",
                    predicate="fractional",
                )

                self.add_variable_to_view(result, output_column)
                return self.view

        return self.map_ghsl(
            handler=handler, stat=stat, output_column=output_column, **kwargs
        )

    def map_wp_pop(
        self,
        country: Optional[Union[str, List[str]]] = None,
        resolution=1000,
        predicate: Literal[
            "centroid_within", "intersects", "fractional"
        ] = "intersects",
        output_column: str = "population",
        handler: Optional[WPPopulationHandler] = None,
        **kwargs,
    ):
        """Map WorldPop population data to zones.

        Args:
            country: Country code or list of country codes. Defaults to the
                generator's country when initialized from a country string.
            resolution: Dataset resolution in meters (100 or 1000). Defaults to 1000.
            predicate: Spatial predicate ("centroid_within", "intersects", or
                "fractional"). Defaults to "intersects".
            output_column: Destination column name. Defaults to "population".
            handler: Optional pre-configured ``WPPopulationHandler``.
                Defaults to None.
            **kwargs: Extra keyword arguments forwarded to ``WPPopulationHandler``.

        Returns:
            pd.DataFrame: Updated view DataFrame with WorldPop values.
        """
        country = self._resolve_country(country)

        # Ensure country is always a list for consistent handling
        countries_list = [country] if isinstance(country, str) else country

        handler = handler or WPPopulationHandler(
            resolution=resolution,
            data_store=self.data_store,
            **kwargs,
        )

        # Force statistic based on project type: categorical data (DU) use median, population counts use sum.
        stat = "median" if handler.config.project == "degree_of_urbanization" else "sum"

        # Restrict to single country for age_structures project
        if handler.config.project == "age_structures" and len(countries_list) > 1:
            raise ValueError(
                "For 'age_structures' project, only a single country can be processed at a time."
            )

        self.logger.info(
            f"Mapping WorldPop Population data (year: {handler.config.year}, resolution: {handler.config.resolution}m)"
        )

        if predicate == "fractional" and handler.config.resolution == 100:
            self.logger.warning(
                "Fractional aggregations only supported for datasets with 1000m resolution. Using `intersects` as predicate"
            )
            predicate = "intersects"

        if predicate == "centroid_within":
            if handler.config.project == "age_structures":
                # Single country enforced above
                iso = countries_list[0]
                raw = handler.load_data(
                    iso,
                    ensure_available=self.config.ensure_available,
                    **handler.select_options(kwargs),
                )
                all_tif_processors = self._ensure_tif_list(raw)

                # Sum results from each tif_processor separately
                all_results_by_zone = {
                    zone_id: 0 for zone_id in self.get_zone_identifiers()
                }
                self.logger.info(
                    f"Sampling individual age_structures rasters using '{stat}' statistic and summing results per zone."
                )
                for tif_processor in all_tif_processors:
                    single_raster_result = self.map_rasters(
                        raster_data=tif_processor, stat=stat
                    )
                    for zone_id, value in single_raster_result.items():
                        all_results_by_zone[zone_id] += value

                result = all_results_by_zone

            else:
                # Non age_structures: aggregate all countries into a flat list
                tif_processors: List[TifProcessor] = []
                for iso in countries_list:
                    raw = handler.load_data(
                        iso,
                        ensure_available=self.config.ensure_available,
                        **handler.select_options(kwargs),
                    )
                    tif_processors.extend(self._ensure_tif_list(raw))

                self.logger.info(
                    f"Sampling WorldPop Population data using '{stat}' statistic"
                )
                result = self.map_rasters(raster_data=tif_processors, stat=stat)
        else:
            gdf_pop = pd.concat(
                [
                    handler.load_into_geodataframe(
                        c,
                        ensure_available=self.config.ensure_available,
                        **kwargs,
                    )
                    for c in countries_list
                ],
                ignore_index=True,
            )

            self.logger.info(
                f"Aggregating WorldPop Population data to the zones using '{stat}' statistic."
            )
            result = self.map_polygons(
                gdf_pop,
                value_columns="pixel_value",
                aggregation=stat,
                predicate=predicate,
            )

        self.add_variable_to_view(result, output_column)

        return self.view

    def map_rwi(
        self,
        country: Optional[Union[str, List[str]]] = None,
        predicate: Literal["intersects", "within", "centroid_within"] = "intersects",
        aggregation: Literal["mean", "median", "max", "min"] = "mean",
        output_column: str = "rwi",
        handler: Optional[RWIHandler] = None,
        **kwargs,
    ):
        """Map Relative Wealth Index (RWI) data to zones.

        Args:
            country: Country name, ISO code, or list of codes. Defaults to the
                generator's country when initialized from a country string.
            predicate: Spatial predicate ("intersects", "within", or
                "centroid_within"). Defaults to "intersects".
            aggregation: Aggregation function ("mean", "median", "max", "min").
                Defaults to "mean".
            output_column: Base column name prefix; outputs ``f"{output_column}_{aggregation}"``.
                Defaults to "rwi".
            handler: Optional pre-configured ``RWIHandler``. Defaults to None.
            **kwargs: Extra keyword arguments forwarded to ``RWIHandler``.

        Returns:
            pd.DataFrame: Updated view DataFrame with RWI metrics, or None if the
                country has no RWI data units.
        """
        country = self._resolve_country(country)

        handler = handler or RWIHandler(
            data_store=self.data_store,
            **kwargs,
        )

        if not handler.config.get_relevant_data_units(country):
            self.logger.warning("Country not exist in rwi context - abort mission")
            return

        if predicate == "centroid_within":
            data = handler.load_data(country)
            result = self.map_points(data, value_columns="rwi", aggregation=aggregation)
        else:
            data = handler.load_as_geodataframe(country)
            result = self.map_polygons(
                data, value_columns="rwi", aggregation=aggregation, predicate=predicate
            )

        self.add_variable_to_view(result, f"{output_column}_{aggregation}")

        return self.view

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

    @staticmethod
    def _ensure_tif_list(obj) -> List["TifProcessor"]:
        """
        Normalize handler.load_data output into a flat list of TifProcessor.

        - If obj is a single TifProcessor, wrap it in a list.
        - If obj is an iterable of TifProcessor, return list(obj).
        - If obj is None or empty, return [].
        """
        if obj is None:
            return []
        # Avoid treating strings as iterables of characters
        if isinstance(obj, TifProcessor):
            return [obj]
        if isinstance(obj, Iterable):
            items = list(obj)
            # If it's already a flat list of TifProcessor, just return it
            if not items:
                return []
            if isinstance(items[0], TifProcessor):
                return items
            # If for some reason handler returns nested, flatten one level
            flat: List[TifProcessor] = []
            for it in items:
                if isinstance(it, TifProcessor):
                    flat.append(it)
                elif isinstance(it, Iterable):
                    flat.extend(list(it))  # best‑effort flatten
            return flat
        # Fallback: unexpected type – be explicit
        raise TypeError(
            f"Unexpected type from load_data: {type(obj).__name__}; "
            f"expected TifProcessor or iterable of TifProcessor."
        )
