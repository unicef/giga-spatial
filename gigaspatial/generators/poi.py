from pathlib import Path
from typing import List, Optional, Union, Tuple, Literal
from pydantic.dataclasses import dataclass, Field

import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import warnings

from scipy.spatial import cKDTree

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.writers import write_dataset

from gigaspatial.handlers.google_open_buildings import GoogleOpenBuildingsHandler
from gigaspatial.handlers.microsoft_global_buildings import MSBuildingsHandler
from gigaspatial.handlers.ghsl import GHSLDataHandler
from gigaspatial.handlers.worldpop import WPPopulationHandler

from gigaspatial.processing.geo import (
    convert_to_geodataframe,
    buffer_geodataframe,
    detect_coordinate_columns,
    aggregate_points_to_zones,
    aggregate_polygons_to_zones,
    get_centroids,
)
from gigaspatial.processing.tif_processor import TifProcessor
from gigaspatial.processing.buildings_engine import GoogleMSBuildingsEngine


@dataclass
class PoiViewGeneratorConfig:
    """
    Configuration for POI (Point of Interest) view generation.

    Attributes:
        base_path (Path): The base directory where generated POI views will be saved.
                          Defaults to a path retrieved from `config`.
        output_format (str): The default format for saving output files (e.g., "csv", "geojson").
                             Defaults to "csv".
    """

    base_path: Path = Field(default=global_config.get_path("poi", "views"))
    output_format: str = "csv"
    ensure_available: bool = True


class PoiViewGenerator:
    """
    POI View Generator for integrating various geospatial datasets
    such as Google Open Buildings, Microsoft Global Buildings, GHSL Built Surface,
    and GHSL Settlement Model (SMOD) data with Points of Interest (POIs).

    This class provides methods to load, process, and map external geospatial
    data to a given set of POIs, enriching them with relevant attributes.
    It leverages handler/reader classes for efficient data access and processing.

    The POIs can be initialized from a list of (latitude, longitude) tuples,
    a list of dictionaries, a pandas DataFrame, or a geopandas GeoDataFrame.
    """

    def __init__(
        self,
        points: Union[
            List[Tuple[float, float]], List[dict], pd.DataFrame, gpd.GeoDataFrame
        ],
        poi_id_column: str = "poi_id",
        config: Optional[PoiViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: logging.Logger = None,
    ):
        """
        Initializes the PoiViewGenerator with the input points and configurations.

        The input `points` are converted into an internal GeoDataFrame
        (`_points_gdf`) for consistent geospatial operations.

        Args:
            points (Union[List[Tuple[float, float]], List[dict], pd.DataFrame, gpd.GeoDataFrame]):
                The input points of interest. Can be:
                - A list of (latitude, longitude) tuples.
                - A list of dictionaries, where each dict must contain 'latitude' and 'longitude' keys.
                - A pandas DataFrame with 'latitude' and 'longitude' columns.
                - A geopandas GeoDataFrame (expected to have a 'geometry' column representing points).
            generator_config (Optional[PoiViewGeneratorConfig]):
                Configuration for the POI view generation process. If None, a
                default `PoiViewGeneratorConfig` will be used.
            data_store (Optional[DataStore]):
                An instance of a data store for managing data access (e.g., LocalDataStore).
                If None, a default `LocalDataStore` will be used.
        """
        if hasattr(points, "__len__") and len(points) == 0:
            raise ValueError("Points input cannot be empty")

        self.config = config or PoiViewGeneratorConfig()
        self.data_store = data_store or LocalDataStore()
        self.logger = logger or global_config.get_logger(self.__class__.__name__)
        self._points_gdf = self._init_points_gdf(points, poi_id_column)
        self._view: pd.DataFrame = self._points_gdf.drop(columns=["geometry"])

    @staticmethod
    def _init_points_gdf(
        points: Union[
            List[Tuple[float, float]], List[dict], pd.DataFrame, gpd.GeoDataFrame
        ],
        poi_id_column: str,
    ) -> gpd.GeoDataFrame:
        """
        Internal static method to convert various point input formats into a GeoDataFrame.

        This method standardizes coordinate column names to 'latitude' and 'longitude'
        for consistent internal representation. It also ensures each point has a unique
        identifier in the 'poi_id' column.

        Args:
            points: Input points in various formats:
                - List of (latitude, longitude) tuples
                - List of dictionaries with coordinate keys
                - DataFrame with coordinate columns
                - GeoDataFrame with point geometries

        Returns:
            gpd.GeoDataFrame: Standardized GeoDataFrame with 'latitude', 'longitude',
                             and 'poi_id' columns

        Raises:
            ValueError: If points format is not supported or coordinate columns cannot be detected
        """
        if isinstance(points, gpd.GeoDataFrame):
            # Check for duplicate indices and reset if found
            if points.index.duplicated().any():
                warnings.warn(
                    "Duplicate indices detected in GeoDataFrame. Resetting index to ensure unique indices.",
                    UserWarning,
                    stacklevel=2,
                )
                points = points.reset_index(drop=True)
            # Convert geometry to lat/lon if needed
            if points.geometry.name == "geometry":
                points = points.copy()
                points["latitude"] = points.geometry.y
                points["longitude"] = points.geometry.x
            if poi_id_column not in points.columns:
                points["poi_id"] = [f"poi_{i}" for i in range(len(points))]
            else:
                points = points.rename(
                    columns={poi_id_column: "poi_id"},
                )
                if points["poi_id"].duplicated().any():
                    raise ValueError(
                        f"Column '{poi_id_column}' provided as 'poi_id_column' contains duplicate values."
                    )

            if points.crs != "EPSG:4326":
                points = points.to_crs("EPSG:4326")
            return points

        elif isinstance(points, pd.DataFrame):
            # Detect and standardize coordinate columns
            try:
                # Check for duplicate indices and reset if found
                if points.index.duplicated().any():
                    warnings.warn(
                        "Duplicate indices detected in DataFrame. Resetting index to ensure unique indices.",
                        UserWarning,
                        stacklevel=2,
                    )
                    points = points.reset_index(drop=True)
                lat_col, lon_col = detect_coordinate_columns(points)
                points = points.copy()
                points["latitude"] = points[lat_col]
                points["longitude"] = points[lon_col]
                if poi_id_column not in points.columns:
                    points[poi_id_column] = [f"poi_{i}" for i in range(len(points))]
                else:
                    points = points.rename(
                        columns={poi_id_column: "poi_id"},
                    )
                    if points["poi_id"].duplicated().any():
                        raise ValueError(
                            f"Column '{poi_id_column}' provided as 'poi_id_column' contains duplicate values."
                        )
                return convert_to_geodataframe(
                    points, lat_col="latitude", lon_col="longitude"
                )
            except ValueError as e:
                raise ValueError(
                    f"Could not detect coordinate columns in DataFrame: {str(e)}"
                )

        elif isinstance(points, list):
            if len(points) == 0:
                return gpd.GeoDataFrame(
                    columns=["latitude", "longitude", "poi_id", "geometry"],
                    geometry="geometry",
                    crs="EPSG:4326",
                )

            if isinstance(points[0], tuple) and len(points[0]) == 2:
                # List of (lat, lon) tuples
                df = pd.DataFrame(points, columns=["latitude", "longitude"])
                df["poi_id"] = [f"poi_{i}" for i in range(len(points))]
                return convert_to_geodataframe(df)

            elif isinstance(points[0], dict):
                # List of dictionaries
                df = pd.DataFrame(points)
                # Check for duplicate indices and reset if found
                if df.index.duplicated().any():
                    warnings.warn(
                        "Duplicate indices detected in DataFrame created from dictionary list. Resetting index to ensure unique indices.",
                        UserWarning,
                        stacklevel=2,
                    )
                    df = df.reset_index(drop=True)
                try:
                    lat_col, lon_col = detect_coordinate_columns(df)
                    df["latitude"] = df[lat_col]
                    df["longitude"] = df[lon_col]
                    if poi_id_column not in df.columns:
                        df["poi_id"] = [f"poi_{i}" for i in range(len(points))]
                    else:
                        df = df.rename(
                            columns={poi_id_column: "poi_id"},
                        )
                        if df["poi_id"].duplicated().any():
                            raise ValueError(
                                f"Column '{poi_id_column}' provided as 'poi_id_column' contains duplicate values."
                            )
                    return convert_to_geodataframe(
                        df, lat_col="latitude", lon_col="longitude"
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Could not detect coordinate columns in dictionary list: {str(e)}"
                    )

        raise ValueError("Unsupported points input type for PoiViewGenerator.")

    @property
    def points_gdf(self) -> gpd.GeoDataFrame:
        """Gets the internal GeoDataFrame of points of interest."""
        return self._points_gdf

    @property
    def view(self) -> pd.DataFrame:
        """The DataFrame representing the current point of interest view."""
        return self._view

    def _update_view(self, new_data: pd.DataFrame) -> None:
        """
        Internal helper to update the main view DataFrame with new columns.
        This method is designed to be called by map_* methods.

        Args:
            new_data (pd.DataFrame): A DataFrame containing 'poi_id' and new columns
                                     to be merged into the main view.
        """
        if "poi_id" not in new_data.columns:
            available_cols = list(new_data.columns)
            raise ValueError(
                f"new_data DataFrame must contain 'poi_id' column. "
                f"Available columns: {available_cols}"
            )

        # Check for poi_id mismatches
        original_poi_ids = set(self._view["poi_id"])
        new_poi_ids = set(new_data["poi_id"])
        missing_pois = original_poi_ids - new_poi_ids

        if missing_pois:
            self.logger.warning(
                f"{len(missing_pois)} POIs will have NaN values for new columns"
            )

        # Ensure poi_id is the index for efficient merging
        # Create a copy to avoid SettingWithCopyWarning if new_data is a slice
        new_data_indexed = new_data.set_index("poi_id").copy()

        # Merge on 'poi_id' (which is now the index of self._view and new_data_indexed)
        # Using left join to keep all POIs from the original view
        self._view = (
            self._view.set_index("poi_id")
            .join(new_data_indexed, how="left")
            .reset_index()
        )

        self.logger.debug(
            f"View updated with columns: {list(new_data_indexed.columns)}"
        )

    def map_nearest_points(
        self,
        points_df: Union[pd.DataFrame, gpd.GeoDataFrame],
        id_column: Optional[str] = None,
        lat_column: Optional[str] = None,
        lon_column: Optional[str] = None,
        output_prefix: str = "nearest",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Maps nearest points from a given DataFrame to the POIs.

        Enriches the `points_gdf` with the ID and distance to the nearest point
        from the input DataFrame for each POI.

        Args:
            points_df (Union[pd.DataFrame, gpd.GeoDataFrame]):
                DataFrame containing points to find nearest neighbors from.
                Must have latitude and longitude columns or point geometries.
            id_column (str, optional):
                Name of the column containing unique identifiers for each point.
                If None, the index of points_df will be used instead.
            lat_column (str, optional):
                Name of the latitude column in points_df. If None, will attempt to detect it
                or extract from geometry if points_df is a GeoDataFrame.
            lon_column (str, optional):
                Name of the longitude column in points_df. If None, will attempt to detect it
                or extract from geometry if points_df is a GeoDataFrame.
            output_prefix (str, optional):
                Prefix for the output column names. Defaults to "nearest".
            **kwargs:
                Additional keyword arguments passed to the data reader (if applicable).

        Returns:
            pd.DataFrame: The updated GeoDataFrame with new columns:
                          '{output_prefix}_id' and '{output_prefix}_distance'.
                          Returns a copy of the current `points_gdf` if no points are found.

        Raises:
            ValueError: If required columns are missing from points_df or if coordinate
                       columns cannot be detected or extracted from geometry.
        """
        self.logger.info(
            f"Mapping nearest points from {points_df.__class__.__name__} to POIs"
        )

        # Validate input DataFrame
        if points_df.empty:
            self.logger.info("No points found in the input DataFrame")
            return self.view

        # Handle GeoDataFrame
        if isinstance(points_df, gpd.GeoDataFrame):
            points_df = points_df.copy()
            if points_df.geometry.name == "geometry":
                points_df["latitude"] = points_df.geometry.y
                points_df["longitude"] = points_df.geometry.x
                lat_column = "latitude"
                lon_column = "longitude"
                self.logger.info("Extracted coordinates from geometry")

        # Detect coordinate columns if not provided
        if lat_column is None or lon_column is None:
            try:
                detected_lat, detected_lon = detect_coordinate_columns(points_df)
                lat_column = lat_column or detected_lat
                lon_column = lon_column or detected_lon
                self.logger.info(
                    f"Detected coordinate columns: {lat_column}, {lon_column}"
                )
            except ValueError as e:
                raise ValueError(f"Could not detect coordinate columns: {str(e)}")

        # Validate required columns
        required_columns = [lat_column, lon_column]
        if id_column is not None:
            required_columns.append(id_column)
        missing_columns = [
            col for col in required_columns if col not in points_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in points_df: {missing_columns}"
            )

        from gigaspatial.processing.geo import calculate_distance

        self.logger.info("Calculating nearest points for each POI")
        tree = cKDTree(points_df[[lat_column, lon_column]])
        points_df_poi = self.points_gdf.copy()
        _, idx = tree.query(points_df_poi[["latitude", "longitude"]], k=1)
        df_nearest = points_df.iloc[idx]
        dist = calculate_distance(
            lat1=points_df_poi.latitude,
            lon1=points_df_poi.longitude,
            lat2=df_nearest[lat_column],
            lon2=df_nearest[lon_column],
        )
        # Create a temporary DataFrame to hold the results for merging
        # Use id_column if provided, otherwise use index
        if id_column is not None:
            nearest_ids = points_df.iloc[idx][id_column].values
        else:
            nearest_ids = points_df.index[idx].values

        temp_result_df = pd.DataFrame(
            {
                "poi_id": points_df_poi["poi_id"],
                f"{output_prefix}_id": nearest_ids,
                f"{output_prefix}_distance": dist,
            }
        )

        self.logger.info(
            f"Nearest points mapping complete with prefix '{output_prefix}'"
        )
        return temp_result_df  # Return the DataFrame

    def map_google_buildings(
        self,
        handler: Optional[GoogleOpenBuildingsHandler] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Maps Google Open Buildings data to the POIs by finding the nearest building.

        Enriches the `points_gdf` with the ID and distance to the nearest
        Google Open Building for each POI.

        Args:
            data_config (Optional[GoogleOpenBuildingsConfig]):
                Configuration for accessing Google Open Buildings data. If None, a
                default `GoogleOpenBuildingsConfig` will be used.
            **kwargs:
                Additional keyword arguments passed to the data reader (if applicable).

        Returns:
            pd.DataFrame: The updated GeoDataFrame with new columns:
                          'nearest_google_building_id' and 'nearest_google_building_distance'.
                          Returns a copy of the current `points_gdf` if no buildings are found.
        """
        self.logger.info("Mapping Google Open Buildings data to POIs")
        handler = handler or GoogleOpenBuildingsHandler(data_store=self.data_store)

        self.logger.info("Loading Google Buildings point data")
        buildings_df = handler.load_points(
            self.points_gdf, ensure_available=self.config.ensure_available, **kwargs
        )
        if buildings_df is None or len(buildings_df) == 0:
            self.logger.info("No Google buildings data found for the provided POIs")
            return self.view

        mapped_data = self.map_nearest_points(
            points_df=buildings_df,
            id_column="full_plus_code",
            output_prefix="nearest_google_building",
            **kwargs,
        )
        self._update_view(mapped_data)
        return self.view

    def map_ms_buildings(
        self,
        handler: Optional[MSBuildingsHandler] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Maps Microsoft Global Buildings data to the POIs by finding the nearest building.

        Enriches the `points_gdf` with the ID and distance to the nearest
        Microsoft Global Building for each POI. If buildings don't have an ID column,
        creates a unique ID using the building's coordinates.

        Args:
            data_config (Optional[MSBuildingsConfig]):
                Configuration for accessing Microsoft Global Buildings data. If None, a
                default `MSBuildingsConfig` will be used.
            **kwargs:
                Additional keyword arguments passed to the data reader (if applicable).

        Returns:
            pd.DataFrame: The updated GeoDataFrame with new columns:
                          'nearest_ms_building_id' and 'nearest_ms_building_distance'.
                          Returns a copy of the current `points_gdf` if no buildings are found.
        """
        self.logger.info("Mapping Microsoft Global Buildings data to POIs")
        handler = handler or MSBuildingsHandler(data_store=self.data_store)
        self.logger.info("Loading Microsoft Buildings polygon data")
        buildings_gdf = handler.load_data(
            self.points_gdf, ensure_available=self.config.ensure_available
        )
        if buildings_gdf is None or len(buildings_gdf) == 0:
            self.logger.info("No Microsoft buildings data found for the provided POIs")
            return self.points_gdf.copy()

        building_centroids = get_centroids(buildings_gdf)

        if "building_id" not in buildings_gdf:
            self.logger.info("Creating building IDs from coordinates")
            building_centroids["building_id"] = building_centroids.apply(
                lambda row: f"{row.geometry.y:.6f}_{row.geometry.x:.6f}",
                axis=1,
            )

        mapped_data = self.map_nearest_points(
            points_df=building_centroids,
            id_column="building_id",
            output_prefix="nearest_ms_building",
            **kwargs,
        )
        self._update_view(mapped_data)
        return self.view

    def map_zonal_stats(
        self,
        data: Union[List[TifProcessor], gpd.GeoDataFrame],
        stat: str = "mean",
        map_radius_meters: Optional[float] = None,
        output_column: str = "zonal_stat",
        value_column: Optional[str] = None,
        predicate: Literal["intersects", "within", "fractional"] = "intersects",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Maps zonal statistics from raster or polygon data to POIs.

        Can operate in three modes:
        1. Raster point sampling: Directly samples raster values at POI locations
        2. Raster zonal statistics: Creates buffers around POIs and calculates statistics within them
        3. Polygon aggregation: Aggregates polygon data to POI buffers with optional area weighting

        Args:
            data (Union[TifProcessor, List[TifProcessor], gpd.GeoDataFrame]):
                Either a TifProcessor object, a list of TifProcessor objects (which will be merged
                into a single TifProcessor for processing), or a GeoDataFrame containing polygon
                data to aggregate.
            stat (str, optional):
                For raster data: Statistic to calculate ("sum", "mean", "median", "min", "max").
                For polygon data: Aggregation method to use.
                Defaults to "mean".
            map_radius_meters (float, optional):
                If provided, creates circular buffers of this radius around each POI
                and calculates statistics within the buffers. If None, samples directly
                at POI locations (only for raster data).
            output_column (str, optional):
                Name of the output column to store the results. Defaults to "zonal_stat".
            value_column (str, optional):
                For polygon data: Name of the column to aggregate. Required for polygon data.
                Not used for raster data.
            predicate (Literal["intersects", "within", "fractional"], optional):
                The spatial relationship to use for aggregation. Defaults to "intersects".
            **kwargs:
                Additional keyword arguments passed to the sampling/aggregation functions.

        Returns:
            pd.DataFrame: The updated GeoDataFrame with a new column containing the
                          calculated statistics. Returns a copy of the current `points_gdf`
                          if no valid data is found.

        Raises:
            ValueError: If no valid data is provided, if parameters are incompatible,
                      or if required parameters (value_column) are missing for polygon data.
        """

        raster_processor: Optional[TifProcessor] = None

        if isinstance(data, TifProcessor):
            raster_processor = data
        elif isinstance(data, list) and all(isinstance(x, TifProcessor) for x in data):
            if not data:
                self.logger.info("No valid raster data provided")
                return self.view

            if len(data) > 1:
                all_source_paths = [tp.dataset_path for tp in data]

                self.logger.info(
                    f"Merging {len(all_source_paths)} rasters into a single TifProcessor for zonal statistics."
                )
                raster_processor = TifProcessor(
                    dataset_path=all_source_paths,
                    data_store=self.data_store,
                    **kwargs,
                )
            else:
                raster_processor = data[0]

        if raster_processor:
            results_df = pd.DataFrame({"poi_id": self.points_gdf["poi_id"]})
            raster_crs = raster_processor.crs

            if map_radius_meters is not None:
                self.logger.info(
                    f"Calculating {stat} within {map_radius_meters}m buffers around POIs"
                )
                # Create buffers around POIs
                buffers_gdf = buffer_geodataframe(
                    self.points_gdf,
                    buffer_distance_meters=map_radius_meters,
                    cap_style="round",
                )

                # Calculate zonal statistics
                sampled_values = raster_processor.sample_by_polygons(
                    polygon_list=buffers_gdf.to_crs(raster_crs).geometry,
                    stat=stat,
                )
            else:
                self.logger.info(f"Sampling {stat} at POI locations")
                # Sample directly at POI locations
                coord_list = (
                    self.points_gdf.to_crs(raster_crs).get_coordinates().to_numpy()
                )
                sampled_values = raster_processor.sample_by_coordinates(
                    coordinate_list=coord_list, **kwargs
                )

            results_df[output_column] = sampled_values

        elif isinstance(data, gpd.GeoDataFrame):
            # Handle polygon data
            if data.empty:
                self.logger.info("No valid GeoDataFrame data provided")
                return pd.DataFrame(
                    columns=["poi_id", output_column]
                )  # Return empty DataFrame

            if map_radius_meters is None:
                raise ValueError(
                    "map_radius_meters must be provided for for GeoDataFrame data"
                )

            # Create buffers around POIs
            buffer_gdf = buffer_geodataframe(
                self.points_gdf,
                buffer_distance_meters=map_radius_meters,
                cap_style="round",
            )

            if any(data.geom_type.isin(["MultiPoint", "Point"])):

                self.logger.info(
                    f"Aggregating point data within {map_radius_meters}m buffers around POIs using predicate '{predicate}'"
                )

                # If no value_column, default to 'count'
                if value_column is None:
                    actual_stat = "count"
                    self.logger.warning(
                        "No value_column provided for point data. Defaulting to 'count' aggregation."
                    )
                else:
                    actual_stat = stat
                    if value_column not in data.columns:
                        raise ValueError(
                            f"Value column '{value_column}' not found in input GeoDataFrame."
                        )

                aggregation_result_gdf = aggregate_points_to_zones(
                    points=data,
                    zones=buffer_gdf,
                    value_columns=value_column,
                    aggregation=actual_stat,
                    point_zone_predicate=predicate,  # can't be `fractional``
                    zone_id_column="poi_id",
                    output_suffix="",
                    drop_geometry=True,
                    **kwargs,
                )

                output_col_from_agg = (
                    f"{value_column}_{actual_stat}" if value_column else "point_count"
                )
                results_df = aggregation_result_gdf[["poi_id", output_col_from_agg]]

                if output_column != "zonal_stat":
                    results_df = results_df.rename(
                        columns={output_col_from_agg: output_column}
                    )

            else:
                if value_column is None:
                    raise ValueError(
                        "value_column must be provided for polygon data aggregation."
                    )
                if value_column not in data.columns:
                    raise ValueError(
                        f"Value column '{value_column}' not found in input GeoDataFrame."
                    )
                self.logger.info(
                    f"Aggregating polygon data within {map_radius_meters}m buffers around POIs using predicate '{predicate}'"
                )

                # Aggregate polygons to buffers
                aggregation_result_gdf = aggregate_polygons_to_zones(
                    polygons=data,
                    zones=buffer_gdf,
                    value_columns=value_column,
                    aggregation=stat,
                    predicate=predicate,
                    zone_id_column="poi_id",
                    output_suffix="",
                    drop_geometry=True,
                    **kwargs,
                )

                output_col_from_agg = value_column

            results_df = aggregation_result_gdf[["poi_id", output_col_from_agg]]

            if output_column != "zonal_stat":
                results_df = results_df.rename(
                    columns={output_col_from_agg: output_column}
                )

        else:
            raise ValueError(
                "data must be either a list of TifProcessor objects or a GeoDataFrame"
            )

        # self._update_view(results_df) # Removed direct view update
        self.logger.info(
            f"Zonal statistics mapping complete for column(s) derived from '{output_column}' or '{value_column}'"
        )
        return results_df  # Return the DataFrame

    def map_built_s(
        self,
        map_radius_meters: float = 150,
        stat: str = "sum",
        dataset_year=2020,
        dataset_resolution=100,
        output_column="built_surface_m2",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Maps GHSL Built Surface (GHS_BUILT_S) data to the POIs.

        Calculates the sum of built surface area within a specified buffer
        radius around each POI. Enriches `points_gdf` with the 'built_surface_m2' column.

        Args:
            data_config (Optional[GHSLDataConfig]):
                Configuration for accessing GHSL Built Surface data. If None, a
                default `GHSLDataConfig` for 'GHS_BUILT_S' will be used.
            map_radius_meters (float):
                The buffer distance in meters around each POI to calculate
                zonal statistics for built surface. Defaults to 150 meters.
            **kwargs:
                Additional keyword arguments passed to the data reader (if applicable).

        Returns:
            pd.DataFrame: The updated GeoDataFrame with a new column:
                          'built_surface_m2'. Returns a copy of the current
                          `points_gdf` if no GHSL Built Surface data is found.
        """
        self.logger.info("Mapping GHSL Built Surface data to POIs")
        handler = GHSLDataHandler(
            product="GHS_BUILT_S",
            year=dataset_year,
            resolution=dataset_resolution,
            data_store=self.data_store,
            **kwargs,
        )
        self.logger.info("Loading GHSL Built Surface raster tiles")
        tif_processors = handler.load_data(
            self.points_gdf.copy(),
            ensure_available=self.config.ensure_available,
            merge_rasters=True,
            **kwargs,
        )

        mapped_data = self.map_zonal_stats(
            data=tif_processors,
            stat=stat,
            map_radius_meters=map_radius_meters,
            output_column=output_column,
            **kwargs,
        )
        self._update_view(mapped_data)
        return self.view

    def map_smod(
        self,
        stat="median",
        dataset_year=2020,
        dataset_resolution=1000,
        output_column="smod_class",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Maps GHSL Settlement Model (SMOD) data to the POIs.

        Samples the SMOD class value at each POI's location. Enriches `points_gdf`
        with the 'smod_class' column.

        Args:
            data_config (Optional[GHSLDataConfig]):
                Configuration for accessing GHSL SMOD data. If None, a
                default `GHSLDataConfig` for 'GHS_SMOD' will be used.
            **kwargs:
                Additional keyword arguments passed to the data reader (if applicable).

        Returns:
            pd.DataFrame: The updated GeoDataFrame with a new column:
                          'smod_class'. Returns a copy of the current
                          `points_gdf` if no GHSL SMOD data is found.
        """
        self.logger.info("Mapping GHSL Settlement Model (SMOD) data to POIs")
        handler = GHSLDataHandler(
            product="GHS_SMOD",
            year=dataset_year,
            resolution=dataset_resolution,
            data_store=self.data_store,
            coord_system=54009,
            **kwargs,
        )

        self.logger.info("Loading GHSL SMOD raster tiles")
        tif_processors = handler.load_data(
            self.points_gdf.copy(),
            ensure_available=self.config.ensure_available,
            merge_rasters=True,
            **kwargs,
        )

        mapped_data = self.map_zonal_stats(
            data=tif_processors,
            stat=stat,
            output_column=output_column,
            **kwargs,
        )
        self._update_view(mapped_data)
        return self.view

    def map_wp_pop(
        self,
        country: Union[str, List[str]],
        map_radius_meters: float,
        resolution=1000,
        predicate: Literal[
            "centroid_within", "intersects", "fractional", "within"
        ] = "intersects",
        output_column: str = "population",
        **kwargs,
    ):
        # Ensure country is always a list for consistent handling
        countries_list = [country] if isinstance(country, str) else country

        handler = WPPopulationHandler(
            resolution=resolution,
            data_store=self.data_store,
            **kwargs,
        )

        # Restrict to single country for age_structures project
        if handler.config.project == "age_structures" and len(countries_list) > 1:
            raise ValueError(
                "For 'age_structures' project, only a single country can be processed at a time."
            )

        self.logger.info(
            f"Mapping WorldPop Population data (year: {handler.config.year}, resolution: {handler.config.resolution}m)"
        )

        if predicate == "fractional" and resolution == 100:
            self.logger.warning(
                "Fractional aggregations only supported for datasets with 1000m resolution. Using `intersects` as predicate"
            )
            predicate = "intersects"

        data_to_process: Union[List[TifProcessor], gpd.GeoDataFrame, pd.DataFrame]

        if predicate == "centroid_within":
            if handler.config.project == "age_structures":
                # Load individual tif processors for the single country
                all_tif_processors = handler.load_data(
                    countries_list[0],
                    ensure_available=self.config.ensure_available,
                    **kwargs,
                )

                # Sum results from each tif_processor separately
                summed_results_by_poi = {
                    poi_id: 0.0 for poi_id in self.points_gdf["poi_id"].unique()
                }

                self.logger.info(
                    f"Sampling individual age_structures rasters using 'sum' statistic and summing per POI."
                )
                for tif_processor in all_tif_processors:
                    single_raster_df = self.map_zonal_stats(
                        data=tif_processor,
                        stat="sum",
                        map_radius_meters=map_radius_meters,
                        value_column="pixel_value",
                        predicate=predicate,
                        output_column=output_column,  # This output_column will be in the temporary DF
                        **kwargs,
                    )
                    # Add values from this single raster to the cumulative sum
                    for _, row in single_raster_df.iterrows():
                        summed_results_by_poi[row["poi_id"]] += row[output_column]

                # Convert the summed dictionary back to a DataFrame
                data_to_process = pd.DataFrame(
                    list(summed_results_by_poi.items()),
                    columns=["poi_id", output_column],
                )

            else:
                # Existing behavior for non-age_structures projects or if merging is fine
                # 'data_to_process' will be a list of TifProcessor objects, which map_zonal_stats will merge
                data_to_process = []
                for c in countries_list:
                    data_to_process.extend(
                        handler.load_data(
                            c, ensure_available=self.config.ensure_available, **kwargs
                        )
                    )
        else:
            # 'data_to_process' will be a GeoDataFrame
            data_to_process = pd.concat(
                [
                    handler.load_into_geodataframe(
                        c, ensure_available=self.config.ensure_available, **kwargs
                    )
                    for c in countries_list  # Original iteration over countries_list
                ],
                ignore_index=True,
            )

        self.logger.info(
            f"Mapping WorldPop Population data into {map_radius_meters}m zones around POIs using 'sum' statistic"
        )

        final_mapped_df: pd.DataFrame

        # If 'data_to_process' is already the summed DataFrame (from age_structures/centroid_within branch),
        # use it directly.
        if (
            isinstance(data_to_process, pd.DataFrame)
            and output_column in data_to_process.columns
            and "poi_id" in data_to_process.columns
        ):
            final_mapped_df = data_to_process
        else:
            # For other cases, proceed with the original call to map_zonal_stats
            final_mapped_df = self.map_zonal_stats(
                data=data_to_process,
                stat="sum",
                map_radius_meters=map_radius_meters,
                value_column="pixel_value",
                predicate=predicate,
                output_column=output_column,
                **kwargs,
            )
        self._update_view(
            final_mapped_df
        )  # Update the view with the final mapped DataFrame
        return self.view

    def save_view(
        self,
        name: str,
        output_format: Optional[str] = None,
    ) -> Path:
        """
        Saves the current POI view (the enriched DataFrame) to a file.

        The output path and format are determined by the `config`
        or overridden by the `output_format` parameter.

        Args:
            name (str): The base name for the output file (without extension).
            output_format (Optional[str]):
                The desired output format (e.g., "csv", "geojson"). If None,
                the `output_format` from `config` will be used.

        Returns:
            Path: The full path to the saved output file.
        """
        format_to_use = output_format or self.config.output_format
        output_path = self.config.base_path / f"{name}.{format_to_use}"

        self.logger.info(f"Saving POI view to {output_path}")
        # Save the current view, which is a pandas DataFrame, not a GeoDataFrame
        # GeoJSON/Shapefile formats would require converting back to GeoDataFrame first.
        # For CSV, Parquet, Feather, this is fine.
        if format_to_use in ["geojson", "shp", "gpkg"]:
            self.logger.warning(
                f"Saving to {format_to_use} requires converting back to GeoDataFrame. Geometry column will be re-added."
            )
            # Re-add geometry for saving to geospatial formats
            view_to_save_gdf = self.view.merge(
                self.points_gdf[["poi_id", "geometry"]], on="poi_id", how="left"
            )
            write_dataset(
                data=view_to_save_gdf,
                path=str(output_path),
                data_store=self.data_store,
            )
        else:
            write_dataset(
                data=self.view,  # Use the internal _view DataFrame
                path=str(output_path),
                data_store=self.data_store,
            )

        return output_path

    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns the current POI view as a DataFrame.

        This method combines all accumulated variables in the view

        Returns:
            pd.DataFrame: The current view.
        """
        return self.view

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Returns the current POI view merged with the original point geometries as a GeoDataFrame.

        This method combines all accumulated variables in the view with the corresponding
        point geometries, providing a spatially-enabled DataFrame for further analysis or export.

        Returns:
            gpd.GeoDataFrame: The current view merged with point geometries.
        """
        return gpd.GeoDataFrame(
            self.view.merge(
                self.points_gdf[["poi_id", "geometry"]], on="poi_id", how="left"
            ),
            crs="EPSG:4326",
        )

    def chain_operations(self, operations: List[dict]) -> "PoiViewGenerator":
        """
        Chain multiple mapping operations for fluent interface.

        Args:
            operations: List of dicts with 'method' and 'kwargs' keys

        Example:
            generator.chain_operations([
                {'method': 'map_google_buildings', 'kwargs': {}},
                {'method': 'map_built_s', 'kwargs': {'map_radius_meters': 200}},
            ])
        """
        for op in operations:
            method_name = op["method"]
            kwargs = op.get("kwargs", {})
            if hasattr(self, method_name):
                getattr(self, method_name)(**kwargs)
            else:
                raise AttributeError(f"Method {method_name} not found")
        return self

    def validate_data_coverage(self, data_bounds: gpd.GeoDataFrame) -> dict:
        """
        Validate how many POIs fall within the data coverage area.

        Returns:
            dict: Coverage statistics
        """
        poi_within = self.points_gdf.within(data_bounds.union_all())
        coverage_stats = {
            "total_pois": len(self.points_gdf),
            "covered_pois": poi_within.sum(),
            "coverage_percentage": (poi_within.sum() / len(self.points_gdf)) * 100,
            "uncovered_pois": (~poi_within).sum(),
        }
        return coverage_stats

    def find_nearest_buildings(
        self,
        country: str,
        search_radius: float = 1000,
        source_filter: Literal["google", "microsoft"] = None,
        find_nearest_globally: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Find the nearest building to each POI within a specified search radius.

        This method processes building data by:
        1. Filtering to only building tiles that intersect POI buffers (partitioned datasets)
        2. Finding the nearest building candidate per POI (nearest-neighbor search)
        3. Computing final POI-to-building distances in **meters** using haversine distance

        Parameters
        ----------
        country : str
            Country code for which to load building data.

        search_radius : float, default=1000
            Search radius in meters. Only buildings within this distance from a POI
            will be considered. For better performance, use the smallest radius
            that meets your requirements.

        source_filter : {'google', 'microsoft'}, optional
            Filter buildings by data source. If None, uses buildings from all sources.

        find_nearest_globally : bool, default=False
            If True, finds the true nearest building regardless of distance.
            This overrides search_radius and may be significantly slower.
            When False, uses the efficient radius-limited search.

        **kwargs : dict
            Additional arguments passed to the building data handler.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - poi_id: Original POI identifier
            - nearest_building_distance_m: Distance to nearest building in meters.
            NaN if no building found within the search constraints.
            - building_within_{search_radius}m: Boolean indicating if a building
            was found within the specified search_radius.

        Notes
        -----
        - Distances are computed in meters using haversine (great-circle) distance via
        `calculate_distance`.
        - Nearest-neighbor candidate selection is performed using coordinates extracted
        from geometries.
        - For countries with a single building file (no partitioning),
        the search is performed globally regardless of search_radius.
        - For partitioned countries, search_radius optimizes performance by
        filtering which tiles to process.

        Examples
        --------
        >>> # Find buildings within 350m of POIs
        >>> result = poi_gdf.find_nearest_buildings("USA", search_radius=350)

        >>> # Find nearest building globally (may be slow for partitioned countries)
        >>> result = poi_gdf.find_nearest_buildings("USA", find_nearest_globally=True)
        """

        # ---------------------------------------------------------
        # 1. PARAMETER VALIDATION AND LOGGING
        # ---------------------------------------------------------

        if find_nearest_globally:
            self.logger.info(
                "Global nearest building search enabled. "
                "This may be slower than radius-limited search."
            )
            return self._find_nearest_building_globally(
                country=country, source_filter=source_filter, **kwargs
            )

        # Warn for large search radii
        if search_radius > 5000:
            self.logger.warning(
                f"Large search radius ({search_radius}m) may impact performance. "
                "Consider using progressive expansion for better efficiency."
            )

        self.logger.info(
            f"Mapping Google-Microsoft Buildings data to POIs within {search_radius}m"
        )

        # ---------------------------------------------------------
        # 2. LOAD BUILDING FILES
        # ---------------------------------------------------------

        from gigaspatial.handlers.google_ms_combined_buildings import (
            GoogleMSBuildingsHandler,
        )

        handler = GoogleMSBuildingsHandler(partition_strategy="s2_grid")

        if self.config.ensure_available:
            if not handler.ensure_data_available(country, **kwargs):
                raise RuntimeError("Could not ensure data availability for loading")

        building_files = handler.reader.resolve_source_paths(country, **kwargs)

        result = GoogleMSBuildingsEngine.nearest_buildings_to_pois(
            handler=handler,
            building_files=building_files,
            pois_gdf=self.points_gdf,
            source_filter=source_filter,
            search_radius_m=search_radius,
            logger=self.logger,
        )

        distances_clean = result.distances_m.replace(np.inf, np.nan)
        result_df = pd.DataFrame(
            {
                "poi_id": result.distances_m.index,
                "nearest_building_distance_m": distances_clean,
                f"building_within_{search_radius}m": result.distances_m
                <= search_radius,
            }
        )

        # Update the view and return
        self._update_view(result_df)
        return self.view

    def _find_nearest_building_globally(
        self,
        country: str,
        source_filter: Literal["google", "microsoft"] = None,
        max_global_search: float = 10000,
        **kwargs,
    ) -> pd.DataFrame:
        """Find the true nearest building for each POI using progressive expansion."""

        self.logger.info("Starting global nearest building search...")

        from gigaspatial.handlers.google_ms_combined_buildings import (
            GoogleMSBuildingsHandler,
        )

        handler = GoogleMSBuildingsHandler(partition_strategy="s2_grid")
        building_files = handler.reader.resolve_source_paths(country, **kwargs)

        # Single file: use optimized path
        if len(building_files) == 1:
            self.logger.info("Single file country: using single-file optimization.")

            result = GoogleMSBuildingsEngine.nearest_buildings_to_pois(
                handler=handler,
                building_files=building_files,
                pois_gdf=self.points_gdf,
                source_filter=source_filter,
                search_radius_m=max_global_search,
                logger=self.logger,
            )

            result_df = pd.DataFrame(
                {
                    "poi_id": result.distances_m.index,
                    "nearest_building_distance_m": result.distances_m.replace(
                        np.inf, np.nan
                    ),
                    f"building_within_{max_global_search}m": result.distances_m
                    <= max_global_search,
                }
            )

            found_count = result_df["nearest_building_distance_m"].notna().sum()
            self.logger.info(
                f"Single file search complete. Found {found_count}/{len(result_df)} POIs."
            )

            self._update_view(result_df)
            return self.view

        # Partitioned case: progressive radius expansion
        self.logger.info(
            f"Partitioned country: progressive expansion up to {max_global_search}m"
        )

        radii = [350, 1000, 2500, 5000, max_global_search]

        # FIX: Track POIs that still need searching (use set for efficiency)
        remaining_poi_ids = set(self.points_gdf.poi_id)
        global_results = pd.Series(np.inf, index=self.points_gdf.poi_id, dtype=float)

        # FIX: Track processed tiles to avoid re-scanning
        processed_tiles = set()

        for radius in radii:
            if not remaining_poi_ids:
                self.logger.info("All POIs found buildings. Stopping early.")
                break

            self.logger.info(
                f"Searching radius {radius}m for {len(remaining_poi_ids)} POIs"
            )

            # Create jobs for this radius
            jobs = GoogleMSBuildingsEngine.create_partitioned_jobs_for_pois(
                self.points_gdf,
                building_files,
                search_radius_m=radius,
            )

            # FIX: Only process tiles we haven't seen yet
            new_jobs = [(f, pois) for f, pois in jobs if f not in processed_tiles]

            if not new_jobs:
                self.logger.debug(f"No new tiles at radius {radius}m")
                continue

            # Process new tiles (don't update the view inside the engine)
            temp_result = GoogleMSBuildingsEngine.nearest_buildings_to_pois(
                handler=handler,
                building_files=[f for f, _ in new_jobs],
                pois_gdf=self.points_gdf,
                source_filter=source_filter,
                search_radius_m=radius,
                logger=self.logger,
            )
            temp_min_dists = temp_result.distances_m

            # Update global results and track found POIs
            found_in_this_iteration = set()

            for poi_id in remaining_poi_ids:
                dist = temp_min_dists[poi_id]
                if dist < global_results[poi_id]:
                    global_results[poi_id] = dist
                    if dist <= radius:  # Found within this radius
                        found_in_this_iteration.add(poi_id)

            # FIX: Remove found POIs from remaining set
            remaining_poi_ids -= found_in_this_iteration

            # Mark these tiles as processed
            processed_tiles.update(f for f, _ in new_jobs)

            self.logger.info(
                f"Found {len(found_in_this_iteration)} POIs at radius {radius}m. "
                f"{len(remaining_poi_ids)} remaining."
            )

        # Create final results
        result_df = pd.DataFrame(
            {
                "poi_id": global_results.index,
                "nearest_building_distance_m": global_results.replace(np.inf, np.nan),
                f"building_within_{max_global_search}m": global_results
                <= max_global_search,
            }
        )

        found_count = result_df["nearest_building_distance_m"].notna().sum()
        self.logger.info(
            f"Global search complete. Found buildings for {found_count}/{len(result_df)} POIs."
        )

        self._update_view(result_df)
        return self.view
