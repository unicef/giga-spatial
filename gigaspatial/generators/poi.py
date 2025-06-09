from pathlib import Path
from typing import List, Optional, Union, Tuple
from pydantic.dataclasses import dataclass, Field

import geopandas as gpd
import pandas as pd
import logging

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.writers import write_dataset
from gigaspatial.config import config as global_config
from gigaspatial.handlers.google_open_buildings import GoogleOpenBuildingsHandler
from gigaspatial.handlers.microsoft_global_buildings import MSBuildingsHandler
from gigaspatial.handlers.ghsl import GHSLDataHandler
from gigaspatial.processing.geo import (
    convert_to_geodataframe,
    buffer_geodataframe,
    detect_coordinate_columns,
    aggregate_polygons_to_zones,
)
from gigaspatial.processing.tif_processor import (
    sample_multiple_tifs_by_polygons,
    sample_multiple_tifs_by_coordinates,
    TifProcessor,
)
from scipy.spatial import cKDTree


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
        self.config = config or PoiViewGeneratorConfig()
        self.data_store = data_store or LocalDataStore()
        self.logger = logger or global_config.get_logger(self.__class__.__name__)
        self._points_gdf = self._init_points_gdf(points)

    @staticmethod
    def _init_points_gdf(
        points: Union[
            List[Tuple[float, float]], List[dict], pd.DataFrame, gpd.GeoDataFrame
        ],
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
            # Convert geometry to lat/lon if needed
            if points.geometry.name == "geometry":
                points = points.copy()
                points["latitude"] = points.geometry.y
                points["longitude"] = points.geometry.x
            if "poi_id" not in points.columns:
                points["poi_id"] = [f"poi_{i}" for i in range(len(points))]
            return points

        elif isinstance(points, pd.DataFrame):
            # Detect and standardize coordinate columns
            try:
                lat_col, lon_col = detect_coordinate_columns(points)
                points = points.copy()
                points["latitude"] = points[lat_col]
                points["longitude"] = points[lon_col]
                if "poi_id" not in points.columns:
                    points["poi_id"] = [f"poi_{i}" for i in range(len(points))]
                return convert_to_geodataframe(points)
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
                try:
                    lat_col, lon_col = detect_coordinate_columns(df)
                    df["latitude"] = df[lat_col]
                    df["longitude"] = df[lon_col]
                    if "poi_id" not in df.columns:
                        df["poi_id"] = [f"poi_{i}" for i in range(len(points))]
                    return convert_to_geodataframe(df)
                except ValueError as e:
                    raise ValueError(
                        f"Could not detect coordinate columns in dictionary list: {str(e)}"
                    )

        raise ValueError("Unsupported points input type for PoiViewGenerator.")

    @property
    def points_gdf(self) -> gpd.GeoDataFrame:
        """Gets the internal GeoDataFrame of points of interest."""
        return self._points_gdf

    def map_nearest_points(
        self,
        points_df: Union[pd.DataFrame, gpd.GeoDataFrame],
        id_column: str,
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
            id_column (str):
                Name of the column containing unique identifiers for each point.
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
            return self.points_gdf.copy()

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
        required_columns = [lat_column, lon_column, id_column]
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
        result = points_df_poi.copy()
        result[f"{output_prefix}_id"] = df_nearest[id_column].to_numpy()
        result[f"{output_prefix}_distance"] = dist
        self.logger.info(
            f"Nearest points mapping complete with prefix '{output_prefix}'"
        )
        self._points_gdf = result
        return result

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
            self.points_gdf, ensure_available=self.config.ensure_available
        )
        if buildings_df is None or len(buildings_df) == 0:
            self.logger.info("No Google buildings data found for the provided POIs")
            return self.points_gdf.copy()

        return self.map_nearest_points(
            points_df=buildings_df,
            id_column="full_plus_code",
            output_prefix="nearest_google_building",
            **kwargs,
        )

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

        if "building_id" not in buildings_gdf:
            self.logger.info("Creating building IDs from coordinates")
            buildings_gdf = buildings_gdf.copy()
            buildings_gdf["building_id"] = buildings_gdf.apply(
                lambda row: f"{row.geometry.y:.6f}_{row.geometry.x:.6f}",
                axis=1,
            )

        return self.map_nearest_points(
            points_df=buildings_gdf,
            id_column="building_id",
            output_prefix="nearest_ms_building",
            **kwargs,
        )

    def map_zonal_stats(
        self,
        data: Union[List[TifProcessor], gpd.GeoDataFrame],
        stat: str = "mean",
        map_radius_meters: Optional[float] = None,
        output_column: str = "zonal_stat",
        value_column: Optional[str] = None,
        area_weighted: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Maps zonal statistics from raster or polygon data to POIs.

        Can operate in three modes:
        1. Raster point sampling: Directly samples raster values at POI locations
        2. Raster zonal statistics: Creates buffers around POIs and calculates statistics within them
        3. Polygon aggregation: Aggregates polygon data to POI buffers with optional area weighting

        Args:
            data (Union[List[TifProcessor], gpd.GeoDataFrame]):
                Either a list of TifProcessor objects containing raster data to sample,
                or a GeoDataFrame containing polygon data to aggregate.
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
            area_weighted (bool, optional):
                For polygon data: Whether to weight values by fractional area of
                intersection. Defaults to False.
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
        if isinstance(data, list) and all(isinstance(x, TifProcessor) for x in data):
            # Handle raster data
            if not data:
                self.logger.info("No valid raster data found for the provided POIs")
                return self.points_gdf.copy()

            if map_radius_meters is not None:
                self.logger.info(
                    f"Calculating {stat} within {map_radius_meters}m buffers around POIs"
                )
                # Create buffers around POIs
                polygon_list = buffer_geodataframe(
                    self.points_gdf,
                    buffer_distance_meters=map_radius_meters,
                    cap_style="round",
                ).geometry

                # Calculate zonal statistics
                sampled_values = sample_multiple_tifs_by_polygons(
                    tif_processors=data, polygon_list=polygon_list, stat=stat, **kwargs
                )
            else:
                self.logger.info(f"Sampling {stat} at POI locations")
                # Sample directly at POI locations
                coord_list = self.points_gdf[["latitude", "longitude"]].to_numpy()
                sampled_values = sample_multiple_tifs_by_coordinates(
                    tif_processors=data, coordinate_list=coord_list, **kwargs
                )

        elif isinstance(data, gpd.GeoDataFrame):
            # Handle polygon data
            if data.empty:
                self.logger.info("No valid polygon data found for the provided POIs")
                return self.points_gdf.copy()

            if map_radius_meters is None:
                raise ValueError("map_radius_meters must be provided for polygon data")

            if value_column is None:
                raise ValueError("value_column must be provided for polygon data")

            self.logger.info(
                f"Aggregating {value_column} within {map_radius_meters}m buffers around POIs"
            )

            # Create buffers around POIs
            buffer_gdf = buffer_geodataframe(
                self.points_gdf,
                buffer_distance_meters=map_radius_meters,
                cap_style="round",
            )

            # Aggregate polygons to buffers
            result = aggregate_polygons_to_zones(
                polygons=data,
                zones=buffer_gdf,
                value_columns=value_column,
                aggregation=stat,
                area_weighted=area_weighted,
                zone_id_column="poi_id",
                **kwargs,
            )

            # Extract values for each POI
            sampled_values = result[value_column].values

        else:
            raise ValueError(
                "data must be either a list of TifProcessor objects or a GeoDataFrame"
            )

        result = self.points_gdf.copy()
        result[output_column] = sampled_values
        self.logger.info(f"Zonal statistics mapping complete: {output_column}")
        self._points_gdf = result
        return result

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
        gdf_points = self.points_gdf.to_crs(handler.config.crs)
        self.logger.info("Loading GHSL Built Surface raster tiles")
        tif_processors = handler.load_data(
            gdf_points, ensure_available=self.config.ensure_available
        )

        return self.map_zonal_stats(
            data=tif_processors,
            stat=stat,
            map_radius_meters=map_radius_meters,
            output_column=output_column,
            **kwargs,
        )

    def map_smod(
        self,
        stat="median",
        dataset_year=2020,
        dataset_resolution=100,
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

        gdf_points = self.points_gdf.to_crs(handler.config.crs)
        self.logger.info("Loading GHSL SMOD raster tiles")
        tif_processors = handler.load_data(
            gdf_points, ensure_available=self.config.ensure_available
        )

        return self.map_zonal_stats(
            data=tif_processors,
            stat=stat,  # Use median for categorical data
            output_column=output_column,
            **kwargs,
        )

    def save_view(
        self,
        name: str,
        output_format: Optional[str] = None,
    ) -> Path:
        """
        Saves the current POI view (the enriched GeoDataFrame) to a file.

        The output path and format are determined by the `generator_config`
        or overridden by the `output_format` parameter.

        Args:
            name (str): The base name for the output file (without extension).
            output_format (Optional[str]):
                The desired output format (e.g., "csv", "geojson"). If None,
                the `output_format` from `generator_config` will be used.

        Returns:
            Path: The full path to the saved output file.
        """
        format_to_use = output_format or self.generator_config.output_format
        output_path = self.generator_config.base_path / f"{name}.{format_to_use}"

        self.logger.info(f"Saving POI view to {output_path}")
        write_dataset(
            df=self.points_gdf,
            path=str(output_path),
            data_store=self.data_store,
            format=format_to_use,
        )

        return output_path
