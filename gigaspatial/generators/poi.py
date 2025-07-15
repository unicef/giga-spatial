from pathlib import Path
from typing import List, Optional, Union, Tuple, Literal
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
from gigaspatial.handlers.worldpop import WPPopulationHandler
from gigaspatial.processing.geo import (
    convert_to_geodataframe,
    buffer_geodataframe,
    detect_coordinate_columns,
    aggregate_polygons_to_zones,
    get_centroids,
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
                lat_col, lon_col = detect_coordinate_columns(points)
                points = points.copy()
                points["latitude"] = points[lat_col]
                points["longitude"] = points[lon_col]
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
        # Create a temporary DataFrame to hold the results for merging
        temp_result_df = pd.DataFrame(
            {
                "poi_id": points_df_poi["poi_id"],
                f"{output_prefix}_id": points_df.iloc[idx][id_column].values,
                f"{output_prefix}_distance": dist,
            }
        )
        self._update_view(temp_result_df)
        self.logger.info(
            f"Nearest points mapping complete with prefix '{output_prefix}'"
        )
        return self.view

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
            return self.view

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

        building_centroids = get_centroids(buildings_gdf)

        if "building_id" not in buildings_gdf:
            self.logger.info("Creating building IDs from coordinates")
            building_centroids["building_id"] = building_centroids.apply(
                lambda row: f"{row.geometry.y:.6f}_{row.geometry.x:.6f}",
                axis=1,
            )

        return self.map_nearest_points(
            points_df=building_centroids,
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

        if isinstance(data, list) and all(isinstance(x, TifProcessor) for x in data):
            results_df = pd.DataFrame({"poi_id": self.points_gdf["poi_id"]})

            # Handle raster data
            if not data:
                self.logger.info("No valid raster data found for the provided POIs")
                return self.view

            raster_crs = data[0].crs

            if not all(tp.crs == raster_crs for tp in data):
                raise ValueError(
                    "All TifProcessors must have the same CRS for zonal statistics."
                )

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
                sampled_values = sample_multiple_tifs_by_polygons(
                    tif_processors=data,
                    polygon_list=buffers_gdf.to_crs(raster_crs).geometry,
                    stat=stat,
                    **kwargs,
                )
            else:
                self.logger.info(f"Sampling {stat} at POI locations")
                # Sample directly at POI locations
                coord_list = (
                    self.points_gdf.to_crs(raster_crs).get_coordinates().to_numpy()
                )
                sampled_values = sample_multiple_tifs_by_coordinates(
                    tif_processors=data, coordinate_list=coord_list, **kwargs
                )

            results_df[output_column] = sampled_values

        elif isinstance(data, gpd.GeoDataFrame):
            # Handle polygon data
            if data.empty:
                self.logger.info("No valid polygon data found for the provided POIs")
                return self.points_gdf.copy()

            if map_radius_meters is None:
                raise ValueError("map_radius_meters must be provided for polygon data")

            if value_column is None:
                raise ValueError("value_column must be provided for polygon data")

            if value_column not in data.columns:
                raise ValueError(
                    f"Value column '{value_column}' not found in input polygon GeoDataFrame."
                )

            self.logger.info(
                f"Aggregating {value_column} within {map_radius_meters}m buffers around POIs using predicate '{predicate}'"
            )

            # Create buffers around POIs
            buffer_gdf = buffer_geodataframe(
                self.points_gdf,
                buffer_distance_meters=map_radius_meters,
                cap_style="round",
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

            results_df = aggregation_result_gdf[["poi_id", value_column]]

            if output_column != "zonal_stat":
                results_df = results_df.rename(columns={value_column: output_column})

        else:
            raise ValueError(
                "data must be either a list of TifProcessor objects or a GeoDataFrame"
            )

        self._update_view(results_df)
        self.logger.info(
            f"Zonal statistics mapping complete for column(s) derived from '{output_column}' or '{value_column}'"
        )
        return self.view

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
            self.points_gdf.copy(), ensure_available=self.config.ensure_available
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
            self.points_gdf.copy(), ensure_available=self.config.ensure_available
        )

        return self.map_zonal_stats(
            data=tif_processors,
            output_column=output_column,
            **kwargs,
        )

    def map_wp_pop(
        self,
        country: Union[str, List[str]],
        map_radius_meters: float,
        resolution=1000,
        predicate: Literal[
            "centroid_within", "intersects", "fractional", "within"
        ] = "fractional",
        output_column: str = "population",
        **kwargs,
    ):
        if isinstance(country, str):
            country = [country]

        handler = WPPopulationHandler(
            project="pop", resolution=resolution, data_store=self.data_store, **kwargs
        )

        self.logger.info(
            f"Mapping WorldPop Population data (year: {handler.config.year}, resolution: {handler.config.resolution}m)"
        )

        if predicate == "fractional" and resolution == 100:
            self.logger.warning(
                "Fractional aggregations only supported for datasets with 1000m resolution. Using `intersects` as predicate"
            )
            predicate = "intersects"
        
        if predicate == "centroid_within":
            data = []
            for c in country:
                data.extend(
                    handler.load_data(c, ensure_available=self.config.ensure_available)
                )
        else:
            data = pd.concat(
                [
                    handler.load_into_geodataframe(
                        c, ensure_available=self.config.ensure_available
                    )
                    for c in country
                ],
                ignore_index=True,
            )

        self.logger.info(f"Mapping WorldPop Population data into {map_radius_meters}m zones around POIs using 'sum' statistic")

        return self.map_zonal_stats(
            data,
            stat="sum",
            map_radius_meters=map_radius_meters,
            value_column="pixel_value",
            predicate=predicate,
            output_column=output_column,
            **kwargs
        )

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
