from pathlib import Path
from typing import List, Optional, Union, Tuple
from pydantic.dataclasses import dataclass, Field

import geopandas as gpd
import pandas as pd

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.writers import write_dataset
from gigaspatial.config import config
from gigaspatial.handlers.google_open_buildings import (
    GoogleOpenBuildingsConfig,
    GoogleOpenBuildingsReader,
)
from gigaspatial.handlers.microsoft_global_buildings import (
    MSBuildingsConfig,
    MSBuildingsReader,
)
from gigaspatial.handlers.ghsl import GHSLDataConfig, GHSLDataReader
from gigaspatial.processing.geo import convert_to_geodataframe, buffer_geodataframe
from gigaspatial.processing.tif_processor import (
    sample_multiple_tifs_by_polygons,
    sample_multiple_tifs_by_coordinates,
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
        n_workers (int): The number of workers to use for parallel processing (if applicable).
                         Defaults to 4.
    """

    base_path: Path = Field(default=config.get_path("poi", "views"))
    output_format: str = "csv"
    n_workers: int = 4


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
        generator_config: Optional[PoiViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
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
        self.generator_config = generator_config or PoiViewGeneratorConfig()
        self.data_store = data_store or LocalDataStore()
        self._points_gdf = self._init_points_gdf(points)
        self.logger = config.get_logger(self.__class__.__name__)

    @staticmethod
    def _init_points_gdf(
        points: Union[
            List[Tuple[float, float]], List[dict], pd.DataFrame, gpd.GeoDataFrame
        ],
    ) -> gpd.GeoDataFrame:
        """Internal static method to convert various point input formats into a GeoDataFrame."""
        if isinstance(points, gpd.GeoDataFrame):
            return points
        elif isinstance(points, pd.DataFrame):
            return convert_to_geodataframe(points)
        elif isinstance(points, list):
            if len(points) == 0:
                return gpd.GeoDataFrame(
                    columns=["geometry"], geometry="geometry", crs="EPSG:4326"
                )
            if isinstance(points[0], tuple) and len(points[0]) == 2:
                # List of (lat, lon) tuples
                df = pd.DataFrame(points, columns=["latitude", "longitude"])
                return convert_to_geodataframe(df)
            elif isinstance(points[0], dict):
                df = pd.DataFrame(points)
                return convert_to_geodataframe(df)
        raise ValueError("Unsupported points input type for PoiViewGenerator.")

    @property
    def points_gdf(self) -> gpd.GeoDataFrame:
        """Gets the internal GeoDataFrame of points of interest."""
        return self._points_gdf

    def map_google_buildings(
        self,
        data_config: Optional[GoogleOpenBuildingsConfig] = None,
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
        data_config = data_config or GoogleOpenBuildingsConfig()
        reader = GoogleOpenBuildingsReader(
            config=data_config, data_store=self.data_store
        )
        self.logger.info("Loading Google Buildings point data")
        buildings_df = reader.load_points(self.points_gdf)
        if buildings_df is None or len(buildings_df) == 0:
            self.logger.info("No Google buildings data found for the provided POIs")
            return self.points_gdf.copy()
        from gigaspatial.processing.geo import calculate_distance

        self.logger.info("Calculating nearest Google building for each POI")
        tree = cKDTree(buildings_df[["latitude", "longitude"]])
        points_df = self.points_gdf.copy()
        _, idx = tree.query(points_df[["latitude", "longitude"]], k=1)
        df_nearest = buildings_df.iloc[idx]
        dist = calculate_distance(
            lat1=points_df.latitude,
            lon1=points_df.longitude,
            lat2=df_nearest.latitude,
            lon2=df_nearest.longitude,
        )
        result = points_df.copy()
        result["nearest_google_building_id"] = df_nearest.full_plus_code.to_numpy()
        result["nearest_google_building_distance"] = dist
        self.logger.info("Google building mapping complete")
        self._points_gdf = result
        return result

    def map_ms_buildings(
        self,
        data_config: Optional[MSBuildingsConfig] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Maps Microsoft Global Buildings data to the POIs by finding the nearest building.

        Enriches the `points_gdf` with the ID and distance to the nearest
        Microsoft Global Building for each POI.

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
        data_config = data_config or MSBuildingsConfig(data_store=self.data_store)
        reader = MSBuildingsReader(config=data_config, data_store=self.data_store)
        self.logger.info("Loading Microsoft Buildings polygon data")
        buildings_gdf = reader.load(self.points_gdf)
        if buildings_gdf is None or len(buildings_gdf) == 0:
            self.logger.info("No Microsoft buildings data found for the provided POIs")
            return self.points_gdf.copy()
        buildings_gdf = buildings_gdf.copy()
        buildings_gdf["x"] = buildings_gdf.geometry.x
        buildings_gdf["y"] = buildings_gdf.geometry.y
        from gigaspatial.processing.geo import calculate_distance

        self.logger.info("Calculating nearest Microsoft building for each POI")
        tree = cKDTree(buildings_gdf[["y", "x"]])
        points_df = self.points_gdf.copy()
        _, idx = tree.query(points_df[["latitude", "longitude"]], k=1)
        df_nearest = buildings_gdf.iloc[idx]
        dist = calculate_distance(
            lat1=points_df.latitude,
            lon1=points_df.longitude,
            lat2=df_nearest.y,
            lon2=df_nearest.x,
        )
        result = points_df.copy()
        result["nearest_ms_building_id"] = df_nearest.get("building_id", None)
        result["nearest_ms_building_distance"] = dist
        self.logger.info("Microsoft building mapping complete")
        self._points_gdf = result
        return result

    def map_built_s(
        self,
        data_config: Optional[GHSLDataConfig] = None,
        map_radius_meters: float = 150,
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
        data_config = data_config or GHSLDataConfig(
            product="GHS_BUILT_S",
            year=2020,
            resolution=100,
            coord_system=4326,
            base_path=config.get_path("ghsl", "silver"),
        )
        reader = GHSLDataReader(config=data_config, data_store=self.data_store)
        gdf_points = self.points_gdf.to_crs(data_config.crs)
        self.logger.info(
            f"Buffering POI points by {map_radius_meters} meters for zonal statistics"
        )
        polygon_list = buffer_geodataframe(
            gdf_points, buffer_distance_meters=map_radius_meters, cap_style="round"
        ).geometry
        self.logger.info("Loading GHSL Built Surface raster tiles")
        tif_processors = reader.load(gdf_points)
        if not tif_processors:
            self.logger.info("No GHSL Built Surface data found for the provided POIs")
            return self.points_gdf.copy()
        self.logger.info("Sampling built surface values for each POI buffer")
        sampled_values = sample_multiple_tifs_by_polygons(
            tif_processors=tif_processors, polygon_list=polygon_list, stat="sum"
        )
        result = gdf_points.copy()
        result["built_surface_m2"] = sampled_values
        self.logger.info("GHSL Built Surface mapping complete")
        self._points_gdf = result
        return result

    def map_smod(
        self,
        data_config: Optional[GHSLDataConfig] = None,
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
        data_config = data_config or GHSLDataConfig(
            product="GHS_SMOD",
            year=2020,
            resolution=1000,
            coord_system=54009,
            base_path=config.get_path("ghsl", "silver"),
        )
        reader = GHSLDataReader(config=data_config, data_store=self.data_store)
        gdf_points = self.points_gdf.to_crs(data_config.crs)
        self.logger.info("Loading GHSL SMOD raster tiles")
        coord_list = [
            (x, y) for x, y in zip(gdf_points["geometry"].x, gdf_points["geometry"].y)
        ]
        tif_processors = reader.load(gdf_points)
        if not tif_processors:
            self.logger.info("No GHSL SMOD data found for the provided POIs")
            return self.points_gdf.copy()
        self.logger.info("Sampling SMOD class values for each POI")
        sampled_values = sample_multiple_tifs_by_coordinates(
            tif_processors=tif_processors, coordinate_list=coord_list
        )
        result = gdf_points.copy()
        result["smod_class"] = sampled_values
        self.logger.info("GHSL SMOD mapping complete")
        self._points_gdf = result
        return result

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
