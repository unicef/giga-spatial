from typing import Dict, List, Optional, Union
from shapely.geometry import Polygon, MultiPolygon

import geopandas as gpd
import pandas as pd
import logging

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.config import config as global_config
from gigaspatial.processing.geo import (
    add_area_in_meters,
    get_centroids,
)
from gigaspatial.handlers.ghsl import GHSLDataHandler
from gigaspatial.handlers.google_open_buildings import GoogleOpenBuildingsHandler
from gigaspatial.handlers.microsoft_global_buildings import MSBuildingsHandler
from gigaspatial.generators.zonal.base import (
    ZonalViewGenerator,
    ZonalViewGeneratorConfig,
    T,
)


class GeometryBasedZonalViewGenerator(ZonalViewGenerator[T]):
    """Mid-level class for zonal view generation based on geometries with identifiers.

    This class serves as an intermediate between the abstract ZonalViewGenerator and specific
    implementations like MercatorViewGenerator or H3ViewGenerator. It handles the common case
    where zones are defined by a mapping between zone identifiers and geometries, either
    provided as a dictionary or as a GeoDataFrame.

    The class extends the base functionality with methods for mapping common geospatial
    datasets including GHSL (Global Human Settlement Layer), Google Open Buildings,
    and Microsoft Global Buildings data.

    Attributes:
        zone_dict (Dict[T, Polygon]): Mapping of zone identifiers to geometries.
        zone_id_column (str): Name of the column containing zone identifiers.
        zone_data_crs (str): Coordinate reference system of the zone data.
        _zone_gdf (gpd.GeoDataFrame): Cached GeoDataFrame representation of zones.
        data_store (DataStore): For accessing input data.
        config (ZonalViewGeneratorConfig): Configuration for view generation.
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
        """Initialize with zone geometries and identifiers.

        Args:
            zone_data (Union[Dict[T, Polygon], gpd.GeoDataFrame]): Zone definitions.
                Either a dictionary mapping zone identifiers to Polygon/MultiPolygon geometries,
                or a GeoDataFrame with geometries and a zone identifier column.
            zone_id_column (str): Name of the column containing zone identifiers.
                Only used if zone_data is a GeoDataFrame. Defaults to "zone_id".
            zone_data_crs (str): Coordinate reference system of the zone data.
                Defaults to "EPSG:4326" (WGS84).
            config (ZonalViewGeneratorConfig, optional): Generator configuration.
                If None, uses default configuration.
            data_store (DataStore, optional): Data store for accessing input data.
                If None, uses LocalDataStore.

        Raises:
            TypeError: If zone_data is not a dictionary or GeoDataFrame, or if dictionary
                values are not Polygon/MultiPolygon geometries.
            ValueError: If zone_id_column is not found in GeoDataFrame, or if the provided
                CRS doesn't match the GeoDataFrame's CRS.
        """
        super().__init__(config=config, data_store=data_store, logger=logger)

        self.zone_id_column = zone_id_column
        self.zone_data_crs = zone_data_crs

        # Store zone data based on input type
        if isinstance(zone_data, dict):
            for zone_id, geom in zone_data.items():
                if not isinstance(geom, (Polygon, MultiPolygon)):
                    raise TypeError(
                        f"Zone {zone_id}: Expected (Multi)Polygon, got {type(geom).__name__}"
                    )

            # Store the original dictionary
            self.zone_dict = zone_data

            # Also create a GeoDataFrame for consistent access
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

            if zone_data_crs != zone_data.crs:
                raise ValueError(
                    f"Provided data crs '{zone_data_crs}' does not match to the crs of the data '{zone_data.crs}'"
                )

            # Store the GeoDataFrame
            self._zone_gdf = zone_data.rename(columns={zone_id_column: "zone_id"})

            # Also create a dictionary for fast lookups
            self.zone_dict = dict(zip(zone_data[zone_id_column], zone_data.geometry))

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
        return self._zone_gdf[self.zone_id_column].tolist()

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert zones to a GeoDataFrame with standardized column names.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame with 'zone_id' and 'geometry' columns.
                The zone_id column is renamed from the original zone_id_column if different.
        """
        # If we already have a GeoDataFrame, just rename the ID column if needed
        result = self._zone_gdf.copy()
        if self.zone_id_column != "zone_id":
            result = result.rename(columns={self.zone_id_column: "zone_id"})
        return result

    def map_built_s(
        self,
        year=2020,
        resolution=100,
        stat: str = "sum",
        name_prefix: str = "built_surface_m2_",
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Map GHSL Built-up Surface data to zones.

        Convenience method for mapping Global Human Settlement Layer Built-up Surface
        data using appropriate default parameters for built surface analysis.

        Args:
            ghsl_data_config (GHSLDataConfig): Configuration for GHSL Built-up Surface data.
                Defaults to GHS_BUILT_S product for 2020 at 100m resolution.
            stat (str): Statistic to calculate for built surface values within each zone.
                Defaults to "sum" which gives total built surface area.
            name_prefix (str): Prefix for the output column name. Defaults to "built_surface_m2_".

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with zones and built surface metrics.
                Adds a column named "{name_prefix}{stat}" containing the aggregated values.
        """
        handler = GHSLDataHandler(
            product="GHS_BUILT_S",
            year=year,
            resolution=resolution,
            data_store=self.data_store,
            **kwargs,
        )

        return self.map_ghsl(
            handler=handler, stat=stat, name_prefix=name_prefix, **kwargs
        )

    def map_smod(
        self,
        year=2020,
        resolution=100,
        stat: str = "median",
        name_prefix: str = "smod_class_",
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Map GHSL Settlement Model data to zones.

        Convenience method for mapping Global Human Settlement Layer Settlement Model
        data using appropriate default parameters for settlement classification analysis.

        Args:
            ghsl_data_config (GHSLDataConfig): Configuration for GHSL Settlement Model data.
                Defaults to GHS_SMOD product for 2020 at 1000m resolution in Mollweide projection.
            stat (str): Statistic to calculate for settlement class values within each zone.
                Defaults to "median" which gives the predominant settlement class.
            name_prefix (str): Prefix for the output column name. Defaults to "smod_class_".

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with zones and settlement classification.
                Adds a column named "{name_prefix}{stat}" containing the aggregated values.
        """
        handler = GHSLDataHandler(
            product="GHS_SMOD",
            year=year,
            resolution=resolution,
            data_store=self.data_store,
            coord_system=54009,
            **kwargs,
        )

        return self.map_ghsl(
            handler=handler, stat=stat, name_prefix=name_prefix, **kwargs
        )

    def map_ghsl(
        self,
        handler: GHSLDataHandler,
        stat: str,
        name_prefix: Optional[str] = None,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Map Global Human Settlement Layer data to zones.

        Loads and processes GHSL raster data for the intersecting tiles, then samples
        the raster values within each zone using the specified statistic.

        Args:
            ghsl_data_config (GHSLDataConfig): Configuration specifying which GHSL
                product, year, resolution, and coordinate system to use.
            stat (str): Statistic to calculate for raster values within each zone.
                Common options: "mean", "sum", "median", "min", "max".
            name_prefix (str, optional): Prefix for the output column name.
                If None, uses the GHSL product name in lowercase followed by underscore.

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with zones and GHSL metrics.
                Adds a column named "{name_prefix}{stat}" containing the sampled values.

        Note:
            The method automatically determines which GHSL tiles intersect with the zones
            and loads only the necessary data for efficient processing.
        """
        handler = handler or GHSLDataHandler(data_store=self.data_store, **kwargs)
        self.logger.info(
            f"Mapping {handler.config.product} data (year: {handler.config.year}, resolution: {handler.config.resolution}m)"
        )
        tif_processors = handler.load_data(
            self.zone_gdf, ensure_available=self.config.ensure_available
        )

        self.logger.info(
            f"Sampling {handler.config.product} data using '{stat}' statistic"
        )
        sampled_values = self.map_rasters(tif_processors=tif_processors, stat=stat)

        name_prefix = (
            name_prefix if name_prefix else handler.config.product.lower() + "_"
        )
        column_name = f"{name_prefix}{stat}"
        self._zone_gdf[column_name] = sampled_values

        self.logger.info(f"Added {column_name} column")

        return self._zone_gdf.copy()

    def map_google_buildings(
        self,
        handler: Optional[GoogleOpenBuildingsHandler] = None,
        use_polygons: bool = False,
    ) -> gpd.GeoDataFrame:
        """Map Google Open Buildings data to zones.

        Processes Google Open Buildings dataset to calculate building counts and total
        building area within each zone. Can use either point centroids (faster) or
        polygon geometries (more accurate) for spatial operations.

        Args:
            google_open_buildings_config (GoogleOpenBuildingsConfig): Configuration
                for accessing Google Open Buildings data. Uses default configuration if not provided.
            use_polygons (bool): Whether to use polygon geometries for buildings.
                If True, uses actual building polygons for more accurate area calculations
                but with slower performance. If False, uses building centroids with
                area values from attributes for faster processing. Defaults to False.

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with zones and building metrics.
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
            self.zone_gdf, ensure_available=self.config.ensure_available
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
                self.zone_gdf, self.config.ensure_available
            )

            self.logger.info(
                "Calculating building areas with area-weighted aggregation"
            )
            area_result = self.map_polygons(buildings_gdf, area_weighted=True)

            self.logger.info("Counting buildings using points data")
            count_result = self.map_points(points=buildings_df, predicate="within")

        self._zone_gdf["google_buildings_count"] = self.zone_gdf.index.map(count_result)
        self._zone_gdf["google_buildings_area_in_meters"] = self.zone_gdf.index.map(
            area_result
        )

        self.logger.info(f"Added Google building data")

        return self._zone_gdf.copy()

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
            ms_buildings_config (MSBuildingsConfig, optional): Configuration for
                accessing Microsoft Global Buildings data. If None, uses default configuration.
            use_polygons (bool): Whether to use polygon geometries for buildings.
                If True, uses actual building polygons for more accurate area calculations
                but with slower performance. If False, uses building centroids with
                area values from attributes for faster processing. Defaults to False.

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with zones and building metrics.
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
        handler = MSBuildingsHandler(data_store=self.data_store)
        buildings_gdf = handler.load_data(
            self.zone_gdf, ensure_available=self.config.ensure_available
        )

        # Check if we found any buildings
        if buildings_gdf.empty:
            self.logger.warning(
                "No Microsoft buildings data found for the provided zones"
            )
            return self._zone_gdf.copy()

        buildings_gdf = add_area_in_meters(buildings_gdf)

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
            area_result = self.map_polygons(buildings_gdf, area_weighted=True)

            self.logger.info("Counting Microsoft buildings per zone")

            count_result = self.map_points(
                points=building_centroids, predicate="within"
            )

        self._zone_gdf["ms_buildings_count"] = self.zone_gdf.index.map(count_result)
        self._zone_gdf["ms_buildings_area_in_meters"] = self.zone_gdf.index.map(
            area_result
        )

        self.logger.info(f"Added Microsoft building data")

        return self._zone_gdf.copy()
