from typing import Dict, List, Optional, Union, Literal
from shapely.geometry import Polygon, MultiPolygon

import geopandas as gpd
import pandas as pd
import logging

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.processing.geo import (
    add_area_in_meters,
    get_centroids,
)
from gigaspatial.processing.tif_processor import TifProcessor
from gigaspatial.handlers.ghsl import GHSLDataHandler
from gigaspatial.handlers.google_open_buildings import GoogleOpenBuildingsHandler
from gigaspatial.handlers.microsoft_global_buildings import MSBuildingsHandler
from gigaspatial.handlers.worldpop import WPPopulationHandler
from gigaspatial.handlers.rwi import RWIHandler
from gigaspatial.processing.buildings_engine import GoogleMSBuildingsEngine
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
        return self._zone_gdf.zone_id.tolist()

    def get_zone_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert zones to a GeoDataFrame with standardized column names.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame with 'zone_id' and 'geometry' columns.
                The zone_id column is renamed from the original zone_id_column if different.
        """
        # Since _zone_gdf is already created with 'zone_id' column in the constructor,
        # we just need to return a copy of it
        return self._zone_gdf.copy()

    @property
    def zone_gdf(self) -> gpd.GeoDataFrame:
        """Override the base class zone_gdf property to ensure correct column names.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame with 'zone_id' and 'geometry' columns.
        """
        return self._zone_gdf.copy()

    def map_built_s(
        self,
        year=2020,
        resolution=100,
        stat: str = "sum",
        output_column: str = "built_surface_m2",
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
        Returns:
            pd.DataFrame: Updated view DataFrame and settlement classification.
                Adds a column with `output_column` containing the aggregated values.
        """
        handler = GHSLDataHandler(
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
        Returns:
            pd.DataFrame: Updated view DataFrame and settlement classification.
                Adds a column with `output_column` containing the aggregated values.
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
            handler=handler, stat=stat, output_column=output_column, **kwargs
        )

    def map_ghsl(
        self,
        handler: GHSLDataHandler,
        stat: str,
        output_column: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Map Global Human Settlement Layer data to zones.

        Loads and processes GHSL raster data for the intersecting tiles, then samples
        the raster values within each zone using the specified statistic.

        Args:
            hander (GHSLDataHandler): Handler for the GHSL data.
            stat (str): Statistic to calculate for raster values within each zone.
                Common options: "mean", "sum", "median", "min", "max".
            output_column (str): The output column name.
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
            **kwargs,
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
            google_open_buildings_config (GoogleOpenBuildingsConfig): Configuration
                for accessing Google Open Buildings data. Uses default configuration if not provided.
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
        country: str,
        source_filter: Literal["google", "microsoft"] = None,
        output_column: str = "building_count",
        **kwargs,
    ):
        """Map Google-Microsoft combined buildings data to zones.

        Efficiently counts buildings within each zone by leveraging spatial indexing
        and S2 grid partitioning. For partitioned datasets, only loads building tiles
        that intersect with zones, significantly improving performance for large countries.

        The method uses Shapely's STRtree for fast spatial queries, enabling efficient
        intersection testing between millions of buildings and zone geometries.

        Parameters
        ----------
        country : str
            ISO 3166-1 alpha-3 country code (e.g., 'USA', 'BRA', 'IND').
            Must match the country codes used in the building dataset.

        source_filter : {'google', 'microsoft'}, optional
            Filter buildings by data source. Options:
            - 'google': Only count buildings from Google Open Buildings
            - 'microsoft': Only count buildings from Microsoft Global Buildings
            - None (default): Count buildings from both sources

        output_column : str, default='building_count'
            Name of the column to add to the view containing building counts.
            Must be a valid pandas column name.

        **kwargs : dict
            Additional keyword arguments passed to GoogleMSBuildingsHandler.
            Common options include data versioning or custom data paths.

        Returns
        -------
        pd.DataFrame
            Updated view DataFrame with building counts added.
            The DataFrame includes all original zone columns plus the new
            output_column containing integer counts of buildings per zone.

        Notes
        -----
        Algorithm Overview:
        1. **Single-file countries**: Processes all zones against the entire dataset
        in one pass (optimal for small countries or non-partitioned data).

        2. **Partitioned countries**: Uses S2 grid spatial index to:
        - Identify which S2 cells intersect with zones
        - Load only intersecting building tiles (not entire country)
        - Process each tile independently for memory efficiency
        - Accumulate counts across tiles

        Performance Characteristics:
        - For partitioned data with N zones and M S2 tiles:
        * Only loads ~sqrt(M) tiles that intersect zones (huge savings)
        * Uses STRtree with O(log k) query time per zone
        * Memory usage: O(max_buildings_per_tile + N)

        - For single-file data:
        * Loads entire building dataset once
        * Single STRtree query for all zones (vectorized)

        The spatial query uses the 'intersects' predicate, meaning a building is
        counted if its polygon boundary touches or overlaps with a zone's boundary.
        Buildings on zone borders may be counted in multiple zones.

        Examples
        --------
        Count all buildings in H3 hexagons across the USA:

        >>> h3_zones = H3ViewGenerator(resolution=8, bbox=usa_bbox)
        >>> h3_zones.map_buildings("USA")
        >>> print(h3_zones.view[['zone_id', 'building_count']].head())
        zone_id  building_count
        0  88...01            1250
        1  88...02             890
        2  88...03               0

        Count only Google buildings with custom column name:

        >>> zones.map_buildings(
        ...     "BRA",
        ...     source_filter="google",
        ...     output_column="google_building_count"
        ... )

        Compare building counts from different sources:

        >>> zones.map_buildings("IND", source_filter="google",
        ...                     output_column="google_buildings")
        >>> zones.map_buildings("IND", source_filter="microsoft",
        ...                     output_column="ms_buildings")
        >>> zones.view['total_buildings'] = (
        ...     zones.view['google_buildings'] + zones.view['ms_buildings']
        ... )

        See Also
        --------
        map_google_buildings : Map Google Open Buildings data only
        map_ms_buildings : Map Microsoft Global Buildings data only
        GoogleMSBuildingsHandler : Handler for combined building datasets

        References
        ----------
        .. [1] Google Open Buildings: https://sites.research.google/open-buildings/
        .. [2] Microsoft Global Buildings: https://github.com/microsoft/GlobalMLBuildingFootprints
        """

        self.logger.info(f"Mapping Google-Microsoft Buildings data to zones")

        from gigaspatial.handlers.google_ms_combined_buildings import (
            GoogleMSBuildingsHandler,
        )

        handler = GoogleMSBuildingsHandler(partition_strategy="s2_grid")

        if self.config.ensure_available:
            if not handler.ensure_data_available(country, **kwargs):
                raise RuntimeError("Could not ensure data availability for loading")

        building_files = handler.reader.resolve_source_paths(country, **kwargs)

        result = GoogleMSBuildingsEngine.count_buildings_in_zones(
            handler=handler,
            building_files=building_files,
            zones_gdf=self.zone_gdf,
            source_filter=source_filter,
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
        **kwargs,
    ):
        handler = GHSLDataHandler(
            product="GHS_POP",
            resolution=resolution,
            data_store=self.data_store,
            **kwargs,
        )

        if predicate == "fractional":
            if resolution == 100:
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
        country: Union[str, List[str]],
        resolution=1000,
        predicate: Literal[
            "centroid_within", "intersects", "fractional"
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

        if predicate == "centroid_within":
            if handler.config.project == "age_structures":
                # Single country enforced above
                iso = countries_list[0]
                raw = handler.load_data(
                    iso,
                    ensure_available=self.config.ensure_available,
                    **kwargs,
                )
                all_tif_processors = self._ensure_tif_list(raw)

                # Sum results from each tif_processor separately
                all_results_by_zone = {
                    zone_id: 0 for zone_id in self.get_zone_identifiers()
                }
                self.logger.info(
                    f"Sampling individual age_structures rasters using 'sum' statistic and summing per zone."
                )
                for tif_processor in all_tif_processors:
                    single_raster_result = self.map_rasters(
                        raster_data=tif_processor, stat="sum"
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
                        **kwargs,
                    )
                    tif_processors.extend(self._ensure_tif_list(raw))

                self.logger.info(
                    f"Sampling WorldPop Population data using 'sum' statistic"
                )
                result = self.map_rasters(raster_data=tif_processors, stat="sum")
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

            self.logger.info(f"Aggregating WorldPop Population data to the zones.")
            result = self.map_polygons(
                gdf_pop,
                value_columns="pixel_value",
                aggregation="sum",
                predicate=predicate,
            )

        self.add_variable_to_view(result, output_column)

        return self.view

    def map_rwi(
        self,
        country: Union[str, List[str]],
        predicate: Literal["intersects", "within", "centroid_within"] = "intersects",
        aggregation: Literal["mean", "median", "max", "min"] = "mean",
        output_column: str = "rwi",
        **kwargs,
    ):

        handler = RWIHandler(
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
