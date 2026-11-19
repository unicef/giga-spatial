import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import Field
from scipy.spatial import cKDTree

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.generators._base import BaseViewGenerator, BaseViewGeneratorConfig
from gigaspatial.handlers.google_open_buildings import GoogleOpenBuildingsHandler
from gigaspatial.handlers.microsoft_global_buildings import MSBuildingsHandler
from gigaspatial.processing.buildings_engine import GoogleMSBuildingsEngine
from gigaspatial.processing.geo import (
    calculate_distance,
    convert_to_geodataframe,
    detect_coordinate_columns,
    get_centroids,
)
from gigaspatial.processing.tif_processor import TifProcessor

if TYPE_CHECKING:
    from gigaspatial.handlers.google_ms_combined_buildings import (
        GoogleMSBuildingsHandler,
    )


class PoiViewGeneratorConfig(BaseViewGeneratorConfig):
    """Configuration for POI (Point of Interest) view generation.

    .. deprecated::
        Pass ``ensure_available=`` directly to :class:`PoiViewGenerator` and
        a target file path directly to :meth:`PoiViewGenerator.save_view`.

    Attributes:
        base_path: The base directory where generated POI views will be saved.
            Defaults to a path retrieved from `config`.
        output_format: The default format for saving output files (e.g., "csv", "geojson").
            Defaults to "csv".
        ensure_available: If True, ensures datasets are downloaded locally during mapping.
    """

    base_path: Path = Field(default=global_config.get_path("poi", "views"))
    output_format: str = "csv"


class PoiViewGenerator(BaseViewGenerator):
    """POI View Generator for multi-source attribute enrichment.

    Integrates geospatial datasets such as Google Open Buildings,
    Microsoft Global Buildings, GHSL (Built Surface/SMOD/Population),
    WorldPop, RWI, and WRI Aqueduct Flood Hazard with Points of Interest (POIs).

    The POIs can be initialized from a list of (latitude, longitude) tuples,
    a list of dictionaries, a pandas DataFrame, or a geopandas GeoDataFrame.
    """

    id_column = "poi_id"
    _default_config_cls = PoiViewGeneratorConfig
    _is_point_generator = True
    _coverage_label = "pois"
    _default_built_s_radius = 150
    _default_aqueduct_buffer_meters = 500

    def __init__(
        self,
        points: Union[
            List[Tuple[float, float]], List[dict], pd.DataFrame, gpd.GeoDataFrame
        ],
        poi_id_column: str = "poi_id",
        config: Optional[PoiViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        ensure_available: bool = True,
    ):
        """Initialize the PoiViewGenerator with input points and options.

        Args:
            points: The input points of interest. Can be a list of (lat, lon)
                tuples, a list of dicts, a pandas DataFrame, or a GeoDataFrame.
            poi_id_column: The name of the column containing unique IDs for
                each POI. Defaults to "poi_id".
            config: Deprecated configuration object.
            data_store: Storage interface for data management. If None,
                LocalDataStore is used.
            logger: Custom logger. If None, uses a default logger.
            ensure_available: If True, ensures datasets are downloaded locally
                during mapping unless overridden per call.
        """
        if hasattr(points, "__len__") and len(points) == 0:
            raise ValueError("Points input cannot be empty")

        points_gdf = self._init_points_gdf(points, poi_id_column)
        super().__init__(
            base_gdf=points_gdf,
            ensure_available=ensure_available,
            config=config,
            data_store=data_store,
            logger=logger,
        )
        self._points_gdf = self._base_gdf

    @staticmethod
    def _normalize_poi_id(df: pd.DataFrame, poi_id_column: str) -> pd.DataFrame:
        """Validate and standardize the POI identifier column to ``'poi_id'``."""
        if poi_id_column not in df.columns:
            if poi_id_column != "poi_id":
                raise ValueError(
                    f"POI ID column '{poi_id_column}' not found in input points."
                )
            df["poi_id"] = [f"poi_{i}" for i in range(len(df))]
            return df

        if poi_id_column != "poi_id" and "poi_id" in df.columns:
            raise ValueError(
                f"Column 'poi_id' already exists in input; cannot rename '{poi_id_column}' to 'poi_id'."
            )

        if poi_id_column != "poi_id":
            df = df.rename(columns={poi_id_column: "poi_id"})

        if df["poi_id"].duplicated().any():
            raise ValueError(
                f"Column '{poi_id_column}' provided as 'poi_id_column' contains duplicate values."
            )
        return df

    @staticmethod
    def _init_points_gdf(
        points: Union[
            List[Tuple[float, float]], List[dict], pd.DataFrame, gpd.GeoDataFrame
        ],
        poi_id_column: str,
    ) -> gpd.GeoDataFrame:
        """Standardize diverse point input formats into an EPSG:4326 GeoDataFrame."""
        if isinstance(points, gpd.GeoDataFrame):
            if points.index.duplicated().any():
                warnings.warn(
                    "Duplicate indices detected in GeoDataFrame. Resetting index to ensure unique indices.",
                    UserWarning,
                    stacklevel=2,
                )
                points = points.reset_index(drop=True)

            points = points.copy()
            if points.crs is None:
                raise ValueError("Input GeoDataFrame must have a CRS set.")
            if points.crs != "EPSG:4326":
                points = points.to_crs("EPSG:4326")
            if points.active_geometry_name != "geometry":
                points = points.rename_geometry("geometry")

            points["latitude"] = points.geometry.y
            points["longitude"] = points.geometry.x
            return PoiViewGenerator._normalize_poi_id(points, poi_id_column)

        if isinstance(points, pd.DataFrame):
            if points.index.duplicated().any():
                warnings.warn(
                    "Duplicate indices detected in DataFrame. Resetting index to ensure unique indices.",
                    UserWarning,
                    stacklevel=2,
                )
                points = points.reset_index(drop=True)
            try:
                lat_col, lon_col = detect_coordinate_columns(points)
            except ValueError as e:
                raise ValueError(
                    f"Could not detect coordinate columns in DataFrame: {str(e)}"
                ) from e

            points = points.copy()
            points["latitude"] = points[lat_col]
            points["longitude"] = points[lon_col]
            points = PoiViewGenerator._normalize_poi_id(points, poi_id_column)
            return convert_to_geodataframe(
                points, lat_col="latitude", lon_col="longitude"
            )

        if isinstance(points, list):
            if len(points) == 0:
                raise ValueError("Points input cannot be empty")

            if isinstance(points[0], tuple) and len(points[0]) == 2:
                df = pd.DataFrame(points, columns=["latitude", "longitude"])
                df = PoiViewGenerator._normalize_poi_id(df, poi_id_column)
                return convert_to_geodataframe(df)

            if isinstance(points[0], dict):
                df = pd.DataFrame(points)
                if df.index.duplicated().any():
                    warnings.warn(
                        "Duplicate indices detected in DataFrame created from dictionary list. Resetting index to ensure unique indices.",
                        UserWarning,
                        stacklevel=2,
                    )
                    df = df.reset_index(drop=True)
                try:
                    lat_col, lon_col = detect_coordinate_columns(df)
                except ValueError as e:
                    raise ValueError(
                        f"Could not detect coordinate columns in dictionary list: {str(e)}"
                    ) from e

                df["latitude"] = df[lat_col]
                df["longitude"] = df[lon_col]
                df = PoiViewGenerator._normalize_poi_id(df, poi_id_column)
                return convert_to_geodataframe(
                    df, lat_col="latitude", lon_col="longitude"
                )

        raise ValueError("Unsupported points input type for PoiViewGenerator.")

    @property
    def points_gdf(self) -> gpd.GeoDataFrame:
        """A copy of the internal GeoDataFrame of points of interest."""
        return self._base_gdf.copy()

    def map_nearest_points(
        self,
        points_df: Union[pd.DataFrame, gpd.GeoDataFrame],
        id_column: Optional[str] = None,
        lat_column: Optional[str] = None,
        lon_column: Optional[str] = None,
        output_prefix: str = "nearest",
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Enrich POIs with IDs and distances to their nearest neighbors and update ``self.view``.

        Args:
            points_df: DataFrame containing the target points to search.
            id_column: Name of the column containing target point IDs.
                If None, uses the DataFrame index.
            lat_column: Name of the latitude column in `points_df`. If None,
                it is automatically detected.
            lon_column: Name of the longitude column in `points_df`. If None,
                it is automatically detected.
            output_prefix: Prefix for the new '{prefix}_id' and
                '{prefix}_distance' columns. Defaults to "nearest".
            overwrite: If True, allows replacing existing output columns in the view.
            **kwargs: Additional parameters.

        Returns:
            The updated POI view DataFrame.
        """
        id_out = f"{output_prefix}_id"
        dist_out = f"{output_prefix}_distance"
        self._validate_output_columns([id_out, dist_out], overwrite=overwrite)

        self.logger.info(
            f"Mapping nearest points from {points_df.__class__.__name__} to POIs"
        )

        if points_df is None or points_df.empty:
            self.logger.warning("No points found in the input DataFrame")
            return self._fill_empty_columns(
                {id_out: np.nan, dist_out: np.nan}, overwrite=overwrite
            )

        if isinstance(points_df, gpd.GeoDataFrame):
            points_df = points_df.copy()
            if points_df.crs is not None and points_df.crs != "EPSG:4326":
                points_df = points_df.to_crs("EPSG:4326")
            points_df["latitude"] = points_df.geometry.y
            points_df["longitude"] = points_df.geometry.x
            lat_column = "latitude"
            lon_column = "longitude"
            self.logger.info("Extracted coordinates from geometry")

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

        self.logger.info("Calculating nearest points for each POI")
        tree = cKDTree(points_df[[lat_column, lon_column]])
        points_df_poi = self.points_gdf
        _, idx = tree.query(points_df_poi[["latitude", "longitude"]], k=1)
        df_nearest = points_df.iloc[idx]
        dist = calculate_distance(
            lat1=points_df_poi.latitude,
            lon1=points_df_poi.longitude,
            lat2=df_nearest[lat_column],
            lon2=df_nearest[lon_column],
        )
        if id_column is not None:
            nearest_ids = points_df.iloc[idx][id_column].values
        else:
            nearest_ids = points_df.index[idx].values

        temp_result_df = pd.DataFrame(
            {
                "poi_id": points_df_poi["poi_id"],
                id_out: nearest_ids,
                dist_out: dist,
            }
        )

        self.logger.info(
            f"Nearest points mapping complete with prefix '{output_prefix}'"
        )
        return self._update_view(temp_result_df, overwrite=overwrite)

    def map_google_buildings(
        self,
        handler: Optional[GoogleOpenBuildingsHandler] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Map Google Open Buildings data to POIs by finding the nearest building.

        Args:
            handler: Optional handler for Google Open Buildings data.
            ensure_available: Override generator's ``ensure_available`` flag.
            overwrite: If True, replaces existing output columns in the view.
            **kwargs: Additional parameters passed to `handler.load_points`.

        Returns:
            The updated POI view DataFrame.
        """
        out_cols = ["nearest_google_building_id", "nearest_google_building_distance"]
        self._validate_output_columns(out_cols, overwrite=overwrite)
        eff_ensure = self._resolve_ensure_available(ensure_available)

        self.logger.info("Mapping Google Open Buildings data to POIs")
        handler = handler or GoogleOpenBuildingsHandler(data_store=self.data_store)

        self.logger.info("Loading Google Buildings point data")
        buildings_df = handler.load_points(
            self.points_gdf, ensure_available=eff_ensure, **kwargs
        )
        if buildings_df is None or len(buildings_df) == 0:
            self.logger.warning("No Google buildings data found for the provided POIs")
            return self._fill_empty_columns(
                {
                    "nearest_google_building_id": np.nan,
                    "nearest_google_building_distance": np.nan,
                },
                overwrite=overwrite,
            )

        return self.map_nearest_points(
            points_df=buildings_df,
            id_column="full_plus_code",
            output_prefix="nearest_google_building",
            overwrite=overwrite,
            **kwargs,
        )

    def map_ms_buildings(
        self,
        handler: Optional[MSBuildingsHandler] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Map Microsoft Global Buildings data to POIs by finding the nearest building.

        Args:
            handler: Optional handler for MS Buildings data.
            ensure_available: Override generator's ``ensure_available`` flag.
            overwrite: If True, replaces existing output columns in the view.
            **kwargs: Additional parameters.

        Returns:
            The updated POI view DataFrame.
        """
        out_cols = ["nearest_ms_building_id", "nearest_ms_building_distance"]
        self._validate_output_columns(out_cols, overwrite=overwrite)
        eff_ensure = self._resolve_ensure_available(ensure_available)

        self.logger.info("Mapping Microsoft Global Buildings data to POIs")
        handler = handler or MSBuildingsHandler(data_store=self.data_store)
        self.logger.info("Loading Microsoft Buildings polygon data")
        buildings_gdf = handler.load_data(self.points_gdf, ensure_available=eff_ensure)
        if buildings_gdf is None or len(buildings_gdf) == 0:
            self.logger.warning(
                "No Microsoft buildings data found for the provided POIs"
            )
            return self._fill_empty_columns(
                {
                    "nearest_ms_building_id": np.nan,
                    "nearest_ms_building_distance": np.nan,
                },
                overwrite=overwrite,
            )

        building_centroids = get_centroids(buildings_gdf)
        if "building_id" not in building_centroids.columns:
            self.logger.info("Creating building IDs from coordinates")
            building_centroids["building_id"] = building_centroids.apply(
                lambda row: f"{row.geometry.y:.6f}_{row.geometry.x:.6f}",
                axis=1,
            )

        return self.map_nearest_points(
            points_df=building_centroids,
            id_column="building_id",
            output_prefix="nearest_ms_building",
            overwrite=overwrite,
            **kwargs,
        )

    def map_zonal_stats(
        self,
        data: Union[TifProcessor, List[TifProcessor], gpd.GeoDataFrame],
        stat: str = "mean",
        map_radius_meters: Optional[float] = None,
        output_column: str = "zonal_stat",
        value_column: Optional[str] = None,
        predicate: Literal["intersects", "within", "fractional"] = "intersects",
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Enrich POIs with zonal statistics from raster, point, or polygon data and update ``self.view``.

        Args:
            data: Data source (TifProcessor(s) or GeoDataFrame).
            stat: Statistic to calculate (e.g., 'sum', 'mean', 'count'). Defaults to 'mean'.
            map_radius_meters: Radius for circular POI buffers. If None, samples
                directly at point locations (raster only).
            output_column: Destination column name. Defaults to "zonal_stat".
            value_column: Target column for point or polygon aggregation.
            predicate: Spatial relationship for aggregation. Defaults to "intersects".
            overwrite: If True, replaces an existing column in the view.
            **kwargs: Extra parameters for raster sampling.

        Returns:
            The updated POI view DataFrame.
        """
        self._validate_output_columns([output_column], overwrite=overwrite)

        if isinstance(data, TifProcessor) or (
            isinstance(data, list) and all(isinstance(x, TifProcessor) for x in data)
        ):
            return self.map_rasters(
                raster_data=data,
                stat=stat,
                output_column=output_column,
                map_radius_meters=map_radius_meters,
                overwrite=overwrite,
                **kwargs,
            )

        if isinstance(data, gpd.GeoDataFrame):
            if data.empty:
                self.logger.warning("No valid GeoDataFrame data provided")
                is_count = stat == "count" or (
                    value_column is None
                    and (data.geom_type.empty or any(data.geom_type.isin(["MultiPoint", "Point"])))
                )
                return self._fill_empty_columns(
                    {output_column: 0 if is_count else np.nan},
                    overwrite=overwrite,
                )

            if map_radius_meters is None:
                raise ValueError(
                    "map_radius_meters must be provided for GeoDataFrame data"
                )

            if any(data.geom_type.isin(["MultiPoint", "Point"])):
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

                return self.map_points(
                    points=data,
                    value_columns=value_column,
                    aggregation=actual_stat,
                    predicate=predicate,
                    output_column=output_column,
                    map_radius_meters=map_radius_meters,
                    overwrite=overwrite,
                )

            if value_column is None:
                if stat == "count":
                    return self.map_polygons(
                        polygons=data,
                        value_columns=None,
                        aggregation="count",
                        predicate=predicate,
                        output_column=output_column,
                        map_radius_meters=map_radius_meters,
                        overwrite=overwrite,
                    )
                raise ValueError(
                    "value_column must be provided for polygon data aggregation."
                )

            if value_column not in data.columns:
                raise ValueError(
                    f"Value column '{value_column}' not found in input GeoDataFrame."
                )

            return self.map_polygons(
                polygons=data,
                value_columns=value_column,
                aggregation=stat,
                predicate=predicate,
                output_column=output_column,
                map_radius_meters=map_radius_meters,
                overwrite=overwrite,
            )

        raise ValueError(
            "data must be either a list of TifProcessor objects or a GeoDataFrame"
        )

    def _resolve_building_files(
        self,
        country: str,
        handler: Optional["GoogleMSBuildingsHandler"],
        ensure_available: Optional[bool],
        **kwargs,
    ) -> tuple["GoogleMSBuildingsHandler", List[Path]]:
        if handler is None:
            from gigaspatial.handlers.google_ms_combined_buildings import (
                GoogleMSBuildingsHandler,
            )

            handler = GoogleMSBuildingsHandler(
                partition_strategy="s2_grid", data_store=self.data_store
            )

        eff_ensure = self._resolve_ensure_available(ensure_available)
        if eff_ensure:
            if not handler.ensure_data_available(country, **kwargs):
                raise RuntimeError(
                    f"Could not ensure Google/Microsoft building data availability for '{country}'."
                )

        building_files = handler.reader.resolve_source_paths(
            country, **handler.select_options(kwargs, handler.config)
        )
        return handler, building_files

    def find_nearest_buildings(
        self,
        country: str,
        search_radius: float = 1000,
        find_nearest_globally: bool = False,
        handler: Optional["GoogleMSBuildingsHandler"] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Find the nearest building to each POI within a search radius and update ``self.view``."""
        if find_nearest_globally:
            self.logger.info(
                "Global nearest building search enabled. "
                "This may be slower than radius-limited search."
            )
            return self._find_nearest_building_globally(
                country=country,
                handler=handler,
                ensure_available=ensure_available,
                overwrite=overwrite,
                **kwargs,
            )

        within_col = f"building_within_{search_radius}m"
        self._validate_output_columns(
            ["nearest_building_distance_m", within_col], overwrite=overwrite
        )

        if search_radius > 5000:
            self.logger.warning(
                f"Large search radius ({search_radius}m) may impact performance. "
                "Consider using progressive expansion for better efficiency."
            )

        self.logger.info(
            f"Mapping combined buildings data to POIs within {search_radius}m"
        )

        handler, building_files = self._resolve_building_files(
            country=country,
            handler=handler,
            ensure_available=ensure_available,
            **kwargs,
        )
        if not building_files:
            self.logger.warning(
                "No Google/Microsoft building files found for %s; filling empty mapping.",
                country,
            )
            return self._fill_empty_columns(
                {"nearest_building_distance_m": np.nan, within_col: False},
                overwrite=overwrite,
            )

        result = GoogleMSBuildingsEngine.nearest_buildings_to_pois(
            handler=handler,
            building_files=building_files,
            pois_gdf=self.points_gdf,
            search_radius_m=search_radius,
            logger=self.logger,
        )

        distances_clean = result.distances_m.replace(np.inf, np.nan)
        result_df = pd.DataFrame(
            {
                "poi_id": result.distances_m.index,
                "nearest_building_distance_m": distances_clean,
                within_col: result.distances_m <= search_radius,
            }
        )

        return self._update_view(result_df, overwrite=overwrite)

    def _find_nearest_building_globally(
        self,
        country: str,
        max_global_search: float = 10000,
        handler: Optional["GoogleMSBuildingsHandler"] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Find the true nearest building for each POI using progressive expansion."""
        within_col = f"building_within_{max_global_search}m"
        self._validate_output_columns(
            ["nearest_building_distance_m", within_col], overwrite=overwrite
        )

        self.logger.info("Starting global nearest building search...")

        handler, building_files = self._resolve_building_files(
            country=country,
            handler=handler,
            ensure_available=ensure_available,
            **kwargs,
        )
        if not building_files:
            self.logger.warning(
                "No Google/Microsoft building files found for %s; filling empty mapping.",
                country,
            )
            return self._fill_empty_columns(
                {"nearest_building_distance_m": np.nan, within_col: False},
                overwrite=overwrite,
            )

        if len(building_files) == 1:
            self.logger.info("Single file country: using single-file optimization.")

            result = GoogleMSBuildingsEngine.nearest_buildings_to_pois(
                handler=handler,
                building_files=building_files,
                pois_gdf=self.points_gdf,
                search_radius_m=max_global_search,
                logger=self.logger,
            )

            result_df = pd.DataFrame(
                {
                    "poi_id": result.distances_m.index,
                    "nearest_building_distance_m": result.distances_m.replace(
                        np.inf, np.nan
                    ),
                    within_col: result.distances_m <= max_global_search,
                }
            )

            found_count = result_df["nearest_building_distance_m"].notna().sum()
            self.logger.info(
                f"Single file search complete. Found {found_count}/{len(result_df)} POIs."
            )

            return self._update_view(result_df, overwrite=overwrite)

        self.logger.info(
            f"Partitioned country: progressive expansion up to {max_global_search}m"
        )

        radii = [350, 1000, 2500, 5000, max_global_search]
        points_gdf = self.points_gdf
        remaining_poi_ids = set(points_gdf.poi_id)
        global_results = pd.Series(np.inf, index=points_gdf.poi_id, dtype=float)
        processed_tiles = set()

        for radius in radii:
            if not remaining_poi_ids:
                self.logger.info("All POIs found buildings. Stopping early.")
                break

            self.logger.info(
                f"Searching radius {radius}m for {len(remaining_poi_ids)} POIs"
            )

            jobs = GoogleMSBuildingsEngine.create_partitioned_jobs_for_pois(
                points_gdf,
                building_files,
                search_radius_m=radius,
            )

            new_jobs = [(f, pois) for f, pois in jobs if f not in processed_tiles]

            if not new_jobs:
                self.logger.debug(f"No new tiles at radius {radius}m")
                continue

            temp_result = GoogleMSBuildingsEngine.nearest_buildings_to_pois(
                handler=handler,
                building_files=[f for f, _ in new_jobs],
                pois_gdf=points_gdf,
                search_radius_m=radius,
                logger=self.logger,
            )
            temp_min_dists = temp_result.distances_m

            found_in_this_iteration = set()

            for poi_id in remaining_poi_ids:
                dist = temp_min_dists[poi_id]
                if dist < global_results[poi_id]:
                    global_results[poi_id] = dist
                    if dist <= radius:
                        found_in_this_iteration.add(poi_id)

            remaining_poi_ids -= found_in_this_iteration
            processed_tiles.update(f for f, _ in new_jobs)

            self.logger.info(
                f"Found {len(found_in_this_iteration)} POIs at radius {radius}m. "
                f"{len(remaining_poi_ids)} remaining."
            )

        result_df = pd.DataFrame(
            {
                "poi_id": global_results.index,
                "nearest_building_distance_m": global_results.replace(np.inf, np.nan),
                within_col: global_results <= max_global_search,
            }
        )

        found_count = result_df["nearest_building_distance_m"].notna().sum()
        self.logger.info(
            f"Global search complete. Found buildings for {found_count}/{len(result_df)} POIs."
        )

        return self._update_view(result_df, overwrite=overwrite)
