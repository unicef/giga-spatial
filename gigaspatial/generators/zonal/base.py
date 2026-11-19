import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import Field
from shapely.geometry import MultiPolygon, Polygon

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.generators._base import BaseViewGenerator, BaseViewGeneratorConfig
from gigaspatial.handlers.google_open_buildings import GoogleOpenBuildingsHandler
from gigaspatial.handlers.microsoft_global_buildings import MSBuildingsHandler
from gigaspatial.processing.buildings_engine import GoogleMSBuildingsEngine
from gigaspatial.processing.geo import add_area_in_meters, get_centroids

if TYPE_CHECKING:
    from gigaspatial.handlers.google_ms_combined_buildings import (
        GoogleMSBuildingsHandler,
    )


class ZonalViewGeneratorConfig(BaseViewGeneratorConfig):
    """Configuration for zonal view generation.

    .. deprecated::
        Pass ``ensure_available=`` directly to :class:`ZonalViewGenerator` and
        a target file path directly to :meth:`ZonalViewGenerator.save_view`.

    Attributes:
        base_path (Path): Base directory path for storing zonal views. Defaults to
            configured zonal views path.
        output_format (str): Default output format for saved views. Defaults to "parquet".
    """

    base_path: Path = Field(default=global_config.get_path("zonal", "views"))
    output_format: str = "parquet"


T = TypeVar("T")  # For zone type


class ZonalViewGenerator(BaseViewGenerator, Generic[T]):
    """Generates enriched zonal views over polygonal zones.

    Handles zones defined as either a mapping of zone identifiers to geometries
    or a GeoDataFrame with an identifier column, and supports point, polygon,
    raster, and dataset enrichment operations (`map_wp_pop`, `map_built_s`,
    `map_smod`, `map_ghsl_pop`, `map_ghsl`, `map_buildings`,
    `map_google_buildings`, `map_ms_buildings`, `map_rwi`,
    `map_aqueduct_flood_hazard`).

    Attributes:
        zone_dict (Dict[T, Polygon]): Mapping of zone identifiers to geometries.
        zone_id_column (str): Name of the source column containing zone identifiers.
        zone_data_crs (str): Coordinate reference system of the zone data.
        _zone_gdf (gpd.GeoDataFrame): Internal GeoDataFrame representation of zones.
        data_store (DataStore): The data store for accessing input data.
        config (ZonalViewGeneratorConfig): Configuration for the generator.
        logger: Logger instance for this class.
    """

    id_column = "zone_id"
    _default_config_cls = ZonalViewGeneratorConfig
    _is_point_generator = False
    _coverage_label = "zones"

    def __init__(
        self,
        zone_data: Union[Dict[T, Polygon], gpd.GeoDataFrame],
        zone_id_column: str = "zone_id",
        zone_data_crs: str = "EPSG:4326",
        config: Optional[ZonalViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        ensure_available: bool = True,
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
            config (ZonalViewGeneratorConfig, optional): Deprecated configuration object.
            data_store (DataStore, optional): Data store for accessing input data.
                If None, uses LocalDataStore.
            logger (logging.Logger, optional): Custom logger instance.
            ensure_available (bool): If True, ensures datasets are downloaded locally
                during mapping unless overridden per call.

        Raises:
            TypeError: If ``zone_data`` is not a dictionary or GeoDataFrame, or if
                dictionary values are not Polygon/MultiPolygon geometries.
            ValueError: If ``zone_id_column`` is missing, contains duplicate values,
                collides with an existing ``'zone_id'`` column, or if
                ``zone_data_crs`` does not match the GeoDataFrame's CRS.
        """
        self.zone_id_column = zone_id_column
        self.zone_data_crs = zone_data_crs

        if isinstance(zone_data, dict):
            for zone_id, geom in zone_data.items():
                if not isinstance(geom, (Polygon, MultiPolygon)):
                    raise TypeError(
                        f"Zone {zone_id}: Expected (Multi)Polygon, got {type(geom).__name__}"
                    )

            self.zone_dict = zone_data
            zone_gdf = gpd.GeoDataFrame(
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

            zone_gdf = zone_data.rename(columns={zone_id_column: "zone_id"}).copy()
            if zone_gdf.active_geometry_name != "geometry":
                zone_gdf = zone_gdf.rename_geometry("geometry")
            self.zone_dict = dict(zip(zone_data[zone_id_column], zone_data.geometry))

        super().__init__(
            base_gdf=zone_gdf,
            ensure_available=ensure_available,
            config=config,
            data_store=data_store,
            logger=logger,
        )
        self._zone_gdf = self._base_gdf

    def get_zonal_geometries(self) -> List[Polygon]:
        """Get the geometry of each zone."""
        return self._base_gdf.geometry.tolist()

    def get_zone_identifiers(self) -> List[T]:
        """Get the identifier for each zone."""
        return self._base_gdf.zone_id.tolist()

    def get_zone_geodataframe(self) -> gpd.GeoDataFrame:
        """Return a copy of the canonical GeoDataFrame with ``'zone_id'`` and ``'geometry'``."""
        return self._base_gdf.copy()

    @property
    def zone_gdf(self) -> gpd.GeoDataFrame:
        """A copy of the canonical GeoDataFrame of zone identifiers and geometries."""
        return self._base_gdf.copy()

    def map_google_buildings(
        self,
        handler: Optional[GoogleOpenBuildingsHandler] = None,
        use_polygons: bool = False,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Map Google Open Buildings data to zones and update ``self.view``.

        Adds ``'google_buildings_count'`` and ``'google_buildings_area_in_meters'``.
        When no buildings are found, populates ``0`` for counts and ``np.nan`` for area
        and returns ``self.view``.
        """
        out_cols = ["google_buildings_count", "google_buildings_area_in_meters"]
        self._validate_output_columns(out_cols, overwrite=overwrite)
        eff_ensure = self._resolve_ensure_available(ensure_available)

        self.logger.info(
            f"Mapping Google Open Buildings data (use_polygons={use_polygons})"
        )
        self.logger.info("Loading Google Buildings point data")
        handler = handler or GoogleOpenBuildingsHandler(data_store=self.data_store)
        buildings_df = handler.load_points(
            self.zone_gdf, ensure_available=eff_ensure, **kwargs
        )

        if buildings_df is None or buildings_df.empty:
            self.logger.warning("No Google buildings data found for the provided zones")
            return self._fill_empty_columns(
                {
                    "google_buildings_count": 0,
                    "google_buildings_area_in_meters": np.nan,
                },
                overwrite=overwrite,
            )

        if not use_polygons:
            self.logger.info("Aggregating building data using points with attributes")
            agg = self._aggregate_points(
                points=buildings_df,
                value_columns=["full_plus_code", "area_in_meters"],
                aggregation={"full_plus_code": "count", "area_in_meters": "sum"},
                predicate="within",
            )
            combined = agg.rename(
                columns={
                    "full_plus_code": "google_buildings_count",
                    "area_in_meters": "google_buildings_area_in_meters",
                }
            )
        else:
            self.logger.info(
                "Loading Google Buildings polygon data for more accurate mapping"
            )
            buildings_gdf = handler.load_polygons(
                self.zone_gdf, ensure_available=eff_ensure, **kwargs
            )
            self.logger.info("Counting buildings using points data")
            count_df = self._aggregate_points(
                points=buildings_df,
                predicate="within",
                output_column="google_buildings_count",
            )
            if buildings_gdf is None or buildings_gdf.empty:
                area_df = pd.DataFrame(
                    {
                        "zone_id": self._base_gdf["zone_id"].to_numpy(),
                        "google_buildings_area_in_meters": np.full(
                            len(self._base_gdf), np.nan, dtype=float
                        ),
                    }
                )
            else:
                self.logger.info(
                    "Calculating building areas with area-weighted aggregation"
                )
                area_df = self._aggregate_polygons(
                    polygons=buildings_gdf,
                    value_columns="area_in_meters",
                    aggregation="sum",
                    predicate="fractional",
                    output_column="google_buildings_area_in_meters",
                )
            combined = count_df.merge(area_df, on="zone_id", how="left")

        return self._update_view(combined, overwrite=overwrite)

    def map_ms_buildings(
        self,
        handler: Optional[MSBuildingsHandler] = None,
        use_polygons: bool = False,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Map Microsoft Global Buildings data to zones and update ``self.view``.

        Adds ``'ms_buildings_count'`` and ``'ms_buildings_area_in_meters'``.
        When no buildings are found, populates ``0`` for counts and ``np.nan`` for area
        and returns ``self.view``.
        """
        out_cols = ["ms_buildings_count", "ms_buildings_area_in_meters"]
        self._validate_output_columns(out_cols, overwrite=overwrite)
        eff_ensure = self._resolve_ensure_available(ensure_available)

        self.logger.info("Mapping Microsoft Global Buildings data")
        self.logger.info("Loading Microsoft Buildings polygon data")
        handler = handler or MSBuildingsHandler(data_store=self.data_store)
        buildings_gdf = handler.load_data(self.zone_gdf, ensure_available=eff_ensure)

        if buildings_gdf is None or buildings_gdf.empty:
            self.logger.warning(
                "No Microsoft buildings data found for the provided zones"
            )
            return self._fill_empty_columns(
                {
                    "ms_buildings_count": 0,
                    "ms_buildings_area_in_meters": np.nan,
                },
                overwrite=overwrite,
            )

        buildings_gdf = add_area_in_meters(
            buildings_gdf.copy(), area_column_name="area_in_meters"
        )
        building_centroids = get_centroids(buildings_gdf)

        if not use_polygons:
            self.logger.info("Aggregating building data using points with attributes")
            count_df = self._aggregate_points(
                points=building_centroids,
                predicate="within",
                output_column="ms_buildings_count",
            )
            area_df = self._aggregate_points(
                points=building_centroids,
                value_columns="area_in_meters",
                aggregation="sum",
                predicate="within",
                output_column="ms_buildings_area_in_meters",
            )
            combined = count_df.merge(area_df, on="zone_id", how="left")
        else:
            self.logger.info(
                "Calculating building areas with area-weighted aggregation"
            )
            area_df = self._aggregate_polygons(
                polygons=buildings_gdf,
                value_columns="area_in_meters",
                aggregation="sum",
                predicate="fractional",
                output_column="ms_buildings_area_in_meters",
            )
            self.logger.info("Counting Microsoft buildings per zone")
            count_df = self._aggregate_points(
                points=building_centroids,
                predicate="within",
                output_column="ms_buildings_count",
            )
            combined = count_df.merge(area_df, on="zone_id", how="left")

        return self._update_view(combined, overwrite=overwrite)

    def map_buildings(
        self,
        country: Optional[str] = None,
        output_column: str = "building_count",
        handler: Optional["GoogleMSBuildingsHandler"] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Map Google-Microsoft combined buildings data to zones and update ``self.view``."""
        self._validate_output_columns([output_column], overwrite=overwrite)
        country = self._resolve_country(country)
        eff_ensure = self._resolve_ensure_available(ensure_available)
        self.logger.info("Mapping Google-Microsoft Buildings data to zones")

        if handler is None:
            from gigaspatial.handlers.google_ms_combined_buildings import (
                GoogleMSBuildingsHandler,
            )

            handler = GoogleMSBuildingsHandler(
                partition_strategy="s2_grid", data_store=self.data_store
            )

        if eff_ensure:
            if not handler.ensure_data_available(country, **kwargs):
                raise RuntimeError("Could not ensure data availability for loading")

        building_files = handler.reader.resolve_source_paths(
            country, **handler.select_options(kwargs, handler.config)
        )
        if not building_files:
            self.logger.warning(
                "No Google/Microsoft building files found for %s; filling zero counts.",
                country,
            )
            return self._fill_empty_columns({output_column: 0}, overwrite=overwrite)

        result = GoogleMSBuildingsEngine.count_buildings_in_zones(
            handler=handler,
            building_files=building_files,
            zones_gdf=self.zone_gdf,
            logger=self.logger,
        )

        total_buildings = result.counts.sum()
        zones_with_buildings = (result.counts > 0).sum()
        self.logger.info(
            f"Mapping complete: {total_buildings:,.0f} buildings across "
            f"{zones_with_buildings}/{len(result.counts)} zones"
        )

        self.add_variable_to_view(result.counts, output_column, overwrite=overwrite)
        return self.view
