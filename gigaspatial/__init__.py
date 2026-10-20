__version__ = "0.9.9"

from typing import TYPE_CHECKING

from gigaspatial._lazy import attach

__getattr__, __dir__, _exports = attach(
    __name__,
    {
        # data stores and clients
        "gigaspatial.core.io.local_data_store": ["LocalDataStore"],
        "gigaspatial.core.io.adls_data_store": ["ADLSDataStore"],
        "gigaspatial.core.io.snowflake_data_store": ["SnowflakeDataStore"],
        "gigaspatial.core.io.delta_sharing_data_store": ["DeltaSharingDataStore"],
        "gigaspatial.core.io.database": ["DBConnection"],
        "gigaspatial.core.io.bigquery_client": [
            "BigQueryClient",
            "BigQueryClientConfig",
        ],
        "gigaspatial.core.io.readers": ["read_dataset"],
        "gigaspatial.core.io.writers": ["write_dataset"],
        # handlers
        "gigaspatial.handlers.boundaries": ["AdminBoundaries"],
        "gigaspatial.handlers.ghsl": ["GHSLDataHandler"],
        "gigaspatial.handlers.google_open_buildings": ["GoogleOpenBuildingsHandler"],
        "gigaspatial.handlers.microsoft_global_buildings": ["MSBuildingsHandler"],
        "gigaspatial.handlers.google_ms_combined_buildings": [
            "GoogleMSBuildingsHandler"
        ],
        "gigaspatial.handlers.worldpop": ["WPPopulationHandler"],
        "gigaspatial.handlers.ookla_speedtest": ["OoklaSpeedtestHandler"],
        "gigaspatial.handlers.opencellid": ["OpenCellIDHandler"],
        "gigaspatial.handlers.hdx": ["HDXHandler"],
        "gigaspatial.handlers.rwi": ["RWIHandler"],
        "gigaspatial.handlers.nasa.srtm": ["NasaSRTMHandler"],
        "gigaspatial.handlers.wri.aqueduct_flood": ["AqueductFloodHandler"],
        # fetchers
        "gigaspatial.fetchers.osm": ["OSMLocationFetcher"],
        "gigaspatial.fetchers.overture": ["OvertureAmenityFetcher"],
        "gigaspatial.fetchers.healthsites": ["HealthSitesFetcher"],
        "gigaspatial.fetchers.giga.school_locations": ["GigaSchoolLocationFetcher"],
        "gigaspatial.fetchers.giga.school_profile": ["GigaSchoolProfileFetcher"],
        "gigaspatial.fetchers.giga.school_measurements": [
            "GigaSchoolMeasurementsFetcher"
        ],
        "gigaspatial.fetchers.giga.countries": ["GigaCountriesFetcher"],
        "gigaspatial.fetchers.unicef.sdg": ["UnicefDataFetcher"],
        "gigaspatial.fetchers.mlab": ["MLabFetcher"],
        "gigaspatial.fetchers.gee.profiler": ["GEEProfiler"],
        # image downloaders
        "gigaspatial.image_downloaders.mapbox_image": ["MapboxImageDownloader"],
        "gigaspatial.image_downloaders.maxar_image": ["MaxarImageDownloader"],
        # generators
        "gigaspatial.generators.poi": ["PoiViewGenerator"],
        "gigaspatial.generators.zonal.geometry": ["GeometryBasedZonalViewGenerator"],
        "gigaspatial.generators.zonal.admin": ["AdminBoundariesViewGenerator"],
        "gigaspatial.generators.zonal.mercator": ["MercatorViewGenerator"],
        "gigaspatial.generators.zonal.h3": ["H3ViewGenerator"],
        "gigaspatial.generators.zonal.s2": ["S2ViewGenerator"],
        # grids
        "gigaspatial.grid.mercator_tiles": ["MercatorTiles"],
        "gigaspatial.grid.h3": ["H3Hexagons"],
        "gigaspatial.grid.s2": ["S2Cells"],
        # processing
        "gigaspatial.processing.tif_processor": ["TifProcessor"],
        "gigaspatial.processing.geo": [
            "convert_to_geodataframe",
            "buffer_geodataframe",
            "aggregate_points_to_zones",
            "aggregate_polygons_to_zones",
            "calculate_distance",
            "map_points_within_polygons",
        ],
    },
    submodules=[
        "core",
        "fetchers",
        "generators",
        "grid",
        "handlers",
        "image_downloaders",
        "processing",
    ],
)

__all__ = ["__version__", *_exports]

if TYPE_CHECKING:
    from gigaspatial.core.io.local_data_store import LocalDataStore
    from gigaspatial.core.io.adls_data_store import ADLSDataStore
    from gigaspatial.core.io.snowflake_data_store import SnowflakeDataStore
    from gigaspatial.core.io.delta_sharing_data_store import DeltaSharingDataStore
    from gigaspatial.core.io.database import DBConnection
    from gigaspatial.core.io.bigquery_client import BigQueryClient, BigQueryClientConfig
    from gigaspatial.core.io.readers import read_dataset
    from gigaspatial.core.io.writers import write_dataset
    from gigaspatial.handlers.boundaries import AdminBoundaries
    from gigaspatial.handlers.ghsl import GHSLDataHandler
    from gigaspatial.handlers.google_open_buildings import GoogleOpenBuildingsHandler
    from gigaspatial.handlers.microsoft_global_buildings import MSBuildingsHandler
    from gigaspatial.handlers.google_ms_combined_buildings import (
        GoogleMSBuildingsHandler,
    )
    from gigaspatial.handlers.worldpop import WPPopulationHandler
    from gigaspatial.handlers.ookla_speedtest import OoklaSpeedtestHandler
    from gigaspatial.handlers.opencellid import OpenCellIDHandler
    from gigaspatial.handlers.hdx import HDXHandler
    from gigaspatial.handlers.rwi import RWIHandler
    from gigaspatial.handlers.nasa.srtm import NasaSRTMHandler
    from gigaspatial.handlers.wri.aqueduct_flood import AqueductFloodHandler
    from gigaspatial.fetchers.osm import OSMLocationFetcher
    from gigaspatial.fetchers.overture import OvertureAmenityFetcher
    from gigaspatial.fetchers.healthsites import HealthSitesFetcher
    from gigaspatial.fetchers.giga.school_locations import GigaSchoolLocationFetcher
    from gigaspatial.fetchers.giga.school_profile import GigaSchoolProfileFetcher
    from gigaspatial.fetchers.giga.school_measurements import (
        GigaSchoolMeasurementsFetcher,
    )
    from gigaspatial.fetchers.giga.countries import GigaCountriesFetcher
    from gigaspatial.fetchers.unicef.sdg import UnicefDataFetcher
    from gigaspatial.fetchers.mlab import MLabFetcher
    from gigaspatial.fetchers.gee.profiler import GEEProfiler
    from gigaspatial.image_downloaders.mapbox_image import MapboxImageDownloader
    from gigaspatial.image_downloaders.maxar_image import MaxarImageDownloader
    from gigaspatial.generators.poi import PoiViewGenerator
    from gigaspatial.generators.zonal.geometry import GeometryBasedZonalViewGenerator
    from gigaspatial.generators.zonal.admin import AdminBoundariesViewGenerator
    from gigaspatial.generators.zonal.mercator import MercatorViewGenerator
    from gigaspatial.generators.zonal.h3 import H3ViewGenerator
    from gigaspatial.generators.zonal.s2 import S2ViewGenerator
    from gigaspatial.grid.mercator_tiles import MercatorTiles
    from gigaspatial.grid.h3 import H3Hexagons
    from gigaspatial.grid.s2 import S2Cells
    from gigaspatial.processing.tif_processor import TifProcessor
    from gigaspatial.processing.geo import (
        convert_to_geodataframe,
        buffer_geodataframe,
        aggregate_points_to_zones,
        aggregate_polygons_to_zones,
        calculate_distance,
        map_points_within_polygons,
    )
