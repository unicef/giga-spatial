__version__ = "0.8.0"

import gigaspatial.core.io as io
import gigaspatial.processing.algorithms as algorithms
import gigaspatial.processing.geo as geo

from .core.io import (
    ADLSDataStore,
    LocalDataStore,
    SnowflakeDataStore,
    DeltaSharingDataStore,
    DBConnection,
    read_dataset,
    write_dataset,
)

from .handlers import (
    AdminBoundaries,
    GoogleOpenBuildingsHandler,
    MSBuildingsHandler,
    GoogleMSBuildingsHandler,
    GEEProfiler,
    GHSLDataHandler,
    GigaSchoolLocationFetcher,
    GigaSchoolMeasurementsFetcher,
    GigaSchoolProfileFetcher,
    OoklaSpeedtestHandler,
    OSMLocationFetcher,
    HealthSitesFetcher,
    OvertureAmenityFetcher,
    HDXHandler,
    WPPopulationHandler,
    RWIHandler,
)

from .generators import (
    GeometryBasedZonalViewGenerator,
    PoiViewGenerator,
    MercatorViewGenerator,
    H3ViewGenerator,
    S2ViewGenerator,
)

from .grid import MercatorTiles, H3Hexagons, S2Cells

from .processing import TifProcessor
from .processing.geo import (
    convert_to_geodataframe,
    buffer_geodataframe,
    aggregate_points_to_zones,
    aggregate_polygons_to_zones,
    calculate_distance,
    map_points_within_polygons,
)

__all__ = [
    "__version__",
    # io
    "io",
    "ADLSDataStore",
    "LocalDataStore",
    "SnowflakeDataStore",
    "DeltaSharingDataStore",
    "DBConnection",
    "read_dataset",
    "write_dataset",
    "algorithms",
    # handlers
    "AdminBoundaries",
    "GoogleOpenBuildingsHandler",
    "MSBuildingsHandler",
    "GoogleMSBuildingsHandler",
    "GEEProfiler",
    "GHSLDataHandler",
    "GigaSchoolLocationFetcher",
    "GigaSchoolMeasurementsFetcher",
    "GigaSchoolProfileFetcher",
    "OoklaSpeedtestHandler",
    "OSMLocationFetcher",
    "HealthSitesFetcher",
    "OvertureAmenityFetcher",
    "HDXHandler",
    "WPPopulationHandler",
    "RWIHandler",
    # generators
    "GeometryBasedZonalViewGenerator",
    "PoiViewGenerator",
    "MercatorViewGenerator",
    "H3ViewGenerator",
    "S2ViewGenerator",
    # grids
    "MercatorTiles",
    "H3Hexagons",
    "S2Cells",
    # processing
    "TifProcessor",
    "geo",
    "convert_to_geodataframe",
    "buffer_geodataframe",
    "aggregate_points_to_zones",
    "aggregate_polygons_to_zones",
    "calculate_distance",
    "map_points_within_polygons",
]
