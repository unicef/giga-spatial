__version__ = "0.7.6"

import gigaspatial.core.io as io

from .handlers import (
    AdminBoundaries,
    GoogleOpenBuildingsHandler,
    MSBuildingsHandler,
    GoogleMSBuildingsHandler,
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
import gigaspatial.processing.algorithms as algorithms
import gigaspatial.processing.geo as geo_processing

__all__ = [
    "__version__",
    "io",
    "algorithms",
    # handlers
    "AdminBoundaries",
    "GoogleOpenBuildingsHandler",
    "MSBuildingsHandler",
    "GoogleMSBuildingsHandler",
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
    "geo_processing",
]
