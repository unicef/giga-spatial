from typing import TYPE_CHECKING

from gigaspatial._lazy import attach

__getattr__, __dir__, __all__ = attach(
    __name__,
    {
        "gigaspatial.handlers.boundaries": ["AdminBoundaries"],
        "gigaspatial.handlers.georepo": ["GeoRepoClient"],
        "gigaspatial.handlers.ghsl": [
            "GHSLDataConfig",
            "GHSLDataDownloader",
            "GHSLDataReader",
            "GHSLDataHandler",
        ],
        "gigaspatial.handlers.google_open_buildings": [
            "GoogleOpenBuildingsConfig",
            "GoogleOpenBuildingsDownloader",
            "GoogleOpenBuildingsReader",
            "GoogleOpenBuildingsHandler",
        ],
        "gigaspatial.handlers.microsoft_global_buildings": [
            "MSBuildingsConfig",
            "MSBuildingsDownloader",
            "MSBuildingsReader",
            "MSBuildingsHandler",
        ],
        "gigaspatial.handlers.google_ms_combined_buildings": [
            "GoogleMSBuildingsConfig",
            "GoogleMSBuildingsDownloader",
            "GoogleMSBuildingsReader",
            "GoogleMSBuildingsHandler",
        ],
        "gigaspatial.handlers.worldpop": [
            "WPPopulationConfig",
            "WPPopulationDownloader",
            "WPPopulationReader",
            "WPPopulationHandler",
            "WorldPopRestClient",
        ],
        "gigaspatial.handlers.ookla_speedtest": [
            "OoklaSpeedtestConfig",
            "OoklaSpeedtestDownloader",
            "OoklaSpeedtestReader",
            "OoklaSpeedtestHandler",
        ],
        "gigaspatial.handlers.opencellid": [
            "OpenCellIDConfig",
            "OpenCellIDDownloader",
            "OpenCellIDReader",
            "OpenCellIDHandler",
        ],
        "gigaspatial.handlers.hdx": [
            "HDXConfig",
            "HDXDownloader",
            "HDXReader",
            "HDXHandler",
        ],
        "gigaspatial.handlers.rwi": [
            "RWIConfig",
            "RWIDownloader",
            "RWIReader",
            "RWIHandler",
        ],
        "gigaspatial.handlers.nasa.srtm": [
            "NasaSRTMConfig",
            "NasaSRTMDownloader",
            "NasaSRTMReader",
            "NasaSRTMHandler",
        ],
        "gigaspatial.handlers.wri.aqueduct_flood": [
            "AqueductFloodConfig",
            "AqueductFloodDownloader",
            "AqueductFloodReader",
            "AqueductFloodHandler",
        ],
        "gigaspatial.handlers.osm.roads": [
            "OSMRoadsConfig",
            "OSMRoadsDownloader",
            "OSMRoadsReader",
            "OSMRoadsHandler",
        ],
    },
)

if TYPE_CHECKING:
    from gigaspatial.handlers.boundaries import AdminBoundaries
    from gigaspatial.handlers.georepo import GeoRepoClient
    from gigaspatial.handlers.ghsl import (
        GHSLDataConfig,
        GHSLDataDownloader,
        GHSLDataReader,
        GHSLDataHandler,
    )
    from gigaspatial.handlers.google_open_buildings import (
        GoogleOpenBuildingsConfig,
        GoogleOpenBuildingsDownloader,
        GoogleOpenBuildingsReader,
        GoogleOpenBuildingsHandler,
    )
    from gigaspatial.handlers.microsoft_global_buildings import (
        MSBuildingsConfig,
        MSBuildingsDownloader,
        MSBuildingsReader,
        MSBuildingsHandler,
    )
    from gigaspatial.handlers.google_ms_combined_buildings import (
        GoogleMSBuildingsConfig,
        GoogleMSBuildingsDownloader,
        GoogleMSBuildingsReader,
        GoogleMSBuildingsHandler,
    )
    from gigaspatial.handlers.worldpop import (
        WPPopulationConfig,
        WPPopulationDownloader,
        WPPopulationReader,
        WPPopulationHandler,
        WorldPopRestClient,
    )
    from gigaspatial.handlers.ookla_speedtest import (
        OoklaSpeedtestConfig,
        OoklaSpeedtestDownloader,
        OoklaSpeedtestReader,
        OoklaSpeedtestHandler,
    )
    from gigaspatial.handlers.opencellid import (
        OpenCellIDConfig,
        OpenCellIDDownloader,
        OpenCellIDReader,
        OpenCellIDHandler,
    )
    from gigaspatial.handlers.hdx import HDXConfig, HDXDownloader, HDXReader, HDXHandler
    from gigaspatial.handlers.rwi import RWIConfig, RWIDownloader, RWIReader, RWIHandler
    from gigaspatial.handlers.nasa.srtm import (
        NasaSRTMConfig,
        NasaSRTMDownloader,
        NasaSRTMReader,
        NasaSRTMHandler,
    )
    from gigaspatial.handlers.wri.aqueduct_flood import (
        AqueductFloodConfig,
        AqueductFloodDownloader,
        AqueductFloodReader,
        AqueductFloodHandler,
    )
    from gigaspatial.handlers.osm.roads import (
        OSMRoadsConfig,
        OSMRoadsDownloader,
        OSMRoadsReader,
        OSMRoadsHandler,
    )
