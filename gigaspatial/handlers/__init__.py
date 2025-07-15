from gigaspatial.handlers.boundaries import AdminBoundaries
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
from gigaspatial.handlers.osm import OSMLocationFetcher
from gigaspatial.handlers.overture import OvertureAmenityFetcher
from gigaspatial.handlers.mapbox_image import MapboxImageDownloader
from gigaspatial.handlers.maxar_image import MaxarConfig, MaxarImageDownloader

from gigaspatial.handlers.worldpop import (
    WPPopulationConfig,
    WPPopulationReader,
    WPPopulationDownloader,
    WPPopulationHandler,
    WorldPopRestClient,
)
from gigaspatial.handlers.ookla_speedtest import (
    OoklaSpeedtestTileConfig,
    OoklaSpeedtestConfig,
)
from gigaspatial.handlers.opencellid import (
    OpenCellIDConfig,
    OpenCellIDDownloader,
    OpenCellIDReader,
)
from gigaspatial.handlers.hdx import HDXConfig, HDXDownloader, HDXReader, HDXHandler
from gigaspatial.handlers.rwi import RWIConfig, RWIDownloader, RWIReader, RWIHandler
from gigaspatial.handlers.unicef_georepo import (
    GeoRepoClient,
    get_country_boundaries_by_iso3,
)
from gigaspatial.handlers.giga import (
    GigaSchoolLocationFetcher,
    GigaSchoolProfileFetcher,
    GigaSchoolMeasurementsFetcher,
)
