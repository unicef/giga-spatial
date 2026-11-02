from typing import TYPE_CHECKING

from gigaspatial._lazy import attach

__getattr__, __dir__, __all__ = attach(
    __name__,
    {
        "gigaspatial.fetchers.osm.pois": ["OSMLocationFetcher"],
        "gigaspatial.fetchers.osm.roads": ["OSMRoadsFetcher"],
        "gigaspatial.fetchers.overture": ["OvertureAmenityFetcher"],
        "gigaspatial.fetchers.healthsites": ["HealthSitesFetcher"],
        "gigaspatial.fetchers.giga.school_locations": ["GigaSchoolLocationFetcher"],
        "gigaspatial.fetchers.giga.school_profile": ["GigaSchoolProfileFetcher"],
        "gigaspatial.fetchers.giga.school_measurements": [
            "GigaSchoolMeasurementsFetcher"
        ],
        "gigaspatial.fetchers.giga.countries": ["GigaCountriesFetcher"],
        "gigaspatial.fetchers.unicef.sdg": ["UnicefDataFetcher"],
        "gigaspatial.fetchers.mlab": ["MLabConfig", "MLabFetcher"],
        "gigaspatial.fetchers.gee.profiler": ["GEEProfiler"],
        "gigaspatial.fetchers.gee.config": ["GEEConfig"],
        "gigaspatial.fetchers.gee.registry": ["GEEDatasetRegistry"],
        "gigaspatial.fetchers.gee.datasets.base": ["GEEDatasetEntry"],
    },
)

if TYPE_CHECKING:
    from gigaspatial.fetchers.osm.pois import OSMLocationFetcher
    from gigaspatial.fetchers.osm.roads import OSMRoadsFetcher
    from gigaspatial.fetchers.overture import OvertureAmenityFetcher
    from gigaspatial.fetchers.healthsites import HealthSitesFetcher
    from gigaspatial.fetchers.giga.school_locations import GigaSchoolLocationFetcher
    from gigaspatial.fetchers.giga.school_profile import GigaSchoolProfileFetcher
    from gigaspatial.fetchers.giga.school_measurements import (
        GigaSchoolMeasurementsFetcher,
    )
    from gigaspatial.fetchers.giga.countries import GigaCountriesFetcher
    from gigaspatial.fetchers.unicef.sdg import UnicefDataFetcher
    from gigaspatial.fetchers.mlab import MLabConfig, MLabFetcher
    from gigaspatial.fetchers.gee.profiler import GEEProfiler
    from gigaspatial.fetchers.gee.config import GEEConfig
    from gigaspatial.fetchers.gee.registry import GEEDatasetRegistry
    from gigaspatial.fetchers.gee.datasets.base import GEEDatasetEntry
