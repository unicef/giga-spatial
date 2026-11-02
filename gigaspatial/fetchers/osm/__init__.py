"""
OpenStreetMap data from the Overpass API.

- ``pois``: ``OSMLocationFetcher``, points of interest such as clinics and schools, as records.
- ``roads``: ``OSMRoadsFetcher``, the road network of an area, through osmnx.

For whole countries, ``gigaspatial.handlers.osm`` builds road networks from
Geofabrik's extracts instead.
"""

from typing import TYPE_CHECKING

from gigaspatial._lazy import attach

__getattr__, __dir__, __all__ = attach(
    __name__,
    {
        "gigaspatial.fetchers.osm.pois": ["OSMLocationFetcher"],
        "gigaspatial.fetchers.osm.roads": ["OSMRoadsFetcher"],
    },
)

if TYPE_CHECKING:
    from gigaspatial.fetchers.osm.pois import OSMLocationFetcher
    from gigaspatial.fetchers.osm.roads import OSMRoadsFetcher
