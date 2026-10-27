"""Spatial networks: places as nodes with coordinates, joined by weighted edges."""


class SpatialNetwork:
    """A spatial network held as a nodes table and an edges table.

    Nodes carry coordinates in EPSG:4326, and edges carry a length in meters
    or another weight. A road network fetched by ``OSMRoadsFetcher`` is one
    kind; points joined to their nearest neighbours are another.
    """
