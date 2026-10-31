"""OpenStreetMap road networks, fetched from Overpass through osmnx."""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Union

import geopandas as gpd
import shapely
from pydantic.dataclasses import dataclass
from pyproj import Geod

from gigaspatial.config import config
from gigaspatial.handlers.base.config import source_geometry
from gigaspatial.handlers.base.source import SourceKind, classify_source
from gigaspatial.processing.geo import estimate_utm_crs_with_fallback
from gigaspatial.processing.network import SpatialNetwork
from gigaspatial.processing.network.attributes import normalize_attributes

# Above this, Overpass is slow and its fair-use limits apply.
LARGE_AREA_KM2 = 50_000
_GEOD = Geod(ellps="WGS84")


@dataclass(kw_only=True)
class OSMRoadsFetcher:
    """Fetch the road network of a source from OpenStreetMap, through osmnx and the Overpass API.

    The source is turned into an area to query: a country's boundary, a
    geometry as given, the union of a frame's polygons, or for points the
    convex hull (``area="hull"``), bounding box (``"bbox"``), or union of
    buffers (``"buffers"``), buffered by ``buffer`` meters. The result is a
    directed :class:`SpatialNetwork`, with list-valued OSM attributes
    normalised so it can be saved; the fetcher saves nothing itself.

    Needs ``pip install 'giga-spatial[roads]'``.

    Attributes:
        network_type: osmnx network type, such as ``"drive_service"``,
            ``"drive"``, ``"walk"``, ``"bike"``, or ``"all"``.
        custom_filter: Overpass way filter replacing ``network_type``, such
            as ``'["highway"~"primary|secondary"]'``.
        simplify: Join edges between junctions, as osmnx does.
        retain_all: Keep every disconnected part, not only the largest.
        truncate_by_edge: Keep edges that leave the area at one end.
        buffer: Meters around point and line sources.
        area: How points become an area: ``"hull"``, ``"bbox"``, or ``"buffers"``.
        overpass_url: Overpass endpoint, such as a mirror when the main server
            is down. The ``OVERPASS_URL`` setting otherwise.
        timeout: Seconds for each Overpass request.
        max_retries: Retries after a failed request, with growing pauses.
        retry_delay: Seconds before the first retry.
    """

    network_type: str = "drive_service"
    custom_filter: Optional[Union[str, List[str]]] = None
    simplify: bool = True
    retain_all: bool = False
    truncate_by_edge: bool = False
    buffer: float = 2000.0
    area: Literal["hull", "bbox", "buffers"] = "hull"
    overpass_url: Optional[str] = None
    timeout: int = 180
    max_retries: int = 3
    retry_delay: float = 5.0

    def __post_init__(self):
        if self.buffer < 0:
            raise ValueError("buffer must not be negative")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.logger = config.get_logger(self.__class__.__name__)

    def fetch(self, source: Any) -> SpatialNetwork:
        """The road network of a source.

        Args:
            source: Country name or code, shapely geometry, GeoDataFrame,
                DataFrame with latitude and longitude columns, or list of
                ``(lat, lon)`` tuples or Points.

        Returns:
            A directed network in EPSG:4326 with lengths in meters. Its
            metadata records the source, the area, and the fetch parameters.

        Raises:
            ImportError: If osmnx is not installed.
            ValueError: If the source has no area, or OSM has no matching
                roads in it.
            RuntimeError: If Overpass keeps failing after the retries.
        """
        area = self.query_area(source)
        area_km2 = abs(_GEOD.geometry_area_perimeter(area)[0]) / 1e6
        if area_km2 > LARGE_AREA_KM2:
            self.logger.warning(
                f"The area is {area_km2:,.0f} km2. Overpass is slow at this size and has fair-use limits; "
                "expect a long download, or fetch smaller parts and combine them with SpatialNetwork.concat."
            )
        graph = self._download(area)
        return self._network(graph, source, area, area_km2)

    def query_area(self, source: Any) -> Union[shapely.Polygon, shapely.MultiPolygon]:
        """The area queried for a source, in EPSG:4326."""
        return query_area(source, buffer=self.buffer, area=self.area, logger=self.logger)

    def _download(self, area: Any):
        try:
            import osmnx as ox
        except ImportError as error:
            raise ImportError(
                "OSMRoadsFetcher requires 'osmnx'. Install it with: pip install 'giga-spatial[roads]'"
            ) from error
        import requests

        errors = getattr(ox, "_errors", None)
        no_data = getattr(errors, "InsufficientResponseError", None)
        status = getattr(errors, "ResponseStatusCodeError", None)
        retryable = tuple(e for e in (requests.exceptions.RequestException, status) if e is not None)
        url = self.overpass_url or config.OVERPASS_URL
        with _osmnx_settings(ox, url, self.timeout):
            for attempt in range(self.max_retries + 1):
                try:
                    return ox.graph_from_polygon(
                        area,
                        network_type=self.network_type,
                        simplify=self.simplify,
                        retain_all=self.retain_all,
                        truncate_by_edge=self.truncate_by_edge,
                        custom_filter=self.custom_filter,
                    )
                except Exception as error:
                    if no_data is not None and isinstance(error, no_data):
                        raise ValueError(
                            f"OpenStreetMap has no {self.custom_filter or self.network_type} roads in the area"
                        ) from error
                    if not isinstance(error, retryable):
                        raise
                    if attempt == self.max_retries:
                        raise RuntimeError(
                            f"Overpass at {url} failed {attempt + 1} times: {error}. If the server is down, "
                            "set overpass_url or the OVERPASS_URL setting to a mirror, such as "
                            "https://overpass.kumi.systems/api/interpreter"
                        ) from error
                    pause = self.retry_delay * 2**attempt
                    self.logger.warning(f"Overpass request failed ({error}); retrying in {pause:.0f} s")
                    time.sleep(pause)

    def _network(self, graph: Any, source: Any, area: Any, area_km2: float) -> SpatialNetwork:
        network = SpatialNetwork.from_networkx(graph)
        edges = normalize_attributes(
            network._edges, int_lists=("osmid",), skip=("u", "v", "key", "length", "kind", "geometry")
        )
        nodes = normalize_attributes(network._nodes, skip=("x", "y", "kind"))
        metadata = {
            **network.metadata,
            "source": _describe(source),
            "area_bounds": [float(v) for v in area.bounds],
            "area_km2": round(area_km2, 3),
            "fetch": {
                "fetcher": "OSMRoadsFetcher",
                "network_type": self.network_type,
                "custom_filter": self.custom_filter,
                "simplify": self.simplify,
                "retain_all": self.retain_all,
                "truncate_by_edge": self.truncate_by_edge,
                "buffer": self.buffer,
                "area": self.area,
                "overpass_url": self.overpass_url or config.OVERPASS_URL,
            },
        }
        return SpatialNetwork._from_tables(
            nodes,
            edges,
            directed=network.directed,
            weight=network.weight,
            metadata=metadata,
            positions=network._edge_positions,
        )


def query_area(
    source: Any, buffer: float = 2000.0, area: str = "hull", logger: Any = None
) -> Union[shapely.Polygon, shapely.MultiPolygon]:
    """The road area of a source in EPSG:4326, shared by ``OSMRoadsFetcher`` and ``OSMRoadsHandler``.

    A country is its boundary, a geometry is used as given, and a frame of
    polygons is their union. Points and lines become their convex hull
    (``area="hull"``), bounding box (``"bbox"``), or union of buffers
    (``"buffers"``), widened by ``buffer`` meters through UTM.

    Raises:
        ValueError: If the source has no area, or points or lines have no buffer.
    """
    kind = classify_source(source)
    if kind in (SourceKind.PATHS, SourceKind.OTHER):
        raise ValueError(f"Cannot fetch roads for a source of type {type(source).__name__}")
    geometry = source_geometry(source, "EPSG:4326")
    if not geometry.is_valid:
        geometry = shapely.make_valid(geometry)
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    parts = getattr(geometry, "geoms", [geometry])
    polygons = [g for g in parts if g.geom_type in ("Polygon", "MultiPolygon")]
    if polygons and len(polygons) == len(parts):
        return shapely.union_all(polygons)
    if area == "hull":
        shape = geometry.convex_hull
    elif area == "bbox":
        shape = shapely.box(*geometry.bounds)
    else:
        shape = geometry
    if buffer == 0 and shape.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("Points and lines need a buffer greater than 0 to make an area")
    series = gpd.GeoSeries([shape], crs="EPSG:4326")
    utm = estimate_utm_crs_with_fallback(gpd.GeoDataFrame(geometry=series), logger=logger)
    return series.to_crs(utm).buffer(buffer).to_crs("EPSG:4326").iloc[0]


@contextmanager
def _osmnx_settings(ox: Any, url: str, timeout: int):
    """osmnx settings for one fetch, restored afterwards. osmnx wants the API base, without /interpreter."""
    base = url.rstrip("/")
    if base.endswith("/interpreter"):
        base = base[: -len("/interpreter")]
    saved = {name: getattr(ox.settings, name) for name in ("overpass_url", "requests_timeout")}
    ox.settings.overpass_url = base
    ox.settings.requests_timeout = timeout
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(ox.settings, name, value)


def _describe(source: Any) -> Dict[str, Any]:
    kind = classify_source(source)
    if kind is SourceKind.COUNTRY:
        return {"kind": "country", "country": source}
    if kind is SourceKind.GEOMETRY:
        return {"kind": "geometry", "type": source.geom_type}
    if kind in (SourceKind.GEODATAFRAME, SourceKind.DATAFRAME):
        return {"kind": "frame", "rows": len(source)}
    return {"kind": "points", "count": len(list(source))}
