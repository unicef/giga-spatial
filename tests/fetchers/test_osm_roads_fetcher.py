import math
import sys
import types
from unittest.mock import MagicMock

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
import requests
from shapely.geometry import LineString, Point, box

import gigaspatial.fetchers.osm_roads as osm_roads
from gigaspatial import OSMRoadsFetcher, SpatialNetwork
from gigaspatial.config import config

UTM_36S = "EPSG:32736"


class InsufficientResponseError(ValueError):
    pass


class ResponseStatusCodeError(ValueError):
    pass


def osmnx_graph():
    """A small simplified graph as osmnx returns it, with list-valued attributes."""
    G = nx.MultiDiGraph(crs="EPSG:4326", created_with="OSMnx 2.0.2", simplified=True)
    G.add_node(1, x=30.06, y=-1.95, street_count=3)
    G.add_node(2, x=30.061, y=-1.95, street_count=1)
    G.add_node(3, x=30.061, y=-1.949, street_count=1)
    G.add_edge(1, 2, osmid=[11, 12], highway=["primary", "secondary"], oneway=False, reversed=[False, True], length=111.0)
    G.add_edge(2, 1, osmid=[11, 12], highway=["primary", "secondary"], oneway=False, reversed=[True, False], length=111.0)
    G.add_edge(2, 3, osmid=13, highway="residential", oneway=True, reversed=False, length=111.0,
               geometry=LineString([(30.061, -1.95), (30.0611, -1.9495), (30.061, -1.949)]))
    return G


class FakeOsmnx(types.ModuleType):
    def __init__(self, failures=0, error=None):
        super().__init__("osmnx")
        self.settings = types.SimpleNamespace(overpass_url="https://overpass-api.de/api", requests_timeout=180)
        self._errors = types.SimpleNamespace(
            InsufficientResponseError=InsufficientResponseError, ResponseStatusCodeError=ResponseStatusCodeError
        )
        self.failures = failures
        self.error = error
        self.calls = []

    def graph_from_polygon(self, polygon, **kwargs):
        self.calls.append(
            {"polygon": polygon, "kwargs": kwargs, "url": self.settings.overpass_url, "timeout": self.settings.requests_timeout}
        )
        if self.failures:
            self.failures -= 1
            raise self.error
        return osmnx_graph()


@pytest.fixture
def fake_osmnx(monkeypatch):
    fake = FakeOsmnx()
    monkeypatch.setitem(sys.modules, "osmnx", fake)
    monkeypatch.setattr(osm_roads.time, "sleep", lambda seconds: fake.calls.append({"slept": seconds}))
    return fake


KIGALI = [(-1.95, 30.06), (-1.955, 30.07), (-1.945, 30.065)]


def meters(geometry):
    return gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(UTM_36S).iloc[0]


# ----- the queried area


def test_points_become_a_buffered_hull():
    area = OSMRoadsFetcher(buffer=500).query_area(KIGALI)
    hull = meters(gpd.GeoSeries([Point(lon, lat) for lat, lon in KIGALI], crs="EPSG:4326").union_all().convex_hull)
    assert meters(area).area == pytest.approx(hull.buffer(500).area, rel=1e-3)


def test_area_modes():
    hull = meters(OSMRoadsFetcher(buffer=100).query_area(KIGALI)).area
    bbox = meters(OSMRoadsFetcher(buffer=100, area="bbox").query_area(KIGALI)).area
    buffers = meters(OSMRoadsFetcher(buffer=100, area="buffers").query_area(KIGALI)).area
    assert buffers < hull < bbox
    single = meters(OSMRoadsFetcher(buffer=1000).query_area([(-1.95, 30.06)])).area
    assert single == pytest.approx(math.pi * 1000**2, rel=1e-2)


def test_frames_of_points_and_lines_are_buffered():
    frame = pd.DataFrame({"lat": [p[0] for p in KIGALI], "lon": [p[1] for p in KIGALI]})
    assert OSMRoadsFetcher(buffer=500).query_area(frame).equals_exact(OSMRoadsFetcher(buffer=500).query_area(KIGALI), 1e-9)
    line = OSMRoadsFetcher(buffer=200).query_area(LineString([(30.06, -1.95), (30.07, -1.95)]))
    assert line.geom_type == "Polygon" and line.contains(Point(30.065, -1.9505))


def test_polygons_are_used_as_given():
    polygon = box(30.05, -1.96, 30.07, -1.94)
    assert OSMRoadsFetcher().query_area(polygon).equals(polygon)
    frame = gpd.GeoDataFrame(geometry=[box(30.05, -1.96, 30.06, -1.94), box(30.06, -1.96, 30.07, -1.94)], crs="EPSG:4326")
    assert OSMRoadsFetcher().query_area(frame).equals(polygon)


def test_countries_use_their_admin_boundary(monkeypatch):
    from gigaspatial.handlers import boundaries

    calls = []

    def create(country_code=None, **kwargs):
        calls.append(country_code)
        return types.SimpleNamespace(to_geoms=lambda: [box(28.8, -2.9, 30.9, -1.0)])

    monkeypatch.setattr(boundaries.AdminBoundaries, "create", staticmethod(create))
    assert OSMRoadsFetcher().query_area("Rwanda").equals(box(28.8, -2.9, 30.9, -1.0))
    assert calls == ["Rwanda"]


@pytest.mark.parametrize("source", ["data/roads.parquet", {"name": "x"}])
def test_sources_without_an_area_raise(source):
    with pytest.raises(ValueError):
        OSMRoadsFetcher().query_area(source)


def test_invalid_settings():
    with pytest.raises(ValueError, match="buffer"):
        OSMRoadsFetcher(buffer=-1)
    with pytest.raises(ValueError, match="greater than 0"):
        OSMRoadsFetcher(buffer=0).query_area([(-1.95, 30.06)])
    assert OSMRoadsFetcher(buffer=0).query_area(KIGALI).geom_type == "Polygon"


# ----- fetching


def test_fetch_builds_a_saveable_network(fake_osmnx, tmp_path):
    network = OSMRoadsFetcher(network_type="drive", buffer=300).fetch(KIGALI)
    assert isinstance(network, SpatialNetwork) and network.directed
    assert (len(network), len(network.edges)) == (3, 3)
    edges = network.edges.sort_values(["u", "v"])
    assert edges["osmid"].tolist() == [[11, 12], [11, 12], [13]]
    assert edges["highway"].astype(str).tolist() == ["primary;secondary", "primary;secondary", "residential"]
    assert edges["reversed"].tolist() == [True, True, False]
    meta = network.metadata
    assert meta["created_with"] == "OSMnx 2.0.2" and meta["source"] == {"kind": "points", "count": 3}
    assert meta["fetch"]["network_type"] == "drive" and meta["fetch"]["buffer"] == 300
    assert meta["area_km2"] > 0 and len(meta["area_bounds"]) == 4
    network.save(tmp_path / "roads")
    assert len(SpatialNetwork.load(tmp_path / "roads").edges) == 3

    call = fake_osmnx.calls[0]
    assert call["kwargs"] == {
        "network_type": "drive",
        "simplify": True,
        "retain_all": False,
        "truncate_by_edge": False,
        "custom_filter": None,
    }
    assert call["polygon"].equals(OSMRoadsFetcher(buffer=300).query_area(KIGALI))


def test_osmnx_settings_are_set_for_the_call_and_restored(fake_osmnx):
    OSMRoadsFetcher(overpass_url="https://overpass.kumi.systems/api/interpreter/", timeout=60).fetch(KIGALI)
    assert fake_osmnx.calls[0]["url"] == "https://overpass.kumi.systems/api"
    assert fake_osmnx.calls[0]["timeout"] == 60
    assert fake_osmnx.settings.overpass_url == "https://overpass-api.de/api"
    assert fake_osmnx.settings.requests_timeout == 180
    OSMRoadsFetcher().fetch(KIGALI)
    assert fake_osmnx.calls[-1]["url"] == config.OVERPASS_URL.removesuffix("/interpreter")


def test_transient_failures_are_retried(fake_osmnx):
    fake_osmnx.failures = 2
    fake_osmnx.error = ResponseStatusCodeError("502 Bad Gateway")
    network = OSMRoadsFetcher(retry_delay=2).fetch(KIGALI)
    assert len(network.edges) == 3
    assert [c["slept"] for c in fake_osmnx.calls if "slept" in c] == [2, 4]


def test_retries_give_up_with_a_hint(fake_osmnx):
    fake_osmnx.failures = 5
    fake_osmnx.error = requests.exceptions.ConnectionError("unreachable")
    with pytest.raises(RuntimeError, match="mirror"):
        OSMRoadsFetcher(max_retries=1).fetch(KIGALI)
    assert len([c for c in fake_osmnx.calls if "polygon" in c]) == 2


def test_no_roads_and_other_errors_are_not_retried(fake_osmnx):
    fake_osmnx.failures = 1
    fake_osmnx.error = InsufficientResponseError("no data")
    with pytest.raises(ValueError, match="no drive_service roads"):
        OSMRoadsFetcher().fetch(KIGALI)
    fake_osmnx.failures = 1
    fake_osmnx.error = TypeError("bad argument")
    with pytest.raises(TypeError):
        OSMRoadsFetcher().fetch(KIGALI)
    assert not [c for c in fake_osmnx.calls if "slept" in c]


def test_large_areas_warn(fake_osmnx, monkeypatch):
    monkeypatch.setattr(osm_roads, "LARGE_AREA_KM2", 1)
    fetcher = OSMRoadsFetcher(buffer=2000)
    fetcher.logger = MagicMock()
    fetcher.fetch(KIGALI)
    assert "Overpass is slow" in fetcher.logger.warning.call_args[0][0]


def test_missing_osmnx_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "osmnx", None)
    with pytest.raises(ImportError, match=r"giga-spatial\[roads\]"):
        OSMRoadsFetcher().fetch(KIGALI)


@pytest.mark.live
def test_live_fetch_of_central_kigali():
    pytest.importorskip("osmnx")
    network = OSMRoadsFetcher(network_type="drive", buffer=300).fetch([(-1.9441, 30.0619)])
    assert len(network.edges) > 10 and network.directed
    assert network.total_length() > 1000
