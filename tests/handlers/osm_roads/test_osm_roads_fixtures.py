import json
from pathlib import Path

import pytest

from gigaspatial.handlers.osm_roads import OSMRoadsHandler, OSMRoadsReadOptions, OSMRoadsResolveOptions
from gigaspatial.processing.network import SpatialNetwork

DATA = Path(__file__).resolve().parents[2] / "data" / "osm_roads"


def test_index_subset_has_geometries_and_codes():
    index = json.loads((DATA / "geofabrik_index_subset.json").read_text())
    by_id = {f["properties"]["id"]: f for f in index["features"]}
    assert {"rwanda", "brazil", "nordeste", "africa", "israel-and-palestine", "polynesie-francaise"} <= set(by_id)
    assert all(f["geometry"]["type"] == "MultiPolygon" for f in index["features"])
    assert by_id["rwanda"]["properties"]["iso3166-1:alpha2"] == ["RW"]
    assert by_id["nordeste"]["properties"]["parent"] == "brazil"
    assert by_id["polynesie-francaise"]["properties"]["iso3166-1:alpha2"] == ["VU"]


def test_whole_index_without_geometries():
    index = json.loads((DATA / "geofabrik_index_nogeom.json").read_text())
    assert len(index["features"]) > 500
    assert all(f["geometry"] is None and "urls" in f["properties"] for f in index["features"])


def test_kigali_extract_is_a_pbf_file():
    head = (DATA / "kigali.osm.pbf").read_bytes()[:32]
    assert b"OSMHeader" in head
    meta = json.loads((DATA / "kigali.json").read_text())
    assert meta["bbox"] == [30.045, -1.965, 30.085, -1.935] and meta["highway_ways"] > 1000


@pytest.mark.parametrize(
    "name", ["drive_service-simplified", "drive_service-raw", "drive-simplified", "walk-simplified", "all-simplified"]
)
def test_osmnx_networks_load(name):
    network = SpatialNetwork.load(DATA / "kigali_osmnx" / name)
    assert network.directed and len(network.edges) > 1000
    assert "street_count" in network.nodes.columns


def test_skeleton_defaults():
    handler = OSMRoadsHandler()
    assert handler.config.base_path.parts[-2:] == ("osm", "geofabrik")
    assert handler.config.index_url == "https://download.geofabrik.de/index-v1.json"
    assert OSMRoadsResolveOptions().regions is False and OSMRoadsResolveOptions().buffer == 2000.0
    assert OSMRoadsReadOptions().network_type == "drive_service" and OSMRoadsReadOptions().simplify
