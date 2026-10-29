import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.osm.pbf import (
    IN_MEMORY_INDEX_MAX_BYTES,
    NETWORK_FILTERS,
    build_road_network,
    node_index_for,
    way_filter,
)
from gigaspatial.handlers.osm.roads import OSMRoadsHandler, OSMRoadsReadOptions, _local_file
from gigaspatial.processing.network import SpatialNetwork

osmium = pytest.importorskip("osmium")

DATA = Path(__file__).resolve().parents[2] / "data" / "osm_roads"
KIGALI = DATA / "kigali.osm.pbf"
RECORDED = ["drive_service-raw", "drive_service-simplified", "drive-simplified", "walk-simplified", "all-simplified"]


# ----- the same network as osmnx


def _parts(value):
    return None if pd.isna(value) else frozenset(str(value).split(";"))


def assert_same_network(ours: SpatialNetwork, theirs: SpatialNetwork, exact_lengths: bool):
    """Same nodes and attributes, same (u, v, key) edges and attributes; text compared as sets of ';' parts."""
    on, tn = ours.nodes, theirs.nodes.loc[ours.nodes.index]
    assert set(on.index) == set(theirs.nodes.index)
    assert set(on.columns) == set(tn.columns)
    assert np.array_equal(on["x"], tn["x"]) and np.array_equal(on["y"], tn["y"])
    assert np.array_equal(on["street_count"], tn["street_count"])
    for name in set(on.columns) - {"x", "y", "kind", "street_count"}:
        assert on[name].astype(object).fillna("-").tolist() == tn[name].astype(object).fillna("-").tolist(), name

    oe = ours.edges.set_index(["u", "v", "key"])
    te = theirs.edges.set_index(["u", "v", "key"])
    assert oe.index.is_unique and set(oe.index) == set(te.index)
    te = te.loc[oe.index]
    assert set(oe.columns) == set(te.columns)
    if exact_lengths:
        assert np.array_equal(oe["length"], te["length"])
    else:
        assert oe["length"].to_numpy() == pytest.approx(te["length"].to_numpy(), abs=1e-9)
    for name in ("oneway", "reversed"):
        assert np.array_equal(oe[name], te[name]), name
    osmids = [frozenset(np.atleast_1d(v).tolist()) for v in oe["osmid"]]
    assert osmids == [frozenset(np.atleast_1d(v).tolist()) for v in te["osmid"]]
    if "geometry" in te.columns:
        ours_geometry, their_geometry = oe["geometry"].to_numpy(), te["geometry"].to_numpy()
        assert np.array_equal(pd.isna(ours_geometry), pd.isna(their_geometry))
        present = ~pd.isna(their_geometry)
        assert shapely.equals_exact(ours_geometry[present], their_geometry[present], 0).all()
    for name in set(oe.columns) - {"length", "kind", "oneway", "reversed", "osmid", "geometry"}:
        assert [_parts(v) for v in oe[name]] == [_parts(v) for v in te[name]], name


@pytest.mark.parametrize("name", RECORDED)
def test_kigali_matches_osmnx(name):
    network_type, variant = name.rsplit("-", 1)
    ours = build_road_network(KIGALI, network_type=network_type, simplify=variant == "simplified")
    theirs = SpatialNetwork.load(DATA / "kigali_osmnx" / name)
    assert_same_network(ours, theirs, exact_lengths=variant == "raw")


def test_filters_are_osmnx_filters():
    _overpass = pytest.importorskip("osmnx._overpass")
    for network_type, excluded in NETWORK_FILTERS.items():
        text = _overpass._get_network_filter(network_type)
        assert text.startswith('["highway"]')
        assert re.findall(r'\["([^"]+)"!~"([^"]*)"\]', text) == list(excluded), network_type
        assert re.sub(r'\["[^"]+"!~"[^"]*"\]', "", text) == '["highway"]'


def test_way_filter_rules():
    def passes(network_type, **tags):
        return way_filter(network_type)[0](tags.get)

    assert passes("drive_service", highway="service", service="driveway")
    assert not passes("drive", highway="service")
    assert not passes("drive_service", highway="service", service="parking_aisle")
    assert not passes("walk", highway="motorway")  # unanchored, as in Overpass
    assert passes("walk", highway="footway") and not passes("drive", highway="footway")
    assert not passes("drive", highway="residential", access="private")
    assert passes("all", highway="residential", access="private")
    assert not passes("all", highway="pedestrian", area="yes")
    assert passes("bike", highway="cycleway") and not passes("bike", highway="cycleway", bicycle="no")
    with pytest.raises(ValueError, match="Unknown network_type"):
        way_filter("boat")


def test_tags_filter():
    passes, keys = way_filter(tags={"highway": ["primary", "secondary"], "oneway": None})
    assert keys == ("highway", "oneway")
    assert passes({"highway": "primary", "oneway": "no"}.get)
    assert not passes({"highway": "primary"}.get) and not passes({"highway": "tertiary", "oneway": "yes"}.get)
    with pytest.raises(ValueError):
        way_filter(tags={})

    network = build_road_network(KIGALI, tags={"highway": ["primary", "secondary"]}, simplify=False)
    assert set(network.edges["highway"].astype(str)) == {"primary", "secondary"}
    assert network.metadata["build"]["tags"] == {"highway": ["primary", "secondary"]}


# ----- osmnx's rules, on a small made-up extract

NODES = {i: (30.06 + 0.001 * i, -1.95 + 0.0005 * (i % 3)) for i in range(1, 9)}
WAYS = {
    10: ([1, 2, 3], {"highway": "residential", "name": "A"}),
    11: ([3, 4], {"highway": "primary", "oneway": "yes"}),
    12: ([5, 4], {"highway": "secondary", "oneway": "-1"}),
    13: ([5, 6], {"highway": "tertiary", "oneway": "T"}),
    14: ([6, 7], {"highway": "residential", "oneway": "F"}),
    15: ([7, 8, 8, 7], {"highway": "residential", "junction": "roundabout"}),
    16: ([1, 2], {"highway": "residential", "surface": "asphalt"}),
    17: ([2, 7], {"highway": "service", "service": "parking_aisle"}),
    18: ([1, 8], {"highway": "footway"}),
}


def write_extract(path, ways=WAYS, nodes=NODES):
    with osmium.SimpleWriter(str(path)) as writer:
        for i, (lon, lat) in nodes.items():
            tags = {"highway": "traffic_signals", "crossing": "zebra"} if i == 2 else {}
            writer.add_node(osmium.osm.mutable.Node(id=i, location=(lon, lat), tags=tags))
        for way_id, (refs, tags) in ways.items():
            writer.add_way(osmium.osm.mutable.Way(id=way_id, nodes=refs, tags=tags))
    return path


def edges_of(network):
    edges = network.edges
    return {(int(u), int(v), int(k)): row for (u, v, k), row in zip(zip(edges.u, edges.v, edges.key), edges.to_dict("records"))}


def test_one_way_rules(tmp_path):
    network = build_road_network(write_extract(tmp_path / "x.osm.pbf"), network_type="all", simplify=False)
    edges = edges_of(network)
    assert edges[(4, 5, 0)]["oneway"] and not edges[(4, 5, 0)]["reversed"] and (5, 4, 0) not in edges  # -1
    assert (6, 5, 0) in edges and (5, 6, 0) not in edges  # T is reversed
    assert (6, 7, 0) in edges and (7, 6, 0) not in edges  # F is one-way
    assert edges[(3, 4, 0)]["oneway"] and (4, 3, 0) not in edges
    assert (7, 8, 0) in edges and (8, 7, 0) in edges and (8, 8, 0) not in edges  # roundabout, repeat dropped
    assert edges[(2, 1, 0)]["reversed"] and not edges[(1, 2, 0)]["reversed"] and not edges[(1, 2, 0)]["oneway"]
    assert edges[(1, 2, 1)]["osmid"] == 16 and edges[(2, 1, 1)]["osmid"] == 16  # parallel ways keyed in order
    assert "surface" not in network.edges.columns
    assert network.nodes.loc[2, "highway"] == "traffic_signals" and "crossing" not in network.nodes.columns
    assert network.nodes.loc[2, "street_count"] == 4 and network.nodes.loc[5, "street_count"] == 2

    walk = edges_of(build_road_network(tmp_path / "x.osm.pbf", network_type="walk", simplify=False))
    assert (4, 3, 0) in walk and not walk[(3, 4, 0)]["oneway"] and (5, 4, 0) in walk  # walk is two-way

    drive = build_road_network(tmp_path / "x.osm.pbf", network_type="drive_service", simplify=False)
    assert set(drive.edges["osmid"]) == {10, 11, 12, 13, 14, 15, 16}


@pytest.mark.parametrize("network_type, bidirectional", [("all", False), ("walk", True)])
def test_made_up_extract_matches_osmnx(tmp_path, network_type, bidirectional):
    ox = pytest.importorskip("osmnx")
    path = write_extract(tmp_path / "x.osm.pbf")
    xml = tmp_path / "x.osm"
    with osmium.SimpleWriter(str(xml)) as writer:
        for entity in osmium.FileProcessor(str(path)):
            writer.add(entity)
    graph = ox.graph_from_xml(xml, bidirectional=bidirectional, simplify=False, retain_all=True)
    for node, count in ox.stats.count_streets_per_node(graph).items():
        graph.nodes[node]["street_count"] = count
    theirs = SpatialNetwork.from_networkx(graph)
    theirs = SpatialNetwork(theirs.nodes, theirs.edges.drop(columns=["surface"], errors="ignore"))
    ours = build_road_network(path, network_type=network_type, simplify=False)
    assert_same_network(ours, theirs, exact_lengths=True)


def test_ways_with_missing_nodes_are_skipped(tmp_path):
    ways = {**WAYS, 19: ([6, 99], {"highway": "residential"})}
    network = build_road_network(write_extract(tmp_path / "x.osm.pbf", ways=ways), network_type="all", simplify=False)
    assert 99 not in network.node_index and 19 not in set(network.edges["osmid"])


def test_no_matching_ways(tmp_path):
    with pytest.raises(ValueError, match="No ways"):
        build_road_network(write_extract(tmp_path / "x.osm.pbf"), tags={"railway": ["rail"]})


# ----- the node index


def test_node_index_by_size():
    assert node_index_for(70 * 2**20) == "flex_mem"
    assert node_index_for(IN_MEMORY_INDEX_MAX_BYTES + 1) == "sparse_file_array"
    assert node_index_for(10, "sparse_file_array") == "sparse_file_array"
    with pytest.raises(ValueError, match="100 GB"):
        node_index_for(10, "dense_mem_array")
    with pytest.raises(ValueError, match="Unknown node_index"):
        node_index_for(10, "fast")


def test_disk_index_gives_the_same_network():
    in_memory = build_road_network(KIGALI, simplify=False)
    on_disk = build_road_network(KIGALI, simplify=False, node_index="sparse_file_array")
    assert on_disk.metadata["build"]["node_index"] == "sparse_file_array"
    assert_same_network(on_disk, in_memory, exact_lengths=True)


# ----- metadata


def test_metadata_records_the_extract(tmp_path):
    stored = tmp_path / "osm/geofabrik/2026-10-26/africa/rwanda.osm.pbf"
    stored.parent.mkdir(parents=True)
    shutil.copy(KIGALI, stored)
    handler = OSMRoadsHandler(data_store=LocalDataStore(tmp_path), base_path=Path("osm/geofabrik"))
    network = handler.reader.build_network(Path("osm/geofabrik/2026-10-26/africa/rwanda.osm.pbf"))
    assert network.metadata["extract"] == {
        "id": "rwanda",
        "date": "2026-10-26",
        "url": "https://download.geofabrik.de/africa/rwanda-261026.osm.pbf",
        "path": "africa/rwanda",
        "replication_timestamp": "2026-10-26T21:24:46Z",
    }
    build = network.metadata["build"]
    assert build["network_type"] == "drive_service" and build["simplified"] and build["node_index"] == "flex_mem"

    network.save(tmp_path / "saved")
    assert SpatialNetwork.load(tmp_path / "saved").metadata["extract"]["date"] == "2026-10-26"
    raw = handler.reader.build_network(
        Path("osm/geofabrik/2026-10-26/africa/rwanda.osm.pbf"), OSMRoadsReadOptions(network_type="walk", simplify=False)
    )
    assert raw.metadata["build"]["network_type"] == "walk" and not raw.metadata["build"]["simplified"]


def test_metadata_of_a_file_outside_the_layout():
    extract = build_road_network(KIGALI).metadata["extract"]
    assert extract == {"id": "kigali", "date": "2026-10-26", "url": None, "replication_timestamp": "2026-10-26T21:24:46Z"}
    config = OSMRoadsHandler(base_path=Path("osm/geofabrik")).config
    assert config.extract_of_path("elsewhere/rwanda.osm.pbf") is None
    assert config.extract_of_path("osm/geofabrik/latest/africa/rwanda.osm.pbf") is None
    assert config.extract_of_path("osm/geofabrik/2026-10-26/south-america/brazil/nordeste.osm.pbf")["id"] == "nordeste"


def test_remote_stores_are_read_from_a_local_copy(tmp_path):
    class Remote:
        def __init__(self, inner):
            self.inner = inner

        def open(self, path, mode="r"):
            return self.inner.open(path, mode)

    shutil.copy(KIGALI, tmp_path / "rwanda.osm.pbf")
    with _local_file(Remote(LocalDataStore(tmp_path)), "rwanda.osm.pbf") as local:
        assert local.read_bytes() == KIGALI.read_bytes() and tmp_path not in local.parents
    assert not local.exists()
