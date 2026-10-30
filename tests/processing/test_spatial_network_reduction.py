import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from networkx.algorithms.approximation import steiner_tree
from shapely.geometry import LineString

from gigaspatial.processing.network import SpatialNetwork
from gigaspatial.processing.network.geodesy import line_lengths

DATA = Path(__file__).resolve().parents[1] / "data" / "spatial_network"
RAW = DATA / "kigali_raw"
OSMNX = DATA / "kigali_raw_osmnx.json"


@pytest.fixture(scope="module")
def raw():
    return SpatialNetwork.load(RAW)


@pytest.fixture(scope="module")
def osmnx_result():
    return json.loads(OSMNX.read_text())


def nodes_on_a_line(ids):
    return pd.DataFrame({"x": 30.0 + 0.001 * np.arange(len(ids)), "y": [-1.95] * len(ids)}, index=ids)


def assert_valid(net):
    SpatialNetwork(net.nodes, net.edges, directed=net.directed, weight=net.weight)
    assert net.degree(directed=True).to_dict() == dict(net.to_networkx().degree())


# ----- simplify, against osmnx


def test_simplify_matches_osmnx(raw, osmnx_result):
    simple = raw.simplify()
    assert set(simple.node_index) == set(osmnx_result["nodes"])
    assert len(simple.edges) == osmnx_result["edges"]
    assert np.round(np.sort(simple.edges["length"].to_numpy()), 6) == pytest.approx(osmnx_result["edge_lengths"], abs=1e-5)
    assert simple.total_length() == pytest.approx(raw.total_length())
    assert_valid(simple)


def test_simplified_geometry_follows_the_joined_edges(raw):
    simple = raw.simplify()
    assert line_lengths(simple.edges_gdf.geometry) == pytest.approx(simple.edges["length"].to_numpy(), rel=1e-9)


def test_simplify_again_changes_nothing(raw):
    simple = raw.simplify()
    assert simple.simplify() is simple


def test_simplify_keeps_protected_and_kept_nodes(raw, osmnx_result):
    removed = [n for n in raw.node_index[:2000] if n not in set(osmnx_result["nodes"])][:3]
    kept = raw.simplify(keep=removed[:2])
    assert set(removed[:2]) <= set(kept.node_index) and removed[2] not in kept.node_index
    marked = raw.set_node_attribute("kind", {removed[2]: "point"}).simplify()
    assert removed[2] in marked.node_index
    assert kept.total_length() == pytest.approx(raw.total_length())


def test_simplify_on_the_simplified_fixture_keeps_length(kigali):
    simple = kigali.simplify()
    assert len(simple) <= len(kigali)
    assert simple.total_length() == pytest.approx(kigali.total_length())


# ----- simplify, attribute joining


def one_way_chain():
    edges = pd.DataFrame(
        {
            "u": ["a", "b"],
            "v": ["b", "c"],
            "length": [100.0, 300.0],
            "highway": ["primary", "secondary"],
            "osmid": [1, 2],
            "oneway": [True, True],
            "reversed": [False, True],
            "maxspeed": [50.0, 30.0],
            "cost": [1.0, 2.0],
        }
    )
    return SpatialNetwork(nodes_on_a_line(["a", "b", "c"]), edges)


def test_simplify_joins_attributes():
    joined = one_way_chain().simplify(sum_columns=("length", "cost"))
    assert list(joined.node_index) == ["a", "c"]
    edge = joined.edges.iloc[0]
    assert (edge["u"], edge["v"], edge["key"]) == ("a", "c", 0)
    assert edge["length"] == 400.0 and edge["cost"] == 3.0
    assert edge["highway"] == "primary;secondary"
    assert list(edge["osmid"]) == [1, 2]
    assert bool(edge["oneway"]) and bool(edge["reversed"])
    assert edge["maxspeed"] == pytest.approx(35.0)
    assert edge["kind"] == "road"
    assert list(edge["geometry"].coords) == [(30.0, -1.95), (30.001, -1.95), (30.002, -1.95)]


def test_simplify_without_sum_columns_averages_numbers():
    edge = one_way_chain().simplify().edges.iloc[0]
    assert edge["cost"] == pytest.approx(1.75)


def test_two_way_roads_stay_two_way():
    edges = pd.DataFrame({"u": ["a", "b", "b", "c"], "v": ["b", "a", "c", "b"], "length": [1.0, 1.0, 2.0, 2.0]})
    joined = SpatialNetwork(nodes_on_a_line(["a", "b", "c"]), edges).simplify()
    pairs = sorted(joined.edges[["u", "v"]].itertuples(index=False, name=None))
    assert pairs == [("a", "c"), ("c", "a")]
    back = joined.edges.set_index(["u", "v"]).loc[("c", "a"), "geometry"]
    assert list(back.coords) == [(30.002, -1.95), (30.001, -1.95), (30.0, -1.95)]


def test_a_change_from_one_way_to_two_way_is_kept():
    edges = pd.DataFrame({"u": ["a", "b", "c"], "v": ["b", "c", "b"], "length": [1.0, 2.0, 2.0]})
    net = SpatialNetwork(nodes_on_a_line(["a", "b", "c"]), edges)
    assert net.simplify() is net


def test_undirected_chain_and_parallel_keys():
    edges = pd.DataFrame({"u": ["a", "c", "c", "a"], "v": ["b", "b", "d", "d"], "length": [1.0, 2.0, 3.0, 10.0]})
    net = SpatialNetwork(nodes_on_a_line(["a", "b", "c", "d"]), edges, directed=False)
    joined = net.simplify(keep=["a", "d"])
    assert sorted(joined.node_index) == ["a", "d"]
    table = joined.edges.sort_values("length")
    assert table["length"].tolist() == [6.0, 10.0]
    assert sorted(table["key"].tolist()) == [0, 1]
    assert_valid(joined)


def test_a_node_with_two_neighbours_through_a_parallel_road_is_removed():
    edges = pd.DataFrame({"u": ["a", "c", "c", "a"], "v": ["b", "b", "d", "d"], "length": [1.0, 2.0, 3.0, 10.0]})
    ring = SpatialNetwork(nodes_on_a_line(["a", "b", "c", "d"]), edges, directed=False).simplify(keep=["d"])
    assert list(ring.node_index) == ["d"] and ring.edges["length"].tolist() == [16.0]


def test_a_ring_keeps_one_node():
    edges = pd.DataFrame({"u": ["a", "b", "c", "d"], "v": ["b", "c", "d", "a"], "length": [1.0, 2.0, 3.0, 4.0]})
    ring = SpatialNetwork(nodes_on_a_line(["a", "b", "c", "d"]), edges, directed=False).simplify()
    assert len(ring) == 1 and len(ring.edges) == 1
    assert ring.edges["length"].iloc[0] == 10.0
    assert ring.edges["u"].iloc[0] == ring.edges["v"].iloc[0]


# ----- connecting subnetwork


@pytest.fixture(scope="module")
def connected(kigali):
    return kigali.largest_component()


@pytest.fixture(scope="module")
def chosen(connected):
    rng = np.random.default_rng(1)
    ids = connected.node_index
    return list(ids[rng.choice(len(ids), 30, replace=False)]), list(ids[rng.choice(len(ids), 3, replace=False)])


def test_steiner_matches_networkx(connected, chosen):
    terminals, _ = chosen
    tree = connected.connecting_subnetwork(terminals)
    simple = nx.Graph(connected.to_undirected().to_networkx(directed=False))
    networkx_best = min(
        steiner_tree(simple, terminals, weight="length", method=m).size(weight="length") for m in ("mehlhorn", "kou")
    )
    # networkx 3.6 gives the same tree; older versions' approximations can be longer.
    assert tree.total_length() <= networkx_best * (1 + 1e-9)
    assert not tree.directed and set(terminals) <= set(tree.node_index)
    assert len(tree.edges) == len(tree) - 1
    degree = tree.degree()
    assert set(degree[degree == 1].index) <= set(terminals)
    assert (degree.drop(terminals) >= 3).all()
    assert line_lengths(tree.edges_gdf.geometry) == pytest.approx(tree.edges["length"].to_numpy(), rel=1e-9)


def test_nearest_source_paths(connected, chosen):
    terminals, sources = chosen
    sub = connected.connecting_subnetwork(terminals, sources=sources)
    nearest = connected.nearest_by_network(terminals, sources, directed=False)
    inside = sub.nearest_by_network(terminals, sources)
    assert inside["distance"].to_numpy() == pytest.approx(nearest["distance"].to_numpy())
    assert set(sources) & set(sub.node_index)


def test_all_pairs_keeps_every_shortest_path(connected, chosen):
    terminals, sources = chosen
    terminals = terminals[:8]
    sub = connected.connecting_subnetwork(terminals, sources=sources, method="all_pairs")
    targets = terminals + sources
    expected = connected.shortest_path_lengths(terminals, targets, directed=False)
    assert sub.shortest_path_lengths(terminals, targets).to_numpy() == pytest.approx(expected.to_numpy())


def test_connecting_subnetwork_edge_cases(kigali, connected, chosen):
    terminals, _ = chosen
    island = kigali.connected_components().idxmax()
    with_island = kigali.connecting_subnetwork(terminals[:5] + [island])
    assert island in with_island.node_index and with_island.degree()[island] == 0
    single = connected.connecting_subnetwork(terminals[:1])
    assert (len(single), len(single.edges)) == (1, 0)
    with pytest.raises(ValueError, match="method"):
        connected.connecting_subnetwork(terminals, method="kou")


def test_connecting_subnetwork_with_a_line_geometry():
    edges = pd.DataFrame(
        {
            "u": ["a", "b", "b"],
            "v": ["b", "c", "d"],
            "geometry": [LineString([(30.0, -1.95), (30.001, -1.95)]), None, None],
        }
    )
    nodes = pd.DataFrame({"x": [30.0, 30.001, 30.002, 30.001], "y": [-1.95, -1.95, -1.95, -1.949]}, index=list("abcd"))
    tree = SpatialNetwork(nodes, edges, directed=False).connecting_subnetwork(["a", "c"])
    assert list(tree.node_index) == ["a", "c"]
    assert list(tree.edges["geometry"].iloc[0].coords) in (
        [(30.0, -1.95), (30.001, -1.95), (30.002, -1.95)],
        [(30.002, -1.95), (30.001, -1.95), (30.0, -1.95)],
    )
