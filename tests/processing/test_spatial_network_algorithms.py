import networkx as nx
import numpy as np
import pandas as pd
import pytest

import gigaspatial.processing.network.algorithms as algorithms
from gigaspatial.processing.network import SpatialNetwork
from gigaspatial.processing.network.geodesy import haversine, line_lengths


def line(directed=True, lengths=(10.0, 20.0, 30.0)):
    """1 -> 2 -> 3 -> 4 along a line, plus a longer parallel edge 1 -> 2."""
    nodes = pd.DataFrame({"x": [30.0, 30.001, 30.002, 30.003], "y": [-1.95] * 4}, index=[1, 2, 3, 4])
    edges = pd.DataFrame({"u": [1, 1, 2, 3], "v": [2, 2, 3, 4], "length": [lengths[0], 99.0, lengths[1], lengths[2]]})
    return SpatialNetwork(nodes, edges, directed=directed)


@pytest.fixture(scope="module")
def graph(kigali):
    return kigali.to_networkx()


@pytest.fixture(scope="module")
def sample(kigali):
    rng = np.random.default_rng(0)
    ids = kigali.node_index
    return list(ids[rng.choice(len(ids), 6, replace=False)]), list(ids[rng.choice(len(ids), 8, replace=False)])


def nx_length(G, a, b, weight="length"):
    try:
        return nx.shortest_path_length(G, a, b, weight=weight)
    except nx.NetworkXNoPath:
        return np.inf


# ----- adjacency


def test_adjacency_keeps_the_shortest_parallel_edge_and_is_cached():
    net = line()
    matrix = net.adjacency()
    assert matrix[0, 1] == 10.0 and matrix[1, 0] == 0 and matrix.nnz == 3
    assert net.adjacency() is matrix
    symmetric = net.adjacency(directed=False)
    assert symmetric[1, 0] == 10.0 and symmetric.nnz == 6
    assert (line(directed=False).adjacency() != symmetric).nnz == 0


def test_weights_must_be_present_and_not_negative():
    net = line().set_edge_attribute("cost", {(1, 2, 0): 1.0})
    with pytest.raises(ValueError, match="have no 'cost'"):
        net.adjacency(weight="cost")
    with pytest.raises(ValueError, match="negative"):
        line().set_edge_attribute("length", -1.0).adjacency()
    with pytest.raises(ValueError, match="not an edge column"):
        line().adjacency(weight="time")


# ----- shortest path lengths


def test_lengths_match_networkx(kigali, graph, sample):
    origins, destinations = sample
    matrix = kigali.shortest_path_lengths(origins, destinations)
    assert list(matrix.index) == origins and list(matrix.columns) == destinations
    for a in origins:
        for b in destinations:
            assert matrix.loc[a, b] == pytest.approx(nx_length(graph, a, b))
    pairwise = kigali.shortest_path_lengths(origins, destinations[:6], pairwise=True)
    assert pairwise == pytest.approx([nx_length(graph, a, b) for a, b in zip(origins, destinations[:6])])


def test_lengths_without_direction_match_undirected_networkx(kigali, graph, sample):
    origins, destinations = sample
    undirected = graph.to_undirected()
    values = kigali.shortest_path_lengths(origins, destinations, directed=False)
    for a in origins:
        for b in destinations:
            assert values.loc[a, b] == pytest.approx(nx_length(undirected, a, b))


def test_one_way_roads_and_zero_weights():
    net = line(lengths=(0.0, 20.0, 30.0))
    lengths = net.shortest_path_lengths([1, 4], [2, 4])
    assert lengths.loc[1, 2] == 0.0 and lengths.loc[1, 4] == 50.0
    assert lengths.loc[4, 2] == np.inf
    assert net.shortest_path_lengths([4], [1], directed=False).loc[4, 1] == 50.0


def test_all_destinations_and_errors(kigali, sample):
    origins, _ = sample
    assert kigali.shortest_path_lengths(origins[:2]).shape == (2, len(kigali))
    with pytest.raises(ValueError, match="as many destinations"):
        kigali.shortest_path_lengths(origins, origins[:2], pairwise=True)
    with pytest.raises(ValueError, match="backend"):
        kigali.shortest_path_lengths(origins, origins, backend="igraph")
    with pytest.raises(KeyError):
        kigali.shortest_path_lengths([-1], origins)


def test_matrix_size_limit(kigali, sample, monkeypatch):
    monkeypatch.setattr(algorithms, "MAX_MATRIX_BYTES", 8)
    with pytest.raises(ValueError, match="batches"):
        kigali.shortest_path_lengths(sample[0], sample[1])


def test_small_blocks_give_the_same_results(kigali, sample, monkeypatch):
    origins, destinations = sample
    expected = kigali.shortest_path_lengths(origins, destinations)
    monkeypatch.setattr(algorithms, "CHUNK_BYTES", 1)
    pd.testing.assert_frame_equal(kigali.shortest_path_lengths(origins, destinations), expected)


# ----- shortest paths


def test_paths_have_the_networkx_length(kigali, graph, sample):
    origins, destinations = sample
    paths = kigali.shortest_paths(origins, destinations[:6])
    for path, a, b in zip(paths, origins, destinations[:6]):
        expected = nx_length(graph, a, b)
        if np.isinf(expected):
            assert path == []
            continue
        assert path[0] == a and path[-1] == b
        walked = sum(min(d["length"] for d in graph.get_edge_data(p, q).values()) for p, q in zip(path, path[1:]))
        assert walked == pytest.approx(expected)


def test_path_geometry(kigali, sample):
    origins, destinations = sample
    routes = kigali.shortest_paths(origins, destinations[:6], geometry=True)
    assert list(routes.columns) == ["origin", "destination", "length", "nodes", "geometry"]
    reached = routes[np.isfinite(routes["length"])]
    assert line_lengths(reached.geometry) == pytest.approx(reached["length"].to_numpy(), rel=1e-9)
    for route in reached.itertuples():
        start = kigali.node_coordinates([route.origin])[0][::-1]
        end = kigali.node_coordinates([route.destination])[0][::-1]
        assert list(route.geometry.coords[0]) == pytest.approx(list(start))
        assert list(route.geometry.coords[-1]) == pytest.approx(list(end))


def test_path_edge_cases():
    net = line()
    assert net.shortest_paths([1, 4, 2], [3, 1, 2]) == [[1, 2, 3], [], [2]]
    routes = net.shortest_paths([4, 2], [1, 2], geometry=True)
    assert routes["length"].tolist() == [np.inf, 0.0] and routes.geometry.isna().all()
    reverse = net.shortest_paths([3], [1], geometry=True, directed=False)
    assert reverse["nodes"].iloc[0] == [3, 2, 1]
    assert list(reverse.geometry.iloc[0].coords) == [(30.002, -1.95), (30.001, -1.95), (30.0, -1.95)]
    assert net.shortest_paths([1], [3, 4], pairwise=False) == [[1, 2, 3], [1, 2, 3, 4]]


# ----- reachable, nearest, counts


def test_reachable_matches_networkx(kigali, graph, sample):
    origin = sample[0][0]
    found = kigali.reachable([origin], 1500)
    expected = nx.single_source_dijkstra_path_length(graph, origin, cutoff=1500, weight="length")
    assert dict(zip(found["node_id"], found["distance"])) == pytest.approx(expected)
    assert found["distance"].is_monotonic_increasing
    assert found["origin"].iloc[0] == origin and found["distance"].iloc[0] == 0


def test_nearest_by_network_matches_brute_force(kigali, graph, sample):
    origins, targets = sample
    nearest = kigali.nearest_by_network(origins, targets)
    for row in nearest.itertuples():
        distances = {t: nx_length(graph, row.origin, t) for t in targets}
        assert row.distance == pytest.approx(min(distances.values()))
        assert distances[row.target] == pytest.approx(row.distance)
    two = kigali.nearest_by_network(origins, targets, k=2)
    assert two.groupby("origin")["distance"].apply(lambda d: d.is_monotonic_increasing).all()
    assert two.groupby("origin").head(1)["distance"].to_numpy() == pytest.approx(nearest["distance"].to_numpy())


def test_nearest_by_network_max_distance():
    found = line().nearest_by_network([1, 4], [3], max_distance=40)
    assert found["target"].tolist() == [3, None]
    assert found["distance"].tolist() == [30.0, np.inf]
    assert line().nearest_by_network([1], [3, 4], k=5, max_distance=40)["target"].tolist() == [3]


@pytest.mark.parametrize("n_targets", [2, 8])
def test_count_within_matches_brute_force(kigali, graph, sample, n_targets):
    origins = sample[0][:4]
    targets = sample[1][:n_targets]
    counts = kigali.count_within(origins, targets, 3000)
    expected = [sum(nx_length(graph, o, t) <= 3000 for t in targets) for o in origins]
    assert counts.tolist() == expected and list(counts.index) == origins


# ----- structure


def test_components_match_networkx(kigali, graph):
    weak = kigali.connected_components()
    sizes = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    assert weak.value_counts().sort_index().tolist() == sizes
    strong = kigali.connected_components(strong=True)
    assert strong.max() + 1 == nx.number_strongly_connected_components(graph)
    assert len(kigali.largest_component()) == sizes[0]


def test_minimum_spanning_tree_matches_networkx(kigali):
    tree = kigali.minimum_spanning_tree()
    simple = nx.Graph(kigali.to_undirected().to_networkx(directed=False))
    expected = sum(d["length"] for *_, d in nx.minimum_spanning_edges(simple, weight="length", data=True))
    assert not tree.directed and set(tree.edges["key"]) == {0}
    assert tree.total_length() == pytest.approx(expected)
    assert len(tree.edges) == len(kigali) - kigali.connected_components().nunique()


def test_minimum_spanning_tree_keeps_zero_weight_edges():
    assert len(line(lengths=(0.0, 20.0, 30.0)).minimum_spanning_tree().edges) == 3


def clusters():
    """Three clusters of two nodes, far apart."""
    xs = [30.0, 30.0001, 30.01, 30.0101, 30.05, 30.0501]
    nodes = pd.DataFrame({"x": xs, "y": [-1.95] * 6}, index=list("abcdef"))
    edges = pd.DataFrame({"u": ["a", "c", "e"], "v": ["b", "d", "f"]})
    return SpatialNetwork(nodes, edges, directed=False)


def test_connect_components_joins_nearest_pairs():
    joined = clusters().connect_components()
    links = joined.edges[joined.edges["kind"] == "component_link"]
    assert sorted(map(sorted, links[["u", "v"]].values.tolist())) == [["b", "c"], ["d", "e"]]
    assert links["length"].iloc[0] == pytest.approx(haversine(-1.95, 30.0001, -1.95, 30.01))
    assert joined.connected_components().nunique() == 1
    close = clusters().connect_components(max_distance=2000)
    assert close.connected_components().nunique() == 2


def test_connect_components_directed(kigali):
    joined = kigali.connect_components()
    assert joined.connected_components().nunique() == 1
    links = joined.edges[joined.edges["kind"] == "component_link"]
    assert len(links) == 2 * (kigali.connected_components().nunique() - 1)
    connected = joined.connect_components()
    assert connected is joined


# ----- pandana


def test_pandana_backend(kigali, sample):
    pytest.importorskip("pandana")
    origins, destinations = sample
    scipy_values = kigali.shortest_path_lengths(origins, destinations[:6], pairwise=True)
    pandana_values = kigali.shortest_path_lengths(origins, destinations[:6], pairwise=True, backend="pandana")
    finite = np.isfinite(scipy_values)
    assert (np.isfinite(pandana_values) == finite).all()
    assert pandana_values[finite] == pytest.approx(scipy_values[finite], abs=1.0)
    assert kigali.to_pandana() is kigali.to_pandana()
    matrix = kigali.shortest_path_lengths(origins[:2], destinations[:3], backend="pandana")
    assert matrix.shape == (2, 3)


def test_pandana_unreachable_and_mixed_ids():
    pytest.importorskip("pandana")
    assert line().shortest_path_lengths([4], [1], pairwise=True, backend="pandana").tolist() == [np.inf]
    nodes = pd.DataFrame({"x": [30.0, 30.001], "y": [-1.95, -1.95]}, index=pd.Index([1, "a"], dtype=object))
    mixed = SpatialNetwork(nodes, pd.DataFrame({"u": [1], "v": ["a"]}))
    with pytest.raises(ValueError, match="all integers or all text"):
        mixed.to_pandana()
    assert mixed.shortest_path_lengths([1], ["a"], pairwise=True, backend="pandana")[0] == pytest.approx(
        mixed.shortest_path_lengths([1], ["a"], pairwise=True)[0], abs=0.01
    )


def test_pandana_backend_with_inserted_points(kigali):
    pytest.importorskip("pandana")
    west, south, east, north = kigali.bounds
    points = pd.DataFrame(
        {"school_id": ["school-a", "school-b", "school-c"], "lon": [west + 0.01, (west + east) / 2, east - 0.01],
         "lat": [south + 0.01, (south + north) / 2, north - 0.01]}
    )
    with_schools = kigali.insert_points(points, id_column="school_id")
    origins = ["school-a", "school-b", "school-c"]
    destinations = ["school-c", "school-a", "school-b"]
    scipy_values = with_schools.shortest_path_lengths(origins, destinations, pairwise=True)
    pandana_values = with_schools.shortest_path_lengths(origins, destinations, pairwise=True, backend="pandana")
    assert np.isfinite(scipy_values).all()
    assert pandana_values == pytest.approx(scipy_values, abs=1.0)
