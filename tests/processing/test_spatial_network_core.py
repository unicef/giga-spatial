import math

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon

from gigaspatial.processing.network import SpatialNetwork
from gigaspatial.processing.network.core import EDGE_KINDS, NODE_KINDS
from gigaspatial.processing.network.geodesy import (
    EARTH_RADIUS_M,
    chord_to_meters,
    haversine,
    line_lengths,
    meters_to_chord,
)

UTM_35S = "EPSG:32735"


def small_nodes():
    return pd.DataFrame(
        {"x": [30.0, 30.001, 30.001, 30.0], "y": [-1.95, -1.95, -1.949, -1.949]},
        index=pd.Index([1, 2, 3, 4], name="node_id"),
    )


def small_edges(**extra):
    edges = pd.DataFrame({"u": [1, 2, 3], "v": [2, 3, 4]})
    for name, values in extra.items():
        edges[name] = values
    return edges


# ----- geodesy


def test_haversine_one_degree_of_latitude():
    assert haversine(0.0, 0.0, 1.0, 0.0) == pytest.approx(math.pi / 180 * EARTH_RADIUS_M)


def test_line_lengths_add_segments_and_leave_missing_as_nan():
    line = LineString([(30.0, -1.95), (30.001, -1.95), (30.001, -1.949)])
    expected = haversine(-1.95, 30.0, -1.95, 30.001) + haversine(-1.95, 30.001, -1.949, 30.001)
    lengths = line_lengths([line, None])
    assert lengths[0] == pytest.approx(expected)
    assert np.isnan(lengths[1])


def test_chord_and_meters_are_inverse():
    meters = np.array([1.0, 1_000.0, 2_000_000.0])
    assert chord_to_meters(meters_to_chord(meters)) == pytest.approx(meters)


# ----- construction and validation


def test_kigali_tables(kigali):
    assert len(kigali) == 1209
    assert len(kigali.edges) == 2802
    assert kigali.directed and kigali.weight == "length"
    assert list(kigali.nodes.columns[:3]) == ["x", "y", "kind"]
    assert kigali.nodes.index.name == "node_id"
    assert list(kigali.nodes["kind"].cat.categories[:3]) == list(NODE_KINDS)
    assert list(kigali.edges.columns[:5]) == ["u", "v", "key", "length", "kind"]
    assert list(kigali.edges["kind"].cat.categories[: len(EDGE_KINDS)]) == list(EDGE_KINDS)


def test_kigali_lengths_match_osmnx_along_geometry(kigali):
    edges = kigali.edges
    curved = edges["geometry"].notna()
    assert line_lengths(edges.loc[curved, "geometry"]) == pytest.approx(
        edges.loc[curved, "length"].to_numpy(), rel=1e-9
    )


def test_missing_length_comes_from_geometry_then_endpoints():
    line = LineString([(30.0, -1.95), (30.0005, -1.9499), (30.001, -1.95)])
    net = SpatialNetwork(small_nodes(), small_edges(geometry=[line, None, None]))
    length = net.edges["length"].to_numpy()
    assert length[0] == pytest.approx(line_lengths([line])[0])
    assert length[1] == pytest.approx(haversine(-1.95, 30.001, -1.949, 30.001))


def test_missing_key_numbers_parallel_edges():
    edges = pd.DataFrame({"u": [1, 1, 2], "v": [2, 2, 3]})
    assert SpatialNetwork(small_nodes(), edges).edges["key"].tolist() == [0, 1, 0]


def test_node_id_column_becomes_the_index():
    nodes = small_nodes().reset_index()
    assert SpatialNetwork(nodes, small_edges()).node_index.tolist() == [1, 2, 3, 4]


def test_geodataframe_nodes_without_xy_use_their_points():
    nodes = gpd.GeoDataFrame(
        index=pd.Index([1, 2, 3, 4]),
        geometry=gpd.points_from_xy(small_nodes()["x"], small_nodes()["y"]),
        crs="EPSG:4326",
    )
    net = SpatialNetwork(nodes, small_edges())
    assert net.nodes["x"].tolist() == small_nodes()["x"].tolist()


def test_projected_input_is_converted_to_wgs84():
    to_utm = Transformer.from_crs("EPSG:4326", UTM_35S, always_xy=True)
    x, y = to_utm.transform(small_nodes()["x"].to_numpy(), small_nodes()["y"].to_numpy())
    nodes = pd.DataFrame({"x": x, "y": y}, index=small_nodes().index)
    net = SpatialNetwork(nodes, small_edges(), crs=UTM_35S)
    assert net.nodes["x"].to_numpy() == pytest.approx(small_nodes()["x"].to_numpy(), abs=1e-9)
    assert net.nodes["y"].to_numpy() == pytest.approx(small_nodes()["y"].to_numpy(), abs=1e-9)


@pytest.mark.parametrize(
    "nodes, edges, kwargs, message",
    [
        (small_nodes().iloc[[0, 0, 1]], small_edges().iloc[:0], {}, "unique"),
        (small_nodes(), pd.DataFrame({"u": [1], "v": [9]}), {}, "not in the network"),
        (small_nodes(), pd.DataFrame({"u": [1, 1], "v": [2, 2], "key": [0, 0]}), {}, "repeat"),
        (small_nodes(), pd.DataFrame({"u": [1, 2], "v": [2, 1], "key": [0, 0]}), {"directed": False}, "repeat"),
        (small_nodes().drop(columns="y"), small_edges(), {}, "'y'"),
        (small_nodes(), small_edges(), {"weight": "travel_time"}, "weight"),
        (small_nodes(), small_edges(geometry=[Point(0, 0), None, None]), {}, "LineStrings"),
    ],
)
def test_invalid_tables_raise(nodes, edges, kwargs, message):
    with pytest.raises(ValueError, match=message):
        SpatialNetwork(nodes, edges, **kwargs)


def test_reverse_edges_with_the_same_key_are_distinct_when_directed():
    edges = pd.DataFrame({"u": [1, 2], "v": [2, 1], "key": [0, 0]})
    assert len(SpatialNetwork(small_nodes(), edges, directed=True).edges) == 2


def test_tables_returned_are_copies(kigali):
    nodes = kigali.nodes
    nodes["extra"] = 1
    assert "extra" not in kigali.nodes.columns


# ----- networkx


def test_networkx_round_trip(kigali):
    G = kigali.to_networkx()
    assert isinstance(G, nx.MultiDiGraph)
    assert (G.number_of_nodes(), G.number_of_edges()) == (1209, 2802)
    assert G.graph["crs"] == "EPSG:4326" and G.graph["bbox"] == kigali.metadata["bbox"]

    back = SpatialNetwork.from_networkx(G)
    assert back.directed and back.metadata == kigali.metadata
    assert set(back.node_index) == set(kigali.node_index)
    original = kigali.edges.set_index(["u", "v", "key"]).sort_index()
    rebuilt = back.edges.set_index(["u", "v", "key"]).sort_index()
    assert rebuilt["length"].to_numpy() == pytest.approx(original["length"].to_numpy())
    assert rebuilt["geometry"].notna().sum() == original["geometry"].notna().sum()
    assert (rebuilt["kind"] == "road").all()


def test_to_networkx_leaves_out_missing_attributes(kigali):
    G = kigali.to_networkx()
    straight = kigali.edges.loc[kigali.edges["geometry"].isna()].iloc[0]
    assert "geometry" not in G.edges[straight["u"], straight["v"], straight["key"]]
    node = G.nodes[kigali.node_index[0]]
    assert {"x", "y", "kind", "street_count"} <= set(node)


def test_from_networkx_simple_graph():
    G = nx.Graph()
    G.add_node("a", x=30.0, y=-1.95)
    G.add_node("b", x=30.001, y=-1.95)
    G.add_edge("a", "b", name="road")
    net = SpatialNetwork.from_networkx(G)
    assert not net.directed
    assert net.edges["key"].tolist() == [0]
    assert net.edges["length"].iloc[0] == pytest.approx(haversine(-1.95, 30.0, -1.95, 30.001))


def test_from_networkx_projected_graph():
    to_utm = Transformer.from_crs("EPSG:4326", UTM_35S, always_xy=True)
    G = nx.MultiDiGraph(crs=UTM_35S)
    for node, (x, y) in zip([1, 2], [(30.0, -1.95), (30.001, -1.95)]):
        px, py = to_utm.transform(x, y)
        G.add_node(node, x=px, y=py)
    G.add_edge(1, 2)
    net = SpatialNetwork.from_networkx(G)
    assert net.nodes.loc[2, "x"] == pytest.approx(30.001, abs=1e-9)
    assert "crs" not in net.metadata


def test_from_networkx_needs_coordinates():
    G = nx.Graph()
    G.add_edge(1, 2)
    with pytest.raises(ValueError, match="'x'"):
        SpatialNetwork.from_networkx(G)


def test_undirected_to_directed_networkx_adds_reversed_edges():
    line = LineString([(30.0, -1.95), (30.0005, -1.9499), (30.001, -1.95)])
    net = SpatialNetwork(small_nodes(), small_edges(geometry=[line, None, None]), directed=False)
    G = net.to_networkx(directed=True)
    assert G.number_of_edges() == 6
    assert list(G.edges[2, 1, 0]["geometry"].coords) == list(line.coords)[::-1]
    assert "geometry" not in G.edges[3, 2, 0]


# ----- from_points


def line_of_points():
    lats = [-1.95, -1.95, -1.95, -1.95]
    lons = [30.0, 30.001, 30.003, 30.006]
    return pd.DataFrame({"site": ["a", "b", "c", "d"], "lat": lats, "lon": lons, "pop": [1, 2, 3, 4]})


def test_from_points_knn():
    net = SpatialNetwork.from_points(line_of_points(), k=1, id_column="site")
    assert not net.directed
    assert list(net.edges[["u", "v"]].itertuples(index=False, name=None)) == [("a", "b"), ("b", "c"), ("c", "d")]
    assert set(net.nodes["kind"]) == {"point"} and set(net.edges["kind"]) == {"point_link"}
    assert list(net.nodes.columns) == ["x", "y", "kind", "pop"]
    assert net.edges["length"].iloc[0] == pytest.approx(haversine(-1.95, 30.0, -1.95, 30.001))


def test_from_points_knn_with_max_distance_and_large_k():
    net = SpatialNetwork.from_points(line_of_points(), k=10, max_distance=250, id_column="site")
    assert len(net.edges) == 2
    assert net.edges["length"].max() <= 250


def test_from_points_distance():
    net = SpatialNetwork.from_points(line_of_points(), connect="distance", max_distance=400)
    assert sorted(map(tuple, net.edges[["u", "v"]].to_numpy())) == [(0, 1), (0, 2), (1, 2), (2, 3)]


def test_from_points_tuples_are_lat_lon():
    net = SpatialNetwork.from_points([(-1.95, 30.0), (-1.949, 30.001)], k=1)
    assert net.nodes["x"].tolist() == [30.0, 30.001]
    assert net.nodes["y"].tolist() == [-1.95, -1.949]


def test_from_points_shapely_points_and_projected_geodataframe():
    from_points = SpatialNetwork.from_points([Point(30.0, -1.95), Point(30.001, -1.95)], k=1)
    gdf = gpd.GeoDataFrame(geometry=[Point(30.0, -1.95), Point(30.001, -1.95)], crs="EPSG:4326").to_crs(UTM_35S)
    from_gdf = SpatialNetwork.from_points(gdf, k=1)
    assert from_gdf.nodes["x"].to_numpy() == pytest.approx(from_points.nodes["x"].to_numpy(), abs=1e-9)


def test_from_points_weight_func():
    net = SpatialNetwork.from_points(
        line_of_points(),
        k=1,
        id_column="site",
        weight="cost",
        weight_func=lambda nodes_u, nodes_v, length: length * (nodes_u["pop"].to_numpy() + nodes_v["pop"].to_numpy()),
    )
    assert net.weight == "cost"
    assert net.edges["cost"].iloc[0] == pytest.approx(net.edges["length"].iloc[0] * 3)


def test_from_points_single_point_has_no_edges():
    net = SpatialNetwork.from_points([(-1.95, 30.0)])
    assert (len(net), len(net.edges)) == (1, 0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"weight": "cost"}, "weight_func"),
        ({"weight_func": lambda u, v, length: length}, "other than 'length'"),
        ({"connect": "delaunay"}, "connect"),
        ({"connect": "distance"}, "max_distance"),
        ({"k": 0}, "at least 1"),
    ],
)
def test_from_points_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SpatialNetwork.from_points(line_of_points(), **kwargs)


def test_from_points_rejects_reserved_columns_and_repeated_ids():
    points = line_of_points().assign(kind="school")
    with pytest.raises(ValueError, match="reserved"):
        SpatialNetwork.from_points(points)
    with pytest.raises(ValueError, match="unique"):
        SpatialNetwork.from_points(line_of_points().assign(site="a"), id_column="site")


def test_from_points_rejects_non_points():
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs="EPSG:4326")
    with pytest.raises(ValueError, match="Point"):
        SpatialNetwork.from_points(gdf)


# ----- accessors


def test_node_coordinates_are_y_x(kigali):
    ids = kigali.node_index[[5, 2]]
    coords = kigali.node_coordinates(ids)
    expected = kigali.nodes.loc[ids, ["y", "x"]].to_numpy()
    assert np.array_equal(coords, expected)
    assert kigali.node_coordinates().shape == (1209, 2)


def test_node_coordinates_projected(kigali):
    ids = kigali.node_index[:3]
    x, y = Transformer.from_crs("EPSG:4326", UTM_35S, always_xy=True).transform(
        kigali.nodes.loc[ids, "x"].to_numpy(), kigali.nodes.loc[ids, "y"].to_numpy()
    )
    assert kigali.node_coordinates(ids, crs=UTM_35S) == pytest.approx(np.column_stack([y, x]))


def test_unknown_node_ids_raise_key_error(kigali):
    with pytest.raises(KeyError, match="not in the network"):
        kigali.node_coordinates([kigali.node_index[0], -12345])


def test_geodataframes(kigali):
    assert kigali.nodes_gdf is kigali.nodes_gdf
    assert kigali.nodes_gdf.crs == "EPSG:4326"
    edges = kigali.edges_gdf
    assert edges.geometry.notna().all()
    straight = kigali.edges["geometry"].isna().to_numpy()
    first = edges[straight].iloc[0]
    start = kigali.nodes.loc[first["u"], ["x", "y"]].tolist()
    end = kigali.nodes.loc[first["v"], ["x", "y"]].tolist()
    assert [list(c) for c in first.geometry.coords] == [start, end]
    nodes_utm, edges_utm = kigali.to_geodataframes(crs=UTM_35S)
    assert nodes_utm.crs == UTM_35S and edges_utm.crs == UTM_35S


def test_degree_matches_networkx(kigali):
    G = kigali.to_networkx()
    assert kigali.degree().to_dict() == dict(G.degree())
    simple = nx.Graph(G.to_undirected())
    assert kigali.degree(directed=False).to_dict() == dict(simple.degree())


def test_bounds_total_length_and_info(kigali):
    minx, miny, maxx, maxy = kigali.bounds
    assert minx <= kigali.nodes["x"].min() and maxy >= kigali.nodes["y"].max()
    assert kigali.total_length() == pytest.approx(kigali.edges["length"].sum())
    by_kind = kigali.total_length(by="kind")
    assert by_kind["road"] == pytest.approx(kigali.total_length())
    info = kigali.info()
    assert info["nodes"] == 1209 and info["edge_kinds"] == {"road": 2802}
    assert info["edges_with_geometry"] == int(kigali.edges["geometry"].notna().sum())
    assert repr(kigali) == "SpatialNetwork(nodes=1,209, edges=2,802, directed=True, weight='length')"


def test_node_index_maps_positions_and_ids(kigali):
    index = kigali.node_index
    assert index.get_indexer(index[[3, 7]]).tolist() == [3, 7]
