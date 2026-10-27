import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, box

from gigaspatial.processing.network import SpatialNetwork
from gigaspatial.processing.network.geodesy import haversine
from gigaspatial.processing.network.kinds import EDGE_KINDS, NODE_KINDS

UTM_35S = "EPSG:32735"
CENTRE = box(30.055, -1.955, 30.07, -1.945)


def square(directed=True):
    """Four nodes in a square, with a parallel edge between 1 and 2."""
    nodes = pd.DataFrame(
        {"x": [30.0, 30.001, 30.001, 30.0], "y": [-1.95, -1.95, -1.949, -1.949], "count": np.array([1, 2, 3, 4], dtype=np.int16)},
        index=pd.Index([1, 2, 3, 4]),
    )
    edges = pd.DataFrame({"u": [1, 1, 2, 3, 4], "v": [2, 2, 3, 4, 1], "length": [111.0, 150.0, 111.0, 111.0, 111.0]})
    return SpatialNetwork(nodes, edges, directed=directed)


def edge_set(net):
    return set(net.edges[["u", "v", "key"]].itertuples(index=False, name=None))


def assert_positions_consistent(net):
    G = net.to_networkx()
    assert net.degree().to_dict() == dict(G.degree())


# ----- nearest_nodes


def test_nearest_nodes_match_brute_force(kigali):
    rng = np.random.default_rng(0)
    minx, miny, maxx, maxy = kigali.bounds
    lat, lon = rng.uniform(miny, maxy, 50), rng.uniform(minx, maxx, 50)
    found = kigali.nearest_nodes(list(zip(lat, lon)), return_distance=True)
    x, y = kigali.nodes["x"].to_numpy(), kigali.nodes["y"].to_numpy()
    for i in range(50):
        d = haversine(lat[i], lon[i], y, x)
        assert found["node_id"].iloc[i] == kigali.node_index[d.argmin()]
        assert found["distance"].iloc[i] == pytest.approx(d.min(), rel=1e-6)


def test_nearest_nodes_max_distance_and_index():
    net = square()
    points = pd.DataFrame({"lat": [-1.95, 0.0], "lon": [30.0, 0.0]}, index=["near", "far"])
    found = net.nearest_nodes(points, max_distance=100)
    assert found.index.tolist() == ["near", "far"]
    assert found["near"] == 1 and found["far"] is pd.NA
    assert str(found.dtype) == "Int64"


def test_nearest_nodes_text_ids_missing_is_none():
    points = pd.DataFrame({"site": ["a", "b"], "lat": [-1.95, -1.95], "lon": [30.0, 30.001]})
    net = SpatialNetwork.from_points(points, k=1, id_column="site")
    assert net.nearest_nodes([(-1.95, 30.0), (0.0, 0.0)], max_distance=10).tolist() == ["a", None]


# ----- filter and remove nodes


def test_filter_nodes_by_ids_keeps_edges_between_them(kigali):
    keep = list(kigali.node_index[:300])
    sub = kigali.filter_nodes(ids=keep)
    assert set(sub.node_index) == set(keep)
    expected = kigali.edges[kigali.edges["u"].isin(keep) & kigali.edges["v"].isin(keep)]
    assert len(sub.edges) == len(expected)
    assert_positions_consistent(sub)


def test_filter_nodes_by_mask(kigali):
    mask = kigali.nodes["street_count"] >= 3
    by_series = kigali.filter_nodes(mask=mask)
    by_array = kigali.filter_nodes(mask=mask.to_numpy())
    assert len(by_series) == int(mask.sum()) == len(by_array)
    with pytest.raises(ValueError, match="one value per node"):
        kigali.filter_nodes(mask=[True, False])
    with pytest.raises(KeyError, match="not node IDs"):
        kigali.filter_nodes(mask=pd.Series([True], index=[-1]))


def test_filter_nodes_by_geometry(kigali):
    inside = kigali.filter_nodes(geometry=CENTRE)
    x, y = kigali.nodes["x"], kigali.nodes["y"]
    expected = (x >= 30.055) & (x <= 30.07) & (y >= -1.955) & (y <= -1.945)
    assert set(inside.node_index) == set(kigali.node_index[expected.to_numpy()])
    assert len(kigali.filter_nodes(geometry=CENTRE, predicate="within")) <= len(inside)
    assert_positions_consistent(inside)


def test_filter_nodes_by_projected_geodataframe(kigali):
    to_utm = Transformer.from_crs("EPSG:4326", UTM_35S, always_xy=True)
    cx, cy = to_utm.transform(30.0625, -1.95)
    area = gpd.GeoDataFrame(geometry=[box(cx - 500, cy - 500, cx + 500, cy + 500)], crs=UTM_35S)
    nx_, ny_ = to_utm.transform(kigali.nodes["x"].to_numpy(), kigali.nodes["y"].to_numpy())
    expected = (np.abs(nx_ - cx) <= 500) & (np.abs(ny_ - cy) <= 500)
    assert len(kigali.filter_nodes(geometry=area)) == int(expected.sum())


def test_filter_nodes_argument_errors(kigali):
    with pytest.raises(ValueError, match="exactly one"):
        kigali.filter_nodes()
    with pytest.raises(ValueError, match="exactly one"):
        kigali.filter_nodes(ids=[kigali.node_index[0]], geometry=CENTRE)
    with pytest.raises(ValueError, match="predicate"):
        kigali.filter_nodes(geometry=CENTRE, predicate="touches")
    with pytest.raises(TypeError, match="shapely geometry"):
        kigali.filter_nodes(geometry=[1, 2, 3])


def test_remove_nodes_removes_their_edges(kigali):
    node = kigali.degree().idxmax()
    removed = kigali.remove_nodes(ids=[node])
    assert node not in removed.node_index
    assert len(removed.edges) == len(kigali.edges) - kigali.degree()[node]
    assert_positions_consistent(removed)


def test_remove_nodes_by_points_and_geometry(kigali):
    target = kigali.node_index[10]
    lat, lon = kigali.node_coordinates([target])[0]
    assert target not in kigali.remove_nodes(points=[(lat, lon)], max_distance=1).node_index
    assert len(kigali.remove_nodes(points=[(0.0, 0.0)], max_distance=1)) == len(kigali)
    outside = kigali.remove_nodes(geometry=CENTRE)
    assert len(outside) == len(kigali) - len(kigali.filter_nodes(geometry=CENTRE))


# ----- node attributes


def test_set_node_attribute_forms():
    net = square()
    assert net.set_node_attribute("school", True).nodes["school"].tolist() == [True] * 4
    assert net.set_node_attribute("rank", [4, 3, 2, 1]).nodes["rank"].tolist() == [4, 3, 2, 1]
    partial = net.set_node_attribute("count", {2: 20})
    assert partial.nodes["count"].tolist() == [1, 20, 3, 4]
    assert partial.nodes["count"].dtype == np.int16
    assert net.set_node_attribute("count", {2: 2.5}).nodes["count"].tolist() == [1, 2.5, 3, 4]
    new = net.set_node_attribute("name", pd.Series(["a"], index=[3])).nodes["name"]
    assert new[3] == "a" and new.drop(3).isna().all()
    assert net.nodes["count"].tolist() == [1, 2, 3, 4]


def test_set_node_kind_stays_categorical():
    kinds = square().set_node_attribute("kind", {1: "split", 2: "school"}).nodes["kind"]
    assert list(kinds.cat.categories[:3]) == list(NODE_KINDS)
    assert kinds.tolist() == ["split", "school", "road", "road"]


def test_set_node_attribute_errors():
    with pytest.raises(ValueError, match="cannot be changed"):
        square().set_node_attribute("x", 0.0)
    with pytest.raises(KeyError, match="not in the network"):
        square().set_node_attribute("count", {99: 1})
    with pytest.raises(ValueError, match="Expected 4 values"):
        square().set_node_attribute("count", [1, 2])


def test_update_nodes_by_id_skips_missing_values():
    frame = pd.DataFrame({"count": [10, np.nan], "name": ["a", "b"]}, index=[1, 2])
    nodes = square().update_nodes(frame).nodes
    assert nodes["count"].tolist() == [10, 2, 3, 4]
    assert nodes["name"].tolist()[:2] == ["a", "b"] and nodes["name"].iloc[2:].isna().all()
    with_column = square().update_nodes(frame.reset_index().rename(columns={"index": "node_id"}))
    assert with_column.nodes["count"].tolist() == [10, 2, 3, 4]


def test_update_nodes_by_coordinates():
    frame = pd.DataFrame({"lat": [-1.95001, 0.0], "lon": [30.00101, 0.0], "school": ["s1", "s2"]})
    nodes = square().update_nodes(frame, by="coordinates", max_distance=50).nodes
    assert nodes.loc[2, "school"] == "s1"
    assert nodes["school"].notna().sum() == 1
    with pytest.raises(ValueError, match="by must be"):
        square().update_nodes(frame, by="name")


def test_add_nodes():
    added = square().add_nodes(pd.DataFrame({"lat": [-1.9495], "lon": [30.0005], "name": ["school"]}, index=["s1"]))
    assert len(added) == 5 and added.nodes.loc["s1", "kind"] == "point"
    assert added.nodes.loc["s1", "name"] == "school"
    assert added.degree()["s1"] == 0
    assert_positions_consistent(added)
    with pytest.raises(ValueError, match="already in the network"):
        square().add_nodes(pd.DataFrame({"lat": [-1.95], "lon": [30.0]}, index=[1]))


# ----- edges


def test_filter_edges(kigali):
    by_mask = kigali.filter_edges(mask=(kigali.edges["highway"] == "primary").to_numpy())
    assert set(by_mask.edges["highway"]) == {"primary"} and len(by_mask) == len(kigali)
    within = kigali.filter_edges(geometry=CENTRE, predicate="within")
    assert all(CENTRE.contains(g) for g in within.edges_gdf.geometry)
    assert_positions_consistent(within)
    with pytest.raises(ValueError, match="exactly one"):
        kigali.filter_edges()


def test_remove_edges_forms():
    net = square()
    assert len(net.remove_edges([(1, 2, 1)]).edges) == 4
    assert len(net.remove_edges([(1, 2)]).edges) == 3
    assert len(net.remove_edges(pd.DataFrame({"u": [2], "v": [3]})).edges) == 4
    assert len(net.remove_edges(mask=[True, False, False, False, False]).edges) == 4
    with pytest.raises(KeyError, match="no edge"):
        net.remove_edges([(2, 1)])
    with pytest.raises(KeyError, match="not in the network"):
        net.remove_edges([(1, 2, 7)])


def test_undirected_edges_match_either_way():
    net = square(directed=False)
    assert len(net.remove_edges([(2, 1, 0)]).edges) == 4
    assert net.set_edge_attribute("name", {(3, 2, 0): "x"}).edges["name"].tolist()[2] == "x"


def test_set_edge_attribute_forms():
    net = square()
    keyed = pd.Series([5.0], index=pd.MultiIndex.from_tuples([(1, 2, 1)], names=["u", "v", "key"]))
    assert net.set_edge_attribute("length", keyed).edges["length"].tolist() == [111.0, 5.0, 111.0, 111.0, 111.0]
    assert net.set_edge_attribute("surface", "paved").edges["surface"].tolist() == ["paved"] * 5
    kinds = net.set_edge_attribute("kind", {(1, 2, 0): "connection"}).edges["kind"]
    assert list(kinds.cat.categories[: len(EDGE_KINDS)]) == list(EDGE_KINDS) and kinds.iloc[0] == "connection"
    with pytest.raises(ValueError, match=r"\(u, v, key\)"):
        net.set_edge_attribute("name", {(1, 2): "x"})
    with pytest.raises(ValueError, match="cannot be changed"):
        net.set_edge_attribute("u", 1)


def test_update_edges():
    frame = pd.DataFrame({"u": [1, 2], "v": [2, 3], "key": [0, 0], "length": [100.0, np.nan], "name": ["a", "b"]})
    edges = square().update_edges(frame).edges
    assert edges["length"].tolist() == [100.0, 150.0, 111.0, 111.0, 111.0]
    names = edges["name"]
    assert names.iloc[0] == "a" and pd.isna(names.iloc[1]) and names.iloc[2] == "b"
    indexed = square().update_edges(frame.set_index(["u", "v", "key"]))
    assert indexed.edges["length"].iloc[0] == 100.0
    with pytest.raises(ValueError, match="key"):
        square().update_edges(frame.drop(columns="key"))


def test_add_weighted_edges_keys_lengths_and_types(kigali):
    net = square()
    added = net.add_weighted_edges([1, 1, 2], [2, 2, 4], [9.0, 8.0, 7.0], weight="cost", kind="connection")
    new = added.edges.iloc[5:]
    assert new["key"].tolist() == [2, 3, 0]
    assert new["cost"].tolist() == [9.0, 8.0, 7.0]
    assert new["length"].iloc[2] == pytest.approx(haversine(-1.95, 30.001, -1.949, 30.0))
    assert set(new["kind"]) == {"connection"} and added.edges["cost"].iloc[:5].isna().all()
    assert_positions_consistent(added)

    a, b = kigali.node_index[:2]
    line = LineString(kigali.node_coordinates([a, b])[:, ::-1])
    grown = kigali.add_weighted_edges([a], [b], [12.5], name="link", geometry=[line])
    assert grown.edges["length"].iloc[-1] == 12.5
    assert isinstance(grown.edges["highway"].dtype, pd.CategoricalDtype)
    with pytest.raises(KeyError):
        net.add_weighted_edges([1], [99], [1.0])


def test_add_weighted_edges_undirected_counts_either_direction():
    added = square(directed=False).add_weighted_edges([2], [1], [1.0])
    assert added.edges["key"].iloc[-1] == 2


def test_clip_keeps_edges_and_their_endpoints(kigali):
    clipped = kigali.clip(CENTRE)
    rows = kigali.edges_gdf.geometry.intersects(CENTRE).to_numpy()
    assert len(clipped.edges) == int(rows.sum())
    ends = set(kigali.edges.loc[rows, "u"]) | set(kigali.edges.loc[rows, "v"])
    assert set(clipped.node_index) == ends
    assert_positions_consistent(clipped)


def test_drop_geometry(kigali):
    plain = kigali.drop_geometry()
    assert "geometry" not in plain.edges.columns and "geometry" in kigali.edges.columns
    assert plain.edges_gdf.geometry.notna().all()


def test_to_undirected_keeps_the_shortest_edge(kigali):
    net = square()
    undirected = net.to_undirected()
    assert not undirected.directed
    assert len(undirected.edges) == 4 and set(undirected.edges["key"]) == {0}
    assert undirected.edges["length"].iloc[0] == 111.0

    simple = kigali.to_undirected()
    pairs = {frozenset(p) for p in kigali.edges[["u", "v"]].itertuples(index=False, name=None)}
    assert len(simple.edges) == len(pairs)
    assert simple.degree().to_dict() == kigali.degree(directed=False).to_dict()


def test_concat(kigali):
    west = kigali.clip(box(30.045, -1.965, 30.066, -1.935))
    east = kigali.clip(box(30.064, -1.965, 30.085, -1.935))
    combined = SpatialNetwork.concat([west, east])
    assert edge_set(combined) == edge_set(kigali)
    assert combined.node_index.is_unique
    assert combined.metadata == kigali.metadata
    assert_positions_consistent(combined)
    with pytest.raises(ValueError, match="all directed"):
        SpatialNetwork.concat([square(), square(directed=False)])


def test_concat_undirected_drops_reversed_duplicates_and_keeps_parts():
    a = square(directed=False)
    nodes = a.nodes.loc[[1, 2]]
    b = SpatialNetwork(nodes, pd.DataFrame({"u": [2], "v": [1], "key": [0]}), directed=False, metadata={"part": "b"})
    combined = SpatialNetwork.concat([a, b])
    assert len(combined.edges) == len(a.edges)
    assert combined.metadata == {"parts": [{}, {"part": "b"}]}


def test_edits_do_not_change_the_original_and_have_their_own_caches(kigali):
    before = (len(kigali), len(kigali.edges))
    sub = kigali.filter_nodes(geometry=CENTRE)
    assert (len(kigali), len(kigali.edges)) == before
    lat, lon = kigali.node_coordinates([kigali.node_index[0]])[0]
    assert kigali.nearest_nodes([(lat, lon)]).iloc[0] == kigali.node_index[0]
    assert sub.nearest_nodes([(lat, lon)]).iloc[0] in set(sub.node_index)
