import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString, Point

from gigaspatial.processing.network import SpatialNetwork
from gigaspatial.processing.network.geodesy import haversine, line_lengths

M = 1 / 111_195  # one meter of latitude, in degrees


def road(directed=True, two_way=True, weight="length"):
    """A straight east-west road a-b, about 1.1 km long, two-way unless said otherwise."""
    nodes = pd.DataFrame({"x": [30.0, 30.01], "y": [-1.95, -1.95]}, index=["a", "b"])
    line = LineString([(30.0, -1.95), (30.005, -1.95), (30.01, -1.95)])
    rows = [("a", "b", line)]
    if two_way and directed:
        rows.append(("b", "a", LineString(line.coords[::-1])))
    edges = pd.DataFrame(rows, columns=["u", "v", "geometry"])
    edges["highway"] = "residential"
    if weight != "length":
        edges[weight] = [60.0] * len(edges)
    return SpatialNetwork(nodes, edges, directed=directed, weight=weight)


def points(*coords, ids=None):
    lat, lon = zip(*coords)
    return pd.DataFrame({"lat": lat, "lon": lon}, index=ids or [f"p{i}" for i in range(len(coords))])


def edge(net, u, v):
    rows = net.edges[(net.edges["u"] == u) & (net.edges["v"] == v)]
    assert len(rows) == 1, rows
    return rows.iloc[0]


def assert_valid(net):
    SpatialNetwork(net.nodes, net.edges, directed=net.directed, weight=net.weight)


def test_a_point_splits_its_road_in_both_directions():
    net = road()
    out = net.insert_points(points((-1.95 + 50 * M, 30.004)))
    assert out.nodes.loc[-1, "kind"] == "split" and out.nodes.loc["p0", "kind"] == "point"
    assert out.nodes.loc[-1, "x"] == pytest.approx(30.004) and out.nodes.loc[-1, "y"] == pytest.approx(-1.95)
    forward = edge(out, "a", -1)["length"] + edge(out, -1, "b")["length"]
    backward = edge(out, "b", -1)["length"] + edge(out, -1, "a")["length"]
    assert forward == pytest.approx(net.total_length() / 2, abs=1e-9) and backward == pytest.approx(forward, abs=1e-9)
    connection = edge(out, "p0", -1)
    assert connection["kind"] == "connection" and edge(out, -1, "p0")["kind"] == "connection"
    assert connection["length"] == pytest.approx(50.0, rel=1e-3)
    assert edge(out, "a", -1)["kind"] == "split" and edge(out, "a", -1)["highway"] == "residential"
    assert np.asarray(edge(out, "a", -1)["geometry"].coords)[-1] == pytest.approx([30.004, -1.95])
    expected = [[30.004, -1.95], [30.005, -1.95], [30.01, -1.95]]
    assert np.asarray(edge(out, -1, "b")["geometry"].coords) == pytest.approx(np.array(expected))
    assert_valid(out)


def test_split_pieces_follow_their_geometry():
    out = road().insert_points(points((-1.95 + 50 * M, 30.004)))
    pieces = out.edges[out.edges["kind"] == "split"]
    assert line_lengths(pieces["geometry"]) == pytest.approx(pieces["length"].to_numpy(), rel=1e-6)


def test_snapping_to_an_end_node():
    out = road().insert_points(points((-1.95 + 20 * M, 30.0 + 0.00002)))
    assert "split" not in set(out.nodes["kind"])
    assert edge(out, "p0", "a")["length"] == pytest.approx(haversine(-1.95 + 20 * M, 30.00002, -1.95, 30.0))
    far = road().insert_points(points((-1.95 + 20 * M, 30.00002)), snap_to_node=0)
    assert "split" in set(far.nodes["kind"])


def test_several_points_split_a_road_once():
    out = road().insert_points(points((-1.95 + 10 * M, 30.007), (-1.95 - 10 * M, 30.003)))
    assert sorted(out.nodes.index[out.nodes["kind"] == "split"]) == [-2, -1]
    assert out.nodes.loc[-1, "x"] < out.nodes.loc[-2, "x"]
    for u, v in [("a", -1), (-1, -2), (-2, "b"), ("b", -2), (-2, -1), (-1, "a")]:
        edge(out, u, v)
    assert out.total_length(by="kind")["split"] == pytest.approx(road().total_length())


def test_points_with_the_same_foot_share_a_split_node():
    out = road().insert_points(points((-1.95 + 30 * M, 30.006), (-1.95 - 30 * M, 30.006)))
    assert (out.nodes["kind"] == "split").sum() == 1
    assert edge(out, "p0", -1)["length"] == pytest.approx(edge(out, "p1", -1)["length"], rel=1e-4)


def test_max_distance_and_skipped_points():
    pts = points((-1.95 + 20 * M, 30.002), (-1.95 + 900 * M, 30.002))
    out, skipped = road().insert_points(pts, max_distance=100, return_skipped=True)
    assert "p0" in out.node_index and "p1" not in out.node_index
    assert list(skipped.index) == ["p1"]
    listed, skipped_list = road().insert_points([(-1.95 + 900 * M, 30.002)], max_distance=100, return_skipped=True)
    assert skipped_list == [(-1.95 + 900 * M, 30.002)] and len(listed) == 2


def test_insert_points_errors():
    with pytest.raises(ValueError, match="already node IDs"):
        road().insert_points(points((-1.95, 30.002), ids=["a"]))
    with pytest.raises(ValueError, match="reserved"):
        road().insert_points(points((-1.95, 30.002)).assign(kind="school"))
    with pytest.raises(ValueError, match="unique"):
        road().insert_points(points((-1.95, 30.002), (-1.95, 30.003), ids=["s", "s"]))


def stacked():
    """Three points 10, 20, and 30 m north of the road."""
    return points(*[(-1.95 + d * M, 30.005) for d in (10, 20, 30)])


def test_chain_links_points_to_nearer_points():
    out = road().insert_points(stacked(), chain=True)
    assert edge(out, "p0", -1)["kind"] == "connection"
    assert edge(out, "p1", "p0")["kind"] == "point_link" and edge(out, "p2", "p1")["kind"] == "point_link"
    assert edge(out, "p1", "p0")["length"] == pytest.approx(10.0, rel=1e-3)
    plain = road().insert_points(stacked())
    assert set(plain.edges["kind"]) == {"split", "connection"}


def test_chain_with_max_distance():
    pts = pd.concat([stacked(), points((-1.95 + 1000 * M, 30.009), ids=["far"])])
    out, skipped = road().insert_points(pts, chain=True, max_distance=15, return_skipped=True)
    assert list(skipped.index) == ["far"] and {"p0", "p1", "p2"} <= set(out.node_index)
    none, skipped = road().insert_points(stacked(), chain=True, max_distance=5, return_skipped=True)
    assert list(skipped.index) == ["p0", "p1", "p2"]


def test_one_way_and_undirected_roads():
    one_way = road(two_way=False).insert_points(points((-1.95 + 50 * M, 30.004)))
    assert len(one_way.edges[one_way.edges["kind"] == "split"]) == 2
    assert len(one_way.edges[one_way.edges["kind"] == "connection"]) == 2
    undirected = road(directed=False).insert_points(points((-1.95 + 50 * M, 30.004)))
    assert len(undirected.edges[undirected.edges["kind"] == "connection"]) == 1
    assert len(undirected.edges[undirected.edges["kind"] == "split"]) == 2
    assert_valid(undirected)


def test_split_ids_continue_below_existing_ones():
    once = road().insert_points(points((-1.95 + 50 * M, 30.004)))
    twice = once.insert_points(points((-1.95 + 50 * M, 30.008), ids=["q"]))
    assert sorted(twice.nodes.index[twice.nodes["kind"] == "split"]) == [-2, -1]


def test_weights_are_shared_by_the_pieces():
    out = road(weight="time").insert_points(points((-1.95 + 50 * M, 30.004)))
    assert edge(out, "a", -1)["time"] + edge(out, -1, "b")["time"] == pytest.approx(60.0)
    assert edge(out, "p0", -1)["time"] == pytest.approx(edge(out, "p0", -1)["length"])


def test_points_join_roads_not_earlier_connections():
    first = road().insert_points(points((-1.95 + 200 * M, 30.005), ids=["p"]))
    second = first.insert_points(points((-1.95 + 100 * M, 30.005 + 5 * M), ids=["q"]))
    target = edge(second, "q", second.edges.loc[second.edges["u"] == "q", "v"].iloc[0])
    assert target["length"] == pytest.approx(100.0, rel=1e-2)
    assert second.nodes.loc[target["v"], "kind"] == "split"


def test_point_inputs_and_attributes():
    gdf = gpd.GeoDataFrame({"name": ["school"]}, geometry=[Point(30.004, -1.95 + 50 * M)], crs="EPSG:4326")
    out = road().insert_points(gdf)
    assert out.nodes.loc[0, "name"] == "school" and out.nodes.loc[0, "kind"] == "point"
    listed = road().insert_points([(-1.95 + 50 * M, 30.004)])
    assert 0 in listed.node_index
    no_geometry = road().drop_geometry().insert_points(points((-1.95 + 50 * M, 30.004)))
    assert no_geometry.total_length(by="kind")["split"] == pytest.approx(road().total_length())


# ----- on the Kigali fixture


@pytest.fixture(scope="module")
def schools(kigali):
    rng = np.random.default_rng(0)
    pick = rng.choice(len(kigali), 200, replace=False)
    return pd.DataFrame(
        {
            "school": [f"s{i}" for i in range(200)],
            "lat": kigali.nodes["y"].to_numpy()[pick] + rng.normal(0, 0.001, 200),
            "lon": kigali.nodes["x"].to_numpy()[pick] + rng.normal(0, 0.001, 200),
            "pupils": rng.integers(10, 500, 200),
        }
    )


def test_kigali_insertion(kigali, schools):
    out = kigali.insert_points(schools, id_column="school", snap_to_node=0)
    assert_valid(out)
    lengths = out.total_length(by="kind")
    assert lengths["road"] + lengths["split"] == pytest.approx(kigali.total_length())
    assert (out.degree()[schools["school"]] >= 2).all()
    assert out.nodes.loc["s0", "pupils"] == schools.loc[0, "pupils"]

    # Brute force: the foot on the nearest edge in UTM, measured by great-circle distance like the library.
    utm = "EPSG:32736"
    roads = kigali.edges_gdf.to_crs(utm).geometry.to_numpy()
    projected = gpd.GeoSeries(gpd.points_from_xy(schools["lon"], schools["lat"]), crs="EPSG:4326").to_crs(utm)
    feet = []
    for p in projected:
        line = roads[np.argmin(shapely.distance(roads, p))]
        feet.append(line.interpolate(line.project(p)))
    feet = gpd.GeoSeries(feet, crs=utm).to_crs("EPSG:4326")
    expected = haversine(schools["lat"].to_numpy(), schools["lon"].to_numpy(), feet.y.to_numpy(), feet.x.to_numpy())
    connections = out.edges[out.edges["u"].isin(schools["school"])].set_index("u")["length"]
    assert connections.loc[schools["school"]].to_numpy() == pytest.approx(expected, abs=0.01)


def test_kigali_points_reach_their_road(kigali, schools):
    out = kigali.insert_points(schools, id_column="school")
    connections = out.edges[out.edges["u"].isin(schools["school"])]
    lengths = out.shortest_path_lengths(connections["u"], connections["v"], pairwise=True)
    assert lengths == pytest.approx(connections["length"].to_numpy())
