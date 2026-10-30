"""Sub-task 7: insert_points against the section 9 target, and against the old merge_points_to_graph.

Usage: insertion.py ours <graph> | insertion.py legacy <graph> <n_points>
Points are road nodes moved by about 200 m at random, so most lie near a road, as schools do.
"""
import gc, os, sys, time, warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

from gigaspatial.processing.network import SpatialNetwork


def sample_points(net, n, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(net), n, replace=False)
    return pd.DataFrame(
        {
            "coordinate_id": [f"school-{i}" for i in range(n)],
            "lat": net.nodes["y"].to_numpy()[pick] + rng.normal(0, 0.0018, n),
            "lon": net.nodes["x"].to_numpy()[pick] + rng.normal(0, 0.0018, n),
        }
    )


def fresh(net):
    return SpatialNetwork._from_tables(
        net._nodes, net._edges, directed=net.directed, weight=net.weight, positions=net._edge_positions
    )


def ours(g):
    net = SpatialNetwork.from_networkx(load_graph(g))
    gc.collect()
    log(f"\n### Sub-task 7 — insert_points, {g} ({len(net):,} nodes, {len(net.edges):,} edges)\n")
    rows = []
    t, _ = timeit(lambda: fresh(net)._projected_roads)
    rows.append(["road edges projected to UTM and indexed (first insertion only)", f"{t:.2f}", ""])
    net._projected_roads
    for n in (1_000, 10_000):
        points = sample_points(net, n)
        for chain in (False, True):
            t, out = timeit(lambda: net.insert_points(points, id_column="coordinate_id", chain=chain), 3)
            kinds = out.edges["kind"].value_counts()
            label = f"{n:,} points, chain={chain} ({kinds.get('split', 0):,} split pieces, {kinds.get('point_link', 0):,} point links)"
            rows.append([label, f"{t:.2f}", "under 10 s" if n == 10_000 and not chain else ""])
    points = sample_points(net, 1_000)
    out = net.insert_points(points, id_column="coordinate_id", chain=True, snap_to_node=0)
    joined = out.edges[out.edges["kind"].isin(["connection", "point_link"]) & out.edges["u"].astype(str).str.startswith("school")]
    rows.append(["1,000 points, chain=True, snap_to_node=0: total connection length (m)", f"{joined['length'].sum():,.0f}", ""])
    table(["operation", "s", "target"], rows)


def legacy(g, n):
    import geopandas as gpd
    from legacy_merge import merge_points_to_graph

    G = load_graph(g)
    net = SpatialNetwork.from_networkx(G)
    points = sample_points(net, n)
    del net
    gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
    t = time.perf_counter()
    merged = merge_points_to_graph(G, gdf)
    elapsed = time.perf_counter() - t
    total = sum(d["length"] for *_, d in merged.edges(data=True) if d.get("name") == "point_merge_edge")
    log(f"| merge_points_to_graph, {n:,} points (legacy) | {elapsed:.1f} | total connection length {total:,.0f} m |")


if __name__ == "__main__":
    if sys.argv[1] == "ours":
        ours(sys.argv[2])
    else:
        legacy(sys.argv[2], int(sys.argv[3]))
