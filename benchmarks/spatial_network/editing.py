"""Sub-task 4: SpatialNetwork editing against the section 9 targets.

Usage: editing.py <graph> [reps]
Needs <graph>.pkl from step0.py.
"""
import gc, os, sys, warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

from gigaspatial.processing.network import SpatialNetwork


def main(g, reps):
    net = SpatialNetwork.from_networkx(load_graph(g))
    gc.collect()
    n, m = len(net), len(net.edges)
    log(f"\n### Sub-task 4 — editing, {g} ({n:,} nodes, {m:,} edges), mean of {reps}\n")
    rng = np.random.default_rng(0)
    minx, miny, maxx, maxy = net.bounds
    # The 4% box of the Rwanda bbox around Kigali used for the baselines (partB.py B5).
    area = box(29.856, -2.138, 30.264, -1.762)
    points = list(zip(rng.uniform(miny, maxy, 10_000), rng.uniform(minx, maxx, 10_000)))
    ids = net.node_index[rng.choice(n, 10_000, replace=False)]
    rows = []

    def add(label, fn, target=""):
        t, out = timeit(fn, reps)
        rows.append([label, f"{t:.3f}", target])
        return out

    add("first STRtree on edges (cached per network)", lambda: SpatialNetwork._from_tables(
        net._nodes, net._edges, directed=True, weight="length", positions=net._edge_positions)._edge_tree)
    clipped = add("clip, 4% of the area (index built)", lambda: net.clip(area), "under 0.1 s")
    rows[-1][0] += f" ({len(clipped.edges):,} edges)"
    add("first KD-tree on nodes (cached per network)", lambda: SpatialNetwork._from_tables(
        net._nodes, net._edges, directed=True, weight="length", positions=net._edge_positions)._node_tree)
    add("nearest_nodes, 10,000 points (tree built)", lambda: net.nearest_nodes(points))
    add("filter_nodes(geometry=area)", lambda: net.filter_nodes(geometry=area))
    add("filter_nodes(ids=10,000)", lambda: net.filter_nodes(ids=ids))
    add("remove_nodes(ids=10,000)", lambda: net.remove_nodes(ids=ids))
    add("filter_edges(mask)", lambda: net.filter_edges(mask=net._edges["length"].to_numpy() > 100))
    add("set_node_attribute, Series for 10,000 nodes", lambda: net.set_node_attribute("school", pd.Series(1, index=ids)))
    add("update_nodes, 10,000 rows by id", lambda: net.update_nodes(pd.DataFrame({"pop": rng.random(10_000)}, index=ids)))
    sample = net._edges.sample(10_000, random_state=0)[["u", "v", "key"]]
    add("update_edges, 10,000 rows", lambda: net.update_edges(sample.assign(cost=1.0)))
    add("remove_edges, 10,000 (u, v, key)", lambda: net.remove_edges(sample))
    u = net.node_index[rng.choice(n, 10_000)]
    v = net.node_index[rng.choice(n, 10_000)]
    add("add_weighted_edges, 10,000 edges (revalidates the network)", lambda: net.add_weighted_edges(u, v, rng.random(10_000)))
    add("drop_geometry", net.drop_geometry)
    add("to_undirected", net.to_undirected)
    west = net.clip(box(minx, miny, (minx + maxx) / 2 + 0.01, maxy))
    east = net.clip(box((minx + maxx) / 2 - 0.01, miny, maxx, maxy))
    add("concat of two overlapping halves", lambda: SpatialNetwork.concat([west, east]))
    table(["operation", "s", "target"], rows)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
