"""Sub-task 2: SpatialNetwork core against the section 9 targets.

Usage: core.py run <graph> | core.py mem <graph>
Needs <graph>.pkl from step0.py and the lean tables from partA.py (fmt/<graph>/lean_f64).
"""
import gc, os, subprocess, sys, time, warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

from gigaspatial.processing.network import SpatialNetwork


def tables(g):
    base = f"{ROOT}/fmt/{g}/lean_f64"
    nodes = pq.read_table(f"{base}/nodes.parquet").to_pandas().rename(columns={"osmid": "node_id"})
    edges = pq.read_table(f"{base}/edges.parquet").to_pandas()
    edges["geometry"] = shapely.from_wkb(edges["geometry"].to_numpy())
    return nodes, edges


def released_rss():
    gc.collect()
    pa.default_memory_pool().release_unused()
    return rss_mb()


def mem_child(g):
    import tracemalloc

    before = released_rss()
    nodes, edges = tables(g)
    read = released_rss() - before
    tracemalloc.start()
    start = tracemalloc.take_snapshot()
    t = time.perf_counter()
    net = SpatialNetwork(nodes, edges)
    build = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    del nodes, edges
    gc.collect()
    retained = sum(d.size_diff for d in tracemalloc.take_snapshot().compare_to(start, "filename")) / 1e6
    tracemalloc.stop()
    print(f"{released_rss() - before:.0f} {read:.0f} {build:.3f} {retained:.1f} {peak:.1f}")


def run(g):
    # snowflake.connector, imported through gigaspatial.core.io when installed, sets
    # ARROW_DEFAULT_MEMORY_POOL=system for the process; a child inheriting it reports about
    # 300 MB more RSS for the same tables, so the child starts without it.
    env = {k: v for k, v in os.environ.items() if k != "ARROW_DEFAULT_MEMORY_POOL"}
    out = subprocess.run(
        [PY, "-u", "-W", "ignore", __file__, "mem", g], capture_output=True, text=True, check=True, env=env
    )
    mem, read, build, retained, peak = out.stdout.split()

    G = load_graph(g)
    log(f"\n### Sub-task 2 — SpatialNetwork core, {g} ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)\n")
    rows = []

    t, net = timeit(lambda: SpatialNetwork.from_networkx(G))
    rows.append(["from_networkx(osmnx graph)", f3(t), "under 2 s"])
    del G
    gc.collect()

    rows.append(["SpatialNetwork(nodes, edges) from lean tables, validation included", build, ""])
    rows.append(["RSS after reading the tables (MB)", read, ""])
    rows.append(["RSS with the network built and the input tables released (MB)", mem, "under 400 MB"])
    rows.append(["allocations the network keeps beyond its input tables, tracemalloc (MB)", retained, ""])
    rows.append(["peak extra allocations while building, tracemalloc (MB)", peak, ""])

    t, _ = timeit(net.node_coordinates, reps=5)
    rows.append(["node_coordinates() (ms)", f"{t * 1e3:.2f}", "under 5 ms"])
    ids = net.node_index[np.random.default_rng(0).choice(len(net), 10_000, replace=False)]
    t, _ = timeit(lambda: net.node_coordinates(ids), reps=5)
    rows.append(["node_coordinates(10,000 ids) (ms)", f"{t * 1e3:.2f}", ""])
    t, _ = timeit(net.to_networkx)
    rows.append(["to_networkx()", f3(t), "under 3 s"])
    t, _ = timeit(lambda: net.nodes_gdf)
    rows.append(["nodes_gdf, first access", f3(t), ""])
    t, _ = timeit(lambda: net.edges_gdf)
    rows.append(["edges_gdf, first access (fills straight lines)", f3(t), ""])
    t, _ = timeit(net.degree, reps=3)
    rows.append(["degree()", f3(t), ""])
    t, _ = timeit(lambda: net.degree(directed=False), reps=3)
    rows.append(["degree(directed=False)", f3(t), ""])
    t, _ = timeit(net.info, reps=3)
    rows.append(["info()", f3(t), ""])

    rng = np.random.default_rng(1)
    minx, miny, maxx, maxy = net.bounds
    points = pd.DataFrame({"lat": rng.uniform(miny, maxy, 100_000), "lon": rng.uniform(minx, maxx, 100_000)})
    t, pn = timeit(lambda: SpatialNetwork.from_points(points, k=5))
    rows.append([f"from_points, 100,000 points, k=5 ({len(pn.edges):,} edges)", f3(t), ""])
    t, pn = timeit(lambda: SpatialNetwork.from_points(points, connect="distance", max_distance=500))
    rows.append([f"from_points, 100,000 points, within 500 m ({len(pn.edges):,} edges)", f3(t), ""])
    table(["operation", "s", "target"], rows)


def f3(x):
    return f"{x:.3f}"


def convert(graphs):
    """from_networkx and to_networkx on each graph, best of two runs."""
    rows = []
    for g in graphs:
        G = load_graph(g)
        gc.collect()
        t, net = timeit(lambda: SpatialNetwork.from_networkx(G), reps=2)
        rows.append([f"from_networkx, {g} ({G.number_of_edges():,} edges)", f3(t), "under 2 s" if g == "G_large" else ""])
        del G
        gc.collect()
        t, _ = timeit(net.to_networkx)
        rows.append([f"to_networkx, {g}", f3(t), "under 3 s" if g == "G_large" else ""])
        del net
        gc.collect()
    log("\n### Sub-task 8 — networkx conversion after the speed-up\n")
    table(["operation", "s", "target"], rows)


if __name__ == "__main__":
    mode, graphs = sys.argv[1], sys.argv[2:]
    if mode == "mem":
        mem_child(graphs[0])
    elif mode == "convert":
        convert(graphs)
    else:
        run(graphs[0])
