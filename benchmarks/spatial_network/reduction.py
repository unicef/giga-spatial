"""Sub-task 6: simplify and connecting_subnetwork against the section 9 targets.

Usage: reduction.py simplify | reduction.py subnetwork [reps]
Needs G_large_raw.pkl and G_large.pkl from step0.py; simplify also times osmnx's simplify_graph.
"""
import gc, os, sys, time, warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

from gigaspatial.processing.network import SpatialNetwork


def simplify():
    import osmnx as ox

    G = load_graph("G_large_raw")
    log(f"\n### Sub-task 6 — simplify, G_large_raw ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)\n")
    rows = []
    t = time.perf_counter()
    net = SpatialNetwork.from_networkx(G)
    rows.append(["SpatialNetwork.from_networkx (not part of simplify)", f"{time.perf_counter() - t:.2f}", ""])
    t = time.perf_counter()
    simple = net.simplify()
    ours = time.perf_counter() - t
    rows.append([f"SpatialNetwork.simplify ({len(simple):,} nodes, {len(simple.edges):,} edges)", f"{ours:.2f}", "faster than osmnx"])
    del net
    gc.collect()
    t = time.perf_counter()
    Gs = ox.simplify_graph(G)
    theirs = time.perf_counter() - t
    rows.append([f"ox.simplify_graph ({Gs.number_of_nodes():,} nodes, {Gs.number_of_edges():,} edges)", f"{theirs:.2f}", ""])
    rows.append(["speed-up over osmnx", f"{theirs / ours:.0f}x", ""])
    table(["operation", "s", "target"], rows)
    same_nodes = set(simple.node_index) == set(Gs.nodes)
    ours_len = np.sort(simple.edges["length"].to_numpy())
    theirs_len = np.sort([d["length"] for *_, d in Gs.edges(data=True)])
    same_lengths = len(ours_len) == len(theirs_len) and np.allclose(ours_len, theirs_len, atol=1e-6)
    log(f"- same nodes as osmnx: {same_nodes}; same edge lengths: {same_lengths}; "
        f"total length {simple.total_length():,.1f} m vs osmnx {theirs_len.sum():,.1f} m\n")


def subnetwork(reps):
    net = SpatialNetwork.from_networkx(load_graph("G_large"))
    n = len(net)
    log(f"\n### Sub-task 6 — connecting_subnetwork, G_large ({n:,} nodes, {len(net.edges):,} edges), mean of {reps}\n")
    rng = np.random.default_rng(0)
    ids = net.node_index
    rows = []
    net.adjacency(directed=False)
    for k in (100, 1000):
        terminals = list(ids[rng.choice(n, k, replace=False)])
        t, tree = timeit(lambda: net.connecting_subnetwork(terminals), reps)
        rows.append([f"steiner, {k:,} terminals ({len(tree.edges):,} edges, {tree.total_length() / 1000:,.0f} km)", f"{t:.3f}"])
    terminals = list(ids[rng.choice(n, 1000, replace=False)])
    roots = list(ids[rng.choice(n, 20, replace=False)])
    t, sub = timeit(lambda: net.connecting_subnetwork(terminals, roots=roots), reps)
    rows.append([f"rooted steiner, 1,000 terminals, 20 roots ({sub.total_length() / 1000:,.0f} km)", f"{t:.3f}"])
    t, sub = timeit(lambda: net.connecting_subnetwork(terminals, roots=roots, method="nearest_root"), reps)
    rows.append([f"nearest root, 1,000 terminals, 20 roots ({sub.total_length() / 1000:,.0f} km)", f"{t:.3f}"])
    few = terminals[:50]
    t, sub = timeit(lambda: net.connecting_subnetwork(few, roots=roots, method="all_pairs"), reps)
    rows.append([f"all_pairs, 50 terminals, 20 roots ({sub.total_length() / 1000:,.0f} km)", f"{t:.3f}"])
    t, _ = timeit(net.simplify, reps)
    rows.append(["simplify of the already simplified network", f"{t:.3f}"])
    table(["operation", "s"], rows)


if __name__ == "__main__":
    simplify() if sys.argv[1] == "simplify" else subnetwork(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
