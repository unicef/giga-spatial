"""Sub-task 5: SpatialNetwork algorithms against the section 9 targets.

Usage: algorithms.py <graph> [reps]
Needs <graph>.pkl from step0.py; pandana rows need pandana installed.
"""
import gc, os, sys, warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

from gigaspatial.processing.network import SpatialNetwork


def fresh(net):
    """The same network without cached structures."""
    return SpatialNetwork._from_tables(
        net._nodes, net._edges, directed=net.directed, weight=net.weight, positions=net._edge_positions
    )


def main(g, reps):
    net = SpatialNetwork.from_networkx(load_graph(g))
    gc.collect()
    n = len(net)
    log(f"\n### Sub-task 5 — algorithms, {g} ({n:,} nodes, {len(net.edges):,} edges), mean of {reps}\n")
    rng = np.random.default_rng(0)
    ids = net.node_index
    origins = list(ids[rng.choice(n, 100, replace=False)])
    destinations = list(ids[rng.choice(n, 1000, replace=False)])
    pair_o = list(ids[rng.choice(n, 100_000)])
    pair_d = list(ids[rng.choice(n, 100_000)])
    rows = []

    def add(label, fn, target=""):
        t, out = timeit(fn, reps)
        rows.append([label, f"{t:.3f}", target])
        return out

    add("adjacency() (first build)", lambda: fresh(net).adjacency(), "under 0.3 s")
    net.adjacency()
    add("shortest_path_lengths, 100 x 1,000, scipy", lambda: net.shortest_path_lengths(origins, destinations), "under 3 s")
    add("shortest_path_lengths, 100 x 1,000, scipy, directed=False", lambda: net.shortest_path_lengths(origins, destinations, directed=False))
    add("shortest_paths, 1,000 pairs (10 origins)", lambda: net.shortest_paths(origins[:10] * 100, destinations))
    add("shortest_paths, 1,000 pairs (10 origins), geometry", lambda: net.shortest_paths(origins[:10] * 100, destinations, geometry=True))
    add("reachable, 100 origins, 5 km", lambda: net.reachable(origins, 5000))
    add("nearest_by_network, 10,000 origins to 100 targets, k=1", lambda: net.nearest_by_network(list(ids[rng.choice(n, 10_000)]), origins))
    add("nearest_by_network, 100 origins to 1,000 targets, k=3", lambda: net.nearest_by_network(origins, destinations, k=3))
    add("count_within, 1,000 origins, 100 targets, 5 km", lambda: net.count_within(destinations, origins, 5000))
    add("connected_components()", net.connected_components)
    add("connected_components(strong=True)", lambda: net.connected_components(strong=True))
    add("minimum_spanning_tree()", net.minimum_spanning_tree)
    joined = add("connect_components() (first; components found)", lambda: fresh(net).connect_components())
    rows[-1][0] += f" ({(joined.edges['kind'] == 'component_link').sum()} link rows)"
    islands = net.remove_nodes(ids=list(ids[rng.choice(n, n * 3 // 100, replace=False)]))
    parts = islands.connected_components().nunique()
    joined = add(f"connect_components() after removing 3% of nodes ({parts:,} components)", lambda: fresh(islands).connect_components())
    rows[-1][0] += f", {joined.connected_components().nunique()} left"
    try:
        import pandana  # noqa: F401

        add("to_pandana() (contraction hierarchy build)", lambda: fresh(net).to_pandana(), "")
        net.to_pandana()
        add("shortest_path_lengths, 100,000 pairs, pandana", lambda: net.shortest_path_lengths(pair_o, pair_d, pairwise=True, backend="pandana"))
        add("shortest_path_lengths, 100 x 1,000, pandana", lambda: net.shortest_path_lengths(origins, destinations, backend="pandana"))
    except ImportError:
        rows.append(["pandana rows", "skipped", "pandana not installed"])
    table(["operation", "s", "target"], rows)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
