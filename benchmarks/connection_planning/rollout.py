"""Sub-task 3: the rollout order (method="shortest_path") against the naive version and the targets.

Usage: rollout.py [terminals ...]
Needs G_large.pkl from benchmarks/spatial_network/step0.py, in $GS_BENCH_DIR (/tmp/gs_bench by default).

The naive version runs one full Dijkstra from the whole connected set per
terminal; the planner updates distances only near each new path.
"""
import os, sys, time, warnings

warnings.filterwarnings("ignore")
import numpy as np
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spatial_network"))
from common import load_graph, log, table  # noqa

from gigaspatial.processing.network import ConnectionPlanner, SpatialNetwork

NAIVE_UP_TO = 1000


def naive(planner, terminal, root):
    """Takahashi and Matsuyama with a full Dijkstra from the connected set at every step."""
    graph = planner._graph
    connected = np.zeros(graph.n, dtype=bool)
    connected[root] = True
    waiting = list(terminal)
    cost = 0.0
    while waiting:
        distance, predecessors = dijkstra(
            graph.matrix, directed=True, indices=np.flatnonzero(connected), min_only=True, return_predecessors=True
        )[:2]
        best = min(waiting, key=lambda t: (distance[t], waiting.index(t)))
        if not np.isfinite(distance[best]):
            break
        cost += distance[best]
        node = best
        while not connected[node]:
            connected[node] = True
            node = predecessors[node]
        waiting = [t for t in waiting if not connected[t]]
    return cost


def main(sizes):
    net = SpatialNetwork.from_networkx(load_graph("G_large"))
    n = len(net)
    log(f"\n### Sub-task 3: rollout order, G_large ({n:,} nodes, {len(net.edges):,} edges), 20 roots\n")
    rng = np.random.default_rng(0)
    ids = net.node_index
    roots = list(ids[rng.choice(n, 20, replace=False)])
    rows = []
    for k in sizes:
        terminals = list(ids[rng.choice(n, k, replace=False)])
        planner = ConnectionPlanner(net, terminals, roots=roots)
        started = time.perf_counter()
        plan = planner.plan("shortest_path")
        ours = time.perf_counter() - started
        steiner = planner.plan("steiner").cost
        row = [f"{k:,}", f"{ours:.2f}", f"{plan.cost / 1000:,.0f} km", f"{steiner / 1000:,.0f} km"]
        if k <= NAIVE_UP_TO:
            started = time.perf_counter()
            reference = naive(planner, planner._terminal, planner._root)
            row += [f"{time.perf_counter() - started:.2f}", "yes" if np.isclose(reference, plan.cost) else "no"]
        else:
            row += ["", ""]
        rows.append(row)
    table(["terminals", "planner s", "rollout", "rooted Steiner", "naive s", "same cost"], rows)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [100, 1000, 10000])
