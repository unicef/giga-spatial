"""Sub-task 4: improve() on each method's plan, against the target of one pass under 10 s.

Usage: improve.py [terminals ...]
Needs G_large.pkl from benchmarks/spatial_network/step0.py, in $GS_BENCH_DIR (/tmp/gs_bench by default).
"""
import os, sys, time, warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spatial_network"))
from common import load_graph, log, table  # noqa

from gigaspatial.processing.network import ConnectionPlanner, SpatialNetwork


def main(sizes):
    net = SpatialNetwork.from_networkx(load_graph("G_large"))
    n = len(net)
    log(f"\n### Sub-task 4: improve(), G_large ({n:,} nodes, {len(net.edges):,} edges), 20 roots\n")
    rng = np.random.default_rng(0)
    ids = net.node_index
    roots = list(ids[rng.choice(n, 20, replace=False)])
    rows = []
    for k in sizes:
        terminals = list(ids[rng.choice(n, k, replace=False)])
        planner = ConnectionPlanner(net, terminals, roots=roots)
        for method in ("steiner", "shortest_path", "nearest_root"):
            started = time.perf_counter()
            plan = planner.plan(method)
            planned = time.perf_counter() - started
            started = time.perf_counter()
            better = planner.improve(plan)
            took = time.perf_counter() - started
            passes = better.metadata["improved"]["pass_seconds"]
            rows.append(
                [
                    f"{k:,}",
                    method,
                    f"{plan.cost / 1000:,.0f} km ({planned:.2f} s)",
                    f"{better.cost / 1000:,.0f} km",
                    f"{100 * (1 - better.cost / plan.cost):.1f}%",
                    f"{better.metadata['improved']['moves']:,}",
                    f"{len(passes)}",
                    f"{max(passes):.1f}",
                    f"{took:.1f}",
                ]
            )
    table(["terminals", "method", "plan", "improved", "saved", "moves", "passes", "longest pass s", "total s"], rows)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [100, 1000])
