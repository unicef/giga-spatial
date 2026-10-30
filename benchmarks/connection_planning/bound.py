"""Sub-task 5: lower_bound() against the target of under 60 s, and the gap it gives each plan.

Usage: bound.py [terminals ...]
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
    log(f"\n### Sub-task 5: lower_bound(), G_large ({n:,} nodes, {len(net.edges):,} edges), 20 roots\n")
    rng = np.random.default_rng(0)
    ids = net.node_index
    roots = list(ids[rng.choice(n, 20, replace=False)])
    rows = []
    for k in sizes:
        terminals = list(ids[rng.choice(n, k, replace=False)])
        planner = ConnectionPlanner(net, terminals, roots=roots)
        plan = planner.plan("steiner")
        started = time.perf_counter()
        planner.lower_bound(plan, time_limit=None)
        took = time.perf_counter() - started
        better = planner.improve(plan, time_limit=120)
        planner.lower_bound(better)
        rows.append(
            [
                f"{k:,}",
                f"{took:.1f}",
                f"{plan.bound / 1000:,.0f} km",
                f"{plan.cost / 1000:,.0f} km ({plan.gap:.1%})",
                f"{better.cost / 1000:,.0f} km ({better.gap:.1%})",
            ]
        )
    table(["terminals", "bound s", "bound", "steiner (gap)", "improved, 120 s at most (gap)"], rows)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [100, 1000, 10000])
