"""Sub-task 6: exact optima of road-network instances with scipy's milp (HiGHS), against the planner.

Usage: exact.py [terminals ...] [--seeds N] [--time-limit S]
       exact.py --rwanda [terminals ...]    (only writes Rwanda's instances; needs G_large.pkl from step0.py)
Uses the Kigali test fixture (tests/data/spatial_network/kigali), its largest component, with 3 random roots.
Writes each instance in SteinLib's STP format to $GS_BENCH_DIR/stp (/tmp/gs_bench by default), with the virtual
root as the last node, joined to every root at weight 0, and listed first among the terminals, so an exact Steiner tree solver such as SCIP-Jack can solve the
same instances.

The model is the multi-commodity flow formulation: a binary variable per arc, and for every terminal one unit
of flow from the root that may only use chosen arcs. Its linear relaxation equals the directed cut relaxation,
which the planner's dual ascent approximates from below.
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, identity, vstack

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spatial_network"))
from common import ROOT, load_graph, log, table  # noqa

from gigaspatial.processing.network import ConnectionPlanner, SpatialNetwork
from gigaspatial.processing.network.planning import _with_virtual_root

REPO = Path(__file__).resolve().parents[2]
KIGALI = REPO / "tests" / "data" / "spatial_network" / "kigali"


def kigali():
    meta = json.loads((KIGALI / "network.json").read_text())
    network = SpatialNetwork(
        pd.read_parquet(KIGALI / "nodes.parquet"),
        gpd.read_parquet(KIGALI / "edges.parquet"),
        directed=meta["directed"],
        weight=meta["weight"],
    )
    return network.largest_component()


def instance(planner):
    """The planner's graph with the virtual root, the root's position, and the terminals other than roots."""
    graph = _with_virtual_root(planner._graph, planner._root)
    terminals = planner._planned_in_given_order(np.unique(planner._terminal))
    return graph, planner._graph.n, terminals[~np.isin(terminals, planner._root)]


def write_stp(path, graph, root, terminals, name):
    """The instance as an undirected SteinLib STP file; node i of the graph is node i + 1, the root a terminal."""
    n = graph.n
    a, b = graph.entry_keys // n, graph.entry_keys % n
    keep = a < b
    lines = ["33D32945 STP File, STP Format Version 1.0", "", "SECTION Comment", f'Name "{name}"',
             'Creator "giga-spatial benchmarks/connection_planning/exact.py"', "END", "", "SECTION Graph",
             f"Nodes {n}", f"Edges {int(keep.sum())}"]
    lines += [f"E {u + 1} {v + 1} {w:.6f}" for u, v, w in zip(a[keep], b[keep], graph.entry_weights[keep])]
    lines += ["END", "", "SECTION Terminals", f"Terminals {len(terminals) + 1}", f"T {root + 1}"]
    lines += [f"T {t + 1}" for t in terminals] + ["END", "", "EOF", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def solve(graph, root, terminals, time_limit):
    """The optimum by multi-commodity flow; (cost, lower bound, solved, seconds)."""
    n = graph.n
    tail, head = graph.entry_keys // n, graph.entry_keys % n
    weight = graph.entry_weights
    arcs, k = len(tail), len(terminals)
    # Flow into the root is never useful, which also removes the arcs of the root's joins pointing at it.
    usable = head != root
    tail, head, weight = tail[usable], head[usable], weight[usable]
    arcs = len(tail)
    blocks = []
    conservation = coo_matrix(
        (np.r_[np.ones(arcs), -np.ones(arcs)], (np.r_[tail, head], np.r_[np.arange(arcs), np.arange(arcs)])),
        shape=(n, arcs),
    ).tocsr()
    rows, lower = [], []
    for i, t in enumerate(terminals):
        supply = np.zeros(n)
        supply[root], supply[t] = 1.0, -1.0
        row = [csr_matrix((n, arcs))] + [conservation if j == i else csr_matrix((n, arcs)) for j in range(k)]
        rows.append(hstack(row))
        lower.append(supply)
    flow = LinearConstraint(vstack(rows).tocsr(), np.concatenate(lower), np.concatenate(lower))
    capacity_rows = [hstack([-identity(arcs)] + [identity(arcs) if j == i else csr_matrix((arcs, arcs)) for j in range(k)])
                     for i in range(k)]
    capacity = LinearConstraint(vstack(capacity_rows).tocsr(), -np.inf, 0.0)
    cost = np.r_[weight, np.zeros(k * arcs)]
    integrality = np.r_[np.ones(arcs), np.zeros(k * arcs)]
    started = time.perf_counter()
    result = milp(cost, constraints=[flow, capacity], integrality=integrality, bounds=Bounds(0, 1),
                  options={"time_limit": time_limit, "mip_rel_gap": 1e-9, "disp": False})
    took = time.perf_counter() - started
    solved = result.status == 0
    return result.fun, getattr(result, "mip_dual_bound", None), solved, took


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("terminals", nargs="*", type=int, default=[30, 60])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--rwanda", action="store_true")
    options = parser.parse_args()
    if options.rwanda:
        return write_rwanda(options.terminals)
    network = kigali()
    ids = network.node_index
    rows = []
    for count in options.terminals:
        for seed in range(options.seeds):
            rng = np.random.default_rng(seed)
            terminals = list(ids[rng.choice(len(ids), count, replace=False)])
            roots = list(ids[rng.choice(len(ids), 3, replace=False)])
            planner = ConnectionPlanner(network, terminals, roots=roots)
            graph, root, targets = instance(planner)
            write_stp(Path(ROOT) / "stp" / f"kigali_{count}_{seed}.stp", graph, root, targets, f"kigali_{count}_{seed}")
            plans = {m: planner.plan(m) for m in ("steiner", "shortest_path")}
            improved = {m: planner.improve(p).cost for m, p in plans.items()}
            bound = planner.lower_bound(time_limit=None)
            optimum, dual, solved, took = solve(graph, root, targets, options.time_limit)
            reference = optimum if solved else np.nan
            gap = lambda value: f"{100 * (value / reference - 1):.2f}%" if solved else ""  # noqa: E731
            rows.append(
                [
                    f"{count}",
                    f"{seed}",
                    f"{optimum:,.0f} m" + ("" if solved else " (best found)"),
                    f"{took:.1f}",
                    gap(plans["steiner"].cost),
                    gap(improved["steiner"]),
                    gap(plans["shortest_path"].cost),
                    gap(improved["shortest_path"]),
                    f"{100 * (1 - bound / reference):.2f}%" if solved else f"{bound:,.0f} m",
                ]
            )
    log(f"\n### Sub-task 6: exact optima on Kigali ({len(network):,} nodes), 3 roots, HiGHS multi-commodity flow\n")
    table(["terminals", "seed", "optimum", "HiGHS s", "steiner", "+ improve", "shortest_path", "+ improve",
           "bound below optimum"], rows)
    log("Plans: percentage above the optimum. Bound: percentage below the optimum.\n")


def write_rwanda(sizes):
    """Rwanda's instances as STP files for an exact solver: 20 roots and seed 0, the draws of bound.py for the same sizes."""
    network = SpatialNetwork.from_networkx(load_graph("G_large"))
    ids = network.node_index
    rng = np.random.default_rng(0)
    roots = list(ids[rng.choice(len(ids), 20, replace=False)])
    for count in sizes:
        terminals = list(ids[rng.choice(len(ids), count, replace=False)])
        planner = ConnectionPlanner(network, terminals, roots=roots)
        graph, root, targets = instance(planner)
        path = Path(ROOT) / "stp" / f"rwanda_{count}.stp"
        write_stp(path, graph, root, targets, f"rwanda_{count}")
        log(f"- {path}: {graph.n:,} nodes, {len(targets):,} terminals and the root")


if __name__ == "__main__":
    main()
