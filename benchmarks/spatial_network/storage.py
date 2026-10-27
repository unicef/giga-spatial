"""Sub-task 3: SpatialNetwork.save and load against the section 9 targets.

Usage: storage.py <graph> [reps]
Needs <graph>.pkl from step0.py. Writes to $GS_BENCH_DIR/storage/<graph>.
"""
import gc, os, shutil, sys, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

from gigaspatial.processing.network import SpatialNetwork
from gigaspatial.processing.network.attributes import normalize_attributes


def main(g, reps):
    G = load_graph(g)
    net = SpatialNetwork.from_networkx(G)
    del G
    gc.collect()
    folder = f"{ROOT}/storage/{g}"
    shutil.rmtree(folder, ignore_errors=True)
    log(f"\n### Sub-task 3 — save and load, {g} ({len(net):,} nodes, {len(net.edges):,} edges), mean of {reps}\n")
    rows = []

    t, _ = timeit(lambda: normalize_attributes(net._edges, skip=("u", "v", "key", "length", "kind", "geometry")), reps)
    rows.append(["normalise raw osmnx edge attributes", f"{t:.3f}", ""])
    t, _ = timeit(lambda: net.save(folder), reps)
    rows.append(["save, raw osmnx attributes (normalisation included)", f"{t:.3f}", "under 1 s"])
    rows.append(["folder size (MB)", f"{size_mb(folder):.1f}", ""])

    t, loaded = timeit(lambda: SpatialNetwork.load(folder), reps)
    rows.append(["load, all columns", f"{t:.3f}", "under 0.5 s"])
    t, _ = timeit(lambda: SpatialNetwork.load(folder, columns=["geometry"]), reps)
    rows.append(["load, required columns and geometry", f"{t:.3f}", ""])
    t, _ = timeit(lambda: SpatialNetwork.load(folder, columns=[]), reps)
    rows.append(["load, topology only (u, v, key, length, kind)", f"{t:.3f}", ""])

    again = f"{folder}_again"
    t, _ = timeit(lambda: loaded.save(again), reps)
    rows.append(["save, a loaded (already normalised) network", f"{t:.3f}", "under 1 s"])
    table(["operation", "s", "target"], rows)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
