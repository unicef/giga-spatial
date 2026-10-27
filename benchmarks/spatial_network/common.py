import gc, os, pickle, resource, time, json, subprocess, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil

ROOT = "/tmp/gs_bench"
OUT = f"{ROOT}/out"
REPORT = f"{OUT}/report.md"
PY = "/Users/utkucanozturk/miniforge3/envs/py310/bin/python"
PROC = psutil.Process()


def log(s=""):
    print(s, flush=True)
    with open(REPORT, "a") as f:
        f.write(s + "\n")


def table(headers, rows):
    log("| " + " | ".join(headers) + " |")
    log("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        log("| " + " | ".join(str(x) for x in r) + " |")
    log()


def rss_mb():
    return PROC.memory_info().rss / 1e6


def maxrss_mb():
    # macOS: ru_maxrss is in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def timeit(fn, reps=1):
    ts = []
    out = None
    for _ in range(reps):
        out = None
        gc.collect()
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return float(np.mean(ts)), out


def load_graph(name):
    with open(f"{ROOT}/{name}.pkl", "rb") as f:
        return pickle.load(f)


def size_mb(path):
    if os.path.isdir(path):
        return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / 1e6
    return os.path.getsize(path) / 1e6


def _join(v):
    if isinstance(v, list):
        return ";".join(str(x) for x in v)
    return v


def nx_to_lean(G, osmid="int", length="float32", geometry="wkb"):
    """Extract lean nodes/edges pyarrow Tables from an osmnx MultiDiGraph.

    geometry: 'wkb' (binary column, null where edge is straight), None (dropped), 'shapely' (object array returned
    separately as edges_geom for geopandas writers).
    """
    import shapely

    nid = np.fromiter(G.nodes, dtype=np.int64, count=G.number_of_nodes())
    nd = G.nodes
    x = np.fromiter((nd[n]["x"] for n in nid), dtype=np.float64, count=len(nid))
    y = np.fromiter((nd[n]["y"] for n in nid), dtype=np.float64, count=len(nid))
    sc = np.fromiter((nd[n].get("street_count", 0) for n in nid), dtype=np.int16, count=len(nid))
    nodes = pa.table({"osmid": nid, "x": x, "y": y, "street_count": sc})

    rows = list(G.edges(keys=True, data=True))
    m = len(rows)
    u = np.fromiter((r[0] for r in rows), dtype=np.int64, count=m)
    v = np.fromiter((r[1] for r in rows), dtype=np.int64, count=m)
    k = np.fromiter((r[2] for r in rows), dtype=np.int16, count=m)
    ln = np.fromiter((r[3]["length"] for r in rows), dtype=np.float64, count=m).astype(length)
    hw = pa.array([_join(r[3].get("highway")) for r in rows], type=pa.string()).dictionary_encode()
    nm = pa.array([_join(r[3].get("name")) for r in rows], type=pa.string()).dictionary_encode()
    ow = np.fromiter((bool(r[3].get("oneway", False)) for r in rows), dtype=bool, count=m)
    if osmid == "int":
        oid = pa.array(np.fromiter((r[3]["osmid"][0] if isinstance(r[3]["osmid"], list) else r[3]["osmid"] for r in rows), dtype=np.int64, count=m))
    else:
        oid = pa.array([r[3]["osmid"] if isinstance(r[3]["osmid"], list) else [r[3]["osmid"]] for r in rows], type=pa.list_(pa.int64()))
    cols = {"u": u, "v": v, "key": k, "length": ln, "highway": hw, "name": nm, "oneway": ow, "osmid": oid}
    geoms = None
    if geometry is not None:
        geoms = np.array([r[3].get("geometry") for r in rows], dtype=object)
        if geometry == "wkb":
            cols["geometry"] = pa.array(shapely.to_wkb(geoms), type=pa.binary())
    edges = pa.table(cols)
    if geometry == "shapely":
        return nodes, edges, geoms
    return nodes, edges


def lean_to_csr(nodes_osmid, u, v, w, n=None):
    """Build scipy CSR (positions) from int64 ids; keeps min weight of parallel edges."""
    import scipy.sparse as sp

    order = np.argsort(nodes_osmid, kind="stable")
    sid = nodes_osmid[order]
    ui = order[np.searchsorted(sid, u)]
    vi = order[np.searchsorted(sid, v)]
    n = len(nodes_osmid)
    # drop parallel edges keeping min weight
    o = np.lexsort((w, vi, ui))
    ui, vi, w = ui[o], vi[o], w[o]
    keep = np.ones(len(ui), bool)
    keep[1:] = (ui[1:] != ui[:-1]) | (vi[1:] != vi[:-1])
    A = sp.csr_matrix((w[keep], (ui[keep], vi[keep])), shape=(n, n))
    return A
