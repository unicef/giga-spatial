"""Part B: in-memory + conversion + algorithm costs. Usage: partB.py conv <graph> | partB.py algo <graph> | partB.py mem <what> <graph>"""
import gc, io, json, os, pickle, subprocess, sys, time, warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

FMT = f"{ROOT}/fmt"
RNG = np.random.default_rng(0)


def f3(x):
    return f"{x:.3f}"


# ---------------------------------------------------------------- memory (fresh process)
def mem_child(what, g):
    gc.collect()
    b = rss_mb()
    deep = float("nan")
    if what == "nx":
        o = load_graph(g)
    elif what == "nx_nogeom":
        o = load_graph(g + "_nogeom")
    elif what == "arrow_wkb":
        o = [pq.read_table(f"{FMT}/{g}/lean_f32/{f}.parquet") for f in ("nodes", "edges")]
        deep = sum(t.nbytes for t in o) / 1e6
    elif what == "arrow_nogeom":
        o = [pq.read_table(f"{FMT}/{g}/lean_nogeom/{f}.parquet") for f in ("nodes", "edges")]
        deep = sum(t.nbytes for t in o) / 1e6
    elif what in ("pandas_wkb", "pandas_nogeom", "geopandas"):
        d = "lean_nogeom" if what == "pandas_nogeom" else "lean_f32"
        o = [pq.read_table(f"{FMT}/{g}/{d}/{f}.parquet").to_pandas() for f in ("nodes", "edges")]
        if what == "geopandas":
            import geopandas as gpd
            o[1] = gpd.GeoDataFrame(o[1], geometry=gpd.GeoSeries.from_wkb(o[1]["geometry"].values), crs=4326)
        gc.collect()
        deep = sum(df.memory_usage(deep=True).sum() for df in o) / 1e6
    elif what == "gdfs":
        import osmnx as ox
        G = load_graph(g)
        o = ox.graph_to_gdfs(G)
        del G
        gc.collect()
        deep = sum(df.memory_usage(deep=True).sum() for df in o) / 1e6
    elif what == "csr":
        nodes = pq.read_table(f"{FMT}/{g}/lean_nogeom/nodes.parquet")
        edges = pq.read_table(f"{FMT}/{g}/lean_nogeom/edges.parquet")
        o = lean_to_csr(nodes["osmid"].to_numpy(), edges["u"].to_numpy(), edges["v"].to_numpy(), edges["length"].to_numpy().astype(np.float64))
        del nodes, edges
        deep = (o.data.nbytes + o.indices.nbytes + o.indptr.nbytes) / 1e6
    gc.collect()
    print(json.dumps({"rss": rss_mb() - b, "deep": deep, "peak": maxrss_mb() - b}))


def mem(what, g):
    r = subprocess.run([PY, "-u", "-W", "ignore", __file__, "mem", what, g], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        print(r.stderr[-500:])
        return {"rss": float("nan"), "deep": float("nan"), "peak": float("nan")}


def part_b1(g):
    G = load_graph(g)
    if not os.path.exists(f"{ROOT}/{g}_nogeom.pkl"):
        for *_, d in G.edges(keys=True, data=True):
            d.pop("geometry", None)
        with open(f"{ROOT}/{g}_nogeom.pkl", "wb") as f:
            pickle.dump(G, f, protocol=5)
    del G
    gc.collect()
    rows = []
    labels = {
        "nx": "nx MultiDiGraph (pickle load), with geometry",
        "nx_nogeom": "nx MultiDiGraph, no geometry",
        "gdfs": "ox.graph_to_gdfs output (all attrs, geometry filled)",
        "geopandas": "lean pandas + GeoSeries (decoded shapely)",
        "pandas_wkb": "lean pandas, geometry as WKB bytes",
        "pandas_nogeom": "lean pandas, no geometry",
        "arrow_wkb": "lean Arrow tables, WKB",
        "arrow_nogeom": "lean Arrow tables, no geometry",
        "csr": "scipy CSR (float64 weights, int32 idx)",
    }
    for w, lab in labels.items():
        m = mem(w, g)
        rows.append([lab, f"{m['rss']:.0f}", "" if np.isnan(m["deep"]) else f"{m['deep']:.0f}", f"{m['peak']:.0f}"])
        print(rows[-1], flush=True)
    log(f"\n### B1 — memory footprint, {g} (fresh process each)\n")
    log("RSS = resident memory retained after load (MB); deep = pandas memory_usage(deep=True) / Arrow nbytes (shapely geometries are counted only as pointers by pandas); peak = max RSS increase during load.\n")
    table(["representation", "RSS MB", "deep/nbytes MB", "peak MB"], rows)


# ---------------------------------------------------------------- conversions
def part_conv(g):
    import networkx as nx
    import osmnx as ox
    import geopandas as gpd
    import shapely

    G = load_graph(g)
    log(f"\n### B2–B5 — conversion costs, {g} ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)\n")
    rows = []
    for lab, fn in [
        ("ox.graph_to_gdfs (nodes+edges)", lambda: ox.graph_to_gdfs(G)),
        ("ox.graph_to_gdfs nodes only", lambda: ox.graph_to_gdfs(G, edges=False)),
        ("ox.graph_to_gdfs edges only", lambda: ox.graph_to_gdfs(G, nodes=False)),
        ("ox.graph_to_gdfs edges only, fill_edge_geometry=False", lambda: ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=False)),
        ("nx.to_pandas_edgelist", lambda: nx.to_pandas_edgelist(G)),
        ("pd.DataFrame.from_dict(G.nodes, orient='index')", lambda: pd.DataFrame.from_dict(G.nodes, orient="index")),
        ("hand-vectorized lean extraction (np.fromiter per column), no geom", lambda: nx_to_lean(G, geometry=None)),
        ("hand-vectorized lean extraction + WKB geometry", lambda: nx_to_lean(G, geometry="wkb")),
        ("generic: pd.DataFrame([dict(u=,v=,key=,**d) for edges]) all attrs", lambda: pd.DataFrame([dict(u=u, v=v, key=k, **d) for u, v, k, d in G.edges(keys=True, data=True)])),
    ]:
        reps = 3 if G.number_of_edges() < 1e6 else 1
        t, _ = timeit(fn, reps)
        rows.append([lab, f3(t)])
        print(rows[-1], flush=True)
    log("#### B2 — nx → tables (s)\n")
    table(["operation", "s"], rows)

    # ---- B3 rebuild nx from lean tables
    nodes_t, edges_t, geoms = nx_to_lean(G, osmid="list", geometry="shapely")
    ndf = nodes_t.to_pandas()
    edf = edges_t.to_pandas()
    edf["highway"] = edf["highway"].astype(object)
    edf["name"] = edf["name"].astype(object)
    edf["geometry"] = geoms
    graph_attrs = dict(G.graph)

    def rebuild(with_geom=True):
        H = nx.MultiDiGraph(**graph_attrs)
        H.add_nodes_from(zip(ndf["osmid"].to_numpy().tolist(), ndf[["x", "y", "street_count"]].to_dict("records")))
        cols = ["length", "highway", "name", "oneway", "osmid"] + (["geometry"] if with_geom else [])
        recs = edf[cols].to_dict("records")
        if with_geom:
            for r in recs:
                if r["geometry"] is None:
                    del r["geometry"]
        H.add_edges_from(zip(edf["u"].to_numpy().tolist(), edf["v"].to_numpy().tolist(), edf["key"].to_numpy().tolist(), recs))
        return H

    def rebuild_fast():
        H = nx.MultiDiGraph(**graph_attrs)
        x, y, sc = ndf["x"].to_numpy().tolist(), ndf["y"].to_numpy().tolist(), ndf["street_count"].to_numpy().tolist()
        H.add_nodes_from((n, {"x": a, "y": b, "street_count": c}) for n, a, b, c in zip(ndf["osmid"].to_numpy().tolist(), x, y, sc))
        L = edf["length"].to_numpy().tolist()
        H.add_edges_from((u, v, k, {"length": l}) for u, v, k, l in zip(edf["u"].to_numpy().tolist(), edf["v"].to_numpy().tolist(), edf["key"].to_numpy().tolist(), L))
        return H

    reps = 3 if G.number_of_edges() < 1e6 else 1
    rows = []
    t, H = timeit(lambda: rebuild(True), reps)
    rows.append(["add_nodes_from(zip(ids, to_dict('records'))) + add_edges_from(zip(u,v,k,records)), 5 attrs + geometry", f3(t), f"{H.number_of_edges():,}"])
    t, _ = timeit(lambda: rebuild(False), reps)
    rows.append(["same, no geometry", f3(t), ""])
    t, _ = timeit(rebuild_fast, reps)
    rows.append(["topology + length only (generator of small dicts)", f3(t), ""])
    del H
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    t, H = timeit(lambda: ox.graph_from_gdfs(nodes_gdf, edges_gdf), reps)
    rows.append(["ox.graph_from_gdfs(graph_to_gdfs output)", f3(t), f"{H.number_of_edges():,}"])
    del H
    log("#### B3 — tables → nx (s)\n")
    table(["operation", "s", "edges"], rows)

    # ---- B4 node coords
    rows = []
    for lab, fn in [
        ("[list(map(G.nodes[n].get, ['y','x'])) for n in G.nodes()]", lambda: [list(map(G.nodes[n].get, ["y", "x"])) for n in G.nodes()]),
        ("np.array([(d['y'], d['x']) for _, d in G.nodes(data=True)])", lambda: np.array([(d["y"], d["x"]) for _, d in G.nodes(data=True)])),
        ("np.fromiter x2 over G.nodes(data=...)", lambda: np.column_stack([np.fromiter((d for _, d in G.nodes(data="y")), float), np.fromiter((d for _, d in G.nodes(data="x")), float)])),
        ("df[['y','x']].to_numpy() (lean nodes DataFrame)", lambda: ndf[["y", "x"]].to_numpy()),
    ]:
        t, _ = timeit(fn, 5)
        rows.append([lab, f"{t*1000:.2f}"])
    log("#### B4 — node coordinate array (ms, mean of 5)\n")
    table(["method", "ms"], rows)

    # ---- B5 sub-region
    big = load_graph("G_large") if g != "G_large" else G
    xs = np.array([d["x"] for _, d in big.nodes(data=True)])
    ys = np.array([d["y"] for _, d in big.nodes(data=True)])
    W0, E0, S0, N0 = xs.min(), xs.max(), ys.min(), ys.max()
    del big
    cx, cy = 30.06, -1.95
    hw, hh = 0.1 * (E0 - W0), 0.1 * (N0 - S0)  # 20% x 20% = 4% area
    W, E, S, N = cx - hw, cx + hw, cy - hh, cy + hh
    box = shapely.box(W, S, E, N)
    rows = []
    eg = gpd.GeoDataFrame(edges_gdf.reset_index(), geometry="geometry", crs=4326)
    t0 = time.perf_counter()
    _ = eg.sindex
    t_sidx = time.perf_counter() - t0
    t, res = timeit(lambda: eg.iloc[eg.sindex.query(box, predicate="intersects")], 5)
    rows.append([f"edges_gdf.sindex.query(box, 'intersects') (+ sindex build {t_sidx:.3f} s, once)", f3(t), f"{len(res):,}"])
    t, res = timeit(lambda: eg.iloc[eg.sindex.query(box)], 5)
    rows.append(["edges_gdf.sindex.query(box) bbox-only", f3(t), f"{len(res):,}"])

    def node_bbox():
        m = (ndf["x"].to_numpy() >= W) & (ndf["x"].to_numpy() <= E) & (ndf["y"].to_numpy() >= S) & (ndf["y"].to_numpy() <= N)
        ids = ndf["osmid"].to_numpy()[m]
        em = np.isin(edf["u"].to_numpy(), ids) | np.isin(edf["v"].to_numpy(), ids)
        return edf[em]

    t, res = timeit(node_bbox, 5)
    rows.append(["lean tables: node-coord bbox mask + np.isin(u|v) on edges", f3(t), f"{len(res):,}"])

    def node_bbox_arrow():
        import pyarrow.compute as pc
        x, y = nodes_t["x"], nodes_t["y"]
        m = pc.and_(pc.and_(pc.greater_equal(x, W), pc.less_equal(x, E)), pc.and_(pc.greater_equal(y, S), pc.less_equal(y, N)))
        ids = pc.filter(nodes_t["osmid"], m)
        em = pc.or_(pc.is_in(edges_t["u"], value_set=ids), pc.is_in(edges_t["v"], value_set=ids))
        return edges_t.filter(em)

    t, res = timeit(node_bbox_arrow, 5)
    rows.append(["lean Arrow tables: same with pyarrow.compute", f3(t), f"{res.num_rows:,}"])

    def nx_sub():
        keep = [n for n, d in G.nodes(data=True) if W <= d["x"] <= E and S <= d["y"] <= N]
        return ox.graph_to_gdfs(G.subgraph(keep), nodes=False)

    t, res = timeit(nx_sub, 3)
    rows.append(["nx: node filter + G.subgraph + graph_to_gdfs(edges)", f3(t), f"{len(res):,}"])
    t, res = timeit(lambda: ox.graph_to_gdfs(G, nodes=False).cx[W:E, S:N], 1 if G.number_of_edges() > 1e6 else 3)
    rows.append(["full ox.graph_to_gdfs(edges) + .cx[box]", f3(t), f"{len(res):,}"])
    log(f"#### B5 — sub-region edges, box {W:.3f},{S:.3f},{E:.3f},{N:.3f} (4% of Rwanda bbox area) (s, mean)\n")
    log("Note: sindex/cx select edges whose geometry intersects the box; node-bbox variants select edges with either endpoint inside; subgraph keeps edges with both endpoints inside.\n")
    table(["method", "s", "edges"], rows)


# ---------------------------------------------------------------- algorithms
def part_algo(g):
    import networkx as nx
    import scipy.sparse as sp
    from scipy.sparse import csgraph
    import pandana

    nodes = pq.read_table(f"{FMT}/{g}/lean_nogeom/nodes.parquet").to_pandas()
    edges = pq.read_table(f"{FMT}/{g}/lean_nogeom/edges.parquet").to_pandas()
    edges["length"] = edges["length"].astype(np.float64)
    n, m = len(nodes), len(edges)
    log(f"\n### B6–B8 — routing engines, {g} ({n:,} nodes, {m:,} edges)\n")
    ids = nodes["osmid"].to_numpy()

    # ---- scipy CSR
    rows = []
    t, A = timeit(lambda: lean_to_csr(ids, edges["u"].to_numpy(), edges["v"].to_numpy(), edges["length"].to_numpy()), 3)
    rows.append(["build CSR from tables (searchsorted id->pos, dedupe parallel edges keep-min)", f3(t), f"nnz={A.nnz:,}"])
    t, (ncc, lab) = timeit(lambda: csgraph.connected_components(A, directed=True, connection="strong"), 3)
    sizes = np.bincount(lab)
    rows.append(["connected_components strong", f3(t), f"{ncc:,} comps, largest {sizes.max():,}"])
    t, (nwc, _) = timeit(lambda: csgraph.connected_components(A, directed=True, connection="weak"), 3)
    rows.append(["connected_components weak", f3(t), f"{nwc:,} comps"])
    big = np.flatnonzero(lab == sizes.argmax())
    src = RNG.choice(big, 100, replace=False)
    tgt = RNG.choice(big, 1000, replace=False)
    t, D = timeit(lambda: csgraph.dijkstra(A, directed=True, indices=src), 1)
    t_full = t
    rows.append(["dijkstra 100 sources, full (100 x n matrix)", f3(t), f"{t/100*1000:.1f} ms/source"])
    t, Dl = timeit(lambda: csgraph.dijkstra(A, directed=True, indices=src, limit=20000), 1)
    rows.append(["dijkstra 100 sources, limit=20000 m", f3(t), f"{np.isfinite(Dl).sum(1).mean():.0f} nodes reached/source"])
    t, D2 = timeit(lambda: csgraph.dijkstra(A, directed=True, indices=src)[:, tgt], 1)
    rows.append(["100x1000 many-to-many (dijkstra + index columns)", f3(t), f"{t/1e5*1e6:.2f} µs/pair"])
    t, (Dp, P) = timeit(lambda: csgraph.dijkstra(A, directed=True, indices=src, return_predecessors=True), 1)
    rows.append(["dijkstra 100 sources, return_predecessors=True", f3(t), f"+{(t/t_full-1)*100:.0f}% vs no predecessors"])
    del Dp, P, Dl

    def mst():
        u = np.concatenate([edges["u"].to_numpy(), edges["v"].to_numpy()])
        v = np.concatenate([edges["v"].to_numpy(), edges["u"].to_numpy()])
        w = np.concatenate([edges["length"].to_numpy()] * 2)
        Au = lean_to_csr(ids, u, v, w)
        return csgraph.minimum_spanning_tree(Au)

    t, T = timeit(mst, 1)
    rows.append(["minimum_spanning_tree (symmetrize from tables + MST)", f3(t), f"total {T.sum()/1000:,.0f} km, {T.nnz:,} edges"])
    log("#### B7 — scipy.sparse.csgraph (s)\n")
    table(["operation", "s", "note"], rows)

    # ---- networkx
    G = load_graph(g + "_nogeom") if os.path.exists(f"{ROOT}/{g}_nogeom.pkl") else load_graph(g)
    t, _ = timeit(lambda: [nx.single_source_dijkstra_path_length(G, int(ids[s]), weight="length") for s in src[:10]], 1)
    nx_per = t / 10
    log(f"- networkx single_source_dijkstra_path_length, 10 sources: {t:.2f} s ({nx_per*1000:.0f} ms/source → ~{nx_per*100:.1f} s for 100 sources) vs scipy {t_full/100*1000:.1f} ms/source → scipy ≈ {nx_per/(t_full/100):.0f}x faster")
    d_nx = nx.single_source_dijkstra_path_length(G, int(ids[src[0]]), weight="length")
    chk = np.array([d_nx.get(int(ids[j]), np.inf) for j in tgt])
    log(f"- check networkx vs scipy distances (source 0, 1000 targets): max |diff| = {np.nanmax(np.abs(chk - D2[0])):.4f} m")

    # ---- pandana
    rows = []
    nx_ = pd.Series(nodes["x"].to_numpy(), index=ids)
    ny_ = pd.Series(nodes["y"].to_numpy(), index=ids)

    def build_net(twoway=False, ix=nx_, iy=ny_, eu=None, ev=None):
        return pandana.Network(ix, iy, edges["u"] if eu is None else eu, edges["v"] if ev is None else ev, edges[["length"]], twoway=twoway)

    t, net = timeit(lambda: build_net(False), 1)
    rows.append(["pandana.Network(...) from lean tables, twoway=False (osmid int64 index)", f3(t), "includes CH contraction"])
    O = ids[RNG.choice(big, 10000)]
    Dd = ids[RNG.choice(big, 10000)]
    t, L = timeit(lambda: net.shortest_path_lengths(O, Dd, imp_name="length"), 3)
    rows.append(["shortest_path_lengths 10k OD pairs", f3(t), f"{t/1e4*1e6:.1f} µs/pair"])
    t, P = timeit(lambda: net.shortest_paths(O[:1000], Dd[:1000], imp_name="length"), 1)
    rows.append(["shortest_paths 1k OD pairs (node lists)", f3(t), f"{t/1e3*1e3:.2f} ms/pair"])
    OO = np.repeat(ids[src], 1000)
    DD = np.tile(ids[tgt], 100)
    t, L2 = timeit(lambda: np.asarray(net.shortest_path_lengths(OO, DD, imp_name="length")), 1)
    rows.append(["100x1000 = 100k pairs via shortest_path_lengths", f3(t), f"{t/1e5*1e6:.2f} µs/pair (scipy: see B7)"])
    diff = np.abs(L2.reshape(100, 1000) - D2)
    rows.append(["check vs scipy 100x1000", "", f"max |diff| {diff.max():.3f} m, median {np.median(diff):.4f} m (pandana uses float32-ish/mm ints)"])
    # one-to-all style
    t, _ = timeit(lambda: net.nodes_in_range(list(ids[src[:10]]), 20000, "length") if hasattr(net, "nodes_in_range") else None, 1)
    rows.append(["nodes_in_range 10 sources, 20 km", f3(t), ""])
    del net
    # id constraints
    notes = []
    try:
        sids = pd.Index(["n" + str(i) for i in ids])
        mp = dict(zip(ids, sids))
        t0 = time.perf_counter()
        netS = pandana.Network(pd.Series(nodes["x"].to_numpy(), index=sids), pd.Series(nodes["y"].to_numpy(), index=sids), edges["u"].map(mp), edges["v"].map(mp), edges[["length"]], twoway=False)
        ls = netS.shortest_path_lengths(["n" + str(i) for i in O[:100]], ["n" + str(i) for i in Dd[:100]], imp_name="length")
        notes.append(f"string node ids: OK (build {time.perf_counter()-t0:.1f} s; lengths match int-id run: {np.allclose(ls, np.asarray(L)[:100])})")
        del netS
    except Exception as ex:
        notes.append(f"string node ids: FAIL {ex!r}"[:200])
    notes.append(f"non-contiguous int64 osmids (min {ids.min()}, max {ids.max()}, > 2^32): OK (used above)")
    try:
        net0 = pandana.Network(nx_, ny_, edges["u"], edges["v"], edges[["length"]], twoway=False)
        r = net0.shortest_path_lengths([int(ids[0])], [int(ids[0]) + 1 if int(ids[0]) + 1 not in set(ids) else 1], imp_name="length")
        notes.append(f"query with unknown node id: returns {r} (no error)")
    except Exception as ex:
        notes.append(f"query with unknown node id: raises {type(ex).__name__}: {str(ex)[:120]}")
    log("#### B6 — pandana 0.7 (s)\n")
    table(["operation", "s", "note"], rows)
    for s in notes:
        log(f"- {s}")

    # user function
    del G
    Gf = load_graph(g)
    t0 = time.perf_counter()
    nd = pd.DataFrame.from_dict(Gf.nodes, orient="index")
    t1 = time.perf_counter()
    ed = nx.to_pandas_edgelist(Gf)
    t2 = time.perf_counter()
    net_u = pandana.Network(nd["x"], nd["y"], ed["source"], ed["target"], ed[["length"]])
    t3 = time.perf_counter()
    log(f"\n- user function on {g}: DataFrame.from_dict(G.nodes) {t1-t0:.2f} s + nx.to_pandas_edgelist {t2-t1:.2f} s = conversion {t2-t0:.2f} s; pandana.Network (default twoway=True, so every edge also added reversed) CH build {t3-t2:.2f} s")
    del net_u, Gf, nd, ed
    gc.collect()

    # ---- igraph
    try:
        sys.path.insert(0, f"{ROOT}/pylib")
        import igraph as ig
        order = np.argsort(ids)
        ui = order[np.searchsorted(ids[order], edges["u"].to_numpy())]
        vi = order[np.searchsorted(ids[order], edges["v"].to_numpy())]
        t0 = time.perf_counter()
        Gi = ig.Graph(n=n, edges=np.column_stack([ui, vi]).tolist(), directed=True, edge_attrs={"length": edges["length"].to_numpy().tolist()})
        tb = time.perf_counter() - t0
        t, Di = timeit(lambda: np.array(Gi.distances(source=src.tolist(), target=tgt.tolist(), weights="length")), 1)
        log(f"\n#### B8 — igraph {ig.__version__}\n")
        log(f"- Graph(n, edges=list, edge_attrs) build: {tb:.2f} s; distances(100 sources, 1000 targets, weights='length'): {t:.2f} s ({t/1e5*1e6:.2f} µs/pair); max |diff| vs scipy {np.abs(Di - D2).max():.4f} m")
        t, _ = timeit(lambda: Gi.distances(source=src.tolist(), weights="length"), 1)
        log(f"- igraph distances 100 sources → all nodes: {t:.2f} s ({t/100*1000:.1f} ms/source)")
    except Exception as ex:
        log(f"- igraph: FAILED {ex!r}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "mem":
        mem_child(sys.argv[2], sys.argv[3])
    elif mode == "b1":
        part_b1(sys.argv[2])
    elif mode == "conv":
        part_conv(sys.argv[2])
    elif mode == "algo":
        part_algo(sys.argv[2])
