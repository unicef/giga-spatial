"""Part A: storage formats. Usage: partA.py run <graph> [reps] [fmt,fmt,...] | partA.py mem <fmt> <path>"""
import gc, gzip, io, json, os, pickle, shutil, subprocess, sys, time, zipfile, warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
import geopandas as gpd
import shapely
import osmnx as ox
import networkx as nx
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

LIST_ATTRS = ["osmid", "highway", "name", "lanes", "maxspeed", "reversed", "ref"]


# ---------------- readers (path -> object) ----------------
def r_graphml(p):
    return ox.load_graphml(p)


def r_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def r_pickle_gz(p):
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def r_gpkg(p):
    return gpd.read_file(p, layer="nodes"), gpd.read_file(p, layer="edges")


def r_gpkg_arrow(p):
    return gpd.read_file(p, layer="nodes", use_arrow=True), gpd.read_file(p, layer="edges", use_arrow=True)


def r_geoparquet(p):
    return gpd.read_parquet(f"{p}/nodes.parquet"), gpd.read_parquet(f"{p}/edges.parquet")


def r_geoparquet_nx(p):
    n, e = r_geoparquet(p)
    return ox.graph_from_gdfs(n, e)


def r_lean_arrow(p):
    return pq.read_table(f"{p}/nodes.parquet"), pq.read_table(f"{p}/edges.parquet")


def r_lean_pandas(p):
    n, e = r_lean_arrow(p)
    return n.to_pandas(), e.to_pandas()


def r_lean_geo(p):
    """read + decode WKB to a GeoDataFrame"""
    n, e = r_lean_pandas(p)
    if "geometry" in e:
        e = gpd.GeoDataFrame(e, geometry=gpd.GeoSeries.from_wkb(e["geometry"].values), crs=4326)
    return n, e


def r_lean_gpq(p):
    return pq.read_table(f"{p}/nodes.parquet").to_pandas(), gpd.read_parquet(f"{p}/edges.parquet")


def r_feather_arrow(p):
    return feather.read_table(f"{p}/nodes.arrow"), feather.read_table(f"{p}/edges.arrow")


def r_feather_pandas(p):
    n, e = r_feather_arrow(p)
    return n.to_pandas(), e.to_pandas()


def r_zip_bytes(p):
    with open(p, "rb") as fh:
        z = zipfile.ZipFile(fh)
        n = pq.read_table(io.BytesIO(z.read("nodes.parquet")))
        e = pq.read_table(io.BytesIO(z.read("edges.parquet")))
    return n.to_pandas(), e.to_pandas()


def r_dir_bytes(p):
    out = []
    for f in ("nodes.parquet", "edges.parquet"):
        with open(f"{p}/{f}", "rb") as fh:
            out.append(pq.read_table(io.BytesIO(fh.read())).to_pandas())
    return tuple(out)


def r_csr(p):
    d = np.load(p)
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=(len(d["ids"]),) * 2)
    return A, d["ids"], d["x"], d["y"]


READERS = {k[2:]: v for k, v in globals().items() if k.startswith("r_")}


# ---------------- writers ----------------
def stringify_lists(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object and c != "geometry":
            if df[c].map(lambda x: isinstance(x, list)).any():
                df[c] = df[c].map(lambda x: str(x) if isinstance(x, list) else x)
            # mixed int/str or bool/str columns also break pyarrow
            types = set(df[c].dropna().map(type))
            if len(types) > 1:
                df[c] = df[c].map(lambda x: x if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))
    return df


def lean_pandas_edges(edges_tbl):
    return edges_tbl.to_pandas()


def write_lean_dir(p, nodes, edges, compression="zstd"):
    os.makedirs(p, exist_ok=True)
    pq.write_table(nodes, f"{p}/nodes.parquet", compression=compression)
    pq.write_table(edges, f"{p}/edges.parquet", compression=compression)


def write_lean_gpq(p, nodes, edges_tbl, geoms, encoding):
    os.makedirs(p, exist_ok=True)
    pq.write_table(nodes, f"{p}/nodes.parquet", compression="zstd")
    df = edges_tbl.to_pandas()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries(geoms, crs=4326))
    gdf.to_parquet(f"{p}/edges.parquet", compression="zstd", geometry_encoding=encoding, index=False)


def mem_of(fmt, path):
    r = subprocess.run([PY, "-u", "-W", "ignore", __file__, "mem", fmt, path], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"peak": float("nan"), "retained": float("nan"), "err": r.stderr[-300:]}


# ---------------- list-attribute round-trip checks ----------------
def edge_lookup(obj):
    if isinstance(obj, nx.MultiDiGraph):
        return lambda u, v, k, a: obj.edges[u, v, k].get(a)
    e = obj[1]
    if not {"u", "v", "key"} <= set(e.columns):
        e = e.reset_index()
    idx = {t: i for i, t in enumerate(zip(e["u"].astype("int64"), e["v"].astype("int64"), e["key"].astype("int64")))}
    cols = {c: e[c].tolist() for c in LIST_ATTRS if c in e.columns}

    def f(u, v, k, a):
        if a not in cols:
            return "<missing col>"
        x = cols[a][idx[(u, v, k)]]
        if isinstance(x, np.ndarray):
            x = x.tolist()
        return x

    return f


def classify(G, obj):
    look = edge_lookup(obj)
    res = {}
    for a in LIST_ATTRS:
        lst = [(u, v, k, d[a]) for u, v, k, d in G.edges(keys=True, data=True) if isinstance(d.get(a), list)]
        sc = [(u, v, k, d[a]) for u, v, k, d in G.edges(keys=True, data=True) if a in d and not isinstance(d[a], list)][:200]
        cats = set()
        for u, v, k, orig in lst[:500]:
            try:
                got = look(u, v, k, a)
            except KeyError:
                cats.add("edge missing")
                continue
            if got == "<missing col>":
                cats.add("column missing")
            elif isinstance(got, list) and [str(x) for x in got] == [str(x) for x in orig]:
                cats.add("faithful" if got == orig else "list (elem type changed)")
            elif isinstance(got, str) and got == str(orig):
                cats.add("stringified")
            elif isinstance(got, str) and got == ";".join(map(str, orig)):
                cats.add("';'-joined")
            elif got == orig[0]:
                cats.add("first elem only")
            else:
                cats.add(f"other({type(got).__name__})")
        st = set()
        for u, v, k, orig in sc:
            try:
                got = look(u, v, k, a)
            except KeyError:
                continue
            if got == "<missing col>":
                st.add("missing")
            elif type(got) == type(orig) or (isinstance(orig, (int, np.integer)) and isinstance(got, (int, np.integer))) or (isinstance(orig, bool) and isinstance(got, (bool, np.bool_))):
                st.add("same type")
            else:
                st.add(f"{type(orig).__name__}->{type(got).__name__}")
        res[a] = f"{'/'.join(sorted(cats)) or 'n/a'}; scalars: {'/'.join(sorted(st)) or 'n/a'}"
    return res


# ---------------- main ----------------
def run(gname, reps=3, only=None):
    G = load_graph(gname)
    base = f"{ROOT}/fmt/{gname}"
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base)
    log(f"\n### Part A — {gname} ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges), read = mean of {reps}\n")
    t = time.perf_counter()
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    log(f"- input prep: ox.graph_to_gdfs {time.perf_counter()-t:.2f} s (not included in GeoParquet write times)")
    t = time.perf_counter()
    lean = {
        "f32": nx_to_lean(G, osmid="int", length="float32", geometry="wkb"),
        "f64": nx_to_lean(G, osmid="int", length="float64", geometry="wkb"),
        "f32_osmidlist": nx_to_lean(G, osmid="list", length="float32", geometry="wkb"),
    }
    n32, e32 = lean["f32"]
    lean["nogeom"] = (n32, e32.drop_columns(["geometry"]))
    _, e_nog, geoms = nx_to_lean(G, osmid="int", length="float32", geometry="shapely")
    log(f"- input prep: lean table extraction (4 variants) {time.perf_counter()-t:.2f} s (not included in lean write times)")

    rows, listres = [], {}

    def bench(label, fmt_dir_or_file, write_fn, readers, input_desc, check=None):
        if only and label not in only:
            return
        path = f"{base}/{fmt_dir_or_file}"
        gc.collect()
        try:
            t = time.perf_counter()
            write_fn(path)
            tw = time.perf_counter() - t
        except Exception as ex:
            rows.append([label, input_desc, "FAIL: " + repr(ex)[:120], "", "", "", ""])
            log(f"  {label}: write FAILED {ex!r}"[:300])
            return
        sz = size_mb(path)
        rts = []
        obj = None
        for rname in readers:
            try:
                tr, obj = timeit(lambda: READERS[rname](path), reps)
                rts.append(f"{rname}: {tr:.3f}")
            except Exception as ex:
                rts.append(f"{rname}: FAIL {ex!r}"[:150])
        mem = mem_of(readers[0], path)
        if check is not None:
            try:
                o = READERS[check](path) if isinstance(check, str) else obj
                listres[label] = classify(G, o)
            except Exception as ex:
                listres[label] = {a: f"check failed {ex!r}"[:80] for a in LIST_ATTRS}
        del obj
        row = [label, input_desc, f"{tw:.3f}", "<br>".join(rts), f"{sz:.2f}", f"{mem['peak']:.0f}", f"{mem['retained']:.0f}"]
        rows.append(row)
        print("   ", row, flush=True)

    def w_graphml(p):
        ox.save_graphml(G, p)

    def w_pickle(p):
        with open(p, "wb") as f:
            pickle.dump(G, f, protocol=5)

    def w_pickle_gz(p):
        with gzip.open(p, "wb", compresslevel=6) as f:
            pickle.dump(G, f, protocol=5)

    def w_gpkg(p):
        ox.save_graph_geopackage(G, p, directed=True)

    def w_geoparquet(comp, raw):
        def w(p):
            os.makedirs(p, exist_ok=True)
            n, e = (nodes_gdf, edges_gdf) if raw else (stringify_lists(nodes_gdf), stringify_lists(edges_gdf))
            n.to_parquet(f"{p}/nodes.parquet", compression=comp)
            e.to_parquet(f"{p}/edges.parquet", compression=comp)
        return w

    def w_lean(var, comp="zstd"):
        return lambda p: write_lean_dir(p, *lean[var], compression=comp)

    def w_feather(var, comp):
        def w(p):
            os.makedirs(p, exist_ok=True)
            n, e = lean[var]
            feather.write_feather(n, f"{p}/nodes.arrow", compression=comp)
            feather.write_feather(e, f"{p}/edges.arrow", compression=comp)
        return w

    def w_zip(p):
        tmp = p + ".d"
        write_lean_dir(tmp, *lean["f32"])
        with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as z:
            z.write(f"{tmp}/nodes.parquet", "nodes.parquet")
            z.write(f"{tmp}/edges.parquet", "edges.parquet")
        shutil.rmtree(tmp)

    def w_csr(p):
        ids = n32.column("osmid").to_numpy()
        A = lean_to_csr(ids, e32.column("u").to_numpy(), e32.column("v").to_numpy(), e32.column("length").to_numpy())
        np.savez_compressed(p, indptr=A.indptr, indices=A.indices, data=A.data.astype(np.float32), ids=ids,
                            x=n32.column("x").to_numpy(), y=n32.column("y").to_numpy())

    NX = "nx graph"
    bench("GraphML", "g.graphml", w_graphml, ["graphml"], NX, check=True)
    bench("pickle p5", "g.pkl", w_pickle, ["pickle"], NX, check=True)
    bench("pickle p5 + gzip(6)", "g.pkl.gz", w_pickle_gz, ["pickle_gz"], NX)
    bench("GeoPackage (ox, directed)", "g.gpkg", w_gpkg, ["gpkg", "gpkg_arrow"], NX, check="gpkg")
    bench("GeoParquet snappy (raw gdfs)", "gpq_raw", w_geoparquet("snappy", True), ["geoparquet"], "gdfs", check="geoparquet")
    bench("GeoParquet snappy (lists stringified)", "gpq_snappy", w_geoparquet("snappy", False), ["geoparquet", "geoparquet_nx"], "gdfs", check="geoparquet_nx")
    bench("GeoParquet zstd (lists stringified)", "gpq_zstd", w_geoparquet("zstd", False), ["geoparquet", "geoparquet_nx"], "gdfs")
    bench("Lean pq zstd f32, osmid int64, WKB", "lean_f32", w_lean("f32"), ["lean_pandas", "lean_arrow", "lean_geo"], "lean tbl", check="lean_pandas")
    bench("Lean pq zstd f64, osmid int64, WKB", "lean_f64", w_lean("f64"), ["lean_pandas", "lean_arrow"], "lean tbl")
    bench("Lean pq zstd f32, osmid list<int64>, WKB", "lean_f32_ol", w_lean("f32_osmidlist"), ["lean_pandas", "lean_arrow"], "lean tbl", check="lean_pandas")
    bench("Lean pq zstd f32, no geometry", "lean_nogeom", w_lean("nogeom"), ["lean_pandas", "lean_arrow"], "lean tbl")
    bench("Lean pq snappy f32, WKB", "lean_f32_snappy", w_lean("f32", "snappy"), ["lean_pandas", "lean_arrow"], "lean tbl")
    bench("Lean GeoParquet zstd (WKB, gpd)", "lean_gpq_wkb", lambda p: write_lean_gpq(p, n32, e_nog, geoms, "WKB"), ["lean_gpq"], "lean tbl")
    bench("Lean GeoParquet zstd (GeoArrow native)", "lean_gpq_geoarrow", lambda p: write_lean_gpq(p, n32, e_nog, geoms, "geoarrow"), ["lean_gpq"], "lean tbl")
    bench("Feather/IPC zstd lean f32 WKB", "feather_zstd", w_feather("f32", "zstd"), ["feather_pandas", "feather_arrow"], "lean tbl")
    bench("Feather/IPC uncompressed lean f32 WKB", "feather_unc", w_feather("f32", "uncompressed"), ["feather_pandas", "feather_arrow"], "lean tbl")
    bench("Pkg (b) zip(stored) of lean parquet, via file obj", "pkg.zip", w_zip, ["zip_bytes"], "lean tbl")
    bench("Pkg (c) dir of lean parquet, via file obj", "lean_f32", lambda p: None, ["dir_bytes"], "(same files)")
    bench("CSR npz_compressed (topology only)", "csr.npz", w_csr, ["csr"], "lean tbl")

    log("Columns: write s (from the stated input), read s per reader (mean), on-disk MB, peak RSS increase during first read in a fresh process (MB, ru_maxrss), RSS retained after read (MB).\n")
    log("Readers: graphml=ox.load_graphml; gpkg=gpd.read_file x2 (pyogrio), gpkg_arrow=same with use_arrow=True; geoparquet=gpd.read_parquet x2; geoparquet_nx=+ox.graph_from_gdfs; lean_arrow=pq.read_table x2; lean_pandas=+to_pandas (WKB left as bytes); lean_geo=+GeoSeries.from_wkb; lean_gpq=pq nodes + gpd.read_parquet edges; zip_bytes/dir_bytes=open(...,'rb') -> BytesIO -> pq.read_table -> pandas.\n")
    table(["format", "input", "write s", "read s", "MB", "peak MB", "retained MB"], rows)
    if listres:
        log(f"\n#### List-valued attribute round trip ({gname})\n")
        table(["format"] + LIST_ATTRS, [[k] + [v[a] for a in LIST_ATTRS] for k, v in listres.items()])


if __name__ == "__main__":
    if sys.argv[1] == "mem":
        fmt, path = sys.argv[2], sys.argv[3]
        gc.collect()
        b, b0 = maxrss_mb(), rss_mb()
        o = READERS[fmt](path)
        gc.collect()
        print(json.dumps({"peak": maxrss_mb() - b, "retained": rss_mb() - b0}))
    else:
        g = sys.argv[2]
        reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        only = set(sys.argv[4].split(";")) if len(sys.argv) > 4 else None
        run(g, reps, only)
