"""Synthetic road-grid micro-benchmark: columnar tables vs networkx for Q2."""
import gc
import os
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandana
import psutil
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

proc = psutil.Process(os.getpid())


def rss():
    gc.collect()
    return proc.memory_info().rss / 1e6


def t(label, fn, *a, **k):
    s = time.perf_counter()
    out = fn(*a, **k)
    print(f"{label:<55s} {time.perf_counter() - s:8.2f} s")
    return out


side = int(os.environ.get("SIDE", 400))
rng = np.random.default_rng(0)
step = 0.001  # ~110 m
ix, iy = np.meshgrid(np.arange(side), np.arange(side))
nid = (iy * side + ix).ravel().astype(np.int64) + 10_000_000
x = 30.0 + ix.ravel() * step + rng.normal(0, 1e-5, side * side)
y = -2.0 + iy.ravel() * step + rng.normal(0, 1e-5, side * side)

right = (ix < side - 1).ravel()
up = (iy < side - 1).ravel()
idx = np.arange(side * side)
u0 = np.concatenate([idx[right], idx[up]])
v0 = np.concatenate([idx[right] + 1, idx[up] + side])
u = np.concatenate([u0, v0])
v = np.concatenate([v0, u0])
n_e = len(u)
mid = np.column_stack([(x[u] + x[v]) / 2 + 2e-5, (y[u] + y[v]) / 2])
coords = np.stack([np.column_stack([x[u], y[u]]), mid, np.column_stack([x[v], y[v]])], axis=1)

base = rss()
geom = shapely.linestrings(coords.reshape(-1, 2), indices=np.repeat(np.arange(n_e), 3))
nodes = pd.DataFrame({"x": x, "y": y, "street_count": 4}, index=pd.Index(nid, name="osmid"))
edges = gpd.GeoDataFrame(
    {
        "u": nid[u],
        "v": nid[v],
        "key": 0,
        "osmid": rng.integers(1, 10**9, n_e),
        "highway": pd.Categorical(rng.choice(["residential", "tertiary", "service"], n_e)),
        "length": np.hypot(x[u] - x[v], y[u] - y[v]) * 111_000,
    },
    geometry=geom,
    crs="EPSG:4326",
)
frames_mb = rss() - base
print(f"graph: {len(nodes):,} nodes, {n_e:,} directed edges")
print(f"{'columnar tables, process memory':<55s} {frames_mb:8.0f} MB")


def frames_to_nx():
    G = nx.MultiDiGraph(crs="EPSG:4326")
    G.add_nodes_from(zip(nodes.index, nodes.to_dict("records")))
    attrs = edges.drop(columns=["u", "v", "key"])
    G.add_edges_from(zip(edges["u"], edges["v"], edges["key"], attrs.to_dict("records")))
    return G


base = rss()
G = t("columnar -> networkx MultiDiGraph (all attributes)", frames_to_nx)
print(f"{'networkx graph, extra process memory':<55s} {rss() - base:8.0f} MB")

t("networkx -> ox.graph_to_gdfs (nodes + edges)", ox.graph_to_gdfs, G)


def user_to_pandana(G):
    nd = pd.DataFrame.from_dict(G.nodes, orient="index")
    ed = nx.to_pandas_edgelist(G)
    return pandana.Network(nd["x"], nd["y"], ed["source"], ed["target"], ed[["length"]])


t("networkx -> pandana (your to_pandana_graph)", user_to_pandana, G)
t(
    "columnar -> pandana (direct)",
    lambda: pandana.Network(nodes["x"], nodes["y"], edges["u"], edges["v"], edges[["length"]]),
)

x0, x1 = 30.0 + side * step * 0.4, 30.0 + side * step * 0.6
y0, y1 = -2.0 + side * step * 0.4, -2.0 + side * step * 0.6
print(f"sub-region: central box, 4% of the area")
_ = edges.sindex
t("columnar: edges in box (sindex query, index prebuilt)", lambda: edges.iloc[edges.sindex.query(shapely.box(x0, y0, x1, y1))])
t("columnar: edges in box (bbox column filter, no index)", lambda: edges[(x[u] >= x0) & (x[u] <= x1) & (y[u] >= y0) & (y[u] <= y1)])


def nx_box():
    keep = [n for n, d in G.nodes(data=True) if x0 <= d["x"] <= x1 and y0 <= d["y"] <= y1]
    return ox.graph_to_gdfs(G.subgraph(keep), nodes=False)


t("networkx: subgraph of nodes in box, then graph_to_gdfs", nx_box)
t("networkx: full graph_to_gdfs, then .cx[] box", lambda: ox.graph_to_gdfs(G, nodes=False).cx[x0:x1, y0:y1])

pos = pd.Series(np.arange(len(nid)), index=nid)
A = t(
    "columnar -> scipy CSR adjacency",
    lambda: csr_matrix((edges["length"].to_numpy(), (pos[edges["u"]].to_numpy(), pos[edges["v"]].to_numpy())), shape=(len(nid),) * 2),
)
src = rng.choice(len(nid), 100, replace=False)
t("scipy dijkstra, 100 sources to all nodes", dijkstra, A, indices=src)
