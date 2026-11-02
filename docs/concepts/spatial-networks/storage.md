# Storage and interoperability

## Saving and loading

`save` writes a network to a folder of three files: `nodes.parquet`, `edges.parquet` as GeoParquet, and `network.json` with whether the network is directed, its CRS, weight, and metadata. `SpatialNetwork.load` reads it back.

```python
import os

started = time.perf_counter()
network.save("rwanda-with-schools")
print(f"Saved in {time.perf_counter() - started:.2f} s")
for name in sorted(os.listdir("rwanda-with-schools")):
    print(f"  {name}: {os.path.getsize(os.path.join('rwanda-with-schools', name)) / 1e6:.1f} MB")

started = time.perf_counter()
loaded = SpatialNetwork.load("rwanda-with-schools")
print(loaded, f"{time.perf_counter() - started:.2f} s")
```

```text
Saved in 0.94 s
  edges.parquet: 48.3 MB
  network.json: 0.0 MB
  nodes.parquet: 2.9 MB
SpatialNetwork(nodes=125,494, edges=333,426, directed=True, weight='length') 0.71 s
```

The files are plain Parquet, compressed with zstd, so other tools read them directly: geopandas reads `edges.parquet` as a GeoDataFrame, and DuckDB, QGIS, and pandas read both tables without the library.

```python
import geopandas as gpd

edges_on_their_own = gpd.read_parquet("rwanda-with-schools/edges.parquet")
print(edges_on_their_own.shape, edges_on_their_own.crs.to_string())
```

```text
(333426, 21) EPSG:4326
```

**Loading less.** `columns=` names the edge columns to load, on top of the ones every network needs (`u`, `v`, `key`, `length`, `kind`, and the weight). With an empty list, only the topology and lengths load, which is the fastest way to get a network for routing:

```python
started = time.perf_counter()
topology = SpatialNetwork.load("rwanda-with-schools", columns=[])
print(list(topology.edges.columns), f"{time.perf_counter() - started:.3f} s")
```

```text
['u', 'v', 'key', 'length', 'kind'] 0.230 s
```

**Remote data stores.** `save(path, data_store=...)` and `load(path, data_store=...)` work with any of the library's data stores, such as Azure Blob Storage, through the same calls.

**Why this format.** We compared GraphML, pickle, GeoPackage, Feather, and Parquet on Rwanda's roads. For the unsimplified network, GraphML took 1.4 GB and 437 s to read, pickle 532 MB and 25 s, and these Parquet files 73 MB and 0.18 s. Pickle cannot be read safely from a data store you do not control, and GeoPackage turns OpenStreetMap's list attributes into strings. The details are in [design decision 4](../../design/spatial-networks-decisions.md#decision-4-a-folder-of-parquet-files-for-storage).

**Attributes on save.** Parquet cannot store a column that mixes lists and single values, as osmnx's attributes do after simplification. The library normalises them when it builds a network: `osmid` becomes a list of integers on every edge, other lists become `;`-joined text, and text becomes categorical. Networks from your own tables are normalised when you save them; the network itself does not change.

## GeoDataFrames

`nodes_gdf` and `edges_gdf` are the tables as GeoDataFrames in EPSG:4326, built once and cached; edges without geometry get straight lines. `to_geodataframes(crs=)` returns both, projected if you ask:

```python
nodes_utm, edges_utm = network.to_geodataframes(crs="EPSG:32735")
print(nodes_utm.crs.name, edges_utm.crs.name)
print(edges_utm.geometry.length.sum() / network.total_length())
```

```text
WGS 84 / UTM zone 35S WGS 84 / UTM zone 35S
0.9985940358403483
```

The projected lengths differ from the network's great-circle lengths by a fraction of a percent near the projection's centre. The network's `length` is what every algorithm uses.

## networkx

`to_networkx()` returns a networkx `MultiDiGraph` (or `MultiGraph` when the network is undirected, or with `directed=False`), with every attribute. It is built on each call and not cached, so keep it in a variable. `from_networkx` goes the other way and takes any osmnx graph.

```python
started = time.perf_counter()
graph = network.to_networkx()
print(type(graph).__name__, f"{graph.number_of_edges():,} edges", f"{time.perf_counter() - started:.2f} s")
```

```text
MultiDiGraph 333,426 edges 1.83 s
```

**When to use it.** For algorithms scipy does not have, such as betweenness centrality or community detection, and for libraries built on networkx, including osmnx's own functions. On a large network, centrality in networkx can take hours; run it on a clipped or reduced network.

## pandana

`to_pandana()` returns a pandana network indexed by node ID, built once per weight and cached, on which pandana's own functions, such as aggregations of points within a distance of every node, become available. It needs the `pandana` extra, and node IDs that are all integers or all text. `backend="pandana"` (see [Algorithms](algorithms.md#very-many-pairs-the-pandana-backend)) does not need either restriction on IDs: it gives pandana node positions instead.
