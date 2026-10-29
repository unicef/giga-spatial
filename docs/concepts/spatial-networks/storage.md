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
Saved in 0.88 s
  edges.parquet: 48.3 MB
  network.json: 0.0 MB
  nodes.parquet: 2.9 MB
SpatialNetwork(nodes=125,490, edges=333,422, directed=True, weight='length') 0.65 s
```

The files are plain Parquet, compressed with zstd, so other tools read them directly: geopandas reads `edges.parquet` as a GeoDataFrame, and DuckDB, QGIS, and pandas read both tables without the library.

```python
import geopandas as gpd

edges_on_their_own = gpd.read_parquet("rwanda-with-schools/edges.parquet")
print(edges_on_their_own.shape, edges_on_their_own.crs)
```

```text
(333422, 21) {"$schema": "https://proj.org/schemas/v0.7/projjson.schema.json", "type": "GeographicCRS", "name": "WGS 84", "datum_ensemble": {"name": "World Geodetic System 1984 ensemble", "members": [{"name": "World Geodetic System 1984 (Transit)"}, {"name": "World Geodetic System 1984 (G730)"}, {"name": "World Geodetic System 1984 (G873)"}, {"name": "World Geodetic System 1984 (G1150)"}, {"name": "World Geodetic System 1984 (G1674)"}, {"name": "World Geodetic System 1984 (G1762)"}, {"name": "World Geodetic System 1984 (G2139)"}, {"name": "World Geodetic System 1984 (G2296)"}], "ellipsoid": {"name": "WGS 84", "semi_major_axis": 6378137, "inverse_flattening": 298.257223563}, "accuracy": "2.0", "id": {"authority": "EPSG", "code": 6326}}, "coordinate_system": {"subtype": "ellipsoidal", "axis": [{"name": "Geodetic latitude", "abbreviation": "Lat", "direction": "north", "unit": "degree"}, {"name": "Geodetic longitude", "abbreviation": "Lon", "direction": "east", "unit": "degree"}]}, "scope": "Horizontal component of 3D system.", "area": "World.", "bbox": {"south_latitude": -90, "west_longitude": -180, "north_latitude": 90, "east_longitude": 180}, "id": {"authority": "EPSG", "code": 4326}}
```

**Loading less.** `columns=` names the edge columns to load, on top of the ones every network needs (`u`, `v`, `key`, `length`, `kind`, and the weight). With an empty list, only the topology and lengths load, which is the fastest way to get a network for routing:

```python
started = time.perf_counter()
topology = SpatialNetwork.load("rwanda-with-schools", columns=[])
print(list(topology.edges.columns), f"{time.perf_counter() - started:.3f} s")
```

```text
['u', 'v', 'key', 'length', 'kind'] 0.233 s
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
0.9994358486727203
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
MultiDiGraph 333,422 edges 1.71 s
```

**When to use it.** For algorithms scipy does not have, such as betweenness centrality or community detection, and for libraries built on networkx, including osmnx's own functions. On a large network, centrality in networkx can take hours; run it on a clipped or reduced network.

## pandana

`to_pandana()` returns a pandana network, built once per weight and cached, and `backend="pandana"` uses it for shortest path lengths (see [Algorithms](algorithms.md#very-many-pairs-the-pandana-backend)). pandana's own functions, such as aggregations of points within a distance of every node, become available on it. It needs the `pandana` extra, and node IDs that are all integers or all text.
