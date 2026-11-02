# Design: spatial networks and the OSM roads fetcher

This is the agreed design for two components: `OSMRoadsFetcher`, which fetches the road network of a source from OpenStreetMap, and `SpatialNetwork`, which holds any spatial network and offers editing, insertion of points, reduction, and network algorithms. It records the decisions from the design interview of October 2026 and splits the work into sub-tasks. Each sub-task starts by reading this document, the glossary, and ADRs 0003 to 0005.

Read first:

- `GLOSSARY.md` and `CONTEXT.md` at the repository root (local files, gitignored): spatial network, road network, insertion, connecting subnetwork, fetcher, source.
- [ADR 0003](../adr/0003-public-import-paths.md): public import paths and lazy loading.
- [ADR 0004](../adr/0004-columnar-spatial-network.md): a spatial network is held as tables; networkx and pandana are exports; edits return a new network.
- [ADR 0005](../adr/0005-spatial-network-storage.md): a saved network is a folder of Parquet files.

This page is not in the docs site navigation. The user-facing description is the [road networks guide](../user-guide/road-networks.md) and the [spatial networks API page](../api/network.md).

## 1. Responsibilities

- `OSMRoadsFetcher` only acquires data. It turns a source into an area, queries Overpass through osmnx, normalises the attributes, and returns a `SpatialNetwork`. It never saves anything.
- `SpatialNetwork` owns everything after that: accessors, editing, weights, insertion of points, simplification, algorithms, the networkx and pandana exports, and saving and loading.
- The fetcher shares no static helpers with the network; it hands over a network object.
- Application workflows stay in applications. The per-admin clustering, the school, fiber, and splitter node types, and the orchestration in the old `OSMRoadGraphModel` are not part of the library. The library provides the building blocks: `fetch`, `insert_points`, `concat`, `connect_components`, and `connecting_subnetwork`.

## 2. Modules, imports, and dependencies

| Item | Location |
|---|---|
| `SpatialNetwork` | `gigaspatial/processing/network.py`, possibly split into a `gigaspatial/processing/network/` package if it grows past one module |
| `OSMRoadsFetcher` | `gigaspatial/fetchers/osm_roads.py` |
| Public names | `from gigaspatial import SpatialNetwork, OSMRoadsFetcher`; `gigaspatial.processing.SpatialNetwork`; `gigaspatial.fetchers.OSMRoadsFetcher` |
| Lazy registration | Add both to the `attach` dictionaries and the `TYPE_CHECKING` blocks, per ADR 0003. `tests/test_public_api.py` checks the two lists match and that every exported name resolves. `SpatialNetwork` is registered in sub-task 1; `OSMRoadsFetcher` is registered in sub-task 8, when it exists. |

Dependencies:

- `giga-spatial[roads]` adds `osmnx>=2.0`. On Python 3.10, pip resolves osmnx 2.0.x; on 3.11 and newer, 2.1.x. Use only API common to both, and test both.
- `giga-spatial[pandana]` adds `pandana>=0.8`. It pulls in scikit-learn and PyTables, which is why it is a separate extra.
- `[all]` includes both.
- Core dependencies already cover the rest: numpy, pandas, geopandas, shapely, scipy, pyarrow, and networkx.
- osmnx and pandana are imported only inside the functions that need them, following the Overture and DuckDB pattern: a module flag, and an `ImportError` naming the extra (`pip install 'giga-spatial[roads]'`). A bare `import gigaspatial` must not import either.
- Python support stays at 3.10 and newer.

Configuration: add `OVERPASS_URL` to `gigaspatial/config.py`, with the default `https://overpass-api.de/api/interpreter`.

## 3. Data model

### Tables

The nodes table is a pandas DataFrame indexed by `node_id`. The index keeps the original IDs (OSM integers, user IDs such as `giga_id`, or negative integers for split nodes). IDs are unique.

| Column | Type | Meaning |
|---|---|---|
| `x` | float64 | longitude, EPSG:4326 |
| `y` | float64 | latitude, EPSG:4326 |
| `kind` | categorical | `road`, `point`, or `split` |
| other | any | attributes, such as `street_count` from OSM or the columns of inserted points |

The edges table is a pandas DataFrame with a default RangeIndex. An edge is identified by `(u, v, key)`.

| Column | Type | Meaning |
|---|---|---|
| `u`, `v` | node ID type | endpoints, which must exist in the nodes table |
| `key` | int16 | distinguishes parallel edges, as in osmnx |
| `length` | float64 | meters (see section 4) |
| `kind` | categorical | `road`, `split`, `connection`, `point_link`, `component_link`, or `added` |
| `geometry` | shapely LineString or missing | optional; missing means a straight line between the endpoints |
| other | any | OSM attributes after normalisation (section 8), or user attributes |

Network-level fields:

- `directed` (bool). In a directed network, a two-way road is two rows, one per direction, as osmnx returns it. In an undirected network, each edge is one row and the order of `u` and `v` has no meaning.
- `weight` (str, default `"length"`) is the default weight column for algorithms.
- `metadata` (dict) holds provenance: fetch parameters, the source description, and the library version. It is saved to `network.json`.

### Immutability and caching

Every edit returns a new `SpatialNetwork` (ADR 0004). There is no `inplace=` option. Unchanged columns are shared between the old and new network, not copied. Never mutate a table that another network may share; build a new column or a new table instead.

Each instance caches its derived structures lazily:

- the ID-to-position mapping (`pd.Index.get_indexer`, int32 positions);
- the scipy CSR adjacency per weight and direction;
- the KD-tree on node coordinates;
- the STRtree on edge geometries;
- `nodes_gdf` and `edges_gdf`;
- a pandana network per weight.

Because networks never change, these caches never need to be invalidated.

## 4. Coordinates, distances, and lengths

- Nodes are stored in EPSG:4326. Constructors that receive another CRS convert to it.
- Lengths are meters. Road edges keep the lengths osmnx gives them: the great-circle (haversine) distance with an Earth radius of 6,371,009 m, added up along the original edges during simplification.
- Every edge the library creates gets its haversine length along its geometry, using the same formula and radius as osmnx. This covers connection edges, split pieces, `from_points` edges, component links, and point links. The calculation is a vectorized numpy function, shared by every module.
- When a road is split, the lengths of the pieces are rescaled so they add up exactly to the original length. Inserting a point never changes the length of a route that passes through it.
- Nearest-edge searches run in a UTM projection estimated from the network's extent (`estimate_utm_crs_with_fallback`). UTM is conformal, so the ranking of nearby candidates is right even outside the zone. Distance thresholds such as `max_distance` and `snap_to_node` are always checked with haversine distances, never projected ones.
- Nearest-node searches use a `cKDTree` on 3D unit-sphere coordinates, with chord distances converted to great-circle meters. This is accurate everywhere and needs no projection.
- Ellipsoidal (geodesic) distances were rejected because they would make new edges inconsistent with osmnx's road lengths. Haversine is within about 0.5% of the ellipsoid.
- Accessors take `crs=` to return projected output. Coordinate arrays are always ordered `(y, x)`, which is `(lat, lon)` in EPSG:4326.

## 5. SpatialNetwork API

### Constructors

- `SpatialNetwork(nodes, edges, directed=True, crs="EPSG:4326", weight="length", metadata=None)` validates and adopts the two tables. Validation checks that IDs are unique, edges reference existing nodes, `x` and `y` are present, and `length` is present or computable. Missing lengths are computed with haversine.
- `SpatialNetwork.from_networkx(G)` takes osmnx graphs or any networkx graph whose nodes have `x` and `y`. `directed` comes from `G.is_directed()`, and the CRS from `G.graph["crs"]` (EPSG:4326 if absent). Missing keys become 0. Directed graphs are read from networkx's adjacency dictionaries rather than its edge views, attributes go through `DataFrame.from_records`, and Python's cyclic garbage collector is paused during the conversion (and in `to_networkx`): collections triggered by millions of new objects tripled the time on unsimplified Rwanda.
- `SpatialNetwork.from_points(points, connect="knn", k=5, max_distance=None, id_column=None, weight="length", weight_func=None)` builds a network whose nodes are the points. `connect="knn"` joins each point to its k nearest neighbours. `connect="distance"` joins every pair within `max_distance` meters. Both use a `cKDTree`, like `build_distance_graph`, and never connect all pairs. Edges get `length` in meters. `weight_func(nodes_u, nodes_v, length) -> array` optionally computes another weight column at construction. The network is undirected.
- `SpatialNetwork.load(path, data_store=None)` (section 7).

`points`, here and everywhere else, accepts what the library already understands as point input: a GeoDataFrame, a DataFrame with latitude and longitude columns (found with `detect_coordinate_columns` and converted with `convert_to_geodataframe`), or a list of `(lat, lon)` tuples, the same order the handlers use.

### Accessors

| Accessor | Returns |
|---|---|
| `nodes`, `edges` | the tables; cheap |
| `nodes_gdf`, `edges_gdf` | GeoDataFrames in EPSG:4326, built on first access and cached; missing edge geometries are filled with straight lines in one vectorized call |
| `to_geodataframes(crs=None)` | `(nodes_gdf, edges_gdf)`, projected if `crs` is given |
| `node_coordinates(ids=None, crs=None)` | numpy array of `(y, x)` |
| `degree(directed=None)` | Series of node degrees |
| `bounds` | `(minx, miny, maxx, maxy)` |
| `total_length(by=None)` | meters, optionally grouped by a column such as `kind` |
| `to_networkx(directed=None)` | a new networkx graph each call, not cached: `MultiDiGraph` or `MultiGraph` with all attributes |
| `to_pandana(weight=None)` | a pandana network with `twoway` set to `not directed`; IDs are validated first, because pandana raises a cryptic dtype error on unknown IDs |
| `adjacency(weight=None, directed=None)` | scipy CSR matrix; parallel edges keep the minimum weight |
| `node_index` | maps adjacency positions to node IDs and back |
| `clip(geometry, predicate="intersects")` | the edges matching the predicate and their endpoints |
| `len()`, `info()`, `repr` | size and summary |

### Editing

Every method takes many items at once and returns a new network. They live in `gigaspatial/processing/network/editing.py`, a mixin of `SpatialNetwork`. Filters and attribute edits wrap their tables with `_from_tables`, passing remapped edge positions; additions (`add_nodes`, `add_weighted_edges`, `concat`) go through the validating constructor. Geometries are in EPSG:4326, or a GeoDataFrame or GeoSeries in any CRS.

- `nearest_nodes(points, max_distance=None, return_distance=False)` returns a Series of the node IDs closest to coordinates, indexed like the points. Points with no node within `max_distance` get a missing ID (`<NA>` for integer IDs). The other "by coordinates" methods build on it.
- `filter_nodes(ids=None, mask=None, geometry=None, predicate="intersects")` keeps the selected nodes and the edges between them. Exactly one selector is allowed.
- `remove_nodes(ids=None, points=None, geometry=None, max_distance=None)` removes nodes and their edges. `points` are resolved with `nearest_nodes`.
- `set_node_attribute(name, values)` takes a scalar, an array in node order, or a Series or dict keyed by node ID.
- `update_nodes(frame, by="id" | "coordinates", max_distance=None)` updates several attributes. With `by="coordinates"`, rows are matched to their nearest node.
- `add_nodes(frame, id_column=None, kind="point")` adds nodes from a points input with IDs.
- Attribute edits keep a column's dtype when nothing is lost (an `int16` column stays `int16`), and text columns stay categorical. `x` and `y` cannot be set, and neither can `u`, `v`, and `key`.
- For edges: `filter_edges(mask=None, geometry=None, predicate=...)` keeps every node; `remove_edges(edges=None, mask=None)`, where `edges` are `(u, v, key)` or `(u, v)` tuples or a frame, and `(u, v)` removes every parallel edge; `set_edge_attribute(name, values)`, keyed by `(u, v, key)`; and `update_edges(frame)` keyed by `(u, v, key)`. In an undirected network, `(v, u)` matches `(u, v)`.
- `add_weighted_edges(u, v, weights, weight="length", kind="added", **attributes)` mirrors networkx's `add_weighted_edges_from`, but takes arrays. Keys continue from the edges already between each pair. When `weight != "length"`, `length` is computed with haversine.
- Edge predicates describe the edge's relation to the area: `intersects`, `within`, `crosses`, `touches`. Node predicates are `intersects` and `within`.
- `drop_geometry()` removes edge geometry to save memory.
- `to_undirected(weight=None)` merges reverse and parallel edges into the one with the smallest weight (the network's weight by default), sets every key to 0, and records `directed=False`. It also collapses parallel edges of an undirected network.
- `SpatialNetwork.concat(networks)` combines networks. Nodes with the same ID are the same node, as when separately fetched areas share boundary nodes, and the first occurrence wins. Edges are deduplicated on `(u, v, key)`. All inputs must have the same `directed` value.

### Weights

Algorithms take `weight=None`, which means the network's `weight` (by default `"length"`), and any edge column can be named. There are three ways to set weights: at construction (`from_points(weight_func=...)`), later with `set_edge_attribute`, or when adding edges with `add_weighted_edges`.

## 6. Insertion, reduction, and algorithms

### `insert_points`

```
insert_points(points, id_column=None, max_distance=None, snap_to_node=5.0,
              chain=False, return_skipped=False) -> SpatialNetwork | tuple
```

`return_skipped` was `return_unmerged` in the interview; it was renamed because the glossary avoids "merging". The method lives in `gigaspatial/processing/network/insertion.py`, a mixin of `SpatialNetwork`.

- IDs come from `id_column`, or from the frame's index if it is `None`. They must be unique and must not collide with existing node IDs; a collision raises.
- The columns of the points frame are carried onto their nodes. A collision with reserved columns (`x`, `y`, `kind`) raises.
- Each point is joined to its nearest edge at the foot of the perpendicular. The edge is split there, a split node is added (`kind="split"`), and a connection edge (`kind="connection"`) joins the point (`kind="point"`) to the split node.
- If the foot is within `snap_to_node` meters (haversine) of one of the edge's endpoints, the point connects to that node instead and no split happens.
- Points farther than `max_distance` from any road are left out. A warning gives their count, and `return_skipped=True` returns `(network, skipped_points)`: the skipped rows of a frame, or the skipped items of a list.
- With `chain=True`, points are handled in order of their distance to the road. A point connects to an already-inserted point (`kind="point_link"`) when that point is closer than any road. This reproduces rule (i) of the old `merge_points_to_graph`. Candidates come from a KD-tree on the points, querying k neighbours and doubling k only for points whose k-th neighbour is still closer than their road.
- Split nodes get negative integer IDs, decreasing from the lowest existing negative ID, as OSM does for new objects.
- In a directed network, a two-way road is two rows with reversed geometries. Both rows must be split at the same split nodes, and connection edges are added in both directions.
- Lengths follow section 4: haversine for connections, and split pieces rescaled to the original length.

- Points join road edges only: edges of kind `connection` and `point_link` from earlier insertions are never split.
- In a directed network, the other direction of a two-way road is the row from `v` to `u` whose geometry is the reverse (or both straight). One-way roads are split in their one direction only; connections still run both ways.
- When the network's weight is not `length`, road pieces get their share of the road's weight in proportion to length, and connection edges get the weight equal to their length.
- The UTM projection of the road edges and its STRtree are cached per network (`_projected_roads`), built on the first insertion. The EPSG:4326 STRtree used by `clip` is not used here: in degrees, a road 100 m east looks farther than one 150 m north at 60° latitude.

The work is done in one batch:

1. Run one `STRtree.query_nearest` in UTM for all points.
2. Compute positions along the edges with vectorized `line_locate_point` and `line_interpolate_point`.
3. Group the points by edge and sort each group by position. Points with the same foot share one split node.
4. Split each affected edge once, at all of its points.
5. Append the new nodes and edges in one concatenation.

Python loops may run over affected edges, never over points with table operations inside.

Problems in the old `merge_points_to_graph` that must not come back:

- the distance from each point to every earlier point, which grows with the square of the number of points;
- a `pd.concat` per point, which also grows with the square;
- the bare `except` that falls back to scanning every edge when two points share an edge;
- projecting and copying the whole graph twice.

### `simplify(keep=None, sum_columns=("length",))`

- Removes nodes that only continue a road: nodes with two neighbours in an undirected network, or the osmnx rule for directed networks.
- Never removes nodes of kind `point` or `split`, or nodes in `keep`.
- Joins the edges across each removed node. Columns in `sum_columns` are added up, and so are any columns the user names as weights. Geometries are joined. Text attributes are `;`-joined when they differ. `osmid` lists are concatenated. Other numeric columns take the length-weighted mean. That last rule is a default set in this spec; revisit it in the sub-task 6 pull request if it surprises anyone.
- A two-way road stays two-way only when both directions can be joined.
- Removable nodes follow osmnx's rule, with two deliberate differences. osmnx also removes nodes between duplicated parallel ways and keeps one way; here such nodes stay. osmnx deletes rings without any junction (`remove_rings=True`); here a ring keeps one node, so no road is lost. On the unsimplified Kigali fixture and on unsimplified Rwanda, the result has the same nodes, edges, and edge lengths as `ox.simplify_graph`.
- Joined edges always get a geometry, as in osmnx, so a network without edge geometry gains a `geometry` column, missing for edges left as they were. Edges without geometry contribute their two node positions.
- Implemented in `gigaspatial/processing/network/reduction.py` without walking chains in Python: each edge into a removable node is linked to the edge that continues it, chains are ranked by pointer jumping, and attributes are aggregated with `reduceat` per chain. Rings are found as connected components of removable nodes that touch no other node. Undirected networks run on half-edges and keep one of each mirrored chain.
- Tests compare against `tests/data/spatial_network/kigali_raw/` (the fixture box cut from unsimplified Rwanda, 8,266 nodes and 15,652 edges) and `kigali_raw_osmnx.json`, osmnx's result for it, both made by `make_fixture.py raw`.

### `connecting_subnetwork(terminals, sources=None, method="steiner", weight=None)`

This works on the undirected form of the network. The result is contracted to the terminals, the sources, and the branching nodes (degree 3 or more), with lengths added up and geometries joined.

- `method="steiner"` (the default) uses Mehlhorn's 2-approximation of the Steiner tree. One multi-source Dijkstra run from the terminals (scipy `dijkstra(min_only=True, return_predecessors=True)`) finds each node's nearest terminal. Edges between different terminals' regions give candidate links. Then come an MST over the terminals, expansion of the links into paths, a second MST, and removal of leaves that are not terminals.
- With `sources=` given, the result is the union of each terminal's shortest path to its nearest source, using one multi-source run from the sources.
- `method="all_pairs"` is the union of shortest paths between every terminal and every node in terminals plus sources, the behaviour of the old `create_cleaned_graph`. Its cost grows with the square of the number of terminals, which is documented.
- Paths are collected from Dijkstra predecessor arrays with a vectorized walk back to the roots, not one path at a time. The contraction is `simplify(keep=terminals and sources)` on an undirected network of the chosen edges, so lengths and geometries are joined the same way. Terminals that cannot reach the others stay as nodes without edges.

### Algorithms

Every algorithm uses scipy by default. `backend="pandana"` is available where noted, and requires `[pandana]`. Directed networks respect one-way roads unless `directed=False` is passed. The algorithms live in `gigaspatial/processing/network/algorithms.py`, a mixin of `SpatialNetwork`.

- The adjacency matrix is cached per weight and direction, together with the edge row behind each entry, so routes can be drawn along edge geometries. Its transpose (for searches towards targets) and pandana networks are cached the same way.
- Dijkstra runs in blocks of sources of at most `CHUNK_BYTES` (256 MiB) of distances, and `shortest_path_lengths` refuses result matrices over `MAX_MATRIX_BYTES` (1 GiB). Both are module constants.
- Weights must be present and non-negative on every edge; zero weights are allowed. scipy's minimum spanning tree drops zero-weight entries, so they are replaced by the smallest positive float for that call only.
- pandana reports unreachable pairs as 4294967.295; the library returns `inf`. pandana stores weights with limited precision (differences of about 0.02 m on the Kigali fixture). `to_pandana` refuses node IDs that mix integers and text.
- `connect_components` uses Borůvka's method: each round, every component finds its nearest node outside it (a KD-tree on all nodes with a growing k for small components, a KD-tree on the component's own nodes for large ones). When the network's weight is not `length`, the links get that weight equal to their length.

| Method | Result |
|---|---|
| `shortest_path_lengths(origins, destinations=None, pairwise=False, weight=None, backend="scipy")` | a matrix DataFrame of origins by destinations, or a 1-D array when `pairwise=True`. Estimate the matrix size first and raise with advice to batch if it would exceed a limit. pandana is supported for the pairwise case. |
| `shortest_paths(origins, destinations, pairwise=True, weight=None, geometry=False)` | lists of node IDs, or a GeoDataFrame with one LineString per route and its length when `geometry=True` |
| `reachable(origins, max_distance, weight=None)` | a long table of origin, node ID, and distance, from `dijkstra(limit=...)` |
| `nearest_by_network(origins, targets, k=1, max_distance=None)` | origin, target, and distance. `k=1` is one multi-source run from the targets (on the transposed matrix when directed). |
| `count_within(origins, targets, max_distance)` | a Series of counts per origin, running from whichever side is smaller |
| `connected_components(strong=False)` | a Series of component labels |
| `largest_component()` | a network |
| `connect_components(max_distance=None)` | a network with `component_link` edges from an MST over components, using the nearest node pairs from a KD-tree |
| `minimum_spanning_tree(weight=None)` | an undirected network |

Deferred: isochrone polygons, centrality (available through `to_networkx()`), and travel speeds (osmnx's `add_edge_speeds`).

## 7. Storage (ADR 0005)

`save(path, data_store=None)` writes a folder through `DataStore.open(..., "wb")`, using in-memory buffers so remote data stores work:

- `nodes.parquet`: plain Parquet, zstd. The `node_id` column holds the index, plus `x`, `y`, `kind`, and attributes. Categoricals are stored as dictionaries.
- `edges.parquet`: GeoParquet 1.0, WKB geometry, zstd, with the file-level `geo` metadata written by the library (not through geopandas' private writer). When the geometry has been dropped, it is a plain Parquet file with no geometry column.
- `network.json`: `{"format": "gigaspatial.spatial_network", "format_version": 1, "directed", "crs", "weight", "node_ids", "metadata", "library_version", "created"}`. It is written last, so a folder with `network.json` always holds complete tables.
- Node IDs are stored natively when they are all integers or all text. When they mix both, as after inserting points with text IDs into an OSM network, `node_ids` is `"mixed"`: `node_id`, `u`, and `v` are stored as text and a `node_id_is_int` column restores the integers on load. An integer and a text ID with the same digits (`5` and `"5"`) cannot be saved; other ID types raise.
- The files carry no pandas metadata, so plain `pd.read_parquet` reads them; pandas would otherwise record dtype names it cannot parse back.
- Before writing, `normalize_attributes` (`gigaspatial/processing/network/attributes.py`) makes every attribute column storable; the network itself is not changed. The fetcher calls the same function with `int_lists=("osmid",)`.

`load(path, data_store=None, columns=None)` checks `format` and `format_version`, and raises a clear error for unknown versions. List columns such as `osmid` load as Arrow-backed columns (`pd.ArrowDtype`), so each value reads as a Python list and the column stays compact. `columns=` names edge attributes to read on top of `u`, `v`, `key`, `length`, `kind`, and the weight column; `"geometry"` is included only if it was saved. Reading a subset of columns is allowed (`columns=` for edges) for users who only need the topology.

## 8. OSMRoadsFetcher

The fetcher is a pydantic dataclass, like the other fetchers (`@dataclass(kw_only=True)`, logger from `config.get_logger`).

```
OSMRoadsFetcher(network_type="drive_service", custom_filter=None, simplify=True,
                retain_all=False, truncate_by_edge=False, buffer=2000, area="hull",
                overpass_url=None, timeout=180, max_retries=3)
    .fetch(source) -> SpatialNetwork
```

How a source becomes the queried area:

| Source | Area |
|---|---|
| Country | the admin 0 boundary through `AdminBoundaries.create(...)`, as handlers do |
| Geometry | used as given |
| Frame of polygons | the union of the polygons |
| Points, or a frame of points | `area="hull"` (default): convex hull; `"bbox"`: bounding rectangle; `"buffers"`: union of a buffer per point. Buffered by `buffer` meters through UTM (`buffer_geodataframe`), never degrees. |

- Point tuples are `(lat, lon)`.
- The library's source classification (`classify_source`) decides the kind. The body of `BaseHandlerConfig.extract_search_geometry` moved into `source_geometry(source, crs, options)` in `gigaspatial/handlers/base/config.py`; the method calls it, and so does the fetcher. Lines are buffered like points, and a single point or a line with `buffer=0` raises. `query_area(source)` is public, so users can see what will be queried.
- The query is `ox.graph_from_polygon(polygon, network_type=..., simplify=..., retain_all=..., truncate_by_edge=..., custom_filter=...)`. Set `ox.settings` for the Overpass URL (`overpass_url` or `config.OVERPASS_URL`) and the timeout inside a context manager that restores the previous settings. osmnx's HTTP cache is left as the user configured it.
- osmnx's `overpass_url` is the API base, while `OVERPASS_URL` is the full endpoint, so a trailing `/interpreter` is removed for osmnx.
- Transient failures are retried up to `max_retries` times, pausing `retry_delay` seconds and doubling each time: connection errors and timeouts (`requests.RequestException`) and osmnx's `ResponseStatusCodeError`. osmnx already retries 429 and 504 itself. osmnx's `InsufficientResponseError` (no matching roads) becomes a `ValueError` and is not retried. The last error names the Overpass URL and suggests a mirror.
- Warn when the area is large (above about 50,000 km²): Overpass is slow at country scale and has fair-use limits, and the Geofabrik handler will be the faster route.
- Normalisation before building the network, with `normalize_attributes(edges, int_lists=("osmid",))`, so saving does not repeat it (ADR 0005):
  - `osmid` becomes a list of int64 on every edge.
  - List-valued text attributes (`highway`, `name`, `maxspeed`, `lanes`, `ref`, and the like) become `;`-joined strings, stored as categoricals.
  - `oneway` and `reversed` become booleans. A list of `oneway` values is true only if every value is true; a list of `reversed` values is true if any value is true. The test fixture uses the same rules (`benchmarks/spatial_network/make_fixture.py`).
- The network is built with the column-wise extraction of `from_networkx`. The fetch parameters and the source description go into `metadata`.
- `directed=True` is what OSM returns. Users call `to_undirected()` when they need it.

## 9. Performance targets

Measured on an Apple M1 Pro (8 cores, 17 GB) with the Geofabrik Rwanda extract. The full results and scripts are in `benchmarks/spatial_network/`.

| Network | Nodes | Edges |
|---|---|---|
| Kigali, simplified | 18,547 | 47,499 |
| Rwanda, simplified | 125,058 | 332,535 |
| Rwanda, unsimplified | 2,224,948 | 4,526,967 |

Targets on the simplified Rwanda network unless stated. A pull request that misses one explains why.

| Operation | Target | Baseline measured |
|---|---|---|
| `load` | under 0.5 s | lean Parquet 0.17 s, 0.29 s with geometry decoded; GraphML 12.0 s |
| `save` | under 1 s | lean Parquet 0.26 s |
| Memory, tables with geometry | under 400 MB | 229 MB with WKB, 393 MB decoded; networkx 731 MB |
| `node_coordinates()` | under 5 ms | 0.6 ms from the table; 43 ms from networkx |
| `clip` of 4% of the area | under 0.1 s | 0.03 s; 1.6 to 3.0 s from networkx |
| `adjacency()` | under 0.3 s | 0.14 s |
| `to_networkx()` | under 3 s | 2.5 s |
| Conversion from an osmnx graph | under 2 s | 0.81 s; 14.5 s for unsimplified Rwanda (4.5 million edges), once 49 s; 4.0 s with `graph_to_gdfs` |
| `shortest_path_lengths`, 100 by 1,000 | under 3 s with scipy | scipy 2.4 s; pandana 1.4 s build plus 0.19 s; networkx about 29 s |
| `insert_points`, 10,000 points | under 10 s | 1.8 s, plus 1.3 s once per network for the UTM road index; the old function took 292 s for 1,000 points |
| `simplify` of unsimplified Rwanda | faster than osmnx | osmnx `simplify_graph` 197 s; `simplify` 9.1 s, same result |

## 10. Tests and benchmarks

- Unit tests run offline, on synthetic networks and on a small saved Kigali network fixture under `tests/data/` (a few hundred KB of Parquet). Build the fixture once from the benchmark data with a script kept in `benchmarks/`.
- The fetcher's tests mock osmnx and `AdminBoundaries.create`. One test marked `@pytest.mark.live` calls Overpass for a small area.
- Each sub-task tests its own methods, including the cases called out above: two-way roads in directed networks, points sharing a split position, ID collisions, list attributes in storage, and unknown IDs given to pandana.
- The benchmark scripts live in `benchmarks/spatial_network/`. They are run by hand, not by pytest. Sub-tasks 2 to 7 rerun the relevant script and report the numbers in the pull request.

## 11. Sub-tasks

Each sub-task is its own chat and pull request, with tests, in this order. The fetcher comes late because everything else can be tested on saved networks.

| # | Sub-task | Delivers |
|---|---|---|
| 1 | Scaffolding | the `[roads]` and `[pandana]` extras; `OVERPASS_URL`; `SpatialNetwork` registered for import and covered by the public API test; an empty `SpatialNetwork` module; benchmark scripts tidied in `benchmarks/spatial_network/`; the Kigali fixture in `tests/data/spatial_network/kigali/` (1,209 nodes, 2,802 edges, 264 KB) |
| 2 | Core | tables, validation, the ID-to-position mapping, constructors (`__init__`, `from_networkx`, `from_points`), accessors, `to_networkx`, `info`, caching, haversine helper |
| 3 | Storage | `save` and `load`, and attribute normalisation (shared with the fetcher) |
| 4 | Editing | `nearest_nodes`, filter, remove, update and add for nodes and edges, `add_weighted_edges`, `clip`, `drop_geometry`, `to_undirected`, `concat` |
| 5 | Algorithms | `adjacency`, shortest paths, `reachable`, `nearest_by_network`, `count_within`, components, `connect_components`, MST, `to_pandana` and the pandana backend |
| 6 | Reduction | `simplify`, `connecting_subnetwork` |
| 7 | Insertion | `insert_points` in one batch, including `chain=True`, benchmarked against the old `merge_points_to_graph` |
| 8 | Fetcher | `OSMRoadsFetcher`: source to area, osmnx, normalisation, `SpatialNetwork`, retries, live test, and its import registration |
| 9 | Docs | user guide page, API reference pages, changelog entry, README mention |

## 12. Deferred

- A Geofabrik handler for country-scale road networks from `.osm.pbf` extracts, returning the same `SpatialNetwork`. Its design is now in [`osm-roads-handler.md`](osm-roads-handler.md), with ADRs 0006 and 0007; the notes below are the starting point it was designed from. This comes next, decided on 28 October 2026:
  - Users need both small areas and whole countries or provinces. Overpass suits the small ones; a country-scale Overpass fetch can take hours, exceed fair-use limits, and needs osmnx's networkx graph in memory (about 2.3 GB for unsimplified Rwanda).
  - It is a separate handler, not an option on `OSMRoadsFetcher`: an extract is a file kept in a data store and reused, which the glossary calls a handler.
  - It reads extracts with pyosmium (`osmium`), added to `giga-spatial[roads]`; pyosmium has wheels for Python 3.10 to 3.14 on macOS, Linux, and Windows (pyrosm needs 3.11 or newer). It builds the node and edge tables directly, without osmnx or networkx, and uses `simplify`. The benchmark's route through OSM XML and osmnx took 300 s for Rwanda, most of it in osmnx; pyosmium filtered the extract in 22.5 s, and `simplify` takes 9 s.
  - Open questions for its design round: the class name; whether it returns exactly what `OSMRoadsFetcher` returns for the same area, which means reproducing osmnx's network-type filters and one-way rules; clipping to admin areas smaller than an extract; Geofabrik's split and combined countries (its `index-v1.json` lists ISO codes); and keeping stored extracts up to date.
  - The road networks guide says country-scale networks from Geofabrik are planned.
- Isochrone polygons, centrality measures, and travel speeds.
- Dropping Python 3.10 (end of life in October 2026) is a library-wide decision, outside this work.
