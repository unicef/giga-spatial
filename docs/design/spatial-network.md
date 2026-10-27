# Design: spatial networks and the OSM roads fetcher

This is the agreed design for two components: `OSMRoadsFetcher`, which fetches the road network of a source from OpenStreetMap, and `SpatialNetwork`, which holds any spatial network and offers editing, insertion of points, reduction, and network algorithms. It records the decisions from the design interview of October 2026 and splits the work into sub-tasks. Each sub-task starts by reading this document, the glossary, and ADRs 0003 to 0005.

Read first:

- `GLOSSARY.md` and `CONTEXT.md` at the repository root (local files, gitignored): spatial network, road network, insertion, connecting subnetwork, fetcher, source.
- [ADR 0003](../adr/0003-public-import-paths.md): public import paths and lazy loading.
- [ADR 0004](../adr/0004-columnar-spatial-network.md): a spatial network is held as tables; networkx and pandana are exports; edits return a new network.
- [ADR 0005](../adr/0005-spatial-network-storage.md): a saved network is a folder of Parquet files.

This page is not in the docs site navigation. The user guide page written in sub-task 9 is the user-facing description.

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
| Lazy registration | Add both to the `attach` dictionaries and the `TYPE_CHECKING` blocks, per ADR 0003. `tests/test_public_api.py` checks the two lists match. |

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
- `SpatialNetwork.from_networkx(G)` takes osmnx graphs or any networkx graph whose nodes have `x` and `y`. `directed` comes from `G.is_directed()`, and the CRS from `G.graph["crs"]` (EPSG:4326 if absent). Missing keys become 0. Attributes are extracted column by column with `np.fromiter` and comprehensions, not with `ox.graph_to_gdfs`: on the Rwanda network that took 0.9 s against 4.0 s.
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

Every method takes many items at once and returns a new network.

- `nearest_nodes(points, max_distance=None, return_distance=False)` returns the node IDs closest to coordinates. The other "by coordinates" methods build on it.
- `filter_nodes(ids=None, mask=None, geometry=None, predicate="intersects")` keeps the selected nodes and the edges between them. Exactly one selector is allowed.
- `remove_nodes(ids=None, points=None, geometry=None, max_distance=None)` removes nodes and their edges. `points` are resolved with `nearest_nodes`.
- `set_node_attribute(name, values)` takes a scalar, an array in node order, or a Series or dict keyed by node ID.
- `update_nodes(frame, by="id" | "coordinates", max_distance=None)` updates several attributes. With `by="coordinates"`, rows are matched to their nearest node.
- `add_nodes(frame)` adds nodes from a points input with IDs.
- For edges: `filter_edges(mask=None, geometry=None, predicate=...)`, `remove_edges(edges=None, mask=None)` (where `edges` are `(u, v, key)` tuples or a frame), `set_edge_attribute(name, values)`, and `update_edges(frame)` keyed by `(u, v, key)`.
- `add_weighted_edges(u, v, weights, weight="length", kind="added", **attributes)` mirrors networkx's `add_weighted_edges_from`, but takes arrays. When `weight != "length"`, `length` is computed with haversine.
- `drop_geometry()` removes edge geometry to save memory.
- `to_undirected()` merges reverse and parallel edges, keeping the shortest, and records `directed=False`.
- `SpatialNetwork.concat(networks)` combines networks. Nodes with the same ID are the same node, as when separately fetched areas share boundary nodes, and the first occurrence wins. Edges are deduplicated on `(u, v, key)`. All inputs must have the same `directed` value.

### Weights

Algorithms take `weight=None`, which means the network's `weight` (by default `"length"`), and any edge column can be named. There are three ways to set weights: at construction (`from_points(weight_func=...)`), later with `set_edge_attribute`, or when adding edges with `add_weighted_edges`.

## 6. Insertion, reduction, and algorithms

### `insert_points`

```
insert_points(points, id_column=None, max_distance=None, snap_to_node=5.0,
              chain=False, return_unmerged=False) -> SpatialNetwork | tuple
```

- IDs come from `id_column`, or from the frame's index if it is `None`. They must be unique and must not collide with existing node IDs; a collision raises.
- The columns of the points frame are carried onto their nodes. A collision with reserved columns (`x`, `y`, `kind`) raises.
- Each point is joined to its nearest edge at the foot of the perpendicular. The edge is split there, a split node is added (`kind="split"`), and a connection edge (`kind="connection"`) joins the point (`kind="point"`) to the split node.
- If the foot is within `snap_to_node` meters (haversine) of one of the edge's endpoints, the point connects to that node instead and no split happens.
- Points farther than `max_distance` from any road are left out. A warning gives their count, and `return_unmerged=True` returns `(network, unmerged_points)`.
- With `chain=True`, points are handled in order of their distance to the road. A point connects to an already-inserted point (`kind="point_link"`) when that point is closer than any road. This reproduces rule (i) of the old `merge_points_to_graph`. Candidates come from a KD-tree on the points, querying k neighbours and doubling k only for points whose k-th neighbour is still closer than their road.
- Split nodes get negative integer IDs, decreasing from the lowest existing negative ID, as OSM does for new objects.
- In a directed network, a two-way road is two rows with reversed geometries. Both rows must be split at the same split nodes, and connection edges are added in both directions.
- Lengths follow section 4: haversine for connections, and split pieces rescaled to the original length.

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
- Suggested approach: the connected components of the subgraph of removable nodes are the chains to join, which keeps the work in scipy instead of walking chains in Python.

### `connecting_subnetwork(terminals, sources=None, method="steiner", weight=None)`

This works on the undirected form of the network. The result is contracted to the terminals, the sources, and the branching nodes (degree 3 or more), with lengths added up and geometries joined.

- `method="steiner"` (the default) uses Mehlhorn's 2-approximation of the Steiner tree. One multi-source Dijkstra run from the terminals (scipy `dijkstra(min_only=True, return_predecessors=True)`) finds each node's nearest terminal. Edges between different terminals' regions give candidate links. Then come an MST over the terminals, expansion of the links into paths, a second MST, and removal of leaves that are not terminals.
- With `sources=` given, the result is the union of each terminal's shortest path to its nearest source, using one multi-source run from the sources.
- `method="all_pairs"` is the union of shortest paths between every terminal and every node in terminals plus sources, the behaviour of the old `create_cleaned_graph`. Its cost grows with the square of the number of terminals, which is documented.

### Algorithms

Every algorithm uses scipy by default. `backend="pandana"` is available where noted, and requires `[pandana]`. Directed networks respect one-way roads unless `directed=False` is passed.

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
- `edges.parquet`: GeoParquet, WKB geometry, zstd. When the geometry has been dropped, it is a plain Parquet file with no geometry column.
- `network.json`: `{"format": "gigaspatial.spatial_network", "format_version": 1, "directed", "crs", "weight", "metadata", "library_version", "created"}`.

`load(path, data_store=None)` checks `format` and `format_version`, and raises a clear error for unknown versions. Reading a subset of columns is allowed (`columns=` for edges) for users who only need the topology.

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
- Reuse the library's source classification (`classify_source`). If `BaseHandlerConfig.extract_search_geometry` cannot be used without a handler config, move its body into a function both can call, without changing handler behaviour.
- The query is `ox.graph_from_polygon(polygon, network_type=..., simplify=..., retain_all=..., truncate_by_edge=..., custom_filter=...)`. Set `ox.settings` for the Overpass URL (`overpass_url` or `config.OVERPASS_URL`) and the timeout inside a context manager that restores the previous settings. osmnx's HTTP cache is left as the user configured it.
- Transient failures are retried up to `max_retries` times with backoff. Errors name the Overpass URL and suggest a mirror.
- Warn when the area is large (above about 50,000 km²): Overpass is slow at country scale and has fair-use limits, and the Geofabrik handler will be the faster route.
- Normalisation before building the network, which is required for saving (ADR 0005):
  - `osmid` becomes a list of int64 on every edge.
  - List-valued text attributes (`highway`, `name`, `maxspeed`, `lanes`, `ref`, and the like) become `;`-joined strings, stored as categoricals.
  - `oneway` and `reversed` become booleans.
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
| Conversion from an osmnx graph | under 2 s | 0.9 s column-wise; 4.0 s with `graph_to_gdfs` |
| `shortest_path_lengths`, 100 by 1,000 | under 3 s with scipy | scipy 2.4 s; pandana 1.4 s build plus 0.19 s; networkx about 29 s |
| `insert_points`, 10,000 points | under 10 s | the old function is measured in sub-task 7 |
| `simplify` of unsimplified Rwanda | faster than osmnx | osmnx about 100 s inside `graph_from_xml` |

## 10. Tests and benchmarks

- Unit tests run offline, on synthetic networks and on a small saved Kigali network fixture under `tests/data/` (a few hundred KB of Parquet). Build the fixture once from the benchmark data with a script kept in `benchmarks/`.
- The fetcher's tests mock osmnx and `AdminBoundaries.create`. One test marked `@pytest.mark.live` calls Overpass for a small area.
- Each sub-task tests its own methods, including the cases called out above: two-way roads in directed networks, points sharing a split position, ID collisions, list attributes in storage, and unknown IDs given to pandana.
- The benchmark scripts live in `benchmarks/spatial_network/`. They are run by hand, not by pytest. Sub-tasks 2 to 7 rerun the relevant script and report the numbers in the pull request.

## 11. Sub-tasks

Each sub-task is its own chat and pull request, with tests, in this order. The fetcher comes late because everything else can be tested on saved networks.

| # | Sub-task | Delivers |
|---|---|---|
| 1 | Scaffolding | the `[roads]` and `[pandana]` extras; `OVERPASS_URL`; lazy-import registration and the public API test; an empty `SpatialNetwork` module; benchmark scripts tidied in `benchmarks/spatial_network/`; the Kigali fixture |
| 2 | Core | tables, validation, the ID-to-position mapping, constructors (`__init__`, `from_networkx`, `from_points`), accessors, `to_networkx`, `info`, caching, haversine helper |
| 3 | Storage | `save` and `load`, and attribute normalisation (shared with the fetcher) |
| 4 | Editing | `nearest_nodes`, filter, remove, update and add for nodes and edges, `add_weighted_edges`, `clip`, `drop_geometry`, `to_undirected`, `concat` |
| 5 | Algorithms | `adjacency`, shortest paths, `reachable`, `nearest_by_network`, `count_within`, components, `connect_components`, MST, `to_pandana` and the pandana backend |
| 6 | Reduction | `simplify`, `connecting_subnetwork` |
| 7 | Insertion | `insert_points` in one batch, including `chain=True`, benchmarked against the old `merge_points_to_graph` |
| 8 | Fetcher | `OSMRoadsFetcher`: source to area, osmnx, normalisation, `SpatialNetwork`, retries, live test |
| 9 | Docs | user guide page, API reference pages, changelog entry, README mention |

## 12. Deferred

- A Geofabrik handler for country-scale road networks from `.osm.pbf` extracts, returning the same `SpatialNetwork`. This is the first follow-up: the benchmark built all of Rwanda from the extract in 300 s while every public Overpass server was down. pyosmium (`osmium`) has wheels for Python 3.10 to 3.14; pyrosm needs 3.11 or newer.
- Isochrone polygons, centrality measures, and travel speeds.
- Dropping Python 3.10 (end of life in October 2026) is a library-wide decision, outside this work.
