# Spatial networks and OSM roads: design and algorithm decisions

This document explains how I built `SpatialNetwork`, `OSMRoadsFetcher`, `OSMRoadsHandler`, and `ConnectionPlanner`, and why. For each decision it gives the question I faced, the options I considered, what I measured, what I chose, and what that choice costs. It is meant for colleagues who use these classes and want to know how far they can trust them, and for whoever changes them next.

It does not repeat the implementation specs or the API. For those, read:

- the [spatial network design](spatial-network.md), the [OSM roads handler design](osm-roads-handler.md), and the [connection planning design](connection-planning.md), which say what was built;
- the ADRs under [Design decisions](../adr/index.md), which record single decisions in a few lines: [0004](../adr/0004-columnar-spatial-network.md) (tables), [0005](../adr/0005-spatial-network-storage.md) (storage), [0006](../adr/0006-osm-extract-storage.md) (extract storage), [0007](../adr/0007-osm-extract-choice.md) (extract choice), and [0008](../adr/0008-connection-planning.md) (connection planning);
- the [spatial networks concept note](../concepts/spatial-networks/index.md), which explains how to use each method;
- the benchmark scripts in `benchmarks/spatial_network/`, `benchmarks/osm_roads/`, and `benchmarks/connection_planning/`, with their last results beside them.

Every number below comes from those scripts. Where a number explains a decision I took earlier, I give the run it came from; everything else is from the run of 29 October 2026 on the code as it is now.

## Environment and data

All benchmarks ran on one laptop:

| | |
|---|---|
| Machine | Apple M1 Pro, 8 CPU cores, 16 GiB of memory, internal SSD |
| System | macOS 15.7 |
| Python 3.10.11 | numpy 2.2.6, pandas 2.3.0, scipy 1.15.2, shapely 2.1.2, geopandas 1.1.4, pyarrow 17.0.0, networkx 3.3, osmnx 2.0.2, pandana 0.7, igraph 1.0.0, pyosmium 4.3.2 |
| Python 3.14.6 | numpy 2.5.2, pandas 3.0.3, scipy 1.18.0, shapely 2.1.2, geopandas 1.1.4, pyarrow 23.0.1, networkx 3.6.1, pyosmium 4.3.2 |

The spatial network benchmarks run on Python 3.10, where osmnx, pandana, and igraph are installed for the comparisons. The OSM road benchmarks run on Python 3.14, with pandas 3. The library supports both, and the test suite runs on both.

The test networks are all Rwanda's drivable roads (osmnx's `drive_service` network type), from Geofabrik's Rwanda extract:

| Network | Built by | Nodes | Edges |
|---|---|---|---|
| Kigali, simplified | osmnx, cut to the box 29.98, -2.03 to 30.20, -1.89 | 18,547 | 47,499 |
| Rwanda, simplified | osmnx's `graph_from_xml` | 125,058 | 332,535 |
| Rwanda, unsimplified | osmnx's `graph_from_xml(simplify=False)` | 2,224,948 | 4,526,967 |
| Rwanda's extract | Geofabrik, 26 October 2026 | 65 MiB file | |

Rwanda is a useful size: big enough that slow code shows (the unsimplified network has 4.5 million edges), small enough to run every comparison on a laptop. Timings are wall-clock times, the mean of three runs unless a table says otherwise. Costs paid once per network, such as building a spatial index, are reported separately from the calls that use them. Memory is the resident set size of the process, or tracemalloc's count where stated.

To reproduce, see `benchmarks/spatial_network/README.md` and `benchmarks/osm_roads/README.md`. A full run takes about ten minutes, most of it osmnx's own `simplify_graph`.

## Part 1: SpatialNetwork

### Decision 1: tables, not networkx graphs

**Question.** What should hold a network in memory? osmnx returns a networkx `MultiDiGraph`, and the original Giga code worked on those directly.

**Options.** A networkx graph; an igraph graph; or two tables, one of nodes and one of edges, with the structures that algorithms need (a sparse adjacency matrix, spatial indexes) derived from them and cached.

**Evidence.** Measured on simplified Rwanda (`partB.py`, 27 October 2026):

| | networkx graph | tables |
|---|---|---|
| Memory, with edge geometry | 731 MB | 229 MB (geometry as WKB), 393 MB (as shapely objects) |
| All node coordinates | 42.7 ms | 0.61 ms |
| The edges of a box covering 4% of the country | 1.63 s | 0.037 s |
| Shortest paths from one source to every node | 292 ms | 24.1 ms (scipy) |

networkx keeps a Python dictionary per node and per edge, so every accessor loops in Python. igraph is compact and fast, but it is one more compiled dependency, and its routing sat between scipy's and pandana's (decision 5).

**Decision.** A `SpatialNetwork` holds a pandas table of nodes, indexed by node ID, and a table of edges identified by `(u, v, key)`. scipy's sparse graph routines work on an adjacency matrix built from the edge table on demand. networkx and pandana are export formats: `to_networkx()` and `to_pandana()`.

On the current code, a built network of simplified Rwanda takes 350 MB with its tables, all node coordinates come back in 0.38 ms, clipping the same 4% box takes 0.066 s, and the adjacency matrix is built in 0.062 s. Converting from osmnx takes 0.85 s and back to networkx 1.35 s, so the bridge to networkx stays cheap for anyone who needs it.

**Consequences and limits.** Editing one node or edge at a time in a Python loop is slow, so every editing method takes many items at once. Algorithms that scipy does not offer, such as centrality, are reached through `to_networkx()`.

### Decision 2: edits return a new network

**Question.** Should edits change the network in place, as networkx and many pandas methods do, or return a new one?

**Options.** In-place edits; an `inplace=` option; or always a new network.

**Evidence.** An in-place option saves no memory, because a new network shares every column it did not change with the old one. It would also make caching dangerous. A network caches its adjacency matrix per weight and direction, a KD-tree on nodes, an R-tree on edges, a projected index of roads for insertion, a pandana network, and its GeoDataFrames. Every in-place edit would have to invalidate the right ones, and missing one gives wrong shortest paths without any error. Returning a new network is cheap: keeping 10,000 nodes by ID takes 0.004 s and setting an attribute 0.004 s. Adding 10,000 edges, which validates the whole network again, takes 0.34 s.

**Decision.** Every edit returns a new `SpatialNetwork`, and there is no `inplace=` option. We discussed keeping one for familiarity, and I left it out for the reasons above.

**Consequences and limits.** The `nodes` and `edges` properties return shallow copies of the tables. Adding a column to them is safe, but changing their values in place could reach another network that shares the column, so the docs say not to.

### Decision 3: lengths by haversine, with osmnx's Earth radius

**Question.** How should the library measure the edges it creates, such as connections to inserted points, split pieces of roads, links between points, and links between components?

**Options.** Great-circle (haversine) distance with osmnx's Earth radius of 6,371,009 m; ellipsoidal (geodesic) distance; or distance in a projected CRS such as UTM.

**Evidence.** osmnx measures road edges with haversine and that radius, so any other formula would make new edges inconsistent with the roads they join. Haversine is within about 0.5% of the ellipsoid. Projected distances drift with distance from the projection's centre: on 1,000 points inserted into Rwanda's roads, the old code's connections added up to 67,985 m in UTM metres, against 68,187 m by haversine.

**Decision.** Every length is haversine in metres, with osmnx's formula and radius, computed by one vectorised function. When a road is split, the pieces are rescaled so they add up exactly to the road's length, so inserting a point never changes the length of a route through it. Searches use whatever ranks candidates correctly: the nearest edge is found in UTM, which is conformal, and the nearest node with a KD-tree on points on the unit sphere, which needs no projection at all. Distance thresholds are always checked in haversine metres.

**Consequences and limits.** Lengths can differ from a geodesic measurement by up to about 0.5%, in exchange for consistency with osmnx's road lengths.

### Decision 4: a folder of Parquet files for storage

**Question.** How should a network be saved?

**Options.** GraphML or pickle of a networkx graph, as is common with osmnx; a GeoPackage with two layers; GeoParquet written by geopandas; Feather; or our own layout of Parquet tables.

**Evidence.** `partA.py`, 27 October 2026. Read times are the fastest reader for each format:

| Format | Simplified Rwanda | | Unsimplified Rwanda | |
|---|---|---|---|---|
| | size | read | size | read |
| GraphML | 216 MB | 12.0 s | 1,389 MB | 437 s |
| pickle | 121 MB | 1.9 s | 532 MB | 25 s |
| GeoPackage | 167 MB | 0.47 s | | |
| Feather, zstd | 62 MB | 0.14 s | 64 MB | 0.11 s |
| Our Parquet tables, zstd | 64 MB | 0.17 s | 73 MB | 0.18 s |

GeoParquet written straight from osmnx's GeoDataFrames failed: simplified osmnx edges mix lists and single values in one column (`osmid` is a number on some edges and a list on others), which Parquet cannot store. GeoPackage and stringified GeoParquet turn those lists into strings. Pickle cannot be read safely from an untrusted data store, or by any other tool.

**Decision.** `save` writes a folder: `nodes.parquet`, `edges.parquet` as GeoParquet with WKB geometry and zstd compression, and `network.json` with whether the network is directed, its CRS, weight, metadata, and the format version. `network.json` is written last, so a folder that has it holds complete tables. Feather was as fast but is read by fewer tools; GeoPandas, DuckDB, QGIS, and pandas all read these Parquet files directly. On the current code, simplified Rwanda saves to 64.9 MB and loads in 0.47 s, or 0.022 s for the topology alone (`load(columns=[])`).

**Consequences and limits.** Saving a network fresh from osmnx takes 1.2 s, above my 1 s target, because its attributes are normalised first (decision 9). A network that was loaded or built by this library saves in 0.62 s. Node IDs that mix integers and text are stored as text with a flag that restores the integers.

### Decision 5: scipy for routing, pandana as an option

**Question.** Which engine should compute shortest paths and the searches built on them?

**Options.** networkx; igraph; scipy's sparse graph routines (`scipy.sparse.csgraph`); pandana, which uses contraction hierarchies.

**Evidence.** On simplified Rwanda, 100 origins by 1,000 destinations (`partB.py`, 27 October 2026):

| Engine | Set-up | 100 by 1,000 | Per pair | Agreement with scipy |
|---|---|---|---|---|
| networkx | none | about 29 s | 292 µs | within 0.003 m |
| igraph | 0.28 s graph build | 1.33 s | 13.3 µs | exact |
| scipy | 0.14 s adjacency build | 2.41 s | 24.1 µs | |
| pandana | 1.42 s contraction hierarchy | 0.19 s | 1.9 µs | median 0.15 m, at most 0.47 m |

scipy is already a dependency of the library. pandana is the fastest once built, but it pulls in scikit-learn and PyTables, stores weights with limited precision, reports unreachable pairs as 4294967.295, and fails with a cryptic dtype error on an unknown node ID. It is maintained: version 0.8 came out in August 2026. igraph would have been one more compiled dependency for a middle speed.

**Decision.** Every algorithm uses scipy by default. `backend="pandana"` is available where it helps, through the `giga-spatial[pandana]` extra. The backend gives pandana node positions rather than IDs, so it works on any network, including one whose IDs mix OpenStreetMap's integers with the text IDs of inserted schools, and it returns `inf` for unreachable pairs. `to_pandana()`, for pandana's own functions, keeps node IDs and so still needs them all integers or all text. My first version passed IDs to pandana in both cases, which ruled out the backend on exactly the networks people build; writing the concept note's examples showed it. networkx is only an export. I did not add igraph.

How the algorithms use scipy, on the current code:

| Operation, simplified Rwanda | Time |
|---|---|
| `shortest_path_lengths`, 100 by 1,000, scipy | 2.58 s |
| the same with pandana, after a 1.43 s build | 0.24 s |
| `reachable`, 100 origins, 5 km | 0.080 s |
| `nearest_by_network`, 10,000 origins to 100 targets | 0.046 s |
| `count_within`, 1,000 origins, 100 targets, 5 km | 0.039 s |
| `minimum_spanning_tree` | 0.069 s |

The searches are fast because they avoid computing a full matrix. `reachable` stops Dijkstra at the distance limit. `nearest_by_network` with one target per origin runs a single multi-source Dijkstra from all the targets, on the reversed graph when the network is directed. `count_within` runs from whichever side is smaller. Dijkstra runs in blocks of at most 256 MiB of distances, and `shortest_path_lengths` refuses result matrices over 1 GiB, with advice to batch.

**Consequences and limits.** scipy's cost grows with the number of origins (about 25 ms per origin on Rwanda), and pandana's with the number of pairs. pandana's 1.43 s build costs as much as about 55 origins of scipy work; below that, scipy is faster end to end. scipy's minimum spanning tree ignores zero-weight edges, so zero weights are replaced by the smallest positive float for that call only.

### Decision 6: our own simplification

**Question.** osmnx's `simplify_graph` removes nodes that only continue a road. Should the library call it, or implement the same rule on tables?

**Options.** Convert to networkx and call osmnx; or implement it on the tables.

**Evidence.** On unsimplified Rwanda (`reduction.py simplify`, current code):

| | Time | Nodes | Edges | Total length |
|---|---|---|---|---|
| `ox.simplify_graph` | 147.5 s | 125,058 | 332,535 | 106,220,098.3 m |
| `SpatialNetwork.simplify` | 7.1 s | 125,058 | 332,535 | 106,220,098.3 m |

The two give the same nodes and the same edge lengths, 21 times faster. The first run, on 27 October, measured 9.1 s against 197 s.

**Decision.** `simplify` uses osmnx's rule for which nodes can go, and never walks a chain of edges in Python. Each edge into a removable node is linked to the edge that continues it; chains are ranked by pointer jumping; attributes are aggregated per chain with numpy's `reduceat`. Joined edges get the sum of their lengths (and of the weight column), one geometry, text joined with `;` where it differs, `osmid` lists concatenated, `oneway` true only if every part is one-way, `reversed` true if any part is, the length-weighted mean of other numbers, and the kind of their longest part.

**Consequences and limits.** I made two deliberate differences from osmnx. osmnx also removes the nodes between two duplicated parallel ways and drops one of the ways; `simplify` keeps them, so no road is lost. osmnx deletes rings with no junction at all; `simplify` keeps one node of the ring. Neither case occurs in Rwanda's roads, which is why the results above are identical. Nodes of kind `point` and `split`, and nodes passed in `keep`, are never removed, so `simplify` can run after points are inserted.

### Decision 7: connecting subnetworks with Mehlhorn's Steiner approximation

**Question.** Fibre planning needs the roads that connect a set of points, such as schools, at the smallest total length. That is the Steiner tree problem, which is NP-hard. How should the library approximate it, and what else should it offer?

**Options.** networkx's `steiner_tree`, after a conversion to networkx; Kou's algorithm, which needs the shortest-path distance between every pair of terminals; Mehlhorn's algorithm, which needs one multi-source Dijkstra; and the old Giga approach, which kept the shortest path between every pair of points (`create_cleaned_graph`).

**Evidence.** On simplified Rwanda (`reduction.py subnetwork`, current code):

| Method | Time | Result |
|---|---|---|
| Steiner tree, 100 terminals | 0.108 s | 1,537 km |
| Steiner tree, 1,000 terminals | 0.178 s | 4,070 km |
| Each of 1,000 terminals to its nearest of 20 sources | 0.178 s | 5,802 km |
| Shortest paths between all pairs, 50 terminals and 20 sources | 1.54 s | 6,453 km |

The test suite checks that our tree on the Kigali fixture is no longer than the shorter of networkx's Mehlhorn and Kou trees.

**Decision.** `connecting_subnetwork` works on the undirected network and offers three methods:

- `method="steiner"` (the default) is Mehlhorn's 2-approximation. One multi-source Dijkstra from the terminals gives each node its nearest terminal. Every edge between two terminals' regions is a candidate link between those terminals, and an MST over the cheapest links, expanded back into road paths, gives a tree. A second MST and pruning of leaves that are not terminals finish it. Its length is at most twice the optimum, and in practice close to it.
- With `sources=`, each terminal takes its shortest path to its nearest source. This is the right model when every school must reach some fibre node, and the cost is one multi-source Dijkstra.
- `method="all_pairs"` keeps the shortest path between every terminal and every terminal or source, as the old code did. Its cost grows with the square of the number of terminals, which is why it is not the default.

The result is contracted to the terminals, the sources, and the branching nodes, with lengths added and geometries joined, using `simplify(keep=...)`.

**Consequences and limits.** The Steiner tree is an approximation, not the optimum. It ignores one-way restrictions, which suits cables laid along roads. Terminals that cannot reach the others stay in the result without edges.

**Later change.** `connecting_subnetwork` became the network of `ConnectionPlanner` (Part 4). `sources=` became `roots=`; with roots, a plan is the rooted Steiner tree, in which a terminal may connect through another one, and each terminal's path to its nearest root is `method="nearest_root"`. The default method became `shortest_path` (decision 25); Mehlhorn's tree is `method="steiner"`.

### Decision 8: inserting points in one batch

**Question.** The old `merge_points_to_graph` joined schools to roads one at a time. How should insertion work at country scale?

**Options.** Keep the old algorithm; or insert all points in one batch.

**Evidence.** On simplified Rwanda (`insertion.py`; the old function was timed on 27 October 2026, one run each):

| | 1,000 points | 2,000 points | 10,000 points |
|---|---|---|---|
| Old `merge_points_to_graph` | 291.7 s | 542.8 s | not run |
| `insert_points`, current code | 0.66 s | | 1.83 s |

`insert_points` also builds a projected index of the road edges once per network, in 1.36 s. The old function compared each point with every earlier point and concatenated a table per point, so its cost grew with the square of the number of points. It fell back to scanning every edge when two points shared an edge, and projected and copied the whole graph twice.

**Decision.** `insert_points` does all points at once. One nearest-edge query in UTM finds each point's road; vectorised `line_locate_point` finds where along the road each point falls; points are grouped by edge and sorted by position; each affected road is split once, at all of its points; and the new nodes and edges are appended in one step. Python loops run only over affected edges. A point within `snap_to_node` metres (5 by default) of a road's end connects to that node instead of splitting the road. In a directed network, both directions of a two-way road are split at the same new nodes. `chain=True` reproduces the old rule that a point may connect to an already inserted point when that point is closer than any road; candidates come from a KD-tree on the points, querying more neighbours only for points that need them.

**Consequences and limits.** Points join roads only, never earlier connections. The 10,000-point time stays far below my 10 s target, so I did not optimise the per-edge loop further.

Writing the concept note's figures showed a bug in the first version. The two directions of a two-way road are split together, so each road row is mapped to its reverse twin, and a one-way road, which has none, was recorded with the twin `-1`. The remapping took `-1` for a lower row and moved every point on a one-way road onto the edge table's last row: in the example, five schools across Kigali shared one split node, and the pieces took the geometry of a road 25 km away. Connection lengths were right, which is what the tests checked; the split nodes' positions and the pieces' geometries were not. The tests now check both, and the fix changed the timings above by less than their noise.

### Decision 9: normalising attributes

**Question.** osmnx attributes mix lists and single values in one column after simplification, and Parquet cannot store that (decision 4). What should those columns become?

**Options.** Stringify the lists, as GeoPackage does; keep Python objects, which cannot be saved; or normalise each column to one storable type.

**Evidence.** Stringified lists lose their types (an `osmid` list becomes the string `"[123, 456]"`) and cannot be filtered or joined. Two library changes made this harder than it looked. pandas 3 reads text as its new `str` dtype rather than `object`, which my first version did not recognise as text. And Arrow list columns made pandas write metadata to the Parquet files that pandas could not read back, so the library writes the GeoParquet metadata itself and leaves out pandas' metadata. Normalising simplified Rwanda's osmnx attributes takes 0.52 s.

**Decision.** `normalize_attributes` gives every attribute column one type. `osmid` becomes a list of integers on every edge (an Arrow list column). Other list columns become lists of integers when they hold only integers, and `;`-joined text otherwise, without repeats. `oneway` is true only if every value is, and `reversed` if any is. Text columns become categoricals. The fetcher and the handler normalise once, when they build a network, so saving does not repeat it.

**Consequences and limits.** A joined text value such as `"primary;secondary"` no longer says which part of the edge is which. That is the same information osmnx gives, as a string.

### Decision 10: fast conversion from networkx

**Question.** Converting osmnx graphs is how most existing work will enter the library. How fast can it be?

**Evidence.** On unsimplified Rwanda (`core.py convert`), my first `from_networkx` took 49 s. Reading networkx's adjacency dictionaries directly instead of its edge views, building attributes with `DataFrame.from_records`, and pausing Python's cyclic garbage collector during the conversion brought it to 14.5 s, and to between 12.9 and 14.6 s in the latest runs. The garbage collector was the largest part: the millions of small objects created during a conversion kept triggering collections over everything already alive. On simplified Rwanda, `from_networkx` takes 0.85 s, against 4.0 s for osmnx's own `graph_to_gdfs`, and `to_networkx` 1.35 s.

**Decision.** Both conversions pause the garbage collector for their duration and restore it afterwards, including when they fail. They create no reference cycles, so nothing is left for the collector.

## Part 2: OSMRoadsFetcher

### Decision 11: osmnx and Overpass for small areas

**Question.** How should users get the roads around a set of schools, a district, or a city?

**Options.** osmnx's `graph_from_polygon` over the Overpass API; our own Overpass client, like `OSMLocationFetcher`'s; or a Geofabrik extract even for small areas.

**Evidence.** osmnx is the reference most people working with OSM roads already know, and it is maintained. Its network types, one-way rules, and simplification define what "the road network" means in most published work. Downloading a whole country's extract (Rwanda is 65 MiB, Nigeria 680 MiB) to get one district is wasteful. I did not time Overpass: it is a shared public service whose speed depends on its load, and osmnx caches its responses.

**Decision.** `OSMRoadsFetcher` turns a source into an area and calls `graph_from_polygon`:

- a country becomes its boundary, a geometry is used as given, and a frame of polygons becomes their union;
- points and lines become their convex hull (`area="hull"`), bounding box (`"bbox"`), or buffers (`"buffers"`), widened by `buffer` metres in UTM, never in degrees;
- `query_area(source)` shows the area before anything is downloaded.

Failed requests are retried with growing pauses, and the last error names the Overpass server and suggests a mirror. A query that finds no roads raises `ValueError` and is not retried. Areas above 50,000 km² log a warning that points to the handler. The osmnx settings for the server and timeout are set for the call and restored afterwards.

**Consequences and limits.** The fetcher needs the internet and Overpass's goodwill. osmnx splits areas over 2,500 km² into several queries, and a country can take hours, which is why decision 12 exists. On Python 3.10 pip resolves osmnx 2.0, and on 3.11 and later 2.1, so the fetcher uses only what both provide.

## Part 3: OSMRoadsHandler

### Decision 12: Geofabrik's extracts for countries

**Question.** How should users get a whole country's roads?

**Options.** Overpass through osmnx, with the area split into parts; or Geofabrik's daily `.osm.pbf` extracts, one file per country or region.

**Evidence.** A country through Overpass can take hours and run into its fair-use limits. Even with the data in hand, osmnx's route is slow: building simplified Rwanda from the extract through OSM XML and `graph_from_xml` took 300 s (`step0.py`, 27 October 2026), and its unsimplified networkx graph held about 2.3 GB. Geofabrik updates every extract daily and keeps the files for download.

**Decision.** A separate handler, `OSMRoadsHandler`, downloads extracts, stores them, and builds the network from them. It is a handler rather than a fetcher option because an extract is a file kept in a data store and reused, which is what our handlers do.

**Consequences and limits.** A small area inside a large country still needs the whole country's file. The fetcher stays the better tool for small areas, and the guide says so.

### Decision 13: choosing extracts by polygon

**Question.** Which extracts does a source need?

**Options.** Look countries up by ISO code; or choose by the extracts' polygons, from Geofabrik's index.

**Evidence.** Geofabrik's index (`index-v1.json`, 554 extracts) gives each extract a polygon, and ISO codes for 191 of them. The codes are not reliable enough to rely on alone. French Polynesia, Wallis and Futuna, Tokelau, American Oceania, and Clipperton carry Vanuatu's code, and Pitcairn carries the Marshall Islands'. Twelve extracts hold several countries, such as `israel-and-palestine` and `malaysia-singapore-brunei`, and several of them carry one country's code or none. About 36 countries, among them Saudi Arabia, Singapore, and Brunei, exist only inside such larger extracts.

**Decision.** Extracts are chosen by polygon ([ADR 0007](../adr/0007-osm-extract-choice.md)):

- A country's ISO code is a fast path: if exactly one extract carries only that code and is not on a curated list of multi-country extracts, it is the country's own file.
- A country without one is covered through its boundary.
- Any other source gets the smallest extracts that together cover its area: touching extracts are taken from the smallest up while part of the area is uncovered, then any extract the others already cover is dropped.
- `regions=True` uses a country's regional extracts. Continents are used only when named in `extracts=`.

The curated lists, of multi-country extracts and of corrected codes, are tested against a stored copy of the whole index: every extract with several codes must be on the list, and no country may end up with two files of its own.

**Consequences and limits.** The curated lists need checking when Geofabrik changes its index. The test runs against the stored copy, so it catches mistakes in our lists but not changes on Geofabrik's side; refreshing the stored copy reveals those.

### Decision 14: storing extracts by date

**Question.** How should extracts be stored and kept up to date?

**Options.** Country folders with dates below them; or date folders with Geofabrik's own folders below them. Always download the latest file; or reuse what is stored.

**Evidence.** Geofabrik keeps daily files for about a week, then the files of the 1st of each month and of 1 January. A dated snapshot is what users keep, compare, or delete together, so the date is the first thing they need to see. `-latest` URLs redirect to the dated file (such as `rwanda-261028.osm.pbf`), except for large extracts like Germany's, which redirect to a mirror under the `-latest` name. For those, the date comes from the extract's replication state file. The date in a file's name is the UTC date of the replication timestamp in its header, which can be read from the file's first 64 KiB.

**Decision.** Extracts are stored as `osm/geofabrik/<date>/<Geofabrik path>.osm.pbf`, with each network saved beside its extract and always covering the whole extract ([ADR 0006](../adr/0006-osm-extract-storage.md)). The newest stored date is used without contacting Geofabrik. `refresh=True` uses Geofabrik's latest, `max_age_days` does when the stored file is older, and `date=` pins a date. Downloads are checked against Geofabrik's `.md5` files while they stream, so a multi-gigabyte file on a remote data store is never read back to check it. A file that fails the check is removed.

**Consequences and limits.** Old dates are never deleted automatically, and extracts take space (Brazil is 2.0 GiB). Reproducing an old result needs its stored extract, since Geofabrik deletes daily files after a week.

### Decision 15: building the network with pyosmium and numpy

**Question.** How should the handler turn an extract into the same network osmnx would build?

**Options.** osmnx's `graph_from_xml`, after converting the extract to OSM XML; pyrosm; quackosm; or pyosmium to read the file, with our own code reproducing osmnx's rules.

**Evidence.** The osmnx route took 300 s for simplified Rwanda: 22.5 s to filter the extract and write 314 MB of XML, and the rest in osmnx. pyrosm needs Python 3.11 or newer, and the library supports 3.10. quackosm converts extracts to GeoParquet features through DuckDB, a heavier dependency, and the network would still have to be built from those features. pyosmium is a thin layer over the C++ library libosmium, with wheels for Python 3.10 to 3.14 on macOS, Linux, and Windows. I only checked pyrosm's and quackosm's versions and wheels; I did not time them.

How to read the file, on Rwanda's extract (`read_strategies.py`, one process each):

| Strategy | Time | Peak memory |
|---|---|---|
| Read the ways, then read only their nodes through an ID filter | 12.2 s | 2,351 MB |
| Read the ways, then read them again with pyosmium's node location index | 7.5 s | 1,047 MB |
| Read once with the location index | 7.3 s | 1,042 MB |

Most of the cost of the first strategy is Python handling 3.8 million node objects. With the location index, pyosmium keeps every node's coordinates while it reads, and writes each way's line as WKB in C++, so Python touches each way but no node.

**Decision.** `build_road_network` reads the extract once with the location index. For each way that passes osmnx's filter for the network type, it keeps the way's ID, node references, and osmnx's kept tags, and has pyosmium write its line. The rest is numpy: drop consecutive repeated nodes, reverse ways that run against their node order, cut ways into edges, number parallel edges as networkx does, measure lengths, and count streets per node. osmnx's rules are reproduced exactly: the network-type filters, including the fact that Overpass's `!~` patterns are unanchored (so `walk`'s exclusion of `motor` also excludes `motorway`), the one-way values including `-1`, `T`, and `F`, the kept tags, and `street_count`.

I measured two reads first and expected one read to be faster. It is not, by much: the location index is what saves the time. I kept one read because it is simpler, with the filter, the tags, and the line coming from the same way, and because it reads the file once.

The result on Rwanda's extract, current code:

| Network | Nodes | Edges | Time |
|---|---|---|---|
| `drive_service`, raw | 2,270,376 | 4,615,754 | 8.4 s |
| `drive_service`, simplified | 129,073 | 338,550 | 13.5 s (fresh process, 2.5 GB peak) |
| `walk`, raw | 3,799,166 | 7,770,848 | 13.7 s |
| `walk`, simplified | 262,964 | 698,444 | 24.4 s |

That is 22 times faster than the osmnx route, and both network types are under my 30 s target. `walk` is the largest of osmnx's presets except `all`, which I did not time.

**Consequences and limits.** The library now owns osmnx's rules, so a change in osmnx would not reach it automatically. Decision 18 describes the tests that guard them. When the builder's output changes, `BUILD_VERSION` in `pbf.py` must be raised so stored networks are built again.

### Decision 16: the node location index

**Question.** Which of pyosmium's location indexes should the build use?

**Evidence.** Dense indexes reserve space for every possible node ID; with OSM's 64-bit IDs that is about 115 GB even for Rwanda. The in-memory `flex_mem` index takes about 16 bytes per node in the file, so about 2 GB for Nigeria. On Rwanda, on this laptop's SSD:

| Index | Time | Peak memory |
|---|---|---|
| `flex_mem`, in memory | 7.3 s | 1,042 MB |
| `sparse_file_array`, on disk | 7.2 s | 846 MB |
| `sparse_mem_array`, in memory | 7.3 s | 1,025 MB |

**Decision.** `flex_mem` up to 1 GiB of extract, and the disk-backed `sparse_file_array` above it, in a temporary file. `node_index=` chooses another, and dense indexes are refused.

**Consequences and limits.** The disk index was as fast as memory and lighter on Rwanda, so the threshold is not about speed on a machine like this one. I kept `flex_mem` as the default for smaller extracts because it needs no temporary disk space. Whether the disk index should be the default everywhere is an open question; it has not been measured on large extracts or slow disks.

### Decision 17: cropping in osmnx's order, and combining extracts as one file

**Question.** A stored network covers a whole extract. How should it be cut to a source, so that the result equals what the fetcher would return for the same area?

**Options.** Clip the stored simplified network, which is small and fast; clip the raw network to the area and simplify afterwards; or replay osmnx's `graph_from_polygon` step by step. For several extracts: `SpatialNetwork.concat`, or a combination that treats them as one file.

**Evidence.** osmnx 2.0.2's `graph_from_polygon` downloads the area widened by 500 m, keeps the largest connected part of that, simplifies it, and only then keeps the nodes inside the area itself and the largest part again. It counts streets on the simplified widened network. Simplifying the widened area first means a road that leaves the area ends where its simplified edge ends, at a real junction outside, and is dropped with that edge.

I first implemented the second option, believing it was osmnx's order, and tested it against my own replay of that order. Writing the concept note's examples showed the mistake: for central Kigali (a 2 by 2 km box), the handler returned 513 nodes and the fetcher 478, from identical raw data (2,824 nodes and 5,117 edges in both). Clipping the raw network first kept the stubs of roads that leave the area. Replaying osmnx's steps on the stored extract gives the fetcher's network exactly: the same 478 nodes and 1,074 edges. The handler now does that, and the tests replay osmnx's steps (decision 18).

For extracts, neighbouring files share the ways that cross between them, with the same IDs, but each numbers parallel edges in its own way order. `concat` keeps one edge per `(u, v, key)`, so depending on which extract came first it can keep a duplicate and drop a different way.

Loading times on Rwanda, from stored networks (`load_rwanda.py`, current code):

| Load | Time |
|---|---|
| Four school points with a 2 km buffer | 1.43 s |
| A 30 by 22 km box around Kigali | 1.85 s |
| The whole country | 0.44 s |
| The first load of a source, which builds and saves the raw network | 11.1 s |

**Decision.** A cropped load replays `graph_from_polygon`: it clips each extract's raw network to the source's area widened by 500 m, combines several extracts with `combine_road_networks`, keeps the largest connected part, simplifies, clips to the area itself, keeps the largest part again, and sets `street_count` as counted on the simplified widened network. `retain_all` and `truncate_by_edge` apply at the same steps as in osmnx. The combination keeps a way's edges once and numbers parallel edges again in way order. A country whose extracts are its own file or its regions is not cropped at all.

**Consequences and limits.** A cropped load reads the raw network, 4.6 million edges for Rwanda, and simplifies the widened area, which together cost 1.4 to 1.9 s for a city-sized area; the first version took about 1 s, for a wrong result. `street_count` near the edge of the widened area counts only the streets inside it, as osmnx's does.

### Decision 18: how equivalence with osmnx is tested

**Question.** The handler claims to return osmnx's network. How is that checked?

**Decision.** The test suite compares against osmnx itself wherever it can:

- osmnx's networks for a small Kigali extract, built once through `graph_from_xml` and stored as fixtures, for `drive_service` raw and simplified, `drive`, `walk`, and `all`. Nodes, `street_count`, node tags, edges by `(u, v, key)`, way IDs, `oneway`, `reversed`, kept tags, and simplified geometries must all match. Raw lengths must be bit-identical.
- The filter presets against osmnx's own filter strings.
- A small made-up extract with every one-way case, repeated nodes, parallel ways, and node tags, against `graph_from_xml`, for a one-way and a two-way network type.
- Cropped loads against osmnx's `graph_from_polygon`, replayed with osmnx's own functions after the download (projection, 500 m buffer, `truncate_graph_polygon`, `largest_component`, `simplify_graph`, `count_streets_per_node`), with and without `truncate_by_edge`, raw and simplified, and with `retain_all`.
- Kigali split in two at a meridian and stored as two extracts, combined in either order, against the whole.
- A live test (`pytest -m live`) that loads central Kigali through the handler and through the fetcher, and requires them to share at least 98% of their nodes, allowing for edits made to OpenStreetMap since the extract's date.

The tests that need osmnx run on Python 3.10, where it is installed, and are skipped elsewhere. One cross-check needs no test: the handler's whole-country load of Rwanda, which keeps the largest connected part, has 125,058 nodes and 332,535 edges, exactly the simplified network osmnx built from the same extract.

The lesson of decision 17 is in the third point: a test that replays my understanding of osmnx can only confirm that understanding. Replaying osmnx's own code, and comparing with the fetcher live, is what catches a wrong understanding.

## Part 4: ConnectionPlanner

The benchmarks of this part are in `benchmarks/connection_planning/`, with their results in `results.md` beside them, and ran on Python 3.14 on 30 October 2026. Rwanda is the simplified network above, with 20 random roots. The design is in the [connection planning design](connection-planning.md), and the reasons for a planner object in [ADR 0008](../adr/0008-connection-planning.md).

### Decision 19: connecting points to an existing network is a rooted Steiner tree

**Question.** A fibre planner's real question is not which roads join the schools to each other, but how to connect every school to the existing fibre at the smallest shared cost, where a school may connect through another school and a trench used by several schools is paid once. Is that a known problem, and how should the library solve it?

**Options.** Each school's shortest path to its nearest fibre node, which `connecting_subnetwork(sources=...)` did; one unrooted Steiner tree over the schools and the fibre nodes; a rooted Steiner tree, with every fibre node joined to one virtual root at cost 0; and an exact solver.

**Evidence.** On the concept note's example, 197 schools and 5 fibre nodes in Kigali: the nearest fibre node for each school took 355 km, one tree over schools and fibre nodes 265 km, and the rooted tree 258 km, in 0.19 s. On Rwanda with 1,000 terminals, the rooted tree took 4,037 km against 5,802 km for the nearest roots, in 0.19 s.

**Decision.** Joining every root to a virtual root makes the problem the Steiner tree problem in graphs, over the terminals and that root. Removing the root leaves a forest whose every tree reaches the existing network. `ConnectionPlanner` holds the problem (network, terminals, roots, weight, bridges) and its solvers, and returns a `ConnectionPlan` with the contracted network, the chosen edges, the cost, and each terminal's status, root, and distance along the plan. `connecting_subnetwork` stays as the planner's network. The existing network's nodes are called roots, since "source" already means the place a dataset is requested for.

**Consequences and limits.** The problem is NP-hard, so the planner returns approximations and a lower bound (decisions 22 and 23). Directions are ignored, and the cost is per edge used, not per cable: a cost that grows with the number of schools behind an edge makes it fixed-charge network design, which is out of scope.

### Decision 20: bridges between parts of the network

**Question.** A clipped network, or one loaded with `retain_all=True`, has parts that no road joins, and schools there cannot be connected. How can a plan cross such gaps, by microwave or a planned road?

**Options.** Join the parts beforehand with `connect_components`, which adds one spanning tree of links whatever they cost; or offer the planner candidate links at a cost per metre, and let it use one only where it is cheaper than the roads.

**Evidence.** On Rwanda cut in two along longitude 30°E (23 parts, the two largest of 70,000 and 55,000 nodes), with all roots in the west, 416 of 1,000 schools could be connected without bridges. With bridges at 10 per metre all 1,000 were, using 2 links of 961 m; at 1,000 per metre, 1 link of 35 m. Finding the 50 candidates took 0.1 s, and the whole plan 0.34 s.

**Decision.** `bridge_cost` offers each part straight-line links to its 5 nearest other parts within `bridge_max_distance`. A part with more than 2,000 nodes, or most of the network, gets its own KD-tree, queried with every node outside it, which gives its closest pair to every other part exactly; smaller parts query a KD-tree of all nodes with their nodes' nearest neighbours, more of them while none lies outside the part. Bridges join the planner's sparse graph as extra entries, not a new network, keep their great-circle length, and count at their cost.

**Consequences and limits.** For two small parts, the search stops at the first outside neighbour, so a small part can miss one of its 5 nearest; the tests check that every part gets its nearest, and the largest its 5 nearest.

### Decision 21: the rollout order by a pruned Dijkstra

**Question.** Takahashi and Matsuyama's heuristic connects the terminal nearest to everything connected so far, one at a time, which is also a model of a phased rollout: each school gets an order and a marginal cost. Run naively, it is one Dijkstra per terminal. How can it run at country scale?

**Options.** The naive version; scipy's Dijkstra from each new path, limited to the largest distance of a waiting terminal; a version that recomputes only when the nearest terminal's distance changed; and a Dijkstra from each new path that stops wherever a distance does not fall, written in Python.

**Evidence.** On Rwanda:

| Terminals | Naive | scipy, limited | Python, pruned |
|---|---|---|---|
| 100 | 1.6 s | 0.6 s | 0.4 s |
| 1,000 | 19.5 s | 5.4 s | 1.1 s |
| 10,000 | not run | 58 s | 4.4 s |

A few remote terminals keep scipy's limit at tens of kilometres, so each update searched about 40,000 nodes: 455 million in the 10,000-terminal run, against 6.2 million nodes updated by the pruned search.

**Decision.** The pruned search in Python. Ties go to the terminal listed first, since node IDs may mix integers and text. The version that recomputes only when the nearest terminal's distance changed is not valid: stale distances are upper bounds, so the top of a heap of them need not be the nearest terminal.

**Consequences and limits.** It is the slowest method, 5 to 20 times slower than Mehlhorn's, but still seconds at country scale. The tests check it, order for order, against the naive version.

### Decision 22: local search with two moves, the second kept local

**Question.** Both heuristics are 2-approximations, and in practice a few percent above the optimum. How can a plan be shortened, without risk of making it worse?

**Options.** Uchoa and Werneck's moves: key-path exchange (replace a path between key vertices by the cheapest path between the two parts it joins), key-vertex elimination (remove a branching node and reconnect the parts by their cheapest links), and key-vertex insertion; and an exact solver.

**Evidence.** On Rwanda with 1,000 terminals, the first version of key-vertex elimination searched for links from every node left in the plan, 20 ms per vertex, and stopped at a 60 s limit after four passes. Searching only from the parts' nodes within the removed weight of the removed nodes gave plans within 0.01% of it on Kigali, in half the time. With it:

| First method | Plan | Improved | Saved | Longest pass |
|---|---|---|---|---|
| steiner | 3,942 km | 3,755 km | 4.7% | 5.8 s |
| shortest_path | 3,845 km | 3,759 km | 2.2% | 5.8 s |
| nearest_root | 5,510 km | 3,759 km | 31.8% | 9.4 s |

**Decision.** `improve()` applies key-path exchange and the local key-vertex elimination until neither saves anything or `time_limit` passes; a move must save a billionth of the weight it replaces. Key paths are listed once per pass, and a path whose end an earlier move left only continuing it is extended to the next key vertex first. Key-vertex insertion is deferred.

**Consequences and limits.** All three first methods end within 0.1% of each other, so after `improve()` the first method matters little. On Kigali's roads improved plans were within 0.86% of the optimum (decision 24); on SteinLib's dense random graphs they were up to 9% above it, where insertion would help.

### Decision 23: the lower bound by Wong's dual ascent, following Wong's rule

**Question.** Without the optimum, a plan cannot say how good it is. How can every plan state how far above the optimum it can be at most?

**Options.** The linear relaxation of a Steiner tree model, solved by an LP solver; and Wong's dual ascent on the directed form of the network, which approximates the directed cut relaxation from below with only shortest-path-like searches. For the dual ascent, which terminal grows next decides the bound: one whole turn per terminal until it reaches the root, nearest first, or Wong's rule, one step at a time for the terminal whose region has the fewest entering arcs.

**Evidence.** The first version gave whole turns, nearest the root first at the current reduced costs. It was valid, but against SteinLib's proven optima it was on average 7.0% below the optimum on set B and 9.7% on set C, at worst 45%. Wong's rule was 0.05% and 0.42% below, at worst 3.1%. On Rwanda the improved plans' gaps fell from 5.5%, 10.9%, and 12.0% to 2.1%, 1.4%, and 0.6%, for 100, 1,000, and 10,000 terminals. Recomputing each region every step took 41 s for 100 terminals on Rwanda; keeping them between steps gave the same bounds in 1.7 s, and 40.5 s for 10,000.

**Decision.** `lower_bound()` follows Wong's rule. Each terminal keeps its region and its entering arcs between steps, since reduced costs only fall and regions only grow, and a step checks only those arcs. The planner keeps a complete bound; one stopped by `time_limit` is still valid, only looser.

**Consequences and limits.** A plan's `gap` is an upper limit on its distance from the optimum. On Kigali the bound was 0.07 to 1.1% below the optimum, so most of the gap there is real.

### Decision 24: validation without SCIP-Jack

**Question.** The design planned to validate the planner against SCIP-Jack, an exact Steiner tree solver, whose source code must now be requested from its developers. How else can the heuristics and the bound be checked against exact optima?

**Options.** SteinLib's test sets with their proven optima; an exact model in scipy's `milp` (HiGHS); and the planner's own bound alone.

**Evidence.** SteinLib's sets B, C, D, and E (78 random sparse graphs of 50 to 2,500 nodes): improved plans were on average 0.4 to 2.2% above the optimum, at worst 9%; the bound 0.05 to 0.42% below it, at worst 4.4%. On Kigali's roads with 30 and 60 terminals and 3 roots (10 instances), the multi-commodity flow model solved each instance in 4 to 110 s: improved plans were 0.00 to 0.86% above the optimum, and the bound 0.07 to 1.1% below. The weaker single-commodity flow model did not solve one in 600 s.

**Decision.** `steinlib.py` scores every method and the bound on SteinLib, and `exact.py` on Kigali against HiGHS; both run only in the benchmarks, so neither SteinLib nor an exact model is a dependency. `exact.py` writes its instances, and Rwanda's, in SteinLib's STP format, so SCIP-Jack can be added as a column when its code arrives.

**Consequences and limits.** HiGHS cannot solve Rwanda-scale instances, so there the bound alone gives the gap. SteinLib's graphs are random, not roads.

### Decision 25: the default method

**Question.** Which method should `plan()` and `connecting_subnetwork()` use by default?

**Options.** `steiner`, Mehlhorn's algorithm, the fastest; and `shortest_path`, Takahashi and Matsuyama's.

**Evidence.** Without `improve()`, `shortest_path` was 0.65 to 5.9% above the optimum on Kigali and `steiner` 3.0 to 8.8%; on SteinLib 2.6 to 4.7% and 3.4 to 9.9% on average. On Rwanda, `shortest_path` took 1.1 s for 1,000 terminals and 4.4 s for 10,000, against 0.2 s.

**Decision.** `shortest_path`: about twice as close to the optimum out of the box, still seconds at country scale, and its order and marginal costs come with it. `steiner` stays one argument away for very large inputs, and `improve()` brings either within about 1% of the optimum on roads.

## Known limits and open questions

Differences from osmnx:

- `simplify` keeps the nodes between duplicated parallel ways and one node of a ring without junctions, where osmnx removes them (decision 6).
- pandana's distances can differ from scipy's by up to about 0.5 m on Rwanda (decision 5).
- An uncropped load counts streets on the whole extract, where `graph_from_xml` sets no `street_count` at all; this follows `graph_from_polygon`, which counts them.

Untested or unmeasured:

- Memory for a large country. Rwanda's simplified `drive_service` build peaks at 2.5 GB; I have not built Nigeria (680 MiB) or larger extracts, so the 16 GB target for Nigeria is an estimate.
- Remote data stores. Loading and building through Azure Blob Storage or another remote store is tested for correctness but not timed.
- pyrosm and quackosm were compared by their requirements, not timed.
- Every benchmark ran on one laptop with a fast SSD.

Open questions:

- Whether the disk-backed node index should be the default for every extract size (decision 16).
- `walk` is the slowest network type at 24 s for Rwanda, most of it in `simplify`. A larger country will take proportionally longer.
- How far the planner's plans are from the optimum at country scale: HiGHS solved Kigali-size instances exactly, but on Rwanda only the bound measures the gap (decision 24). SCIP-Jack would give exact optima there.
- On dense random graphs, improved plans were up to 9% above the optimum (decision 22); key-vertex insertion is the next move to add if real data shows such gaps.
- `SpatialNetwork.concat` keeps one edge per `(u, v, key)`. That is right for networks that share no ways, but OSM networks from overlapping areas should be combined like extracts are, and `concat` does not yet do that for networks from the fetcher.

Deferred from the designs: isochrone polygons, centrality measures, and travel speeds; prize-collecting and capacitated connection plans, separate root networks, and a cost per cable; one entry point that chooses the fetcher or the handler by the size of the area; and dropping Python 3.10, which reached its end of life this month and is a decision for the whole library.
