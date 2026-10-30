# Road networks

This guide fetches the OpenStreetMap roads around a set of schools, joins the schools to the roads, and finds the roads that connect them to a fiber node. The road network is a `SpatialNetwork`: two tables, one of nodes and one of edges, with lengths in meters.

`OSMRoadsFetcher` is a fetcher. It queries the Overpass API through osmnx and returns the network; it does not save it. You save the network yourself, into any data store. For whole countries and regions, `OSMRoadsHandler` builds the same network from Geofabrik's extracts instead (see [Whole countries](#whole-countries)). Install the `roads` extra first:

```console
pip install "giga-spatial[roads]"
```

## Fetch the roads

```python
import pandas as pd
from gigaspatial import OSMRoadsFetcher

schools = pd.DataFrame(
    {
        "giga_id": ["school-1", "school-2", "school-3"],
        "lat": [-1.9441, -1.9536, -1.9580],
        "lon": [30.0619, 30.0588, 30.0750],
    }
)
roads = OSMRoadsFetcher(network_type="drive_service", buffer=2000).fetch(schools)
print(roads)
```

The source can be points, a frame of points or polygons, a shapely geometry, or a country. Points become their convex hull, widened by `buffer` meters. Set `area="bbox"` for the bounding box, or `area="buffers"` for a circle around each point. `fetcher.query_area(source)` returns the area before anything is downloaded.

`network_type` is an osmnx network type, such as `"drive"`, `"drive_service"`, `"walk"`, or `"all"`. `custom_filter` takes an Overpass filter instead. The network is directed, as OSM is: a two-way road is two edges, one per direction.

### Which areas suit Overpass

Overpass is a shared public service. It handles a city, a district, or the area around a group of schools well, usually in seconds to minutes.

- osmnx splits an area larger than 2,500 km² into several queries. The fetcher warns above 50,000 km². A whole country can take hours and run into Overpass's fair-use limits; use `OSMRoadsHandler` for it (see [Whole countries](#whole-countries)).
- osmnx caches every response, in a `cache` folder in the current directory. When a large fetch fails part-way, run it again: the queries that already succeeded come from the cache. Set `osmnx.settings.cache_folder` to keep the cache elsewhere, or `osmnx.settings.use_cache = False` to turn it off.
- Failed requests are retried `max_retries` times, waiting `retry_delay` seconds and doubling the wait each time. When the main Overpass server is down, point the fetcher at a mirror, either for one fetcher or for every fetch through the `OVERPASS_URL` setting (see [Configuration](../getting-started/configuration.md)):

```python
roads = OSMRoadsFetcher(overpass_url="https://overpass.kumi.systems/api/interpreter").fetch(schools)
```

## Whole countries

`OSMRoadsHandler` downloads Geofabrik's OpenStreetMap extracts, one `.osm.pbf` file per country or region, stores them, and builds the road network from them with pyosmium. The result is the network `OSMRoadsFetcher` returns for the same area, with the same nodes, edges, and attributes, but a country takes seconds to minutes instead of hours, and works offline once stored.

```python
from gigaspatial import OSMRoadsHandler

handler = OSMRoadsHandler()
rwanda = handler.load_data("RWA")
around_schools = handler.load_data(schools, buffer=2000)
```

The first call downloads Rwanda's extract (65 MiB), builds its network (about 15 s), and saves both. Later calls read the saved network: the whole country in under a second, the area around a set of schools in about a second. The options are the fetcher's, plus those that choose the extract:

- `network_type`, `simplify`, `retain_all`, `truncate_by_edge`, `buffer`, and `area` work as in the fetcher. `tags={"highway": ["primary", "secondary"]}` replaces `network_type` with the allowed values of each OSM key.
- A country uses its own extract, whole. `regions=True` uses its regional extracts instead, such as Brazil's five, which are smaller downloads. A country without an extract of its own, such as Singapore inside `malaysia-singapore-brunei`, is cut from the extract that holds it along its boundary.
- Any other source uses the smallest extracts that cover its area, such as both Rwanda and Uganda for schools along their border, and is cropped to that area as the fetcher would. `extracts=["africa"]` names extracts yourself; continents are used only when named.
- `crop_to_source=False` returns whole extracts for any source.

### Storage and freshness

Extracts are stored by date, in Geofabrik's own folders, under `base_path` (by default `osm/geofabrik` in the bronze data folder). Each extract's networks are saved beside it, one folder per network type and simplification:

```text
osm/geofabrik/index-v1.json
osm/geofabrik/2026-10-28/africa/rwanda.osm.pbf
osm/geofabrik/2026-10-28/africa/rwanda.networks/drive_service-raw/
osm/geofabrik/2026-10-28/africa/rwanda.networks/drive_service-simplified/
osm/geofabrik/2026-10-28/south-america/brazil/nordeste.osm.pbf
```

A stored network always covers its whole extract; cropping happens when loading. Pass `data_store=` to store everything in a remote data store, such as Azure Blob Storage.

The newest stored date is used, without contacting Geofabrik. `refresh=True` uses Geofabrik's latest extract instead, and `max_age_days=30` does when the stored one is older than that. `date="2026-10-01"` pins a date, stored or downloaded. Geofabrik updates its extracts daily and keeps the daily files for about a week, then those of the 1st of each month and of January 1st; to reproduce an old result, keep its stored extract. Old dates are never deleted for you. Extracts take space: Nigeria is 680 MiB, Brazil 2.0 GiB, and Germany 4.6 GiB.

A network's metadata records its extract's ID, date, URL, and the time its data was taken from OpenStreetMap:

```python
rwanda.metadata["extract"]
# {'id': 'rwanda', 'date': '2026-10-28', 'url': 'https://download.geofabrik.de/africa/rwanda-261028.osm.pbf', ...}
```

Building reads the extract once, with a node location index that pyosmium keeps in memory for extracts up to 1 GiB (about 2 GB of memory for Nigeria) and on disk above that; `node_index=` chooses it yourself. A saved network folder, or a stored `.pbf` file, can also be passed to `load_data` as the source.

## Join the schools to the roads

```python
fiber = pd.DataFrame({"fiber_id": ["fiber-1"], "lat": [-1.9490], "lon": [30.0700]})

network = roads.insert_points(schools, id_column="giga_id", max_distance=1000)
network = network.insert_points(fiber, id_column="fiber_id", max_distance=1000)
```

Each point becomes a node, joined to its nearest road at the foot of the perpendicular. The road is split there at a new node of kind `split`, and a `connection` edge joins the point to it. A point within `snap_to_node` meters (5 by default) of a road's end connects to that end instead. The split pieces add up exactly to the road's length. Points farther than `max_distance` from any road are skipped, with a warning; `return_skipped=True` also returns them.

`chain=True` joins a point to an already inserted point when that point is closer than any road, nearest the road first. The frame's other columns become node attributes. Every edit returns a new network; the original does not change.

## Shortest paths

```python
school_ids = schools["giga_id"].tolist()

lengths = network.shortest_path_lengths(school_ids, ["fiber-1"], directed=False)
nearest = network.nearest_by_network(school_ids, ["fiber-1"], directed=False)
routes = network.shortest_paths(school_ids, ["fiber-1"] * 3, geometry=True, directed=False)
```

`lengths` is a table of meters, one row per school. `routes` is a GeoDataFrame with one line per route. A directed network follows one-way roads; `directed=False` ignores them, which suits a cable laid along a road. `reachable` lists every node within a distance, and `count_within` counts targets within a distance, such as clinics within 5 km by road.

For very many origin-destination pairs, `backend="pandana"` is faster once its index is built (`pip install "giga-spatial[pandana]"`).

## The roads that connect the schools

```python
to_fiber = network.connecting_subnetwork(school_ids, roots=["fiber-1"])
print(to_fiber.total_length())

tree = network.connecting_subnetwork(school_ids)
branches = tree.degree()[tree.degree() >= 3]
```

A connecting subnetwork keeps only the roads that join the points, contracted to the points and the nodes where the paths branch. With `roots`, nodes already connected to each other such as fibre nodes, the schools are joined to the roots at close to the smallest total length: a school may connect through another school, and a road shared by several schools counts once. Without roots, the schools are joined by a tree of close to the smallest total length. Both are Steiner tree approximations. `method="nearest_root"` gives each school its own shortest path to its nearest root instead, and `method="all_pairs"` the shortest path between every pair; its cost grows with the square of the number of points.

`tree.edges_gdf` is the result as lines, ready to map or export.

`ConnectionPlanner` makes the same plans and says more about them: each school's status (connected, itself a root, or unreachable), the root it reaches and its distance along the plan, the plan's uncontracted edges, and its cost.

```python
from gigaspatial import ConnectionPlanner

plan = ConnectionPlanner(network, school_ids, roots=["fiber-1"], weight="length").plan()
print(plan.cost, plan.terminals["status"].value_counts())
```

`plan("shortest_path")` models a phased rollout: schools are connected one at a time, nearest first, to everything connected so far, and the terminals table gains each school's `order` and `marginal_cost`, the cost its connection added.

`planner.improve(plan)` shortens any plan by local search, and never makes it worse; `time_limit=` stops it after that many seconds. On Rwanda's roads it shortens the Steiner tree for 1,000 schools by about 5%.

Schools on roads that do not join the rest, such as an island or a clipped corner, are unreachable. `bridge_cost` lets the plan use straight-line links between such parts, such as microwave links, at a weight per metre and up to `bridge_max_distance` metres (5,000 by default). A link is used only where it is cheaper than the roads:

```python
plan = ConnectionPlanner(network, school_ids, roots=["fiber-1"], bridge_cost=3.0).plan()
links = plan.edges[plan.edges["kind"] == "bridge"]
```

## Save and load

```python
from gigaspatial import SpatialNetwork

network.save("roads/kigali")
network = SpatialNetwork.load("roads/kigali")
topology = SpatialNetwork.load("roads/kigali", columns=[])
```

A saved network is a folder of three files: `nodes.parquet`, `edges.parquet` as GeoParquet, and `network.json`. GIS tools, DuckDB, and pandas read the Parquet files directly. Pass `data_store=` to save to or load from a remote data store. `columns=` loads only some edge columns: with an empty list, only the topology and lengths, which is fastest.

## More

- `simplify()` removes nodes that only continue a road, as osmnx does. `clip(area)` keeps the roads in an area.
- `connected_components()`, `largest_component()`, and `connect_components()` find and join the disconnected parts.
- `nodes_gdf` and `edges_gdf` are GeoDataFrames; `to_networkx()` and `to_pandana()` convert the network for other libraries.

The [spatial network API](../api/network.md) lists every method.
