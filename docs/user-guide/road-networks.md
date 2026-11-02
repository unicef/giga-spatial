# Road networks

This guide fetches the OpenStreetMap roads around a set of schools, joins the schools to the roads, and finds the roads that connect them to a fiber node. The road network is a `SpatialNetwork`: two tables, one of nodes and one of edges, with lengths in meters.

`OSMRoadsFetcher` is a fetcher. It queries the Overpass API through osmnx and returns the network; it does not save it. You save the network yourself, into any data store. Install the `roads` extra first:

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

- osmnx splits an area larger than 2,500 km² into several queries. The fetcher warns above 50,000 km². A whole country can take hours and run into Overpass's fair-use limits; fetch its regions and combine them with `SpatialNetwork.concat`, which joins the shared boundary nodes.
- osmnx caches every response, in a `cache` folder in the current directory. When a large fetch fails part-way, run it again: the queries that already succeeded come from the cache. Set `osmnx.settings.cache_folder` to keep the cache elsewhere, or `osmnx.settings.use_cache = False` to turn it off.
- Failed requests are retried `max_retries` times, waiting `retry_delay` seconds and doubling the wait each time. When the main Overpass server is down, point the fetcher at a mirror, either for one fetcher or for every fetch through the `OVERPASS_URL` setting (see [Configuration](../getting-started/configuration.md)):

```python
roads = OSMRoadsFetcher(overpass_url="https://overpass.kumi.systems/api/interpreter").fetch(schools)
```

Country-scale road networks from Geofabrik's OpenStreetMap extracts are planned as a separate handler.

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
to_fiber = network.connecting_subnetwork(school_ids, sources=["fiber-1"])
print(to_fiber.total_length())

tree = network.connecting_subnetwork(school_ids)
branches = tree.degree()[tree.degree() >= 3]
```

A connecting subnetwork keeps only the roads that join the points, contracted to the points and the nodes where the paths branch. With `sources`, each school takes its shortest path to its nearest source. Without them, the schools are joined by a tree of close to the smallest total length (a Steiner tree approximation). `method="all_pairs"` keeps the shortest path between every pair instead; its cost grows with the square of the number of points.

`tree.edges_gdf` is the result as lines, ready to map or export.

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
