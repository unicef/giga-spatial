# Getting a network

There are three ways to get a `SpatialNetwork`: OpenStreetMap roads through `OSMRoadsHandler` or `OSMRoadsFetcher`, your own node and edge tables, or points joined to their neighbours. This page shows each, and sets up the example the rest of the note uses.

## Roads for a whole country: `OSMRoadsHandler`

`OSMRoadsHandler` downloads Geofabrik's OpenStreetMap extract of a country, stores it, and builds the road network from it. It needs the `roads` extra (`pip install "giga-spatial[roads]"`).

```python
import time

from gigaspatial import OSMRoadsHandler

handler = OSMRoadsHandler()

started = time.perf_counter()
roads = handler.load_data("RWA")
print(roads)
print(f"First load, with the download and the build: {time.perf_counter() - started:.1f} s")

started = time.perf_counter()
roads = handler.load_data("RWA")
print(f"Later loads, from the saved network: {time.perf_counter() - started:.2f} s")
print(roads.metadata["extract"])
```

```text
SpatialNetwork(nodes=125,095, edges=332,633, directed=True, weight='length')
First load, with the download and the build: 24.0 s
Later loads, from the saved network: 0.62 s
{'id': 'rwanda', 'date': '2026-10-28', 'url': 'https://download.geofabrik.de/africa/rwanda-261028.osm.pbf', 'path': 'africa/rwanda', 'replication_timestamp': '2026-10-28T21:21:02Z'}
```

The first call downloads Rwanda's extract (65 MiB), builds the network, and saves it beside the extract. Every later call reads the saved network. The metadata records which extract the network came from, so a result can always be traced to its data.

`load_data` takes any source: a country, a geometry, a GeoDataFrame, a DataFrame of points, or a list of `(lat, lon)` pairs. For anything but a country with its own extract, the network is cropped to the source's area, as the fetcher would crop it (see [Reduction](reduction.md) for how). The options you are most likely to need:

| Option | Default | Effect |
|---|---|---|
| `network_type` | `"drive_service"` | osmnx's network types: `"drive"`, `"drive_service"`, `"walk"`, `"bike"`, `"all_public"`, `"all"` |
| `tags` | none | Keep only these values of these OSM keys, such as `{"highway": ["primary", "secondary"]}`, instead of a network type |
| `simplify` | `True` | Join the edges between junctions (see [Reduction](reduction.md)) |
| `buffer`, `area` | `2000`, `"hull"` | How points become an area: their convex hull, bounding box, or buffers, widened by `buffer` metres |
| `retain_all` | `False` | Keep every disconnected part, not only the largest |
| `regions` | `False` | Use a country's regional extracts, such as Brazil's five |
| `date`, `refresh`, `max_age_days` | | Which extract date to use: a pinned date, Geofabrik's latest, or the latest when the stored one is too old |

## Roads for a small area: `OSMRoadsFetcher`

For a city, a district, or the area around a few schools, `OSMRoadsFetcher` queries the Overpass API through osmnx and downloads only that area. It saves nothing; the network is yours to keep or discard.

```python
from shapely.geometry import box

from gigaspatial import OSMRoadsFetcher

central_kigali = box(30.055, -1.960, 30.075, -1.940)
fetched = OSMRoadsFetcher(network_type="drive_service").fetch(central_kigali)
from_extract = handler.load_data(central_kigali)
print("Overpass:", fetched)
print("Extract: ", from_extract)
```

```text
Overpass: SpatialNetwork(nodes=478, edges=1,074, directed=True, weight='length')
Extract:  SpatialNetwork(nodes=478, edges=1,074, directed=True, weight='length')
```

The two give the same network for the same area, built by the same rules, apart from edits made to OpenStreetMap between the extract's date and the moment of the query. Which to use depends on the area:

| | `OSMRoadsFetcher` | `OSMRoadsHandler` |
|---|---|---|
| Suits | a city, a district, the area around a set of schools | regions and whole countries |
| Downloads | only the area, from Overpass | the whole country's extract, once |
| Time for Rwanda | hours, if Overpass allows it | about 25 s the first time, then under a second |
| Offline after the first run | no | yes |
| Data date | the moment of the query | the extract's date, recorded in the metadata |

## Your own tables

Any network with coordinates can be a `SpatialNetwork`: water pipes, power lines, bus routes, or roads from another source. Pass a nodes table with `x` (longitude) and `y` (latitude) and an edges table with `u` and `v`. Lengths are computed when the edges do not have them, along an edge's geometry if it has one and as a straight line between its ends otherwise.

```python
import pandas as pd

from gigaspatial import SpatialNetwork

nodes = pd.DataFrame(
    {"node_id": ["pump", "tank", "school"], "x": [30.060, 30.070, 30.071], "y": [-1.950, -1.950, -1.942]}
)
edges = pd.DataFrame({"u": ["pump", "tank"], "v": ["tank", "school"], "material": ["PVC", "steel"]})
pipes = SpatialNetwork(nodes, edges, directed=False)
print(pipes.edges)
```

```text
      u       v  key       length  kind material
0  pump    tank    0  1111.306909  road      PVC
1  tank  school    0   896.475473  road    steel
```

The constructor checks the tables: node IDs must be unique, every edge must reference existing nodes, and `(u, v, key)` must not repeat. A missing `key` is numbered for you. Coordinates in another CRS are converted with `crs=`.

## The example

From here on, the note works with Rwanda's roads and three sets of made-up points around Kigali: 200 schools, 5 fibre nodes, and 10 clinics. The seed makes them the same for every reader.

```python
import numpy as np

rng = np.random.default_rng(7)


def points_around_kigali(prefix, count):
    return pd.DataFrame(
        {
            f"{prefix}_id": [f"{prefix}-{i:03d}" for i in range(count)],
            "lat": rng.uniform(-2.02, -1.90, count),
            "lon": rng.uniform(30.00, 30.18, count),
        }
    )


schools = points_around_kigali("school", 200).assign(students=rng.integers(80, 1200, 200))
fibre_nodes = points_around_kigali("fibre", 5)
clinics = points_around_kigali("clinic", 10)
print(schools.head(3))
```

```text
    school_id       lat        lon  students
0  school-000 -1.944989  30.146086      1167
1  school-001 -1.912334  30.118445       448
2  school-002 -1.926918  30.109935      1025
```

## Points joined to their neighbours: `from_points`

`SpatialNetwork.from_points` builds an undirected network whose nodes are the points, joined to their `k` nearest neighbours (`connect="knn"`) or to every point within `max_distance` metres (`connect="distance"`). Neighbours come from a KD-tree on the sphere, so no pair of points is compared that does not need to be, and the cost grows with the number of points, not its square.

```python
school_links = SpatialNetwork.from_points(schools, k=4, id_column="school_id")
print(school_links)
print(school_links.edges["length"].describe().round(0))
```

```text
SpatialNetwork(nodes=200, edges=488, directed=False, weight='length')
count     488.0
mean     1089.0
std       468.0
min        89.0
25%       737.0
50%      1095.0
75%      1434.0
max      2512.0
Name: length, dtype: float64
```

**Use cases.** A network of point-to-point links is the model for wireless or microwave links between schools, where the question is which towers to connect rather than which roads to follow. Its minimum spanning tree (see [Algorithms](algorithms.md)) is the shortest set of links that connects every school. `weight_func` can give the links another weight, such as a cost that grows faster than distance.

## networkx graphs

`SpatialNetwork.from_networkx` takes any networkx graph whose nodes have `x` and `y`, which includes every osmnx graph. Use it to bring existing work into the library; `to_networkx` goes back (see [Storage and interoperability](storage.md)).

```python
graph = fetched.to_networkx()
print(type(graph).__name__, graph.number_of_nodes(), graph.number_of_edges())
print(SpatialNetwork.from_networkx(graph))
```

```text
MultiDiGraph 478 1074
SpatialNetwork(nodes=478, edges=1,074, directed=True, weight='length')
```
