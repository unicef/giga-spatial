# Editing

Every editing method takes many nodes or edges at once and returns a new network. The original stays as it was (see [The data model](data-model.md#a-network-never-changes)).

## Finding nodes by coordinates

`nearest_nodes` returns the node closest to each point, by great-circle distance. It uses a KD-tree on the nodes' positions on the unit sphere, built once per network, so it needs no projection and is accurate everywhere. Distances come back in metres.

```python
nearest = roads.nearest_nodes(schools, return_distance=True)
print(nearest.head(3))
print(nearest["distance"].describe().round(0))
```

```text
      node_id    distance
0  1612911622  148.126799
1  6290263967  244.771110
2  2168896033   78.298345
count    200.0
mean     144.0
std      141.0
min        2.0
25%       50.0
50%       99.0
75%      201.0
max      882.0
Name: distance, dtype: float64
```

`max_distance` leaves points with no node that close without a match (a missing ID). The nearest node is not the nearest road: a school beside a long, straight road can be far from both of its ends. To join points to roads, use [`insert_points`](inserting-points.md), which finds the nearest point on the nearest edge.

## Selecting part of a network

`clip` keeps the edges that intersect an area, and their end nodes. `predicate="within"` keeps only edges entirely inside it. The edges are found with an R-tree on their geometries, built once per network.

```python
from shapely.geometry import box

kigali = box(29.98, -2.05, 30.25, -1.85)
started = time.perf_counter()
kigali_roads = roads.clip(kigali)
print(kigali_roads, f"{time.perf_counter() - started:.2f} s")
print(roads.clip(kigali, predicate="within"))
```

```text
SpatialNetwork(nodes=23,833, edges=61,909, directed=True, weight='length') 0.09 s
SpatialNetwork(nodes=23,630, edges=61,453, directed=True, weight='length')
```

`clip` cuts nothing: an edge that crosses the boundary is kept whole. To cut a road network to an area the way osmnx does, keeping the nodes inside, use `filter_nodes(geometry=area)`; to load roads for an area in the first place, pass the area to `OSMRoadsHandler.load_data`, which also keeps the largest connected part.

`filter_edges` keeps the edges you select and every node, so some nodes may be left without edges. Here, the main roads:

```python
highway = kigali_roads.edges["highway"].astype(str)
main_roads = kigali_roads.filter_edges(mask=highway.str.contains("trunk|primary|secondary").to_numpy())
connected = main_roads.filter_nodes(mask=(main_roads.degree() > 0).to_numpy())
print(main_roads)
print(connected)
```

```text
SpatialNetwork(nodes=23,833, edges=5,587, directed=True, weight='length')
SpatialNetwork(nodes=2,996, edges=5,587, directed=True, weight='length')
```

## Attributes and weights

`set_edge_attribute` and `set_node_attribute` set one column, from a single value, an array in table order, or a Series keyed by ID. `update_nodes` and `update_edges` set several columns from a table keyed by ID, and `update_nodes(by="coordinates")` matches rows to their nearest nodes instead.

Custom weights are edge columns like any other. For fibre planning, the cost of laying fibre depends on the road: trenching along a trunk road costs more than along a residential street, and the last stretch from the road to a school has its own price. The rates below are round, illustrative numbers per metre, not real tariffs:

```python
cost_per_metre = {"trunk": 40, "primary": 30, "secondary": 20, "tertiary": 15}  # illustrative


def with_fibre_cost(net, other_roads=10, connections=25):
    """The network with ``rate`` (cost per metre) and ``cost`` (the edge's total) columns."""
    road_class = net.edges["highway"].astype(str).str.split(";").str[0]
    rate = road_class.map(cost_per_metre).fillna(other_roads).to_numpy()
    rate[(net.edges["kind"] == "connection").to_numpy()] = connections
    net = net.set_edge_attribute("rate", rate)
    return net.set_edge_attribute("cost", net.edges["length"].to_numpy() * rate)


costed = with_fibre_cost(kigali_roads)
print(costed.edges[["highway", "length", "rate", "cost"]].head(4).round(1))
```

```text
       highway  length  rate   cost
0  residential    34.6  10.0  346.3
1  residential    23.9  10.0  238.5
2  residential    55.1  10.0  550.7
3  residential    55.1  10.0  550.7
```

Any algorithm can then minimise the new column with `weight="cost"`. Attribute edits keep a column's type when nothing is lost, and text stays categorical. `x`, `y`, `u`, `v`, and `key` cannot be set, because they define the network.

**Rates and totals.** The function stores both a rate per metre and a total per edge, because they behave differently when edges are split or joined. A rate is like a speed limit: when [inserting points](inserting-points.md#lengths-are-kept) splits a road, both pieces keep it, and when [simplification](reduction.md) joins edges, their length-weighted mean is still the right rate, so `rate × length` gives a joined edge's total exactly. A total is added up when edges are joined only if it is the weight of the call or named in `simplify(sum_columns=...)`, and averaged otherwise; when a road is split, it is copied to both pieces, which counts it twice. So compute totals after inserting points, and keep a rate next to any total you will need to add up later.

## Adding nodes and edges

`add_nodes` adds nodes from a points table, without connecting them. `add_weighted_edges` adds edges between existing nodes, like networkx's `add_weighted_edges_from` but with arrays, and continues the keys of edges already between each pair. A planned road between two nodes, for example:

```python
ends = kigali_roads.nodes.index[[0, 1]]
planned = kigali_roads.add_weighted_edges([ends[0]], [ends[1]], [1200.0], kind="added", name="planned road")
print(planned.edges[planned.edges["kind"] == "added"][["u", "v", "key", "length", "kind", "name"]])
```

```text
               u          v  key  length   kind          name
61909  279244409  279244410    1  1200.0  added  planned road
```

A road already joins those two nodes, so the new edge gets key 1. When the weight is not `length`, the edges' lengths are measured from their geometry or ends, so `length` always means metres.

## Directions

`to_undirected` gives an undirected network with one edge per pair of nodes: the two directions of a two-way road, and any parallel edges, become the one with the smallest weight, which keeps its attributes and geometry.

```python
undirected = kigali_roads.to_undirected()
print(kigali_roads)
print(undirected)
```

```text
SpatialNetwork(nodes=23,833, edges=61,909, directed=True, weight='length')
SpatialNetwork(nodes=23,833, edges=31,443, directed=False, weight='length')
```

Algorithms can also ignore directions for one call with `directed=False`, which leaves the network as it is.

## Combining networks

`SpatialNetwork.concat` combines networks into one. Nodes with the same ID are the same node, so two areas fetched separately join where they meet. An edge whose `(u, v, key)` appears in several networks is kept once.

```python
west = roads.clip(box(29.98, -2.05, 30.12, -1.85))
east = roads.clip(box(30.10, -2.05, 30.25, -1.85))
both = SpatialNetwork.concat([west, east])
print(west, east, both, sep="\n")
```

```text
SpatialNetwork(nodes=13,641, edges=34,805, directed=True, weight='length')
SpatialNetwork(nodes=13,626, edges=35,616, directed=True, weight='length')
SpatialNetwork(nodes=23,833, edges=61,909, directed=True, weight='length')
```

The two halves overlap, and together they are exactly the Kigali box clipped earlier. `concat` suits networks that share nodes but not roads, such as neighbouring areas fetched from Overpass. Networks loaded from overlapping Geofabrik extracts are combined by `OSMRoadsHandler` itself, which matches roads by their OpenStreetMap way rather than by key (see [Reduction](reduction.md#how-a-source-is-cut-from-an-extract)).

## Other edits

| Method | What it does |
|---|---|
| `filter_nodes(ids=, mask=, geometry=)` | Keep the selected nodes and the edges between them |
| `remove_nodes(ids=, points=, geometry=)` | Remove nodes and their edges; `points` are matched to their nearest nodes |
| `remove_edges(edges=, mask=)` | Remove edges by `(u, v, key)`, by `(u, v)` (all parallel edges), or by mask; nodes stay |
| `filter_edges(mask=, geometry=)` | Keep the selected edges and every node |
| `set_node_attribute`, `set_edge_attribute` | Set one column |
| `update_nodes`, `update_edges` | Set several columns from a table |
| `add_nodes` | Add unconnected nodes from a points table |
| `drop_geometry()` | Remove edge geometry, to save memory when only the topology matters |
| `nodes_gdf`, `edges_gdf` | The tables as GeoDataFrames, built once and cached; missing edge geometries become straight lines |
| `to_geodataframes(crs=)` | Both GeoDataFrames, projected if asked |
| `degree(directed=)` | Edges per node |
| `total_length(by=)` | Total length in metres, optionally by a column such as `kind` |
