# The data model

A `SpatialNetwork` is two pandas tables and a few network-level fields. Knowing their columns, and the rules they follow, explains most of what the methods do.

## Nodes

The nodes table is indexed by `node_id` and has a longitude `x`, a latitude `y`, and a `kind`. Every other column is an attribute: OpenStreetMap tags, `street_count` (how many streets meet at the node, as osmnx counts them), or the columns of points you insert.

```python
print(roads.nodes[["x", "y", "kind", "street_count", "highway"]].head())
print(roads.nodes["highway"].value_counts().head(3))
```

```text
                   x         y  kind  street_count highway
node_id                                                   
271317577  29.431325 -1.620477  road             3     NaN
279244409  30.086600 -1.953869  road             3     NaN
279244410  30.085752 -1.953824  road             3     NaN
279244411  30.085278 -1.954051  road             3     NaN
279244419  30.085398 -1.956300  road             3     NaN
highway
crossing           11
traffic_signals     9
mini_roundabout     1
Name: count, dtype: int64
```

Node IDs keep their origin: OpenStreetMap's integer IDs for roads, your own IDs (such as `"school-007"`) for inserted points, and negative integers for the nodes the library creates where it splits a road, as OpenStreetMap does for new objects. IDs are unique, and every method takes and returns them, never row positions.

## Edges

The edges table has one row per edge, identified by `(u, v, key)`. `length` is in metres and `kind` says where the edge came from. `geometry` holds the road's shape as a LineString, or is missing when the edge is a straight line between its ends, which saves memory on the many short, straight road segments.

```python
edges = roads.edges
print(edges[["u", "v", "key", "length", "kind", "osmid", "highway", "oneway", "reversed"]].head())
print(f"Edges with no geometry (straight): {edges['geometry'].isna().mean():.0%}")
```

```text
             u            v  key  ...      highway oneway reversed
0   6365469082   6365469315    0  ...    secondary  False    False
1   6365469315   6365469082    0  ...    secondary  False     True
2    279245648  10130088941    0  ...  residential   True    False
3  10130088941    281686871    0  ...  residential   True    False
4   6371510564   6371497657    0  ...      primary  False    False

[5 rows x 9 columns]
Edges with no geometry (straight): 17%
```

`osmid` holds the OpenStreetMap ways an edge was made from: one way for most edges, several where simplification joined roads. Attributes that differ between those ways are joined with `;`, such as `"residential;tertiary"`.

## Directions, two-way roads, and keys

A road network is directed by default, as OpenStreetMap is. A two-way road is two rows, one per direction, with the second marked `reversed`. A one-way road is one row, with `oneway` true.

```python
two_way = edges[~edges["oneway"]].iloc[0]
pair = edges[
    ((edges["u"] == two_way["u"]) & (edges["v"] == two_way["v"]))
    | ((edges["u"] == two_way["v"]) & (edges["v"] == two_way["u"]))
]
print(pair[["u", "v", "key", "length", "oneway", "reversed"]])
print(f"Parallel edges (key above 0): {(edges['key'] > 0).sum():,}")
```

```text
            u           v  key    length  oneway  reversed
0  6365469082  6365469315    0  6.544992   False     False
1  6365469315  6365469082    0  6.544992   False      True
Parallel edges (key above 0): 2,780
```

The `key` tells apart parallel edges between the same two nodes, such as two roads that leave a junction and meet again. It counts from 0 for each `(u, v)` pair, as in osmnx and networkx.

An undirected network, such as one from `from_points` or `to_undirected`, has one row per edge, and the order of `u` and `v` has no meaning. Methods that should ignore one-way restrictions, such as measuring a cable route, take `directed=False` and treat a directed network as undirected for that call.

## Lengths

Every length is the great-circle distance in metres, with the formula and Earth radius osmnx uses for roads:

\[
d = 2R \arcsin \sqrt{\sin^2\frac{\varphi_2 - \varphi_1}{2} + \cos\varphi_1 \cos\varphi_2 \sin^2\frac{\lambda_2 - \lambda_1}{2}}, \qquad R = 6{,}371{,}009 \text{ m}
\]

where \(\varphi\) is latitude and \(\lambda\) longitude, in radians. An edge with a geometry is measured along it, segment by segment. The library uses the same formula for every edge it creates (connections to inserted points, split pieces of roads, links between points), so they are consistent with the roads. The sphere is within about 0.5% of the true ellipsoid, a difference that never decides which road is nearer.

## Weights

Algorithms minimise or add up a weight column, `length` by default. The network's `weight` field names the default, and every algorithm takes `weight=` to use another edge column for one call: a travel time, a cost of laying fibre, or a risk score. Weights must be present and not negative on every edge.

## Kinds and the library's terms

A node's or edge's `kind` records where it came from, so a result can always be traced back. The library's own terms:

| Term | Meaning |
|---|---|
| node kind `road` | A node of the road network: a junction, a dead end, or a point along a road |
| node kind `point` | An inserted point of interest, such as a school |
| node kind `split` | A node the library added where it split a road to connect a point |
| edge kind `road` | A road segment, or a joined chain of them |
| edge kind `split` | A piece of a road that was split at a `split` node |
| edge kind `connection` | The link from an inserted point to its road |
| edge kind `point_link` | A link between two points: from `from_points`, or from `insert_points(chain=True)` |
| edge kind `component_link` | A link `connect_components` added between disconnected parts |
| edge kind `added` | An edge you added with `add_weighted_edges` |
| insertion | Adding points to a network as nodes, each joined to its nearest road |
| connecting subnetwork | The part of a network that joins a set of points, reduced to those points and the places where the paths branch |
| key | The number that tells apart parallel edges between the same two nodes |
| weight | The edge column algorithms minimise, `length` by default |

## Coordinates and CRS

Nodes are stored in EPSG:4326 (longitude and latitude). Constructors convert other CRSs, and accessors take `crs=` to return projected output. Arrays of coordinates are ordered `(y, x)`, which is `(lat, lon)`, the order the rest of the library uses for point tuples.

```python
print(roads.node_coordinates(roads.node_index[:2]))
print(roads.bounds)
```

```text
[[-1.6204771 29.431325 ]
 [-1.9538695 30.0866001]]
(28.8611577, -2.866518, 30.8968143, -0.9896697)
```

## Metadata and summaries

`metadata` is a dictionary of provenance: the extract or the fetch parameters, how the network was built, and how it was loaded. It is saved with the network. `info()` summarises everything at once.

```python
summary = roads.info()
print({key: summary[key] for key in ("nodes", "edges", "total_length_m", "edges_with_geometry")})
print(roads.metadata["build"]["network_type"], roads.metadata["load"])
```

```text
{'nodes': 125095, 'edges': 332633, 'total_length_m': 106227700.31817828, 'edges_with_geometry': 274942}
drive_service {'retain_all': False, 'truncate_by_edge': False, 'cropped': False}
```

## A network never changes

Every method that edits returns a new network, and the old one stays as it was. The new network shares every column it did not change with the old one, so an edit costs only what it changes. Because a network never changes, it can cache what it derives from its tables (the adjacency matrix for routing, the spatial indexes for nearest searches, the GeoDataFrames) and never has to work out whether a cache is stale.

The one rule this asks of you: `nodes` and `edges` return copies of the tables that share their columns with the network. Adding a column to them is fine, but do not change their values in place, because another network may share the column. To change values, use the editing methods on the next page.
