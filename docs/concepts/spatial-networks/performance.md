# Performance and pitfalls

## What things cost

The costs below are for Rwanda's simplified road network (125,058 nodes, 332,535 edges) on one laptop, from the library's benchmarks. They show how costs compare and grow; your machine will differ.

**Paid once per network.** These are built on first use and cached, so the second call is free. Since a network never changes, a cache never goes stale; but every edit makes a new network with empty caches.

| Built on first use | Cost | Used by |
|---|---|---|
| Adjacency matrix, per weight and direction | 0.06 s | every routing method |
| KD-tree on nodes | 0.03 s | `nearest_nodes`, `update_nodes(by="coordinates")`, `remove_nodes(points=)` |
| R-tree on edges | 0.10 s | `clip`, `filter_edges(geometry=)` |
| Projected index of roads | 1.3 s | `insert_points` |
| pandana network | 1.4 s | `backend="pandana"` |
| `edges_gdf` | 0.2 s | maps and exports |

So insert all your points in one call where you can, rather than in many calls on successive networks: each new network builds its own road index. The example inserts schools, fibre nodes, and clinics in three calls for clarity; one call with all three would be faster.

**Per call.**

| Operation | Cost |
|---|---|
| Load the saved network; the topology only | 0.47 s; 0.02 s |
| `insert_points`, 1,000 points; 10,000 points | 0.6 s; 1.7 s |
| `shortest_path_lengths`, 100 origins by 1,000 destinations | 2.6 s, about 25 ms per origin |
| `reachable`, 100 origins within 5 km | 0.08 s |
| `nearest_by_network`, 10,000 origins to 100 targets | 0.05 s |
| `count_within`, 1,000 origins to 100 targets within 5 km | 0.04 s |
| `ConnectionPlanner.plan`, 1,000 schools and 20 fibre nodes: `shortest_path` (the default); `steiner` | 1.1 s; 0.2 s |
| `improve`, the same plan: one pass of local search; until it stops | about 6 s; about 25 s |
| `lower_bound`, the same schools | 5 s |
| `simplify` of the unsimplified network (4.5 million edges) | 7 s |
| Build Rwanda from its extract, then load it from storage | about 14 s, then 0.4 s |

**How costs grow.** Shortest path lengths grow with the number of origins, since each origin is one Dijkstra search over the whole network; destinations are almost free. Put the smaller set first, or swap origins and destinations on an undirected network. `nearest_by_network` with `k=1` and `count_within` do not grow with the number of origins in the same way, because they search from the targets. A plan by `steiner` costs about the same for 100 or 1,000 points, and one by `shortest_path` grows slowly with them: 4.4 s for 10,000 schools. `method="all_pairs"` grows with the square. `improve` and `lower_bound` take `time_limit=`.

**Memory.** Rwanda's simplified network takes about 350 MB in memory with its tables and geometries. A shortest path matrix of 10,000 origins by 10,000 destinations is 800 MB of floats, and the library refuses matrices over 1 GiB, with advice to batch. `drop_geometry()` saves most of a network's memory when only routing matters, and `load(columns=[])` loads only the topology.

## Pitfalls

**Coordinate order.** Point tuples are `(lat, lon)`, and coordinate arrays from `node_coordinates` are `(y, x)`, the order the rest of the library uses. Tables use `x` for longitude and `y` for latitude, or `lat` and `lon` columns. Swapping them puts Rwanda's schools in the Indian Ocean, and `insert_points` will skip them all as too far from any road.

**Directed or not.** A road network is directed, and its distances follow one-way streets. For walking, cycling against the flow, or a cable along a road, pass `directed=False`, or the distances will be longer than they should be and some pairs unreachable.

**Unreachable pairs are `inf`.** They are not errors and not missing values. Count them before you average distances, and decide whether they mean "too far" or "missing road".

**Mixed node IDs.** Inserting points with text IDs into a road network with integer IDs gives a network with both. Everything works with that, including saving and `backend="pandana"`, except `to_pandana()`, which hands you pandana's network indexed by node ID and needs one kind. If you need pandana's own functions, give your points integer IDs (negative ones are taken by split nodes; use large positive numbers).

**Totals on split and joined edges.** Only the weight column is shared between the pieces of a split road and added up when edges are joined. A total such as a cost per edge is copied to both pieces when a road is split. Keep a rate per metre beside it and compute totals after inserting points (see [Editing](editing.md#attributes-and-weights)).

**Do not change the tables in place.** `network.nodes` and `network.edges` share their columns with the network. Assigning to a cell of them can change other networks that share the column. Use `set_node_attribute`, `update_edges`, and the other editing methods.

**`concat` and overlapping roads.** `SpatialNetwork.concat` keeps one edge per `(u, v, key)`. Two networks fetched separately from Overpass for overlapping areas can number parallel edges differently, so the same road can appear twice. Load overlapping areas through `OSMRoadsHandler`, which combines extracts by OpenStreetMap way, or fetch one area that covers both.

**Simplifying after inserting points.** It is safe: `simplify` never removes points or split nodes. But a road between two inserted schools stays split at both, so the simplified network has more edges than before the schools were inserted.

**Data, not just code.** Every distance is only as good as OpenStreetMap's roads. A missing bridge makes a school look far; an unmapped track makes it look unreachable. `connected_components`, the skipped points of `insert_points`, and the schools with the largest detours are the places to look first.
