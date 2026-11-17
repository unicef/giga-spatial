# Networks between two point sets are built with `from_point_sets`, tagged with `bipartite`

`SpatialNetwork.from_point_sets(origins, targets, ...)` builds an undirected network across two sets of points, such as schools and clinics or schools and fibre nodes, using the same unit-sphere KD-tree and haversine lengths as `from_points` and `connect_points`. Each origin links to its `k` nearest targets (`connect="knn"`, `k=1` by default) or to every target within `max_distance` metres (`connect="distance"`), and all points from both sets stay as nodes even when `max_distance` leaves them without edges. `within` (`"none"`, `"origins"`, `"targets"`, or `"both"`, with optional `within_k` and `within_max_distance` overrides) adds links inside either set: `within="origins"` lets `ConnectionPlanner` route schools through nearby schools to reach a fibre root instead of forcing every school onto a direct link to its nearest root.

Every node keeps `kind="point"` and carries `bipartite` (`0` for `origins`, `1` for `targets`) in all `within` modes, and `SpatialNetwork.distance_table` uses `bipartite == 0` and `1` as its default origins and targets when producing a long `["origin", "target", "distance"]` table of finite reachable pairs for `spopt` and `access`. `build_distance_graph` in `gigaspatial.processing.algorithms` is deprecated (to be removed in 0.12) and delegates to `from_points` and `from_point_sets`, and `EntityTable.to_distance_graph` and `to_distance_graph_with` index entities by `e.id` rather than positional row numbers.

## Considered options

- **Chaining `from_points` and `connect_points` instead of a constructor.** Rejected: `within="none"` has no edge-less starting network from `from_points`; `from_points(origins).connect_points(targets)` searches from each target to `k` origins (the reverse KNN direction); and `connect_points` drops isolated points beyond `max_distance`.
- **New `NODE_KINDS` (`"origin"`, `"target"`) instead of a `bipartite` column.** Rejected: `kind` records structural roles (`"road"`, `"point"`, `"split"`) that `simplify` and `connect_points` branch on, whereas `bipartite` (`0` and `1`) is the node attribute `networkx.algorithms.bipartite` expects after `to_networkx()`.
- **Setting `bipartite` only when `within="none"`.** Rejected: keeping `bipartite` in all `within` modes gives one way to select origins and targets (`nodes["bipartite"] == 0`) and lets `distance_table()` find its default sets even when within-set edges are present.
- **Auto-prefixing clashing IDs between `origins` and `targets`.** Rejected: rewriting node IDs behind the caller's back breaks downstream calls such as `ConnectionPlanner(net, terminals=schools.index, roots=fibre.index)` because the network's node IDs no longer match the caller's tables. Colliding IDs raise `ValueError`, as `insert_points`, `connect_points`, and `add_nodes` do, and empty `origins` or `targets` raise `ValueError` pointing to `from_points`.

## Consequences

- When `within != "none"`, the network has intra-set edges and is not mathematically bipartite (`nx.is_bipartite(G)` is `False`), even though nodes carry `bipartite` (`0` and `1`) to identify their set.
- `EntityTable.to_distance_graph_with` no longer silently merges row `0` of both tables into a single graph node; if two tables share an entity `id`, `from_point_sets` raises `ValueError`.
