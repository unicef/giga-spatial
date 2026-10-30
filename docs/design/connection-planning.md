# Design: connection planning

This is the agreed design for `ConnectionPlanner`, which plans how to connect a set of points (schools, clinics, villages) to an existing network (fibre, a power grid, water mains) along a spatial network at the smallest shared cost. It records the decisions from the design round of 30 October 2026 and splits the work into sub-tasks. Each sub-task starts by reading this document, the glossary, [ADR 0004](../adr/0004-columnar-spatial-network.md), [ADR 0008](../adr/0008-connection-planning.md), and the [spatial network design](spatial-network.md).

This page is not in the docs site navigation. The user-facing description will be in the [spatial networks concept note](../concepts/spatial-networks/reduction.md), and the evidence in [design and algorithm decisions](spatial-networks-decisions.md).

## 1. The problem

Given a spatial network with a non-negative weight on every edge, a set of **terminals** that must be connected, and a set of **roots** that are already connected to each other through an existing network, find the set of edges of smallest total weight such that every terminal can reach some root. A terminal may reach a root directly or through other terminals, and an edge used by several terminals counts once.

Joining every root to one virtual root at weight 0 makes this the Steiner tree problem in graphs over the terminals and the root ([ADR 0008](../adr/0008-connection-planning.md)). Removing the root from the tree leaves a forest whose every tree contains at least one root.

- **Directions are ignored.** A cable or a duct does not follow one-way streets.
- **The cost is per edge used, paid once.** That is trench length, or a cost per metre of trench through `weight=`. A cost per cable, which grows with the number of terminals behind an edge, is out of scope: it makes the problem fixed-charge network design.
- **The roots are one backbone.** Reaching any root is enough. Separate networks, each terminal assigned to one of them, are deferred.
- **No roots** gives an unrooted Steiner tree joining the terminals, as `connecting_subnetwork(terminals)` does today.

Complexity: NP-hard, and NP-hard to approximate within 96/95. The best known approximation is ln 4 + ε ≈ 1.39 (Byrka et al., 2010), through large linear programs. Mehlhorn's algorithm gives at most 2(1 − 1/ℓ) times the optimum for ℓ leaves in O(m + n log n), and is within a few percent in practice. SCIP-Jack solves instances of a few hundred thousand edges exactly, typically in minutes.

On the concept note's example (197 schools and 5 fibre nodes in Kigali, on Rwanda's roads):

| Method | Length | Illustrative cost |
|---|---|---|
| Each school to its nearest fibre node (today's `sources=`) | 355 km | 5.52 million |
| One unrooted tree over schools and fibre nodes | 265 km | 3.92 million |
| Rooted tree, fibre nodes as one root, by length | 258 km | 3.83 million |
| Rooted tree, by cost | 269 km | 3.33 million |

The rooted tree was computed with today's code by joining the fibre nodes with zero-length edges; it took 0.19 s.

## 2. Names and modules

| Item | Decision |
|---|---|
| Classes | `ConnectionPlanner` and `ConnectionPlan` in `gigaspatial/processing/network/planning.py` |
| Public names | `from gigaspatial import ConnectionPlanner`; `gigaspatial.processing.ConnectionPlanner`, `ConnectionPlan` (ADR 0003) |
| Terms | terminals, roots, bridges, connection plan. Glossary entries for **Connection planner**, **Connection plan**, and **Root** beside **Connecting subnetwork** |
| Steiner routines | `_steiner_edges`, `_nearest_source_edges` (renamed `_nearest_root_edges`), `_all_pairs_edges`, `_forest_edges`, `_spanning_tree`, and `_prune_leaves` move from `reduction.py` to `planning.py`; `reduction.py` keeps `simplify` |
| Dependencies | none new: numpy, scipy, pandas, shapely |

The docs use fibre as the main example, but the names and docstrings stay generic, so the planner reads as naturally for a power grid or water mains.

## 3. API

```python
planner = ConnectionPlanner(
    network,                    # a SpatialNetwork, with terminals and roots already inserted
    terminals,                  # node IDs to connect
    roots=(),                   # node IDs already connected to each other
    weight=None,                # edge column to minimise; the network's weight otherwise
    bridge_cost=None,           # weight per metre of a straight-line link; no bridges when None
    bridge_max_distance=5000,   # metres
)
plan = planner.plan(method="steiner")        # or "shortest_path", "nearest_root", "all_pairs"
plan = planner.improve(plan, time_limit=None)
bound = planner.lower_bound(time_limit=60)   # also sets plan.bound and plan.gap when given a plan
```

`SpatialNetwork.connecting_subnetwork(terminals, roots=None, method="steiner", weight=None)` builds a planner and returns `planner.plan(method).network`. Its `sources=` parameter becomes `roots=`, and `method="nearest_source"` becomes `"nearest_root"`; neither is released yet.

| `method` | Meaning |
|---|---|
| `"steiner"` (default) | Mehlhorn's 2-approximation. With roots, the rooted tree (section 4); without, the unrooted tree joining the terminals |
| `"shortest_path"` | Takahashi and Matsuyama's heuristic: connect terminals one at a time, cheapest first, to everything connected so far. Gives the rollout order (section 5) |
| `"nearest_root"` | Each terminal's shortest path to its nearest root; the union of those paths. Today's behaviour of `sources=` |
| `"all_pairs"` | The shortest path between every terminal and every terminal or root, as the old Giga code did; grows with the square of the terminals |

Inputs are node IDs of the network. Inserting points is the user's step (`insert_points`, then plan), so insertion's options and its skipped points stay visible. A root that is also a terminal is connected from the start. Unknown IDs raise `KeyError`. Weights must be present on every edge and not negative; zero is allowed.

### `ConnectionPlan`

| Field | Meaning |
|---|---|
| `network` | The connecting subnetwork, contracted with `simplify(keep=terminals and roots)`, as `connecting_subnetwork` returns today. Undirected |
| `edges` | The chosen edges of the original network, uncontracted, one row per undirected edge with its attributes and geometry; bridges included with `kind="bridge"` |
| `terminals` | One row per terminal: `status` (`connected`, `root`, or `unreachable`), `root` (a root its tree reaches), `distance` (weight along the plan to the nearest root in its tree), `bridged` (whether its path uses a bridge), and for `shortest_path` `order` and `marginal_cost` |
| `cost` | The total weight of `edges` |
| `bound`, `gap` | Filled by `lower_bound`: the bound, and `cost / bound - 1`, so the plan costs at most `1 + gap` times the optimum |
| `metadata` | Method, weight, `bridge_cost`, counts, and the time each step took |

Order and marginal cost come only from `shortest_path`. A Steiner tree's shared trenches have no natural per-terminal marginal cost, and a made-up one would mislead.

## 4. The rooted Steiner tree

The adjacency matrix of the undirected network (`_graph(weight, directed=False)`) gets one extra node, the virtual root, joined to every root node. scipy's minimum spanning tree drops zero entries, so these joins get the smallest positive float, as zero weights already do. Mehlhorn's routine then runs unchanged with terminals ∪ {root}:

1. One multi-source Dijkstra from terminals and root gives every node its nearest terminal. Because the virtual root is joined to every root node at (almost) zero, its Voronoi region is everything nearer to some root node than to any terminal.
2. Edges between regions give candidate links; an MST over the cheapest link per pair of regions picks the pairs.
3. The links are expanded into paths, a second MST removes cycles, and leaves that are neither terminals nor the root are pruned.
4. The root and its joins are removed. What is left is the plan's edges.

The cost is the same as today's Steiner tree: one Dijkstra and two MSTs over a matrix with one more row.

## 5. The rollout order (`method="shortest_path"`)

Takahashi and Matsuyama's heuristic, a 2-approximation, is the planner's model of a phased rollout. Start with the roots as the connected set. Repeatedly connect the unconnected terminal nearest (by weight) to anything connected so far, along its shortest path, and add that path to the connected set. A terminal's `order` is when it was connected, and its `marginal_cost` the weight of its path, which uses only new edges. Ties go to the lower terminal ID, so plans are reproducible. Terminals that the connected set cannot reach are `unreachable`.

Implemented incrementally:

- One multi-source Dijkstra from the connected set gives every node its distance to it.
- After each addition, distances can only fall, and only near the new path. A Dijkstra from the new path's nodes, limited to the largest distance of any unconnected terminal (`limit=` in scipy), updates them. Nodes beyond the limit cannot become any terminal's best connection.
- Terminals are kept in a heap by distance.

Sub-task 3 benchmarks this against the naive version (one full Dijkstra per terminal), and against a version that recomputes only when the nearest terminal's distance changed. A compiled Dijkstra would need a new build dependency and is not planned.

## 6. Bridges

With `bridge_cost=c`, the planner may use straight-line links between the network's connected components, for terminals that no road connects to a root: microwave, wireless, or a planned road.

- Candidates are found with a KD-tree on all nodes, on the unit sphere as `connect_components` does: for each component, its nearest node pairs to up to 5 other components within `bridge_max_distance`.
- Each candidate gets the weight `c` times its great-circle length in metres, in the units of `weight`, and is added to the planner's internal network with `kind="bridge"` and a straight-line geometry.
- Only links between components are offered. Within a component, the roads already connect.
- The Steiner tree uses a bridge only when it is cheaper than the alternatives. A terminal unreachable even with bridges is `unreachable`.

As built:

- A component with more than 2,000 nodes, or most of the network, gets its own KD-tree, queried once with every node outside it, as `connect_components` does. That gives its closest pair to every other component exactly. Smaller components query the KD-tree of all nodes with their nodes' 16 nearest neighbours, doubled while none lies outside the component and the farthest is within `bridge_max_distance`. The closest pair per pair of components is kept, and each component keeps its 5 nearest. On Rwanda's roads cut in two along a meridian (23 components, two of 70,000 and 55,000 nodes), finding the candidates takes 0.1 s.
- Bridges join the planner's graph as extra entries, not as a new `SpatialNetwork`, so offering them costs no copy of the tables. A bridge's `length` stays its great-circle length; its weight is `bridge_cost` times that, stored in the weight column unless the weight is `length`. `cost` and `distance` always count bridges at their weight.
- Bridge ends are kept as nodes when the plan is contracted, so a bridge stays one edge of kind `bridge` in `plan.network`.
- `bridged` is whether a terminal's path to its root along the plan crosses a bridge. Without roots, it is whether the terminal lies outside the road component that holds the most planned terminals.

A network loaded through `OSMRoadsHandler` keeps only its largest connected part by default, so bridges matter for networks loaded with `retain_all=True`, combined with `concat`, or clipped.

## 7. Local search (`improve`)

`improve(plan)` applies two of Uchoa and Werneck's moves until neither improves the plan or `time_limit` passes. Neither ever makes a plan worse.

A **key vertex** is a node of the tree that is a terminal, a root, or a Steiner node of degree 3 or more; a **key path** is a path of the tree between two key vertices whose inner nodes are Steiner nodes of degree 2.

- **Key-path exchange.** Remove one key path; the tree falls into two parts. Find the cheapest path joining the two parts, with a Dijkstra from one part limited to the removed path's weight. If it is cheaper, replace the key path with it.
- **Key-vertex elimination.** Remove one Steiner key vertex and its key paths; the tree falls into three or more parts. A multi-source Dijkstra labelled by part, as in Mehlhorn's algorithm, gives the cheapest links between parts, and an MST over them reconnects all parts. If the reconnection is cheaper than what was removed, keep it.

Moves are tried cheapest-to-evaluate first (key paths in decreasing weight), and each Dijkstra is limited by the weight it must beat, so most searches stop early. Vertex insertion and elimination are deferred, until real data shows a large remaining gap.

## 8. Lower bound (`lower_bound`)

Wong's dual ascent (1984) gives a lower bound on the optimum, and with it the gap of any plan.

- The undirected network becomes a directed one with both directions of every edge, rooted at the virtual root (or at one terminal when there are no roots). Each arc starts with its weight as its reduced cost.
- Repeatedly, pick a terminal that cannot yet be reached from the root along arcs of reduced cost 0, and the set of nodes that reach it along such arcs. Raise the bound by the smallest reduced cost of the arcs entering that set, and lower those arcs' reduced costs by it.
- Stop when every terminal is reachable along arcs of reduced cost 0.

The sum of the raises is a lower bound on the optimal tree's weight. Each round costs a search over arcs of reduced cost 0, and the number of rounds can be large, so `time_limit` stops early and returns the bound reached so far, which is still valid, only looser. Terminals unreachable from the root are left out of the bound and reported.

## 9. Edge cases

- No roots: the unrooted tree over the terminals.
- One terminal and no roots: an empty plan.
- A terminal that is a root: `status="root"`, distance 0.
- A terminal unreachable, even through bridges: `status="unreachable"`, left out of the plan and the bound.
- Weights: zero allowed, negative or missing refused with `ValueError`, as the algorithms do today.
- `bridge_cost` set without any component to bridge: no bridges, no warning.

## 10. Performance targets

On Rwanda's simplified network (125,058 nodes, 332,535 edges), with 1,000 terminals and 20 roots, on the benchmark laptop:

| Step | Target |
|---|---|
| `plan(method="steiner")` | under 1 s (today's Steiner tree: 0.18 s) |
| `plan(method="shortest_path")` | under 10 s; 10,000 terminals under 60 s |
| `improve()`, one pass of both moves | under 10 s |
| `lower_bound()` | under 60 s, with `time_limit` returning the best bound so far |

## 11. Validation

- **Tests.** The rooted tree against networkx's `steiner_tree` on the virtual-root graph; every method on small graphs against brute-force optima; `improve` never worse, and better on constructed cases; the bound never above the brute-force optimum; and every edge case above.
- **Benchmarks.** `benchmarks/connection_planning/` writes instances in SteinLib's STP format and solves them exactly with SCIP-Jack, which is installed only there, never as a dependency. Instances: Kigali, and Rwanda with 100 and 1,000 terminals, by length and by cost. Reported for each method: its gap to the optimum, with and without `improve`; and the bound's gap. If SCIP-Jack cannot be built, networkx's `steiner_tree` gives another upper bound, and the planner's own bound the lower one.

## 12. Sub-tasks

Each sub-task is its own pull request on `feat/connection-planner`, stacked on #21.

| # | Sub-task | Delivers |
|---|---|---|
| 1 | Planner and rooted Steiner tree | `ConnectionPlanner`, `ConnectionPlan`; the virtual root in Mehlhorn's algorithm; `connecting_subnetwork` rewired with its four methods; the terminals table; public imports; tests against networkx on the virtual-root graph and against today's results; glossary entries |
| 2 | Bridges | `bridge_cost`, `bridge_max_distance`; candidate links between components; `kind="bridge"`; unreachable terminals; tests |
| 3 | Rollout order | `method="shortest_path"`, incremental; `order` and `marginal_cost`; tests against the naive version; performance targets |
| 4 | Local search | `improve()` with key-path exchange and key-vertex elimination; tests; performance target |
| 5 | Lower bound | `lower_bound(time_limit=)` by dual ascent; `bound` and `gap`; tests against brute force; performance target |
| 6 | Validation and docs | SCIP-Jack benchmarks; new decisions in the design and algorithm decisions document; the concept note's reduction and use-cases pages; changelog |

## 13. Deferred

- **Prize-collecting plans** (`prizes=`, a value per terminal such as its pupils): connect only the terminals worth their cost, or the most value within a budget. Goemans and Williamson's 2-approximation, with Hegde, Indyk, and Schmidt's nearly linear version (`pcst_fast`), fits the planner's virtual root.
- **Capacities** (`demands=` per terminal, `capacities=` per root): the capacitated Steiner tree, with the capacitated shortest-path heuristic (PLOS One, 2022); and Ito, Zhou, and Nishizeki's partitioning of a tree among capacitated roots as a second stage.
- **Separate root networks**, each terminal assigned to one.
- **Cost per cable**: fixed-charge network design or buy-at-bulk.
- **Vertex insertion and elimination** in `improve`.

## 14. Why "roots"

The existing network's nodes are called **roots**, not sources. "Source" is already a glossary term: the place a dataset is requested for (a country, a geometry, points, a frame), which every handler and fetcher uses. Using it for fibre nodes would give one word two meanings in the same library. "Root" is the Steiner literature's term and matches the virtual root the roots join. `connecting_subnetwork`'s `sources=` and `method="nearest_source"` become `roots=` and `"nearest_root"` in sub-task 1, before their first release. "Anchors" and "connected nodes" were considered and rejected as less precise.

## 15. References

- Mehlhorn, K. (1988). A faster approximation algorithm for the Steiner problem in graphs. *Information Processing Letters* 27(3).
- Takahashi, H., and Matsuyama, A. (1980). An approximate solution for the Steiner problem in graphs. *Math. Japonica* 24.
- Wong, R. T. (1984). A dual ascent approach for Steiner tree problems on a directed graph. *Mathematical Programming* 28.
- Uchoa, E., and Werneck, R. F. (2012). [Fast local search for the Steiner problem in graphs](https://doi.org/10.1145/2133803.2184448). *ACM Journal of Experimental Algorithmics* 17.
- Byrka, J., Grandoni, F., Rothvoß, T., and Sanità, L. (2013). [Steiner tree approximation via iterative randomized rounding](https://doi.org/10.1145/2432622.2432628). *Journal of the ACM* 60(1).
- Gamrath, G., Koch, T., Maher, S., Rehfeldt, D., and Shinano, Y. (2017). [SCIP-Jack: a solver for STP and variants with parallelization extensions](https://doi.org/10.1007/s12532-016-0114-x). *Mathematical Programming Computation* 9.
- Hegde, C., Indyk, P., and Schmidt, L. (2015). [A nearly-linear time framework for graph-structured sparsity](https://proceedings.mlr.press/v37/hegde15.pdf). ICML.
- Ito, T., Zhou, X., and Nishizeki, T. (2005). [Partitioning trees of supply and demand](https://doi.org/10.1142/s0129054105003303). *International Journal of Foundations of Computer Science* 16(4).
- [A construction heuristic for the capacitated Steiner tree problem](https://doi.org/10.1371/journal.pone.0270147) (2022). *PLOS One*.
