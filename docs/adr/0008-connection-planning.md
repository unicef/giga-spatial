# Connecting points to an existing network is a rooted Steiner tree, solved by a planner object

Connecting schools, clinics, or villages to an existing network, such as fibre, a power grid, or water mains, at the smallest shared cost is the Steiner tree problem once the existing network is treated as one root. Every root node (a fibre node, or a place already connected) joins a virtual root at zero cost, and the plan is a minimum Steiner tree over the terminals and that virtual root. Removing it leaves a forest whose every tree reaches the existing network, so a terminal may connect directly or through other terminals, and a trench used by several terminals is paid once. `ConnectionPlanner` holds the problem (network, terminals, roots, weight, and later prizes, demands, and capacities) and its solvers; `SpatialNetwork.connecting_subnetwork` stays as a thin wrapper that returns the planned network.

On the concept note's example (197 schools and 5 fibre nodes in Kigali), the rooted tree is 258 km. Joining each school to its nearest fibre node, the old meaning of `sources=`, takes 355 km, because it shares only overlapping routes. Joining the schools and the fibre nodes in one unrooted tree takes 265 km, because it also pays to connect fibre nodes that the backbone already connects.

The existing network's nodes are called roots, not sources, because "Source" already means the place a dataset is requested for across the handlers and fetchers. `connecting_subnetwork`'s `sources=` and `method="nearest_source"` become `roots=` and `"nearest_root"` before their first release.

## Considered options

- **Shortest paths to the nearest root.** Kept as `method="nearest_root"`, but not the default: it never lets a terminal connect through another one, and it was 38% longer on the example.
- **An unrooted Steiner tree over terminals and roots.** Rejected for this question: it pays to join the roots to each other. The gap grows with how far apart the roots are.
- **Methods on `SpatialNetwork` only.** Rejected: prize-collecting and capacitated versions need prizes per terminal, demands, capacities per root, solver options, and a result with more than a network. A problem object takes them without growing `SpatialNetwork`'s API, and new solvers are added to it.
- **A fibre-specific planner.** Rejected: the same problem arises for power grids, water, and evacuation routes. The planner uses generic terms (terminals, roots, bridges), and the docs use fibre as their main example.
- **An exact solver as a dependency.** Rejected: SCIP-Jack solves instances of this size exactly, but it is a compiled external program. It is used only to validate the heuristics, in the benchmarks.

## Consequences

- `connecting_subnetwork(terminals, sources=...)` becomes `connecting_subnetwork(terminals, roots=...)` and changes meaning before its first release: with roots and `method="steiner"`, it is the rooted tree.
- The problem is NP-hard. The planner returns approximations (Mehlhorn's algorithm, at most twice the optimum) and a lower bound, so every plan can state how far from the optimum it can be at most.
- Cost per cable, which grows with the number of terminals behind an edge, is out of scope; it makes the problem a fixed-charge network design problem, with very different solvers.
