# Spatial networks are held as tables, not as networkx graphs

A `SpatialNetwork` keeps its nodes and edges as two columnar tables. The adjacency matrix that scipy's graph routines need is built from them on demand and cached. networkx and pandana are export formats, not the representation, and every edit returns a new network so that cached structures never go stale. We measured this on the simplified Rwanda drive network (125,058 nodes, 332,535 edges). The networkx graph held 731 MB, against 229 MB for the tables. Node coordinates took 43 ms from the graph and 0.6 ms from the table. The edges of a sub-region took 1.6 to 3 s from the graph and 0.03 s from the tables. scipy computed shortest paths about 12 times faster than networkx.

## Considered options

- **A networkx `MultiDiGraph`, as osmnx returns.** Rejected: it costs three times the memory, and every accessor loops in Python. Building one from the tables takes 0.8 to 2.5 s on the network above, so users who need networkx pay that once, only when they ask for it.
- **igraph.** Rejected as the representation: it is one more compiled dependency, and its many-to-many shortest paths (13 µs per pair) were between scipy (24 µs) and pandana (1.9 µs).
- **An `inplace=` option on edits.** Rejected: it saves no memory, because unchanged columns are shared between the old and new network, and every in-place edit would have to invalidate the cached adjacency matrix, KD-tree, spatial index, and pandana network. Missing one gives wrong shortest paths without an error.

## Consequences

- Adding or removing edges one at a time in a Python loop is slow. The editing methods take many nodes or edges at once.
- pandana takes the tables directly. Its contraction-hierarchy build took 1.4 s on the network above and then answered origin-destination pairs at about 1.9 µs each. That makes pandana the faster backend for very large sets of pairs, while scipy needs no build step.
