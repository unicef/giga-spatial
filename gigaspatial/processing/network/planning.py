"""Planning how to connect points to an existing network along a spatial network, at the smallest shared cost."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree

from gigaspatial.processing.network._arrays import as_positions
from gigaspatial.processing.network.algorithms import Graph, _dijkstra_blocks
from gigaspatial.processing.network.editing import _with_columns

if TYPE_CHECKING:
    from gigaspatial.processing.network.core import SpatialNetwork

METHODS = ("steiner", "nearest_root", "all_pairs")


@dataclass
class ConnectionPlan:
    """The result of a connection planner.

    Attributes:
        network: The connecting subnetwork, undirected, contracted to the
            terminals, the roots, and the nodes where paths branch, with
            lengths added up and geometries joined as :meth:`SpatialNetwork.simplify`
            does. Unreachable terminals and unused roots are kept without edges.
        edges: The chosen edges of the original network, uncontracted, one row
            per pair of nodes with the attributes of its cheapest edge and a
            geometry (a straight line where the network has none).
        terminals: One row per terminal, indexed by node ID: ``status``
            (``"connected"``, ``"root"`` for a terminal that is itself a root,
            or ``"unreachable"``), ``root`` (the root its tree reaches nearest
            along the plan), and ``distance`` (the weight along the plan to that
            root). Without roots, ``root`` is missing and ``distance`` is NaN.
        cost: The total weight of ``edges``.
        metadata: The method, the weight, the numbers of terminals and roots,
            terminals by status, and the seconds each step took.
    """

    network: "SpatialNetwork"
    edges: gpd.GeoDataFrame
    terminals: pd.DataFrame
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        reached = int(self.terminals["status"].isin(["connected", "root"]).sum())
        return (
            f"ConnectionPlan(method={self.metadata.get('method')!r}, cost={self.cost:,.1f}, "
            f"terminals={reached:,} of {len(self.terminals):,} connected, edges={len(self.edges):,})"
        )


class ConnectionPlanner:
    """Plans how to connect terminals to roots along a spatial network at the smallest shared cost.

    Terminals are the points to connect, such as schools. Roots are nodes
    already connected to each other through an existing network, such as
    fibre nodes, a power grid's substations, or water mains; reaching any root
    is enough. A terminal may reach a root directly or through other
    terminals, and an edge used by several terminals is paid once. Directions
    are ignored, since a cable or a pipe does not follow one-way streets.

    With roots, this is the Steiner tree problem once every root is joined to
    one virtual root at weight 0: the plan is a forest whose every tree
    reaches a root. Without roots, it is the Steiner tree joining the
    terminals. Both are NP-hard; the planner returns approximations.

    Insert the terminals and roots into the network first, with
    :meth:`SpatialNetwork.insert_points`, and pass their node IDs.

    Args:
        network: The network to plan along.
        terminals: Node IDs to connect.
        roots: Node IDs already connected to each other.
        weight: Edge column to minimise, such as a cost. The network's weight
            otherwise. Zero is allowed.

    Raises:
        KeyError: If a terminal or root is not a node of the network.
        ValueError: If the weight is missing or negative on some edge.
    """

    def __init__(self, network: "SpatialNetwork", terminals: Any, roots: Any = (), weight: Optional[str] = None):
        self.network = network
        self.weight = weight or network.weight
        self.terminals = _unique_ids(terminals)
        self.roots = _unique_ids(roots)
        self._terminal = network._positions(self.terminals)
        self._root = np.unique(network._positions(self.roots)) if len(self.roots) else np.empty(0, dtype=np.int64)
        self._graph = network._graph(self.weight, False)

    def __repr__(self) -> str:
        return (
            f"ConnectionPlanner(terminals={len(self.terminals):,}, roots={len(self.roots):,}, "
            f"weight={self.weight!r}, network={self.network!r})"
        )

    def plan(self, method: str = "steiner") -> ConnectionPlan:
        """Plan the connections.

        Args:
            method: ``"steiner"``: Mehlhorn's 2-approximation of the Steiner
                tree, with roots the rooted tree, without them the tree joining
                the terminals; within a few percent of the optimum in practice.
                ``"nearest_root"``: each terminal's shortest path to its nearest
                root, and the union of those paths; needs roots, and never lets a
                terminal connect through another one. ``"all_pairs"``: the
                shortest paths between every terminal and every terminal or
                root; their number grows with the square of the terminals.

        Returns:
            The plan. With roots, terminals that cannot reach any root are
            unreachable. Without roots, the plan joins the terminals of the
            network's part that holds the most of them, and the others are
            unreachable.

        Raises:
            ValueError: If the method is unknown, or is ``"nearest_root"``
                without roots.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {', '.join(repr(m) for m in METHODS)}, got {method!r}")
        if method == "nearest_root" and not len(self._root):
            raise ValueError('method="nearest_root" needs roots')
        started = time.perf_counter()
        graph = self._graph
        terminal = np.unique(self._terminal)
        planned = terminal[self._reachable(terminal)]
        if method == "steiner":
            if len(self._root):
                a, b = _rooted_steiner_edges(graph, planned, self._root)
            else:
                a, b = _steiner_edges(graph, planned)
        elif method == "nearest_root":
            a, b = _nearest_root_edges(graph, planned, self._root)
        else:
            a, b = _all_pairs_edges(graph, planned, np.union1d(planned, self._root))
        planned_at = time.perf_counter()

        kept = np.union1d(terminal, self._root)
        sub = _edge_subnetwork(self.network, graph, a, b, kept)
        ids = self.network.node_index
        terminals = self._terminals_table(sub, planned)
        summed = ("length",) if self.weight == "length" else ("length", self.weight)
        contracted = sub.simplify(keep=ids.take(kept), sum_columns=summed)
        status = terminals["status"].value_counts()
        metadata = {
            "method": method,
            "weight": self.weight,
            "terminals": len(terminals),
            "roots": len(self.roots),
            "status": {s: int(status.get(s, 0)) for s in ("connected", "root", "unreachable")},
            "seconds": {"plan": planned_at - started, "assemble": time.perf_counter() - planned_at},
        }
        return ConnectionPlan(
            network=contracted,
            edges=sub.edges_gdf.reset_index(drop=True),
            terminals=terminals,
            cost=float(sub.edges[self.weight].sum()) if len(sub.edges) else 0.0,
            metadata=metadata,
        )

    def _reachable(self, terminal: np.ndarray) -> np.ndarray:
        """Which terminals the plan can connect: those sharing a part of the network with a root,
        or without roots, those in the part that holds the most terminals."""
        if not len(terminal):
            return np.zeros(0, dtype=bool)
        labels = self._labels
        if len(self._root):
            return np.isin(labels[terminal], labels[self._root])
        counts = np.bincount(labels[terminal])
        # argmax takes the first of equal counts, the part of the lowest label, so plans are reproducible.
        return labels[terminal] == np.argmax(counts)

    @property
    def _labels(self) -> np.ndarray:
        if "_component_labels" not in self.__dict__:
            self.__dict__["_component_labels"] = connected_components(self._graph.matrix, directed=False)[1]
        return self.__dict__["_component_labels"]

    def _terminals_table(self, sub: "SpatialNetwork", planned: np.ndarray) -> pd.DataFrame:
        ids = self.network.node_index
        n = len(self.terminals)
        reached = np.isin(self._terminal, planned)
        status = np.where(reached, "connected", "unreachable").astype(object)
        root = np.full(n, None, dtype=object)
        distance = np.full(n, np.nan)
        if len(self._root):
            is_root = np.isin(self._terminal, self._root)
            status[is_root] = "root"
            here = sub.node_index.get_indexer(ids.take(self._root))
            dist, _, nearest = dijkstra(
                sub._graph(self.weight, False).matrix,
                directed=True,
                indices=here,
                min_only=True,
                return_predecessors=True,
            )
            at = sub.node_index.get_indexer(self.terminals)
            found = reached & (nearest[at] >= 0)
            distance[found] = dist[at[found]]
            root[found] = sub.node_index.take(nearest[at[found]]).to_numpy()
        return pd.DataFrame(
            {"status": status, "root": root, "distance": distance},
            index=pd.Index(self.terminals, name="node_id"),
        )


def _unique_ids(values: Any) -> pd.Index:
    if values is None:
        return pd.Index([])
    if np.ndim(values) == 0:
        values = [values]
    return pd.Index(values).unique()


def _edge_subnetwork(
    network: "SpatialNetwork", graph: Graph, a: np.ndarray, b: np.ndarray, kept: np.ndarray
) -> "SpatialNetwork":
    """An undirected network of the cheapest edge behind each (a, b) pair, plus the kept nodes."""
    low, high = np.minimum(a, b), np.maximum(a, b)
    pairs = np.unique(low.astype(np.int64) * graph.n + high)
    rows = np.sort(graph.entry_rows[np.searchsorted(graph.entry_keys, pairs)])
    pu, pv = network._edge_positions
    members = np.union1d(np.union1d(pu[rows], pv[rows]), kept)
    nodes = network._nodes.iloc[members]
    edges = _with_columns(network._edges.iloc[rows], {"key": np.zeros(len(rows), dtype=np.int16)})
    edges.index = pd.RangeIndex(len(rows))
    positions = (
        as_positions(np.searchsorted(members, pu[rows]), len(members)),
        as_positions(np.searchsorted(members, pv[rows]), len(members)),
    )
    return type(network)._from_tables(
        nodes, edges, directed=False, weight=network._weight, metadata=network._metadata, positions=positions
    )


# ----- Steiner routines, on node positions


def _with_virtual_root(graph: Graph, root: np.ndarray) -> Graph:
    """The graph with one more node, joined to every root at weight 0, numbered ``graph.n``."""
    n = graph.n
    size = n + 1
    virtual = np.full(len(root), n, dtype=np.int64)
    a = np.concatenate([graph.entry_keys // n, root, virtual])
    b = np.concatenate([graph.entry_keys % n, virtual, root])
    weights = np.concatenate([graph.entry_weights, np.zeros(2 * len(root))])
    rows = np.concatenate([graph.entry_rows, np.full(2 * len(root), -1)])
    keys = a * size + b
    order = np.argsort(keys, kind="stable")
    matrix = csr_matrix((weights[order], (a[order], b[order])), shape=(size, size))
    return Graph(matrix, keys[order], rows[order], weights[order], size)


def _rooted_steiner_edges(graph: Graph, terminal: np.ndarray, root: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mehlhorn's algorithm over the terminals and a virtual root joined to every root, without the joins."""
    virtual = graph.n
    a, b = _steiner_edges(_with_virtual_root(graph, root), np.append(terminal, virtual))
    real = (a != virtual) & (b != virtual)
    return a[real], b[real]


def _forest_edges(predecessors: np.ndarray, starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Edges on the predecessor paths from ``starts`` back to their roots."""
    visited = np.zeros(len(predecessors), dtype=bool)
    a_parts, b_parts = [], []
    frontier = np.unique(starts)
    visited[frontier] = True
    while len(frontier):
        parent = predecessors[frontier]
        step = parent >= 0
        a_parts.append(frontier[step])
        b_parts.append(parent[step])
        frontier = np.unique(parent[step][~visited[parent[step]]])
        visited[frontier] = True
    if not a_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(a_parts), np.concatenate(b_parts)


def _nearest_root_edges(graph: Graph, terminal: np.ndarray, root: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, predecessors, _ = dijkstra(graph.matrix, directed=True, indices=root, min_only=True, return_predecessors=True)
    return _forest_edges(predecessors, terminal)


def _steiner_edges(graph: Graph, terminal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mehlhorn's 2-approximation: Voronoi regions of the terminals, an MST of their cheapest links, then pruning.

    ``terminal`` must be sorted.
    """
    empty = np.empty(0, dtype=np.int64)
    if len(terminal) < 2:
        return empty, empty
    n = graph.n
    distance, predecessors, nearest = dijkstra(
        graph.matrix, directed=True, indices=terminal, min_only=True, return_predecessors=True
    )
    a, b = graph.entry_keys // n, graph.entry_keys % n
    crossing = (a < b) & (nearest[a] >= 0) & (nearest[b] >= 0) & (nearest[a] != nearest[b])
    a, b = a[crossing], b[crossing]
    cost = distance[a] + graph.entry_weights[crossing] + distance[b]
    ta = np.searchsorted(terminal, np.minimum(nearest[a], nearest[b]))
    tb = np.searchsorted(terminal, np.maximum(nearest[a], nearest[b]))
    order = np.lexsort((cost, tb, ta))
    first = order[np.r_[True, (ta[order][1:] != ta[order][:-1]) | (tb[order][1:] != tb[order][:-1])]]
    k = len(terminal)
    link_cost = np.where(cost[first] == 0, np.nextafter(0, 1), cost[first])
    tree = minimum_spanning_tree(csr_matrix((link_cost, (ta[first], tb[first])), shape=(k, k))).tocoo()
    low, high = np.minimum(tree.row, tree.col).astype(np.int64), np.maximum(tree.row, tree.col)
    chosen = first[np.searchsorted(ta[first].astype(np.int64) * k + tb[first], low * k + high)]
    pa_, pb_ = _forest_edges(predecessors, np.concatenate([a[chosen], b[chosen]]))
    sub_a = np.concatenate([pa_, a[chosen]])
    sub_b = np.concatenate([pb_, b[chosen]])
    sub_a, sub_b = _spanning_tree(graph, sub_a, sub_b)
    return _prune_leaves(sub_a, sub_b, terminal, n)


def _spanning_tree(graph: Graph, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = graph.n
    low, high = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
    keys = np.unique(low * n + high)
    weights = graph.entry_weights[np.searchsorted(graph.entry_keys, keys)]
    weights = np.where(weights == 0, np.nextafter(0, 1), weights)
    tree = minimum_spanning_tree(csr_matrix((weights, (keys // n, keys % n)), shape=(n, n))).tocoo()
    return tree.row.astype(np.int64), tree.col.astype(np.int64)


def _prune_leaves(a: np.ndarray, b: np.ndarray, kept: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    protected = np.zeros(n, dtype=bool)
    protected[kept] = True
    while len(a):
        degree = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
        dangling = ((degree[a] == 1) & ~protected[a]) | ((degree[b] == 1) & ~protected[b])
        if not dangling.any():
            break
        a, b = a[~dangling], b[~dangling]
    return a, b


def _all_pairs_edges(graph: Graph, terminal: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a_parts, b_parts = [], []
    for _, _, predecessors in _dijkstra_blocks(graph.matrix, terminal, predecessors=True):
        for row in predecessors:
            pa_, pb_ = _forest_edges(row, targets)
            a_parts.append(pa_)
            b_parts.append(pb_)
    if not a_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(a_parts), np.concatenate(b_parts)
