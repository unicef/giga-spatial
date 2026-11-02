"""Planning how to connect points to an existing network along a spatial network, at the smallest shared cost."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from scipy.spatial import cKDTree

from gigaspatial.processing.network._arrays import as_positions
from gigaspatial.processing.network.algorithms import _OWN_TREE_SIZE, Graph, _dijkstra_blocks
from gigaspatial.processing.network.editing import _concat_tables, _with_columns
from gigaspatial.processing.network.geodesy import haversine, meters_to_chord, unit_vectors
from gigaspatial.processing.network.kinds import EDGE_KINDS, as_kinds

if TYPE_CHECKING:
    from gigaspatial.processing.network.core import SpatialNetwork

METHODS = ("steiner", "nearest_root", "all_pairs")
# Each part of the network is offered bridges to at most this many of its nearest other parts.
BRIDGES_PER_COMPONENT = 5
_FIRST_NEIGHBOURS = 16


@dataclass
class ConnectionPlan:
    """The result of a connection planner.

    Attributes:
        network: The connecting subnetwork, undirected, contracted to the
            terminals, the roots, the ends of bridges, and the nodes where
            paths branch, with lengths added up and geometries joined as
            :meth:`SpatialNetwork.simplify` does. Unreachable terminals and
            unused roots are kept without edges.
        edges: The chosen edges, uncontracted, one row per pair of nodes with
            the attributes of its cheapest edge and a geometry (a straight line
            where the network has none). Bridges have kind ``"bridge"``, their
            great-circle ``length``, and their weight in the weight column
            unless the weight is ``length``.
        terminals: One row per terminal, indexed by node ID: ``status``
            (``"connected"``, ``"root"`` for a terminal that is itself a root,
            or ``"unreachable"``); ``root``, the root its tree reaches nearest
            along the plan; ``distance``, the weight along the plan to that
            root; and ``bridged``, whether that path uses a bridge. Without
            roots, ``root`` is missing, ``distance`` is NaN, and ``bridged``
            says whether the terminal is joined to the others only through a
            bridge.
        cost: The total weight of ``edges``, with each bridge at
            ``bridge_cost`` per metre.
        metadata: The method, the weight, ``bridge_cost``, the numbers of
            terminals and roots, terminals by status, bridges offered and used,
            and the seconds each step took.
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


class _Bridges(NamedTuple):
    """Straight-line links between parts of the network, by node position, with their lengths in metres."""

    a: np.ndarray
    b: np.ndarray
    length: np.ndarray


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
        bridge_cost: Weight per metre of a straight-line link between parts of
            the network that no road joins, such as a microwave link or a
            planned road, in the units of ``weight``. No bridges when ``None``.
            Each part is offered links to its nearest other parts, at most
            five, and a plan uses a link only where it is cheaper than the
            roads.
        bridge_max_distance: Longest link, in metres.

    Raises:
        KeyError: If a terminal or root is not a node of the network.
        ValueError: If the weight is missing or negative on some edge, or the
            bridge options are negative.
    """

    def __init__(
        self,
        network: "SpatialNetwork",
        terminals: Any,
        roots: Any = (),
        weight: Optional[str] = None,
        bridge_cost: Optional[float] = None,
        bridge_max_distance: float = 5000.0,
    ):
        if bridge_cost is not None and not (np.isfinite(bridge_cost) and bridge_cost >= 0):
            raise ValueError(f"bridge_cost must be a number of at least 0, got {bridge_cost!r}")
        if not (np.isfinite(bridge_max_distance) and bridge_max_distance > 0):
            raise ValueError(f"bridge_max_distance must be a positive number of metres, got {bridge_max_distance!r}")
        self.network = network
        self.weight = weight or network.weight
        self.terminals = _unique_ids(terminals)
        self.roots = _unique_ids(roots)
        self.bridge_cost = None if bridge_cost is None else float(bridge_cost)
        self.bridge_max_distance = float(bridge_max_distance)
        self._terminal = network._positions(self.terminals)
        self._root = np.unique(network._positions(self.roots)) if len(self.roots) else np.empty(0, dtype=np.int64)
        roads = network._graph(self.weight, False)
        self._road_labels = connected_components(roads.matrix, directed=False)[1]
        if self.bridge_cost is None:
            empty = np.empty(0, dtype=np.int64)
            self._bridges = _Bridges(empty, empty, np.empty(0))
        else:
            self._bridges = _bridge_candidates(network, self._road_labels, self.bridge_max_distance)
        self._graph = _with_bridges(roads, self._bridges, self.bridge_cost, len(network._edges))
        if len(self._bridges.a):
            self._labels = connected_components(self._graph.matrix, directed=False)[1]
        else:
            self._labels = self._road_labels

    def __repr__(self) -> str:
        return (
            f"ConnectionPlanner(terminals={len(self.terminals):,}, roots={len(self.roots):,}, "
            f"weight={self.weight!r}, bridges={len(self._bridges.a):,}, network={self.network!r})"
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
            The plan. With roots, terminals that cannot reach any root, even
            through bridges, are unreachable. Without roots, the plan joins the
            terminals of the network's part that holds the most of them, and
            the others are unreachable.

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

        low, high = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
        pairs = np.unique(low * graph.n + high)
        entries = np.searchsorted(graph.entry_keys, pairs)
        rows, weights = graph.entry_rows[entries], graph.entry_weights[entries]
        bridge = rows >= len(self.network._edges)
        ends = np.concatenate([pairs[bridge] // graph.n, pairs[bridge] % graph.n])
        kept = np.union1d(np.union1d(terminal, self._root), ends)
        sub = _plan_subnetwork(self.network, rows, self._bridges, self.bridge_cost, self.weight, kept)
        terminals = self._terminals_table(pairs, weights, bridge, planned)
        summed = ("length",) if self.weight == "length" else ("length", self.weight)
        contracted = sub.simplify(keep=self.network.node_index.take(kept), sum_columns=summed)
        status = terminals["status"].value_counts()
        metadata = {
            "method": method,
            "weight": self.weight,
            "bridge_cost": self.bridge_cost,
            "terminals": len(terminals),
            "roots": len(self.roots),
            "status": {s: int(status.get(s, 0)) for s in ("connected", "root", "unreachable")},
            "bridges": {"offered": len(self._bridges.a), "used": int(bridge.sum())},
            "seconds": {"plan": planned_at - started, "assemble": time.perf_counter() - planned_at},
        }
        return ConnectionPlan(
            network=contracted,
            edges=sub.edges_gdf.reset_index(drop=True),
            terminals=terminals,
            cost=float(weights.sum()),
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

    def _terminals_table(
        self, pairs: np.ndarray, weights: np.ndarray, bridge: np.ndarray, planned: np.ndarray
    ) -> pd.DataFrame:
        n = self._graph.n
        count = len(self.terminals)
        reached = np.isin(self._terminal, planned)
        status = np.where(reached, "connected", "unreachable").astype(object)
        root = np.full(count, None, dtype=object)
        distance = np.full(count, np.nan)
        bridged = np.zeros(count, dtype=bool)
        if len(self._root):
            status[np.isin(self._terminal, self._root)] = "root"
            a, b = pairs // n, pairs % n
            plan = csr_matrix((np.concatenate([weights, weights]), (np.r_[a, b], np.r_[b, a])), shape=(n, n))
            dist, predecessors, nearest = dijkstra(
                plan, directed=True, indices=self._root, min_only=True, return_predecessors=True
            )
            at = self._terminal
            found = reached & (nearest[at] >= 0)
            distance[found] = dist[at[found]]
            root[found] = self.network.node_index.take(nearest[at[found]]).to_numpy()
            if bridge.any():
                bridged = _uses_bridge(predecessors, pairs[bridge], n)[at] & found
        elif bridge.any():
            planned_roads = self._road_labels[planned]
            main = np.argmax(np.bincount(planned_roads))
            bridged = reached & (self._road_labels[self._terminal] != main)
        return pd.DataFrame(
            {"status": status, "root": root, "distance": distance, "bridged": bridged},
            index=pd.Index(self.terminals, name="node_id"),
        )


def _unique_ids(values: Any) -> pd.Index:
    if values is None:
        return pd.Index([])
    if np.ndim(values) == 0:
        values = [values]
    return pd.Index(values).unique()


def _uses_bridge(predecessors: np.ndarray, bridge_pairs: np.ndarray, n: int) -> np.ndarray:
    """Whether each node's predecessor path back to its root crosses a bridge, by pointer jumping."""
    node = np.arange(n)
    parent = predecessors.astype(np.int64)
    has = parent >= 0
    step = np.zeros(n, dtype=bool)
    keys = np.minimum(node[has], parent[has]) * n + np.maximum(node[has], parent[has])
    step[has] = np.isin(keys, bridge_pairs)
    while (parent >= 0).any():
        up = parent >= 0
        step[up] |= step[parent[up]]
        parent[up] = parent[parent[up]]
    return step


def _plan_subnetwork(
    network: "SpatialNetwork",
    rows: np.ndarray,
    bridges: _Bridges,
    bridge_cost: Optional[float],
    weight: str,
    kept: np.ndarray,
) -> "SpatialNetwork":
    """An undirected network of the chosen edge rows, where rows past the network's edges are bridges."""
    m = len(network._edges)
    road = np.sort(rows[rows < m])
    link = np.sort(rows[rows >= m] - m)
    pu, pv = network._edge_positions
    a = np.concatenate([pu[road], bridges.a[link]])
    b = np.concatenate([pv[road], bridges.b[link]])
    members = np.union1d(np.union1d(a, b), kept)
    tables = [_with_columns(network._edges.iloc[road], {"key": np.zeros(len(road), dtype=np.int16)})]
    if len(link):
        tables.append(_bridge_table(network, bridges, link, bridge_cost, weight))
    edges = _concat_tables(tables, ignore_index=True) if len(tables) > 1 else tables[0]
    edges.index = pd.RangeIndex(len(edges))
    positions = (
        as_positions(np.searchsorted(members, a), len(members)),
        as_positions(np.searchsorted(members, b), len(members)),
    )
    return type(network)._from_tables(
        network._nodes.iloc[members],
        edges,
        directed=False,
        weight=network._weight,
        metadata=network._metadata,
        positions=positions,
    )


def _bridge_table(
    network: "SpatialNetwork", bridges: _Bridges, link: np.ndarray, bridge_cost: float, weight: str
) -> pd.DataFrame:
    a, b, length = bridges.a[link], bridges.b[link], bridges.length[link]
    ids = network.node_index
    columns = {
        "u": ids.take(a),
        "v": ids.take(b),
        "key": np.zeros(len(link), dtype=np.int16),
        "length": length,
        "kind": as_kinds("bridge", "bridge", EDGE_KINDS, len(link)),
    }
    if weight != "length":
        columns[weight] = bridge_cost * length
    if "geometry" in network._edges.columns:
        x, y = network._nodes["x"].to_numpy(), network._nodes["y"].to_numpy()
        ends = np.stack([np.column_stack([x[a], y[a]]), np.column_stack([x[b], y[b]])], axis=1)
        columns["geometry"] = shapely.linestrings(ends)
    return pd.DataFrame(columns)


# ----- bridges


def _bridge_candidates(network: "SpatialNetwork", labels: np.ndarray, max_distance: float) -> _Bridges:
    """The closest node pair between each part of the network and each of its nearest other parts.

    A part with more than ``_OWN_TREE_SIZE`` nodes, or most of the network,
    gets its own KD-tree, queried with every node outside it; that gives its
    closest pair to every other part exactly. Smaller parts query a KD-tree of
    all nodes with their nodes' nearest neighbours, more of them while none
    lies outside the part and the farthest is within ``max_distance``.
    """
    empty = np.empty(0, dtype=np.int64)
    count = int(labels.max()) + 1 if len(labels) else 0
    if count < 2:
        return _Bridges(empty, empty, np.empty(0))
    lat, lon = network._nodes["y"].to_numpy(), network._nodes["x"].to_numpy()
    points = unit_vectors(lat, lon)
    radius = float(meters_to_chord(max_distance))
    n = len(points)
    sizes = np.bincount(labels, minlength=count)
    large = np.flatnonzero((sizes > _OWN_TREE_SIZE) | (sizes * 2 > n))
    found_a, found_b = [empty], [empty]
    for part in large:
        inside = np.flatnonzero(labels == part)
        outside = np.flatnonzero(labels != part)
        chord, j = cKDTree(points[inside]).query(points[outside], k=1, distance_upper_bound=radius)
        near = np.isfinite(chord)
        found_a.append(inside[j[near]])
        found_b.append(outside[near])
    pending = np.flatnonzero(~np.isin(labels, large))
    if len(pending):
        tree = cKDTree(points)
        linked = np.zeros(count, dtype=bool)
        k = min(_FIRST_NEIGHBOURS, n)
        while len(pending):
            chord, j = tree.query(points[pending], k=k, distance_upper_bound=radius)
            chord, j = chord.reshape(len(pending), -1), j.reshape(len(pending), -1)
            near = np.isfinite(chord)
            outside = near & (labels[np.where(near, j, 0)] != labels[pending][:, None])
            row, col = np.nonzero(outside)
            found_a.append(pending[row])
            found_b.append(j[row, col])
            linked[labels[pending[row]]] = True
            if k >= n:
                break
            pending = pending[~linked[labels[pending]] & near[:, -1]]
            k = min(2 * k, n)
    a, b = np.concatenate(found_a), np.concatenate(found_b)
    if not len(a):
        return _Bridges(empty, empty, np.empty(0))
    length = haversine(lat[a], lon[a], lat[b], lon[b])
    low, high = np.minimum(labels[a], labels[b]).astype(np.int64), np.maximum(labels[a], labels[b]).astype(np.int64)
    key = low * count + high
    order = np.lexsort((length, key))
    first = order[np.r_[True, key[order][1:] != key[order][:-1]]]
    a, b, length, low, high = a[first], b[first], length[first], low[first], high[first]
    part = np.concatenate([low, high])
    candidate = np.concatenate([np.arange(len(a)), np.arange(len(a))])
    order = np.lexsort((np.concatenate([length, length]), part))
    starts = np.r_[0, np.flatnonzero(part[order][1:] != part[order][:-1]) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    chosen = np.unique(candidate[order[rank < BRIDGES_PER_COMPONENT]])
    return _Bridges(a[chosen].astype(np.int64), b[chosen].astype(np.int64), length[chosen])


def _with_bridges(graph: Graph, bridges: _Bridges, bridge_cost: Optional[float], first_row: int) -> Graph:
    """The graph with each bridge added in both directions, as edge rows numbered from ``first_row``."""
    if not len(bridges.a):
        return graph
    n = graph.n
    k = len(bridges.a)
    weights = bridge_cost * bridges.length
    a = np.concatenate([graph.entry_keys // n, bridges.a, bridges.b])
    b = np.concatenate([graph.entry_keys % n, bridges.b, bridges.a])
    w = np.concatenate([graph.entry_weights, weights, weights])
    rows = np.concatenate([graph.entry_rows, first_row + np.arange(k), first_row + np.arange(k)])
    keys = a * n + b
    order = np.argsort(keys, kind="stable")
    matrix = csr_matrix((w[order], (a[order], b[order])), shape=(n, n))
    return Graph(matrix, keys[order], rows[order], w[order], n)


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
