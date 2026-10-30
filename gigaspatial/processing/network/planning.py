"""Planning how to connect points to an existing network along a spatial network, at the smallest shared cost."""

import heapq
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

METHODS = ("steiner", "shortest_path", "nearest_root", "all_pairs")
# Each part of the network is offered bridges to at most this many of its nearest other parts.
BRIDGES_PER_COMPONENT = 5
_FIRST_NEIGHBOURS = 16
# A move must save at least this share of the weight it replaces, so rounding never loops.
_GAIN = 1e-9
# The dual ascent recomputes distances after this fraction (1 / value) of its waiting terminals.
_REFRESH_SHARE = 50


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
            bridge. Plans by ``"shortest_path"`` add ``order``, when the
            terminal was connected (0 for a terminal that is a root, missing
            when unreachable), and ``marginal_cost``, the weight of the edges
            its connection added.
        cost: The total weight of ``edges``, with each bridge at
            ``bridge_cost`` per metre.
        metadata: The method, the weight, ``bridge_cost``, the numbers of
            terminals and roots, terminals by status, bridges offered and used,
            and the seconds each step took.
        bound: A lower bound on the cost of any plan connecting the same
            terminals, set by :meth:`ConnectionPlanner.lower_bound`.
        gap: ``cost / bound - 1``: the plan costs at most ``1 + gap`` times
            the optimum.
    """

    network: "SpatialNetwork"
    edges: gpd.GeoDataFrame
    terminals: pd.DataFrame
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    bound: Optional[float] = None
    gap: Optional[float] = None

    def __repr__(self) -> str:
        reached = int(self.terminals["status"].isin(["connected", "root"]).sum())
        gap = "" if self.gap is None else f", gap={self.gap:.1%}"
        return (
            f"ConnectionPlan(method={self.metadata.get('method')!r}, cost={self.cost:,.1f}{gap}, "
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
                ``"shortest_path"``: Takahashi and Matsuyama's 2-approximation,
                a model of a phased rollout: starting from the roots (or the
                first terminal), repeatedly connect the terminal nearest to
                anything connected so far, along its shortest path. Gives each
                terminal's order and marginal cost; ties go to the terminal
                listed first. ``"nearest_root"``: each terminal's shortest path to its nearest
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
        rollout = None
        if method == "steiner":
            if len(self._root):
                a, b = _rooted_steiner_edges(graph, planned, self._root)
            else:
                a, b = _steiner_edges(graph, planned)
        elif method == "shortest_path":
            a, b, rollout = _shortest_path_edges(graph, self._planned_in_given_order(planned), self._root)
        elif method == "nearest_root":
            a, b = _nearest_root_edges(graph, planned, self._root)
        else:
            a, b = _all_pairs_edges(graph, planned, np.union1d(planned, self._root))
        return self._assemble(method, a, b, planned, rollout, {"plan": time.perf_counter() - started})

    def improve(self, plan: ConnectionPlan, time_limit: Optional[float] = None) -> ConnectionPlan:
        """Improve a plan by local search; the result never costs more.

        Two of Uchoa and Werneck's moves are applied until neither improves
        the plan or ``time_limit`` passes. A key vertex is a terminal, the
        roots (all counted as one), or a node where the plan branches; a key
        path joins two key vertices through nodes that only continue it.

        - Key-path exchange: remove a key path, which splits the plan in two,
          and reconnect the two parts by the cheapest path between them, when
          that is cheaper.
        - Key-vertex elimination: remove a branching node and its key paths,
          which splits the plan in three or more parts, and reconnect them by
          a spanning tree of their cheapest links, when that is cheaper.

        Key paths are tried heaviest first. Each search is limited to the
        weight it must beat, so most stop early. ``order`` and
        ``marginal_cost`` of a ``"shortest_path"`` plan are dropped, since the
        improved plan no longer connects terminals in that order.

        Args:
            plan: A plan made by this planner.
            time_limit: Seconds to stop after, finishing the move under way.
                Until no move improves the plan otherwise.

        Returns:
            The improved plan. Its metadata records the cost before, the
            moves applied, and the seconds each pass took.

        Raises:
            ValueError: If the plan was made for other terminals.
        """
        if not plan.terminals.index.equals(pd.Index(self.terminals, name="node_id")):
            raise ValueError("The plan was made for other terminals; improve it with the planner that made it")
        started = time.perf_counter()
        deadline = np.inf if time_limit is None else started + float(time_limit)
        ids = self.network.node_index
        reached = plan.terminals["status"].isin(["connected", "root"]).to_numpy()
        planned = np.unique(self._terminal[reached])
        graph = self._graph
        rooted = len(self._root) > 0
        if rooted:
            graph = _with_virtual_root(graph, self._root)
        u = ids.get_indexer(plan.edges["u"]).astype(np.int64)
        v = ids.get_indexer(plan.edges["v"]).astype(np.int64)
        required = planned
        if rooted:
            virtual = self._graph.n
            touched = np.intersect1d(np.union1d(np.union1d(u, v), planned), self._root)
            u = np.concatenate([u, touched])
            v = np.concatenate([v, np.full(len(touched), virtual)])
            required = np.append(planned, virtual)
        tree = _Tree(graph, u, v, required)
        before = float(plan.cost)
        moves, passes = tree.search(deadline)
        a, b = tree.edges()
        if rooted:
            real = (a != virtual) & (b != virtual)
            a, b = a[real], b[real]
        method = plan.metadata.get("method", "steiner")
        improved = self._assemble(method, a, b, planned, None, {"improve": time.perf_counter() - started})
        improved.metadata["improved"] = {"cost_before": before, "moves": moves, "pass_seconds": passes}
        return improved

    def lower_bound(self, plan: Optional[ConnectionPlan] = None, time_limit: Optional[float] = 60.0) -> float:
        """A lower bound on the cost of any plan connecting the reachable terminals, by Wong's dual ascent.

        The network becomes directed, with both directions of every edge,
        rooted at the virtual root, or without roots at the first terminal
        that can be connected. Each terminal in turn grows the set of nodes
        that reach it along arcs of reduced cost 0, raising the bound by the
        smallest reduced cost of the arcs entering the set and lowering theirs
        by as much, until the root is in the set. The sum of the raises is a
        lower bound on the optimum. Unreachable terminals are left out.

        A bound reached when ``time_limit`` stops the ascent is still valid,
        only looser. A complete bound is kept, so later calls return it at once.

        Args:
            plan: A plan of this planner, whose ``bound`` and ``gap`` are set.
            time_limit: Seconds to stop after. No limit when ``None``.

        Returns:
            The bound, in the units of the weight.
        """
        cached = self.__dict__.get("_bound")
        if cached is None or not cached["complete"]:
            started = time.perf_counter()
            deadline = np.inf if time_limit is None else started + float(time_limit)
            terminal = np.unique(self._terminal)
            planned = self._planned_in_given_order(terminal[self._reachable(terminal)])
            graph = self._graph
            if len(self._root):
                graph, root, targets = _with_virtual_root(graph, self._root), self._graph.n, planned
            elif len(planned):
                root, targets = int(planned[0]), planned[1:]
            else:
                root, targets = None, planned[:0]
            value, complete = _dual_ascent(graph, root, targets, deadline) if root is not None else (0.0, True)
            cached = {"bound": value, "complete": complete, "seconds": time.perf_counter() - started}
            self.__dict__["_bound"] = cached
        if plan is not None:
            plan.bound = cached["bound"]
            plan.gap = plan.cost / cached["bound"] - 1 if cached["bound"] > 0 else (0.0 if plan.cost == 0 else np.inf)
            plan.metadata["bound"] = dict(cached)
        return cached["bound"]

    def _assemble(
        self,
        method: str,
        a: np.ndarray,
        b: np.ndarray,
        planned: np.ndarray,
        rollout: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        seconds: Dict[str, float],
    ) -> ConnectionPlan:
        """The plan of the edges (a, b), by node position in the planner's graph."""
        assembling = time.perf_counter()
        graph = self._graph
        terminal = np.unique(self._terminal)
        low, high = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
        pairs = np.unique(low * graph.n + high)
        entries = np.searchsorted(graph.entry_keys, pairs)
        rows, weights = graph.entry_rows[entries], graph.entry_weights[entries]
        bridge = rows >= len(self.network._edges)
        ends = np.concatenate([pairs[bridge] // graph.n, pairs[bridge] % graph.n])
        kept = np.union1d(np.union1d(terminal, self._root), ends)
        sub = _plan_subnetwork(self.network, rows, self._bridges, self.bridge_cost, self.weight, kept)
        terminals = self._terminals_table(pairs, weights, bridge, planned)
        if rollout is not None:
            terminals = _with_rollout(terminals, self._terminal, self._root, *rollout)
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
            "seconds": {**seconds, "assemble": time.perf_counter() - assembling},
        }
        return ConnectionPlan(
            network=contracted,
            edges=sub.edges_gdf.reset_index(drop=True),
            terminals=terminals,
            cost=float(weights.sum()),
            metadata=metadata,
        )

    def _planned_in_given_order(self, planned: np.ndarray) -> np.ndarray:
        """The planned terminals' positions in the order the terminals were given, the order that breaks ties."""
        given = self._terminal
        return given[np.isin(given, planned)]

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


# ----- lower bound


class _DualAscent:
    """Wong's dual ascent on a graph's arcs, one terminal's turn at a time.

    A turn grows the set of nodes that reach the terminal along arcs of
    reduced cost 0 like a reverse Dijkstra: arcs entering the set wait in a
    heap keyed by reduced cost plus the bound raised so far in the turn, so
    all their reduced costs fall together without being touched. When the
    turn ends, each arc's reduced cost is lowered by the raises made while it
    entered the set.
    """

    def __init__(self, graph: Graph, root: int):
        matrix = graph.matrix
        n = graph.n
        self.graph, self.root, self.n = graph, root, n
        self.indptr, self.indices = matrix.indptr.tolist(), matrix.indices.tolist()
        a, b = graph.entry_keys // n, graph.entry_keys % n
        self.reverse = np.searchsorted(graph.entry_keys, b * n + a).tolist()
        self.cost = graph.entry_weights.tolist()
        self.reduced = csr_matrix((graph.entry_weights.copy(), matrix.indices, matrix.indptr), shape=matrix.shape)
        self.member = [-1] * n
        self.turns = 0
        self.bound = 0.0

    def reduced_distances(self) -> np.ndarray:
        """Each node's distance from the root along arcs at their reduced costs."""
        return dijkstra(self.reduced, directed=True, indices=self.root)

    def turn(self, terminal: int, deadline: float, cap: float = np.inf) -> bool:
        """Raise the bound until the root reaches ``terminal`` at reduced cost 0, or the deadline; whether it does."""
        indptr, indices, reverse, cost, member = self.indptr, self.indices, self.reverse, self.cost, self.member
        turn = self.turns
        self.turns += 1
        offset = 0.0
        heap, waiting, joined = [], [], {}
        stack = [terminal]
        member[terminal] = turn
        rooted = False
        while True:
            while stack:
                x = stack.pop()
                joined[x] = offset
                if x == self.root:
                    rooted = True
                for e in range(indptr[x], indptr[x + 1]):
                    y = indices[e]
                    if member[y] == turn:
                        continue
                    arc = reverse[e]
                    if cost[arc] <= 0.0:
                        member[y] = turn
                        stack.append(y)
                    else:
                        key = cost[arc] + offset
                        heapq.heappush(heap, (key, arc, y))
                        waiting.append((arc, key, y))
            if rooted or not heap or time.perf_counter() >= deadline:
                break
            if heap[0][0] > cap:
                offset = cap
                break
            key, arc, y = heapq.heappop(heap)
            if member[y] == turn:
                continue
            offset = key
            member[y] = turn
            stack.append(y)
        changed = []
        for arc, key, tail in waiting:
            cost[arc] = max(0.0, key - min(joined.get(tail, offset), offset))
            changed.append(arc)
        if changed:
            self.reduced.data[changed] = [cost[arc] for arc in changed]
        self.bound += offset
        return rooted


def _dual_ascent(graph: Graph, root: int, terminals: np.ndarray, deadline: float) -> Tuple[float, bool]:
    """The dual ascent's bound, and whether every terminal reached the root before the deadline.

    Terminals take turns nearest the root first, at the current reduced
    costs: a turn raises the bound by exactly that distance. Distances are
    recomputed after every 2% of the waiting terminals have had their turn;
    on Rwanda with 1,000 terminals that was 7 times faster than after every
    turn, and the bound came out 0.5% higher, while on Kigali with 60
    terminals, every turn, it gave the highest bound.
    """
    ascent = _DualAscent(graph, root)
    waiting = np.asarray(terminals, dtype=np.int64)
    queue: list = []
    since = every = 0
    while len(waiting):
        if time.perf_counter() >= deadline:
            return ascent.bound, False
        if not queue or since >= every:
            distance = ascent.reduced_distances()[waiting]
            waiting = waiting[distance > 0]
            if not len(waiting):
                break
            queue = waiting[np.argsort(distance[distance > 0], kind="stable")].tolist()[::-1]
            since, every = 0, max(1, len(waiting) // _REFRESH_SHARE)
        terminal = queue.pop()
        if not ascent.turn(terminal, deadline):
            return ascent.bound, False
        since += 1
        waiting = waiting[waiting != terminal]
    return ascent.bound, True


# ----- local search


class _Tree:
    """A Steiner tree under local search, by node position in a graph that may hold a virtual root."""

    def __init__(self, graph: Graph, a: np.ndarray, b: np.ndarray, required: np.ndarray):
        self.graph = graph
        n = graph.n
        self.required_array = np.asarray(required, dtype=np.int64)
        self.required = np.zeros(n, dtype=bool)
        self.required[self.required_array] = True
        self._build(a, b)

    def _build(self, a: np.ndarray, b: np.ndarray) -> None:
        n = self.graph.n
        if len(a):
            a, b = _spanning_tree(self.graph, a, b)
            a, b = _prune_leaves(a, b, self.required_array, n)
        self.adjacent: Dict[int, set] = {}
        self.degree = np.zeros(n, dtype=np.int64)
        self.slot: Dict[Tuple[int, int], int] = {}
        size = max(16, 2 * len(a))
        self.lo = np.empty(size, dtype=np.int64)
        self.hi = np.empty(size, dtype=np.int64)
        self.on = np.zeros(size, dtype=bool)
        self.count = 0
        for x, y in zip(a.tolist(), b.tolist()):
            self._add(x, y)

    def _add(self, x: int, y: int) -> None:
        key = (x, y) if x < y else (y, x)
        if key in self.slot or x == y:
            return
        if self.count == len(self.lo):
            grow = len(self.lo)
            self.lo = np.concatenate([self.lo, np.empty(grow, dtype=np.int64)])
            self.hi = np.concatenate([self.hi, np.empty(grow, dtype=np.int64)])
            self.on = np.concatenate([self.on, np.zeros(grow, dtype=bool)])
        self.slot[key] = self.count
        self.lo[self.count], self.hi[self.count] = key
        self.on[self.count] = True
        self.count += 1
        self.adjacent.setdefault(x, set()).add(y)
        self.adjacent.setdefault(y, set()).add(x)
        self.degree[x] += 1
        self.degree[y] += 1

    def _remove(self, x: int, y: int) -> None:
        self.on[self.slot.pop((x, y) if x < y else (y, x))] = False
        for p, q in ((x, y), (y, x)):
            self.adjacent[p].discard(q)
            if not self.adjacent[p]:
                del self.adjacent[p]
        self.degree[x] -= 1
        self.degree[y] -= 1

    def edges(self) -> Tuple[np.ndarray, np.ndarray]:
        on = self.on[: self.count]
        return self.lo[: self.count][on], self.hi[: self.count][on]

    def weight(self, path: list) -> float:
        nodes = np.asarray(path, dtype=np.int64)
        return float(_pair_weights(self.graph, nodes[:-1], nodes[1:]).sum())

    def _is_key(self, x: int) -> bool:
        return bool(self.required[x]) or self.degree[x] >= 3

    def _walk(self, start: int, first: int) -> list:
        """The key path from key vertex ``start`` through its neighbour ``first``."""
        path, previous, current = [start, first], start, first
        while not self._is_key(current):
            following = next(x for x in self.adjacent[current] if x != previous)
            path.append(following)
            previous, current = current, following
        return path

    def key_paths(self) -> list:
        seen, paths = set(), []
        for k in list(self.adjacent):
            if not self._is_key(k):
                continue
            for first in self.adjacent[k]:
                if (k, first) in seen:
                    continue
                path = self._walk(k, first)
                seen.add((path[-1], path[-2]))
                paths.append(path)
        return paths

    def _parts(self, removed_edges: list, removed_nodes: list) -> Tuple[np.ndarray, np.ndarray]:
        """The tree's nodes without ``removed_nodes``, and the part each falls in without ``removed_edges``."""
        on = self.on[: self.count].copy()
        on[[self.slot[(x, y) if x < y else (y, x)] for x, y in removed_edges]] = False
        a, b = self.lo[: self.count][on], self.hi[: self.count][on]
        keep = self.degree > 0
        keep[removed_nodes] = False
        nodes = np.flatnonzero(keep)
        k = len(nodes)
        links = csr_matrix((np.ones(len(a)), (np.searchsorted(nodes, a), np.searchsorted(nodes, b))), shape=(k, k))
        return nodes, connected_components(links, directed=False)[1]

    def exchange(self, path: list) -> bool:
        """Replace a key path by the cheapest path between the two parts it joins, when cheaper."""
        pairs = list(zip(path[:-1], path[1:]))
        if any(((x, y) if x < y else (y, x)) not in self.slot for x, y in pairs):
            return False
        inner = path[1:-1]
        if any(self.degree[x] != 2 or self.required[x] for x in inner):
            return False
        # An earlier move may have left an end only continuing the path; extend it to the next key vertex.
        if not self._is_key(path[-1]):
            path = path[:-2] + self._walk(path[-2], path[-1])
        if not self._is_key(path[0]):
            path = (path[::-1][:-2] + self._walk(path[1], path[0]))[::-1]
        pairs = list(zip(path[:-1], path[1:]))
        inner = path[1:-1]
        beat = self.weight(path)
        if beat <= 0:
            return False
        nodes, labels = self._parts(pairs, inner)
        side_a = labels[np.searchsorted(nodes, path[0])]
        side_b = labels[np.searchsorted(nodes, path[-1])]
        size_a, size_b = int((labels == side_a).sum()), int((labels == side_b).sum())
        source, target = (side_a, side_b) if size_a <= size_b else (side_b, side_a)
        found = self._cheapest_link(nodes[labels == source], nodes[labels == target], beat)
        if found is None:
            return False
        for x, y in pairs:
            self._remove(x, y)
        for x, y in zip(found[:-1], found[1:]):
            self._add(x, y)
        return True

    def _cheapest_link(self, sources: np.ndarray, targets: np.ndarray, beat: float) -> Optional[list]:
        """The cheapest path from ``sources`` to ``targets`` lighter than ``beat``, from its target end."""
        distance, predecessors = dijkstra(
            self.graph.matrix, directed=True, indices=sources, min_only=True, limit=beat, return_predecessors=True
        )[:2]
        reach = distance[targets]
        best = int(np.argmin(reach))
        if not reach[best] < beat * (1 - _GAIN):
            return None
        is_target = np.zeros(self.graph.n, dtype=bool)
        is_target[targets] = True
        walk = [int(targets[best])]
        while predecessors[walk[-1]] >= 0:
            walk.append(int(predecessors[walk[-1]]))
        # Start at the target node nearest the sources, so the new path meets the target part once.
        first = max(i for i, x in enumerate(walk) if is_target[x])
        return walk[first:]

    def eliminate(self, vertex: int) -> bool:
        """Remove a branching node and its key paths, and reconnect the parts by their cheapest links, when cheaper.

        Links start only from the parts' nodes within the removed weight of
        the removed nodes, so the search stays near them. On Kigali that found
        plans within 0.01% of searching from every node, and on Rwanda it
        halved the time of a search.
        """
        if self.required[vertex] or self.degree[vertex] < 3:
            return False
        paths = [self._walk(vertex, first) for first in self.adjacent[vertex]]
        pairs = [(x, y) for path in paths for x, y in zip(path[:-1], path[1:])]
        removed = [vertex] + [x for path in paths for x in path[1:-1]]
        beat = sum(self.weight(path) for path in paths)
        if beat <= 0:
            return False
        nodes, labels = self._parts(pairs, removed)
        near = dijkstra(self.graph.matrix, directed=True, indices=removed, min_only=True, limit=beat)
        local = np.isfinite(near[nodes])
        nodes, labels = nodes[local], labels[local]
        distance, predecessors, nearest = dijkstra(
            self.graph.matrix, directed=True, indices=nodes, min_only=True, limit=beat, return_predecessors=True
        )
        part = np.full(self.graph.n, -1, dtype=np.int64)
        part[nodes] = labels
        reached = np.flatnonzero(nearest >= 0)
        matrix = self.graph.matrix
        starts, counts = matrix.indptr[reached], np.diff(matrix.indptr)[reached]
        entry = np.arange(counts.sum()) + np.repeat(starts - np.r_[0, np.cumsum(counts)[:-1]], counts)
        a, b, w = np.repeat(reached, counts), matrix.indices[entry], matrix.data[entry]
        pa, pb = part[nearest[a]], np.where(nearest[b] >= 0, part[np.maximum(nearest[b], 0)], -1)
        crossing = (pb >= 0) & (pa < pb)
        a, b, pa, pb = a[crossing], b[crossing], pa[crossing], pb[crossing]
        cost = distance[a] + w[crossing] + distance[b]
        count = int(labels.max()) + 1
        order = np.lexsort((cost, pb, pa))
        first = order[np.r_[True, (pa[order][1:] != pa[order][:-1]) | (pb[order][1:] != pb[order][:-1])]]
        if not len(first):
            return False
        link = np.where(cost[first] == 0, np.nextafter(0, 1), cost[first])
        tree = minimum_spanning_tree(csr_matrix((link, (pa[first], pb[first])), shape=(count, count))).tocoo()
        if tree.nnz != count - 1 or not tree.data.sum() < beat * (1 - _GAIN):
            return False
        low, high = np.minimum(tree.row, tree.col), np.maximum(tree.row, tree.col)
        chosen = first[np.searchsorted(pa[first] * count + pb[first], low * count + high)]
        fa, fb = _forest_edges(predecessors, np.concatenate([a[chosen], b[chosen]]))
        old_a, old_b = self.edges()
        keep = np.ones(len(old_a), dtype=bool)
        gone = {(x, y) if x < y else (y, x) for x, y in pairs}
        keep[[i for i, key in enumerate(zip(old_a.tolist(), old_b.tolist())) if key in gone]] = False
        self._build(
            np.concatenate([old_a[keep], fa, a[chosen]]),
            np.concatenate([old_b[keep], fb, b[chosen]]),
        )
        return True

    def search(self, deadline: float) -> Tuple[int, list]:
        """Apply both moves until neither improves the tree or the deadline passes; the moves, and seconds per pass."""
        moves, seconds = 0, []
        while time.perf_counter() < deadline:
            began = time.perf_counter()
            improved = False
            paths = self.key_paths()
            weights = [self.weight(path) for path in paths]
            for i in np.argsort(weights, kind="stable")[::-1]:
                if time.perf_counter() >= deadline:
                    break
                if self.exchange(paths[i]):
                    moves += 1
                    improved = True
            branching = sorted(x for x in self.adjacent if not self.required[x] and self.degree[x] >= 3)
            for vertex in branching:
                if time.perf_counter() >= deadline:
                    break
                if vertex in self.adjacent and self.eliminate(vertex):
                    moves += 1
                    improved = True
            seconds.append(time.perf_counter() - began)
            if not improved:
                break
        return moves, seconds


def _pair_weights(graph: Graph, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return graph.entry_weights[np.searchsorted(graph.entry_keys, a.astype(np.int64) * graph.n + b)]


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


def _shortest_path_edges(
    graph: Graph, terminal: np.ndarray, root: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Takahashi and Matsuyama's heuristic, with distances to the connected set updated incrementally.

    ``terminal`` is in the order that breaks ties; it holds only terminals the
    roots, or without roots its first terminal, can reach. One Dijkstra from
    the connected set gives every node its distance to it and the path there.
    After each connection, distances can only fall, and only for nodes nearer
    to the new path than to anything connected before. A Dijkstra from the
    path's new nodes that stops wherever a distance does not fall updates
    them, so its work is the region the path wins, not the whole network.
    It runs in Python, since scipy's Dijkstra cannot stop that way; on
    Rwanda's roads that was 15 times faster than scipy limited to the largest
    distance of a waiting terminal.

    Returns:
        The plan's edges, and the connected terminals with their order and
        marginal cost.
    """
    empty = np.empty(0, dtype=np.int64)
    n = graph.n
    order_of, marginal_of, done = [], [], []
    if len(root):
        start = root
    elif len(terminal):
        start = terminal[:1]
        done, order_of, marginal_of = [int(terminal[0])], [1], [0.0]
    else:
        return empty, empty, (empty, empty, np.empty(0))
    distance, predecessors = dijkstra(
        graph.matrix, directed=True, indices=start, min_only=True, return_predecessors=True
    )[:2]
    indptr, indices, weights = graph.matrix.indptr.tolist(), graph.matrix.indices.tolist(), graph.matrix.data.tolist()
    dist, pred = distance.tolist(), predecessors.tolist()
    connected = [False] * n
    for s in start.tolist():
        connected[s] = True
    waiting = [False] * n
    rank = {}
    for i, t in enumerate(terminal.tolist()):
        if not connected[t]:
            waiting[t] = True
            rank.setdefault(t, i)
    heap = [(dist[t], rank[t], t) for t in rank]
    heapq.heapify(heap)
    a_parts, b_parts = [], []
    step = len(done)
    while heap:
        d, _, t = heapq.heappop(heap)
        if not waiting[t] or d != dist[t]:
            continue
        if d == np.inf:
            break
        path = [t]
        while not connected[path[-1]]:
            path.append(pred[path[-1]])
        outward = path[::-1]
        last = 0.0
        for v in outward[1:]:
            if waiting[v]:
                step += 1
                done.append(v)
                order_of.append(step)
                marginal_of.append(dist[v] - last)
                last = dist[v]
        a_parts.append(outward[:-1])
        b_parts.append(outward[1:])
        front = []
        for v in outward[1:]:
            connected[v] = True
            waiting[v] = False
            dist[v] = 0.0
            pred[v] = -9999
            front.append((0.0, v))
        while front:
            dv, v = heapq.heappop(front)
            if dv > dist[v]:
                continue
            for e in range(indptr[v], indptr[v + 1]):
                u = indices[e]
                nd = dv + weights[e]
                if nd < dist[u]:
                    dist[u] = nd
                    pred[u] = v
                    heapq.heappush(front, (nd, u))
                    if waiting[u]:
                        heapq.heappush(heap, (nd, rank[u], u))
    if not a_parts:
        a, b = empty, empty
    else:
        a = np.fromiter((x for part in a_parts for x in part), dtype=np.int64)
        b = np.fromiter((x for part in b_parts for x in part), dtype=np.int64)
    return a, b, (np.array(done, dtype=np.int64), np.array(order_of, dtype=np.int64), np.array(marginal_of))


def _with_rollout(
    terminals: pd.DataFrame,
    position: np.ndarray,
    root: np.ndarray,
    done: np.ndarray,
    order_of: np.ndarray,
    marginal_of: np.ndarray,
) -> pd.DataFrame:
    """The terminals table with each terminal's order and marginal cost."""
    order = pd.array(np.full(len(position), pd.NA), dtype="Int64")
    marginal = np.full(len(position), np.nan)
    at_root = np.isin(position, root)
    order[at_root] = 0
    marginal[at_root] = 0.0
    lookup = pd.Series(np.arange(len(done)), index=done)
    which = lookup.reindex(position).to_numpy()
    found = ~np.isnan(which) & ~at_root
    picks = which[found].astype(np.int64)
    order[found] = order_of[picks]
    marginal[found] = marginal_of[picks]
    return terminals.assign(order=order, marginal_cost=marginal)


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
