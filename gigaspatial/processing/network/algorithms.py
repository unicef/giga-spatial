"""Network algorithms on scipy's sparse graph routines, with pandana as an optional backend."""

from typing import Any, List, NamedTuple, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _connected_components
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import minimum_spanning_tree as _minimum_spanning_tree
from scipy.spatial import cKDTree

from gigaspatial.processing.network.editing import _with_columns
from gigaspatial.processing.network.geodesy import WGS84, chord_to_meters, haversine, unit_vectors

# Largest result matrix shortest_path_lengths builds before asking the caller to batch.
MAX_MATRIX_BYTES = 2**30
# Largest block of Dijkstra distances (sources x nodes) held at once.
CHUNK_BYTES = 2**28
# pandana reports unreachable pairs as this length.
_PANDANA_UNREACHABLE = 4294967.0


class Graph(NamedTuple):
    """A sparse adjacency matrix and, for each stored entry, the edge row it came from."""

    matrix: csr_matrix
    entry_keys: np.ndarray
    entry_rows: np.ndarray
    n: int


class AlgorithmsMixin:
    """Network algorithms of ``SpatialNetwork``."""

    # ----- adjacency, cached per network

    def adjacency(self, weight: Optional[str] = None, directed: Optional[bool] = None) -> csr_matrix:
        """Sparse adjacency matrix by node position (see ``node_index``).

        Parallel edges keep the smallest weight. An undirected network, or
        ``directed=False``, gives a symmetric matrix.

        Args:
            weight: Edge column. The network's weight otherwise.
            directed: ``False`` ignores one-way restrictions of a directed network.

        Raises:
            ValueError: If the weight is missing or negative on some edge.
        """
        return self._graph(weight, directed).matrix

    def _graph(self, weight: Optional[str] = None, directed: Optional[bool] = None) -> Graph:
        weight = weight or self._weight
        directed = self._directed and (directed is None or bool(directed))
        cache = self.__dict__.setdefault("_graph_cache", {})
        if (weight, directed) not in cache:
            cache[(weight, directed)] = self._build_graph(weight, directed)
        return cache[(weight, directed)]

    def _build_graph(self, weight: str, directed: bool) -> Graph:
        values = self._edge_weights(weight)
        pu, pv = self._edge_positions
        rows = np.arange(len(values))
        if directed:
            a, b, w, r = pu.astype(np.int64), pv.astype(np.int64), values, rows
        else:
            a = np.concatenate([pu, pv]).astype(np.int64)
            b = np.concatenate([pv, pu]).astype(np.int64)
            w = np.concatenate([values, values])
            r = np.concatenate([rows, rows])
        n = len(self._nodes)
        keys = a * n + b
        order = np.lexsort((w, keys))
        first = np.ones(len(order), dtype=bool)
        first[1:] = keys[order][1:] != keys[order][:-1]
        chosen = order[first]
        entry_keys = keys[chosen]
        matrix = csr_matrix((w[chosen], (entry_keys // n, entry_keys % n)), shape=(n, n))
        return Graph(matrix, entry_keys, r[chosen], n)

    def _transposed(self, weight: Optional[str], directed: Optional[bool]) -> csr_matrix:
        graph = self._graph(weight, directed)
        if not (self._directed and (directed is None or directed)):
            return graph.matrix
        cache = self.__dict__.setdefault("_transposed_cache", {})
        key = (weight or self._weight, True)
        if key not in cache:
            cache[key] = graph.matrix.T.tocsr()
        return cache[key]

    def _edge_weights(self, weight: str) -> np.ndarray:
        if weight not in self._edges.columns:
            raise ValueError(f"The weight column {weight!r} is not an edge column")
        values = pd.to_numeric(self._edges[weight], errors="coerce").to_numpy(dtype=np.float64)
        missing = np.isnan(values)
        if missing.any():
            raise ValueError(f"{int(missing.sum())} edges have no {weight!r}; set it before using it as a weight")
        if (values < 0).any():
            raise ValueError(f"Weights must not be negative; {int((values < 0).sum())} edges have a negative {weight!r}")
        return values

    # ----- pandana

    def to_pandana(self, weight: Optional[str] = None):
        """A pandana network with ``twoway`` set to whether the network is undirected.

        Built once per weight and cached. Needs ``pip install 'giga-spatial[pandana]'``.

        Raises:
            ValueError: If node IDs mix integers and text, which pandana cannot index.
        """
        return self._pandana(weight, twoway=not self._directed)

    def _pandana(self, weight: Optional[str], twoway: bool):
        weight = weight or self._weight
        cache = self.__dict__.setdefault("_pandana_cache", {})
        if (weight, twoway) in cache:
            return cache[(weight, twoway)]
        try:
            import pandana
        except ImportError as error:
            raise ImportError(
                "The pandana backend requires 'pandana'. Install it with: pip install 'giga-spatial[pandana]'"
            ) from error
        if pd.api.types.infer_dtype(self._nodes.index, skipna=False) not in ("integer", "string"):
            raise ValueError("pandana needs node IDs that are all integers or all text")
        values = self._edge_weights(weight)
        network = pandana.Network(
            self._nodes["x"],
            self._nodes["y"],
            self._edges["u"],
            self._edges["v"],
            pd.DataFrame({weight: values}),
            twoway=twoway,
        )
        cache[(weight, twoway)] = network
        return network

    # ----- shortest paths

    def shortest_path_lengths(
        self,
        origins: Any,
        destinations: Any = None,
        pairwise: bool = False,
        weight: Optional[str] = None,
        backend: str = "scipy",
        directed: Optional[bool] = None,
    ) -> Any:
        """Shortest path lengths between nodes.

        Args:
            origins: Origin node IDs.
            destinations: Destination node IDs. All nodes otherwise.
            pairwise: Pair ``origins[i]`` with ``destinations[i]`` instead of
                building the full matrix.
            weight: Edge column to add up. The network's weight otherwise.
            backend: ``"scipy"``, or ``"pandana"`` for very many pairs once its
                contraction hierarchy is built. pandana stores weights with
                limited precision, so its lengths can differ by a fraction of
                a meter.
            directed: ``False`` ignores one-way restrictions.

        Returns:
            With ``pairwise``, an array with one length per pair. Otherwise a
            DataFrame indexed by origin with one column per destination.
            Unreachable pairs are ``inf``.

        Raises:
            ValueError: If the matrix would exceed ``MAX_MATRIX_BYTES``; split
                the origins into batches.
            KeyError: If a node ID is not in the network.
        """
        origin = self._positions(origins)
        target = np.arange(len(self._nodes)) if destinations is None else self._positions(destinations)
        if pairwise and len(origin) != len(target):
            raise ValueError("pairwise needs as many destinations as origins")
        if not pairwise:
            size = len(origin) * len(target) * 8
            if size > MAX_MATRIX_BYTES:
                raise ValueError(
                    f"A {len(origin):,} by {len(target):,} matrix needs {size / 2**30:.1f} GiB; "
                    f"split the origins into batches, or use pairwise=True"
                )
        if backend == "pandana":
            values = self._pandana_lengths(origin, target, pairwise, weight, directed)
        elif backend == "scipy":
            values = self._scipy_lengths(origin, target, pairwise, weight, directed)
        else:
            raise ValueError(f'backend must be "scipy" or "pandana", got {backend!r}')
        if pairwise:
            return values
        ids = self._nodes.index
        return pd.DataFrame(values, index=ids.take(origin), columns=ids.take(target))

    def _scipy_lengths(self, origin, target, pairwise, weight, directed) -> np.ndarray:
        matrix = self._graph(weight, directed).matrix
        sources, inverse = np.unique(origin, return_inverse=True)
        out = np.empty(len(origin)) if pairwise else np.empty((len(origin), len(target)))
        for start, block in _dijkstra_blocks(matrix, sources):
            in_block = (inverse >= start) & (inverse < start + len(block))
            local = inverse[in_block] - start
            if pairwise:
                out[in_block] = block[local, target[in_block]]
            else:
                out[in_block] = block[local][:, target]
        return out

    def _pandana_lengths(self, origin, target, pairwise, weight, directed) -> np.ndarray:
        twoway = not (self._directed and (directed is None or directed))
        network = self._pandana(weight, twoway)
        ids = self._nodes.index
        if pairwise:
            a, b = origin, target
        else:
            a, b = np.repeat(origin, len(target)), np.tile(target, len(origin))
        values = np.asarray(network.shortest_path_lengths(list(ids.take(a)), list(ids.take(b))), dtype=np.float64)
        values[values >= _PANDANA_UNREACHABLE] = np.inf
        return values if pairwise else values.reshape(len(origin), len(target))

    def shortest_paths(
        self,
        origins: Any,
        destinations: Any,
        pairwise: bool = True,
        weight: Optional[str] = None,
        geometry: bool = False,
        directed: Optional[bool] = None,
    ) -> Any:
        """Shortest paths between nodes, as node sequences or lines.

        Args:
            origins, destinations: Node IDs. With ``pairwise``, ``origins[i]``
                goes to ``destinations[i]``; otherwise every origin goes to
                every destination, origin by origin.
            weight: Edge column to minimise. The network's weight otherwise.
            geometry: Return routes as lines with their lengths.
            directed: ``False`` ignores one-way restrictions.

        Returns:
            A list of node ID lists, empty where there is no path. With
            ``geometry``, a GeoDataFrame with ``origin``, ``destination``,
            ``length``, ``nodes``, and a LineString ``geometry`` (missing when
            there is no path or the origin is the destination).
        """
        origin = self._positions(origins)
        target = self._positions(destinations)
        if pairwise:
            if len(origin) != len(target):
                raise ValueError("pairwise needs as many destinations as origins")
        else:
            origin, target = np.repeat(origin, len(target)), np.tile(target, len(origin))
        graph = self._graph(weight, directed)
        sources, inverse = np.unique(origin, return_inverse=True)
        paths: List[List[int]] = [[] for _ in origin]
        lengths = np.full(len(origin), np.inf)
        for start, block, predecessors in _dijkstra_blocks(graph.matrix, sources, predecessors=True):
            for i in np.flatnonzero((inverse >= start) & (inverse < start + len(block))):
                row = inverse[i] - start
                lengths[i] = block[row, target[i]]
                paths[i] = _walk(predecessors[row], sources[inverse[i]], target[i])
        ids = self._nodes.index
        node_paths = [list(ids.take(p)) for p in paths]
        if not geometry:
            return node_paths
        lines = [self._path_line(graph, p) for p in paths]
        return gpd.GeoDataFrame(
            {
                "origin": ids.take(origin),
                "destination": ids.take(target),
                "length": lengths,
                "nodes": node_paths,
            },
            geometry=lines,
            crs=WGS84,
        )

    def _path_line(self, graph: Graph, path: List[int]) -> Optional[shapely.LineString]:
        if len(path) < 2:
            return None
        a = np.asarray(path[:-1], dtype=np.int64)
        b = np.asarray(path[1:], dtype=np.int64)
        rows = graph.entry_rows[np.searchsorted(graph.entry_keys, a * graph.n + b)]
        pu, _ = self._edge_positions
        pieces = []
        for row, start in zip(rows, a):
            coords = shapely.get_coordinates(self._filled_geometries[row])
            if pu[row] != start:
                coords = coords[::-1]
            pieces.append(coords if not pieces else coords[1:])
        return shapely.LineString(np.concatenate(pieces))

    def reachable(
        self, origins: Any, max_distance: float, weight: Optional[str] = None, directed: Optional[bool] = None
    ) -> pd.DataFrame:
        """Every node within a network distance of each origin.

        Returns:
            A long table with ``origin``, ``node_id``, and ``distance``, sorted
            by origin and distance; each distinct origin appears once, reaching
            itself at distance 0.
        """
        matrix = self._graph(weight, directed).matrix
        sources = np.unique(self._positions(origins))
        parts = []
        for start, block in _dijkstra_blocks(matrix, sources, limit=max_distance):
            row, node = np.nonzero(np.isfinite(block))
            parts.append((sources[start + row], node, block[row, node]))
        origin = np.concatenate([p[0] for p in parts]) if parts else np.empty(0, dtype=np.int64)
        node = np.concatenate([p[1] for p in parts]) if parts else np.empty(0, dtype=np.int64)
        distance = np.concatenate([p[2] for p in parts]) if parts else np.empty(0)
        order = np.lexsort((distance, origin))
        ids = self._nodes.index
        return pd.DataFrame(
            {"origin": ids.take(origin[order]), "node_id": ids.take(node[order]), "distance": distance[order]}
        )

    def nearest_by_network(
        self,
        origins: Any,
        targets: Any,
        k: int = 1,
        max_distance: Optional[float] = None,
        weight: Optional[str] = None,
        directed: Optional[bool] = None,
    ) -> pd.DataFrame:
        """The nearest targets to each origin by network distance, such as each school's nearest fiber node.

        Args:
            origins: Origin node IDs.
            targets: Target node IDs.
            k: Number of targets per origin. ``k=1`` is a single multi-source
                search from all targets.
            max_distance: Ignore targets farther than this.

        Returns:
            A table with ``origin``, ``target``, and ``distance``: one row per
            origin for ``k=1`` (``target`` missing and ``distance`` ``inf`` when
            none is reachable), and up to ``k`` rows per origin otherwise,
            nearest first.
        """
        origin = self._positions(origins)
        target = np.unique(self._positions(targets))
        limit = np.inf if max_distance is None else float(max_distance)
        ids = self._nodes.index
        if k == 1:
            dist, _, source = dijkstra(
                self._transposed(weight, directed),
                directed=True,
                indices=target,
                min_only=True,
                return_predecessors=True,
                limit=limit,
            )
            found = source[origin] >= 0
            nearest = pd.Series(ids.take(np.where(found, source[origin], 0)), dtype=object)
            nearest[~found] = None
            return pd.DataFrame({"origin": ids.take(origin), "target": nearest.to_numpy(), "distance": dist[origin]})
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        matrix = self._graph(weight, directed).matrix
        sources, inverse = np.unique(origin, return_inverse=True)
        best = {}
        for start, block in _dijkstra_blocks(matrix, sources, limit=limit):
            to_targets = block[:, target]
            take = min(k, len(target))
            nearest = np.argsort(to_targets, axis=1, kind="stable")[:, :take]
            for row in range(len(block)):
                columns = nearest[row][np.isfinite(to_targets[row, nearest[row]])]
                best[start + row] = (target[columns], to_targets[row, columns])
        rows = [(o, t, d) for i, o in enumerate(origin) for t, d in zip(*best[inverse[i]])]
        return pd.DataFrame(
            {
                "origin": ids.take([r[0] for r in rows]),
                "target": ids.take([r[1] for r in rows]),
                "distance": np.array([r[2] for r in rows], dtype=np.float64),
            }
        )

    def count_within(
        self,
        origins: Any,
        targets: Any,
        max_distance: float,
        weight: Optional[str] = None,
        directed: Optional[bool] = None,
    ) -> pd.Series:
        """How many targets each origin reaches within a network distance, such as clinics within 5 km by road.

        Targets are counted once each; an origin that is also a target counts itself.

        Returns:
            Counts indexed by origin ID, in the order given.
        """
        origin = self._positions(origins)
        target = np.unique(self._positions(targets))
        sources, inverse = np.unique(origin, return_inverse=True)
        counts = np.zeros(len(sources), dtype=np.int64)
        if len(target) < len(sources):
            matrix = self._transposed(weight, directed)
            for _, block in _dijkstra_blocks(matrix, target, limit=max_distance):
                counts += np.isfinite(block[:, sources]).sum(axis=0)
        else:
            matrix = self._graph(weight, directed).matrix
            for start, block in _dijkstra_blocks(matrix, sources, limit=max_distance):
                counts[start : start + len(block)] = np.isfinite(block[:, target]).sum(axis=1)
        return pd.Series(counts[inverse], index=self._nodes.index.take(origin), name="count")

    # ----- structure

    def connected_components(self, strong: bool = False) -> pd.Series:
        """Component label of each node; label 0 is the largest component, then by size.

        Args:
            strong: In a directed network, components in which every node can
                reach every other along one-way roads. Weak components ignore
                direction. The same for undirected networks.
        """
        labels = self._component_labels(strong)
        return pd.Series(labels, index=self._nodes.index, name="component")

    def _component_labels(self, strong: bool = False) -> np.ndarray:
        matrix = self._graph("length", None).matrix
        _, labels = _connected_components(
            matrix, directed=self._directed, connection="strong" if strong else "weak"
        )
        sizes = np.bincount(labels)
        rank = np.empty(len(sizes), dtype=np.int64)
        rank[np.argsort(-sizes, kind="stable")] = np.arange(len(sizes))
        return rank[labels]

    def largest_component(self, strong: bool = False) -> "SpatialNetwork":
        """The network's largest connected component."""
        return self.filter_nodes(mask=self._component_labels(strong) == 0)

    def minimum_spanning_tree(self, weight: Optional[str] = None) -> "SpatialNetwork":
        """An undirected network of the edges that join every component at the smallest total weight.

        Directions are ignored, all nodes are kept (a forest when the network is
        disconnected), and keys become 0.
        """
        graph = self._graph(weight, False)
        matrix = graph.matrix.copy()
        # scipy drops zero-weight edges from the tree; the smallest positive float keeps them.
        matrix.data[matrix.data == 0] = np.nextafter(0, 1)
        tree = _minimum_spanning_tree(matrix).tocoo()
        rows = np.sort(graph.entry_rows[np.searchsorted(graph.entry_keys, tree.row.astype(np.int64) * graph.n + tree.col)])
        pu, pv = self._edge_positions
        edges = _with_columns(self._edges.iloc[rows], {"key": np.zeros(len(rows), dtype=np.int16)})
        edges.index = pd.RangeIndex(len(rows))
        return self._derive(self._nodes, edges, (pu[rows], pv[rows]), directed=False)

    def connect_components(self, max_distance: Optional[float] = None) -> "SpatialNetwork":
        """Join disconnected parts with ``component_link`` edges between their nearest nodes.

        The links form a minimum spanning tree over the components by
        great-circle distance, found with Borůvka's method. In a directed
        network each link is added in both directions.

        Args:
            max_distance: Meters. Parts farther apart than this stay apart.

        Returns:
            The network with the links added, or the same network when it is
            already connected.
        """
        labels = self._component_labels(strong=False)
        links = _component_links(
            unit_vectors(self._nodes["y"].to_numpy(), self._nodes["x"].to_numpy()), labels, max_distance
        )
        if not links:
            return self
        a = np.array([link[0] for link in links])
        b = np.array([link[1] for link in links])
        if self._directed:
            a, b = np.concatenate([a, b]), np.concatenate([b, a])
        x = self._nodes["x"].to_numpy()
        y = self._nodes["y"].to_numpy()
        length = haversine(y[a], x[a], y[b], x[b])
        ids = self._nodes.index
        new_weight = {} if self._weight == "length" else {self._weight: length}
        return self.add_weighted_edges(ids.take(a), ids.take(b), length, kind="component_link", **new_weight)


# ----- helpers


def _dijkstra_blocks(matrix: csr_matrix, sources: np.ndarray, limit: float = np.inf, predecessors: bool = False):
    """Dijkstra from the sources in blocks of rows small enough to hold in memory."""
    n = matrix.shape[0]
    per_block = max(1, CHUNK_BYTES // (8 * max(n, 1) * (2 if predecessors else 1)))
    for start in range(0, len(sources), per_block):
        chunk = sources[start : start + per_block]
        result = dijkstra(matrix, directed=True, indices=chunk, limit=limit, return_predecessors=predecessors)
        if predecessors:
            yield start, np.atleast_2d(result[0]), np.atleast_2d(result[1])
        else:
            yield start, np.atleast_2d(result)


def _walk(predecessors: np.ndarray, source: int, target: int) -> List[int]:
    if source == target:
        return [int(source)]
    if predecessors[target] < 0:
        return []
    path = [int(target)]
    node = target
    while node != source:
        node = predecessors[node]
        path.append(int(node))
    return path[::-1]


_OWN_TREE_SIZE = 2_000


def _component_links(points: np.ndarray, labels: np.ndarray, max_distance: Optional[float]) -> List[Tuple[int, int]]:
    """Borůvka over components: each round joins every component to its nearest other component."""
    parent = np.arange(labels.max() + 1)

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    links = []
    tree = cKDTree(points)
    while True:
        groups = np.array([find(c) for c in range(len(parent))])[labels]
        members = pd.Series(np.arange(len(groups))).groupby(groups).indices
        if len(members) < 2:
            return links
        candidates = []
        for group, nodes in members.items():
            if len(nodes) > _OWN_TREE_SIZE or len(nodes) * 2 > len(points):
                outside = np.flatnonzero(groups != group)
                chord, j = cKDTree(points[nodes]).query(points[outside], k=1)
                best = int(np.argmin(chord))
                pair, chord_best = (int(nodes[j[best]]), int(outside[best])), float(chord[best])
            else:
                pair, chord_best = _nearest_outside(tree, points, groups, group, nodes)
            distance = float(chord_to_meters(chord_best))
            if max_distance is None or distance <= max_distance:
                candidates.append((distance, pair))
        merged = False
        for _, (p, q) in sorted(candidates):
            gp, gq = find(labels[p]), find(labels[q])
            if gp != gq:
                parent[gp] = gq
                links.append((p, q))
                merged = True
        if not merged:
            return links


def _nearest_outside(
    tree: cKDTree, points: np.ndarray, groups: np.ndarray, group: int, nodes: np.ndarray
) -> Tuple[Tuple[int, int], float]:
    """The closest pair between a component's nodes and any node outside it.

    Grows k only for nodes whose k-th neighbour is still closer than the best
    pair found, since only those can still improve on it.
    """
    n = len(points)
    best, pair = np.inf, None
    pending = nodes
    k = min(16, n)
    while len(pending):
        chord, j = tree.query(points[pending], k=k)
        chord, j = chord.reshape(len(pending), -1), j.reshape(len(pending), -1)
        outside = (groups[np.minimum(j, n - 1)] != group) & np.isfinite(chord)
        masked = np.where(outside, chord, np.inf)
        row = int(np.argmin(masked.min(axis=1)))
        if masked[row].min() < best:
            best = float(masked[row].min())
            pair = (int(pending[row]), int(j[row, int(np.argmin(masked[row]))]))
        if k >= n:
            break
        unresolved = ~outside.any(axis=1) & (chord[:, -1] < best)
        pending = pending[unresolved]
        k = min(2 * k, n)
    return pair, best
