"""Reducing spatial networks: joining chains of edges, and keeping only what joins a set of points."""

from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree

from gigaspatial.processing.network._arrays import as_positions, next_keys
from gigaspatial.processing.network.algorithms import Graph, _dijkstra_blocks
from gigaspatial.processing.network.attributes import _BOOLEAN_RULES, normalize_attributes
from gigaspatial.processing.network.editing import _concat_tables, _with_columns

_PROTECTED_KINDS = ("point", "split")
_STRUCTURAL = ("u", "v", "key", "kind", "geometry")


class ReductionMixin:
    """Reduction methods of ``SpatialNetwork``."""

    def simplify(self, keep: Any = None, sum_columns: Iterable[str] = ("length",)) -> "SpatialNetwork":
        """Remove nodes that only continue a road, joining the edges on either side.

        A node is removed when it has exactly two neighbours and its edges
        continue through it: one edge in and one out of a one-way road, or both
        directions of a two-way road (in an undirected network, exactly two
        edges). This is the rule osmnx uses, except that osmnx also removes
        nodes between duplicated parallel ways and drops one of the ways, and
        deletes rings with no junction at all; here those nodes stay, and a
        ring keeps one node. Nodes of kind ``point`` or ``split`` and nodes in
        ``keep`` are never removed.

        Joined edges get:

        - the sum of ``sum_columns`` and of the network's weight column;
        - one geometry along the joined edges;
        - text joined with ``;`` where values differ, without repeats;
        - ``osmid`` and other list columns concatenated, without repeats;
        - ``oneway`` true only if every part is one-way, ``reversed`` true if
          any part is, other booleans true only if all parts are;
        - the length-weighted mean of other numeric columns;
        - the kind of their longest part.

        Each direction of a two-way road is joined separately, so it stays
        two-way when both directions can be joined. Joined edges always get a
        geometry, as in osmnx, so a network without edge geometry gains a
        ``geometry`` column that is missing for the edges left as they were.

        Args:
            keep: Node IDs to keep.
            sum_columns: Edge columns to add up.

        Returns:
            The simplified network, or the same network when no node can be removed.
        """
        n = len(self._nodes)
        m = len(self._edges)
        if m == 0:
            return self
        protected = np.array(self._nodes["kind"].isin(_PROTECTED_KINDS), dtype=bool)
        if keep is not None:
            protected[self._positions(keep)] = True
        pu, pv = self._edge_positions
        rows = np.arange(m)
        if self._directed:
            hu, hv, hrow, flipped = pu.astype(np.int64), pv.astype(np.int64), rows, np.zeros(m, dtype=bool)
        else:
            hu = np.concatenate([pu, pv]).astype(np.int64)
            hv = np.concatenate([pv, pu]).astype(np.int64)
            hrow = np.concatenate([rows, rows])
            flipped = np.concatenate([np.zeros(m, dtype=bool), np.ones(m, dtype=bool)])

        removable, out_first, out_second = _interstitial(hu, hv, n)
        removable &= ~protected
        removable = _break_rings(hu, hv, removable, n)
        if not removable.any():
            return self

        chain_edges, starts = _chains(hu, hv, removable, out_first, out_second)
        if not self._directed:
            first_row = hrow[chain_edges[starts]]
            last_row = hrow[chain_edges[np.r_[starts[1:], len(chain_edges)] - 1]]
            keep_chain = first_row < last_row
            chain_edges, starts = _select_chains(chain_edges, starts, keep_chain)

        in_chain = np.zeros(m, dtype=bool)
        in_chain[hrow[chain_edges]] = True
        untouched = np.flatnonzero(~in_chain)

        edges = self._edges
        if "osmid" in edges.columns and not isinstance(edges["osmid"].dtype, pd.ArrowDtype):
            int_lists = ("osmid",)
        else:
            int_lists = ()
        edges = normalize_attributes(edges, int_lists=int_lists, skip=("u", "v", "key", "length", "kind", "geometry"))
        if "geometry" not in edges.columns:
            edges = _with_columns(edges, {"geometry": np.full(m, None, dtype=object)})
        summed = set(sum_columns) | {"length", self._weight}

        chain_rows = hrow[chain_edges]
        ends = np.r_[starts[1:], len(chain_edges)] - 1
        first_node, last_node = hu[chain_edges[starts]], hv[chain_edges[ends]]
        lengths = edges["length"].to_numpy(dtype=np.float64)[chain_rows]
        joined = {}
        for name in edges.columns:
            if name in ("u", "v", "key"):
                continue
            if name == "geometry":
                xy = self._nodes[["x", "y"]].to_numpy()
                joined[name] = _joined_geometry(
                    edges["geometry"].to_numpy(dtype=object)[chain_rows],
                    flipped[chain_edges],
                    xy[hu[chain_edges]],
                    xy[hv[chain_edges]],
                    starts,
                )
            elif name == "kind":
                joined[name] = _longest_value(edges[name], chain_rows, lengths, starts)
            else:
                joined[name] = _aggregate(name, edges[name], chain_rows, lengths, starts, name in summed)

        keep_nodes = ~removable
        position = np.cumsum(keep_nodes) - 1
        old_a, old_b = pu[untouched], pv[untouched]
        new_a, new_b = first_node, last_node
        if not self._directed:
            old_a, old_b = np.minimum(old_a, old_b), np.maximum(old_a, old_b)
            new_a, new_b = np.minimum(new_a, new_b), np.maximum(new_a, new_b)
        keys = next_keys(old_a, old_b, edges["key"].to_numpy()[untouched], new_a, new_b)
        ids = self._nodes.index
        new_edges = pd.DataFrame(
            {
                "u": ids.take(first_node),
                "v": ids.take(last_node),
                "key": keys.astype(np.int16),
                **joined,
            },
            copy=False,
        )[list(edges.columns)]
        old_edges = edges.iloc[untouched]
        table = _concat_tables([old_edges, new_edges], ignore_index=True)
        table["key"] = table["key"].to_numpy().astype(np.int16)
        nodes = self._nodes[keep_nodes]
        n_kept = len(nodes)
        positions = (
            as_positions(np.concatenate([position[pu[untouched]], position[first_node]]), n_kept),
            as_positions(np.concatenate([position[pv[untouched]], position[last_node]]), n_kept),
        )
        return self._derive(nodes, table, positions)

    def connecting_subnetwork(
        self,
        terminals: Any,
        sources: Any = None,
        method: str = "steiner",
        weight: Optional[str] = None,
    ) -> "SpatialNetwork":
        """The part of the network that joins a set of points, contracted to the points and branching nodes.

        Directions are ignored. The result is undirected; it keeps the
        terminals, the sources, and the nodes where paths branch, with joined
        edges as :meth:`simplify` makes them, so lengths are added up and
        geometries joined.

        Args:
            terminals: Node IDs to join, such as schools.
            sources: Node IDs the terminals connect to, such as fiber nodes.
            method: ``"steiner"``: without ``sources``, a tree joining all
                terminals with close to the smallest total weight (Mehlhorn's
                2-approximation of the Steiner tree); with ``sources``, each
                terminal's shortest path to its nearest source. ``"all_pairs"``:
                the shortest paths between every terminal and every terminal
                and source; their number grows with the square of the
                terminals.
            weight: Edge column to minimise. The network's weight otherwise.

        Returns:
            An undirected network. Terminals that cannot reach the others are
            kept without edges.
        """
        graph = self._graph(weight, False)
        terminal = np.unique(self._positions(terminals))
        source = np.unique(self._positions(sources)) if sources is not None else np.empty(0, dtype=np.int64)
        if method == "steiner":
            if len(source):
                a, b = _nearest_source_edges(graph, terminal, source)
            else:
                a, b = _steiner_edges(graph, terminal)
        elif method == "all_pairs":
            a, b = _all_pairs_edges(graph, terminal, np.union1d(terminal, source))
        else:
            raise ValueError(f'method must be "steiner" or "all_pairs", got {method!r}')
        kept = np.union1d(terminal, source)
        sub = self._edge_subnetwork(graph, a, b, kept)
        summed = ("length",) if weight is None else ("length", weight)
        return sub.simplify(keep=self._nodes.index.take(kept), sum_columns=summed)

    def _edge_subnetwork(self, graph: Graph, a: np.ndarray, b: np.ndarray, kept: np.ndarray) -> "SpatialNetwork":
        """An undirected network of the cheapest edge behind each (a, b) pair, plus the kept nodes."""
        low, high = np.minimum(a, b), np.maximum(a, b)
        pairs = np.unique(low.astype(np.int64) * graph.n + high)
        rows = np.sort(graph.entry_rows[np.searchsorted(graph.entry_keys, pairs)])
        pu, pv = self._edge_positions
        members = np.union1d(np.union1d(pu[rows], pv[rows]), kept)
        nodes = self._nodes.iloc[members]
        edges = _with_columns(self._edges.iloc[rows], {"key": np.zeros(len(rows), dtype=np.int16)})
        edges.index = pd.RangeIndex(len(rows))
        positions = (
            as_positions(np.searchsorted(members, pu[rows]), len(members)),
            as_positions(np.searchsorted(members, pv[rows]), len(members)),
        )
        return type(self)._from_tables(
            nodes, edges, directed=False, weight=self._weight, metadata=self._metadata, positions=positions
        )


# ----- simplify helpers


def _nth(order: np.ndarray, start: np.ndarray, degree: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(degree), -1, dtype=np.int64)
    has = degree > k
    out[has] = order[start[has] + k]
    return out


def _interstitial(hu: np.ndarray, hv: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes that only continue a road, and their first and second outgoing edges."""
    out_degree = np.bincount(hu, minlength=n)
    in_degree = np.bincount(hv, minlength=n)
    loops = np.zeros(n, dtype=bool)
    loops[hu[hu == hv]] = True
    in_order = np.argsort(hv, kind="stable")
    out_order = np.argsort(hu, kind="stable")
    in_start = np.concatenate([[0], np.cumsum(in_degree)[:-1]])
    out_start = np.concatenate([[0], np.cumsum(out_degree)[:-1]])
    in1, in2 = _nth(in_order, in_start, in_degree, 0), _nth(in_order, in_start, in_degree, 1)
    out1, out2 = _nth(out_order, out_start, out_degree, 0), _nth(out_order, out_start, out_degree, 1)
    pred1, pred2 = np.where(in1 >= 0, hu[in1], -1), np.where(in2 >= 0, hu[in2], -2)
    succ1, succ2 = np.where(out1 >= 0, hv[out1], -3), np.where(out2 >= 0, hv[out2], -4)
    one_way = (in_degree == 1) & (out_degree == 1) & (pred1 != succ1)
    two_way = (
        (in_degree == 2)
        & (out_degree == 2)
        & (pred1 != pred2)
        & (((pred1 == succ1) & (pred2 == succ2)) | ((pred1 == succ2) & (pred2 == succ1)))
    )
    return (one_way | two_way) & ~loops, out1, out2


def _break_rings(hu: np.ndarray, hv: np.ndarray, removable: np.ndarray, n: int) -> np.ndarray:
    """Keep one node of every ring made only of removable nodes."""
    inner = removable[hu] & removable[hv]
    if not inner.any():
        return removable
    links = csr_matrix((np.ones(int(inner.sum())), (hu[inner], hv[inner])), shape=(n, n))
    count, label = connected_components(links, directed=True, connection="weak")
    touches = np.zeros(count, dtype=bool)
    boundary = removable[hu] != removable[hv]
    touches[label[np.where(removable[hu], hu, hv)[boundary]]] = True
    candidates = np.flatnonzero(removable)
    first = np.full(count, n, dtype=np.int64)
    np.minimum.at(first, label[candidates], candidates)
    rings = np.flatnonzero(~touches & (first < n))
    removable = removable.copy()
    removable[first[rings]] = False
    return removable


def _chains(
    hu: np.ndarray, hv: np.ndarray, removable: np.ndarray, out1: np.ndarray, out2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Edges of every chain through removable nodes, ordered along each chain, and where each chain starts."""
    h = len(hu)
    following = np.full(h, -1, dtype=np.int64)
    into = np.flatnonzero(removable[hv])
    node = hv[into]
    first, second = out1[node], out2[node]
    following[into] = np.where((second < 0) | (hv[first] != hu[into]), first, second)
    previous = np.full(h, -1, dtype=np.int64)
    linked = following >= 0
    previous[following[linked]] = np.flatnonzero(linked)

    head = np.where(previous >= 0, previous, np.arange(h))
    while True:
        jumped = head[head]
        if np.array_equal(jumped, head):
            break
        head = jumped
    rank = (previous >= 0).astype(np.int64)
    pointer = previous.copy()
    while True:
        active = np.flatnonzero(pointer >= 0)
        if not len(active):
            break
        target = pointer[active]
        rank_next = rank.copy()
        rank_next[active] += rank[target]
        pointer_next = pointer.copy()
        pointer_next[active] = pointer[target]
        rank, pointer = rank_next, pointer_next

    in_chain = np.flatnonzero(removable[hu] | removable[hv])
    order = in_chain[np.lexsort((rank[in_chain], head[in_chain]))]
    starts = np.flatnonzero(np.r_[True, head[order][1:] != head[order][:-1]])
    return order, starts


def _select_chains(edges: np.ndarray, starts: np.ndarray, keep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sizes = np.diff(np.r_[starts, len(edges)])
    chosen = np.repeat(keep, sizes)
    kept_sizes = sizes[keep]
    return edges[chosen], np.concatenate([[0], np.cumsum(kept_sizes)[:-1]]).astype(np.int64)


def _joined_geometry(
    geoms: np.ndarray, flipped: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray, starts: np.ndarray
) -> np.ndarray:
    """One line per chain: each piece's own geometry, or its two endpoints when it has none."""
    pieces = len(geoms)
    has = ~pd.isna(geoms)
    oriented = geoms[has]
    reverse = flipped[has]
    if reverse.any():
        oriented = oriented.copy()
        oriented[reverse] = shapely.reverse(oriented[reverse])
    counts = np.full(pieces, 2, dtype=np.int64)
    counts[has] = shapely.get_num_coordinates(oriented)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    coords = np.empty((offsets[-1], 2))
    straight = np.flatnonzero(~has)
    coords[offsets[straight]] = start_xy[straight]
    coords[offsets[straight] + 1] = end_xy[straight]
    if len(oriented):
        values, which = shapely.get_coordinates(oriented, return_index=True)
        first_of_piece = np.searchsorted(which, which)
        coords[offsets[np.flatnonzero(has)[which]] + np.arange(len(values)) - first_of_piece] = values
    chain = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, pieces]))
    piece = np.repeat(np.arange(pieces), counts)
    drop = np.zeros(len(coords), dtype=bool)
    not_first = np.ones(pieces, dtype=bool)
    not_first[starts] = False
    drop[offsets[:-1][not_first]] = True
    return shapely.linestrings(coords[~drop], indices=chain[piece[~drop]])


def _longest_value(column: pd.Series, rows: np.ndarray, lengths: np.ndarray, starts: np.ndarray) -> pd.Categorical:
    chain = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(rows)]))
    order = np.lexsort((-lengths, chain))
    first = order[np.r_[0, np.flatnonzero(chain[order][1:] != chain[order][:-1]) + 1]]
    if isinstance(column.dtype, pd.CategoricalDtype):
        return pd.Categorical.from_codes(column.cat.codes.to_numpy()[rows[first]], dtype=column.dtype)
    return pd.Categorical(column.to_numpy()[rows[first]])


def _aggregate(
    name: str, column: pd.Series, rows: np.ndarray, lengths: np.ndarray, starts: np.ndarray, summed: bool
) -> Any:
    dtype = column.dtype
    if summed:
        return np.add.reduceat(column.to_numpy(dtype=np.float64)[rows], starts)
    if isinstance(dtype, pd.ArrowDtype) and pa.types.is_list(dtype.pyarrow_dtype):
        return _joined_lists(column, rows, starts)
    if pd.api.types.is_bool_dtype(dtype) and not pd.api.types.is_extension_array_dtype(dtype):
        values = column.to_numpy()[rows].astype(np.int8)
        rule = _BOOLEAN_RULES.get(name, all)
        reduce = np.maximum.reduceat if rule is any else np.minimum.reduceat
        return reduce(values, starts).astype(bool)
    if pd.api.types.is_numeric_dtype(dtype) and not isinstance(dtype, pd.CategoricalDtype):
        values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=np.float64)[rows]
        present = ~np.isnan(values)
        weighted = np.add.reduceat(np.where(present, values * lengths, 0.0), starts)
        total = np.add.reduceat(np.where(present, lengths, 0.0), starts)
        plain = np.add.reduceat(np.where(present, values, 0.0), starts)
        count = np.add.reduceat(present.astype(np.int64), starts)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, weighted / total, np.where(count > 0, plain / np.maximum(count, 1), np.nan))
    if isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(dtype):
        return _joined_text(column, rows, starts)
    return column.to_numpy()[rows[starts]]


def _joined_text(column: pd.Series, rows: np.ndarray, starts: np.ndarray) -> pd.Categorical:
    if isinstance(column.dtype, pd.CategoricalDtype):
        codes = column.cat.codes.to_numpy()[rows].astype(np.int64)
        uniques = column.cat.categories
    else:
        codes, uniques = pd.factorize(column.to_numpy(dtype=object)[rows])
    if len(uniques) == 0:
        return pd.Categorical([None] * len(starts))
    low = np.minimum.reduceat(codes, starts)
    high = np.maximum.reduceat(codes, starts)
    values = np.asarray(uniques, dtype=object)
    out = np.where(low >= 0, values[np.maximum(low, 0)], None).astype(object)
    out[high < 0] = None
    ends = np.r_[starts[1:], len(rows)]
    for i in np.flatnonzero(low != high):
        parts = [values[c] for c in codes[starts[i] : ends[i]] if c >= 0]
        out[i] = ";".join(dict.fromkeys(t for p in parts for t in str(p).split(";")))
    return pd.Categorical(out)


def _joined_lists(column: pd.Series, rows: np.ndarray, starts: np.ndarray) -> Any:
    lists = pa.array(column.array).take(pa.array(rows))
    if isinstance(lists, pa.ChunkedArray):
        lists = lists.combine_chunks()
    values = pc.list_flatten(lists).to_numpy(zero_copy_only=False)
    parent = pc.list_parent_indices(lists).to_numpy()
    chain = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(rows)]))[parent]
    unique = ~pd.DataFrame({"chain": chain, "value": values}).duplicated().to_numpy()
    counts = np.bincount(chain[unique], minlength=len(starts))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    joined = pa.ListArray.from_arrays(pa.array(offsets), pa.array(values[unique], type=lists.type.value_type))
    return pd.arrays.ArrowExtensionArray(joined)


# ----- connecting subnetwork helpers


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


def _nearest_source_edges(graph: Graph, terminal: np.ndarray, source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, predecessors, _ = dijkstra(
        graph.matrix, directed=True, indices=source, min_only=True, return_predecessors=True
    )
    return _forest_edges(predecessors, terminal)


def _steiner_edges(graph: Graph, terminal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mehlhorn's 2-approximation: Voronoi regions of the terminals, an MST of their cheapest links, then pruning."""
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
