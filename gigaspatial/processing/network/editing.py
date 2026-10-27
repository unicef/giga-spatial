"""Editing spatial networks. Every method takes many items at once and returns a new network."""

from functools import cached_property
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from gigaspatial.config import config
from gigaspatial.processing.network._arrays import as_positions, next_keys, repeated_rows
from gigaspatial.processing.network.geodesy import WGS84, chord_to_meters, meters_to_chord, unit_vectors
from gigaspatial.processing.network.kinds import EDGE_KINDS, NODE_KINDS, as_kinds
from gigaspatial.processing.network.points import is_wgs84, point_coordinates, points_table

LOGGER = config.get_logger("SpatialNetwork")

_NODE_PREDICATES = {"intersects": shapely.intersects_xy, "within": shapely.contains_xy}
# STRtree.query tests predicate(area, edge); these give the edge's relation to the area.
_EDGE_PREDICATES = {"intersects": "intersects", "within": "contains", "crosses": "crosses", "touches": "touches"}
_PROTECTED_NODE_COLUMNS = ("x", "y")
_PROTECTED_EDGE_COLUMNS = ("u", "v", "key")


class EditingMixin:
    """Editing methods of ``SpatialNetwork``."""

    # ----- search structures, cached per network

    @cached_property
    def _node_tree(self) -> cKDTree:
        return cKDTree(unit_vectors(self._nodes["y"].to_numpy(), self._nodes["x"].to_numpy()))

    @cached_property
    def _edge_tree(self) -> shapely.STRtree:
        return shapely.STRtree(self._filled_geometries)

    def _derive(self, nodes, edges, positions, directed=None):
        return type(self)._from_tables(
            nodes,
            edges,
            directed=self._directed if directed is None else directed,
            weight=self._weight,
            metadata=self._metadata,
            positions=positions,
        )

    def _rebuild(self, nodes, edges):
        return type(self)(nodes, edges, directed=self._directed, weight=self._weight, metadata=self._metadata)

    # ----- nodes

    def nearest_nodes(
        self, points: Any, max_distance: Optional[float] = None, return_distance: bool = False
    ) -> Any:
        """The node closest to each point, by great-circle distance.

        Args:
            points: A GeoDataFrame of points, a DataFrame with latitude and
                longitude columns, or a list of ``(lat, lon)`` tuples or Points.
            max_distance: Meters. Points with no node this close get a missing ID.
            return_distance: Also return the distance in meters.

        Returns:
            A Series of node IDs indexed like the points, or with
            ``return_distance`` a DataFrame with ``node_id`` and ``distance``.
        """
        attributes, x, y = point_coordinates(points)
        chord, position, found = self._nearest_positions(x, y, max_distance)
        ids = _ids_or_missing(self._nodes.index, position, found, attributes.index)
        if not return_distance:
            return ids
        distance = np.where(found, chord_to_meters(np.where(found, chord, 0.0)), np.nan)
        return pd.DataFrame({"node_id": ids, "distance": distance}, index=attributes.index)

    def _nearest_positions(
        self, x: np.ndarray, y: np.ndarray, max_distance: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self._nodes) == 0:
            raise ValueError("The network has no nodes")
        bound = np.inf if max_distance is None else float(meters_to_chord(max_distance))
        chord, position = self._node_tree.query(unit_vectors(y, x), k=1, distance_upper_bound=bound)
        return chord, position, np.isfinite(chord)

    def filter_nodes(
        self, ids: Any = None, mask: Any = None, geometry: Any = None, predicate: str = "intersects"
    ) -> "SpatialNetwork":
        """Keep the selected nodes and the edges between them.

        Pass exactly one of ``ids``, ``mask``, or ``geometry``.

        Args:
            ids: Node IDs to keep.
            mask: Booleans in node order, or a boolean Series indexed by node ID.
            geometry: Area in EPSG:4326, a shapely geometry or a GeoDataFrame
                or GeoSeries in any CRS.
            predicate: ``"intersects"`` or ``"within"``, for ``geometry``.
        """
        return self._keep_nodes(self._node_selection(ids, mask, geometry, predicate))

    def remove_nodes(
        self,
        ids: Any = None,
        points: Any = None,
        geometry: Any = None,
        max_distance: Optional[float] = None,
        predicate: str = "intersects",
    ) -> "SpatialNetwork":
        """Remove nodes and the edges attached to them.

        Pass exactly one of ``ids``, ``points``, or ``geometry``.

        Args:
            ids: Node IDs to remove.
            points: Remove the node nearest each point, within ``max_distance`` meters.
            geometry: Remove the nodes in this area (see :meth:`filter_nodes`).
            max_distance: Meters, for ``points``.
            predicate: ``"intersects"`` or ``"within"``, for ``geometry``.
        """
        if points is not None:
            if ids is not None or geometry is not None:
                raise ValueError("Pass exactly one of ids, points, or geometry")
            ids = self.nearest_nodes(points, max_distance=max_distance).dropna().unique()
        return self._keep_nodes(~self._node_selection(ids, None, geometry, predicate))

    def set_node_attribute(self, name: str, values: Any) -> "SpatialNetwork":
        """Set one node attribute.

        Args:
            name: Column to set or create. ``x`` and ``y`` cannot be set.
            values: A single value for every node, values in node order, or a
                Series or dict keyed by node ID. With a Series or dict, other
                nodes keep their value, or get a missing value in a new column.

        Raises:
            KeyError: If a key is not a node ID.
        """
        _check_writable(name, _PROTECTED_NODE_COLUMNS, "node")
        current = self._nodes[name] if name in self._nodes.columns else None
        column = _assigned(values, len(self._nodes), current, self._positions)
        if name == "kind":
            column = as_kinds(column, "road", NODE_KINDS, len(self._nodes))
        nodes = _with_columns(self._nodes, {name: column})
        return self._derive(nodes, self._edges, self._edge_positions)

    def update_nodes(
        self, frame: pd.DataFrame, by: str = "id", max_distance: Optional[float] = None
    ) -> "SpatialNetwork":
        """Update several node attributes from a table.

        Values that are present overwrite; missing values leave the node's value
        as it is. New columns are created, missing for other nodes.

        Args:
            frame: With ``by="id"``, a table indexed by node ID (or with a
                ``node_id`` column). With ``by="coordinates"``, a points input
                whose rows are matched to their nearest node; when two rows
                match the same node, the later row wins.
            by: ``"id"`` or ``"coordinates"``.
            max_distance: Meters, for ``by="coordinates"``. Rows with no node
                this close are skipped, with a warning.

        Raises:
            KeyError: If ``by="id"`` and a row is not a node ID.
        """
        if by == "id":
            table = frame.set_index("node_id") if "node_id" in frame.columns else frame
            positions = self._positions(table.index)
        elif by == "coordinates":
            table, x, y = point_coordinates(frame)
            _, position, found = self._nearest_positions(x, y, max_distance)
            if not found.all():
                LOGGER.warning(f"{int((~found).sum())} rows have no node within {max_distance} m and were skipped")
            table = table[found]
            positions = position[found]
        else:
            raise ValueError(f'by must be "id" or "coordinates", got {by!r}')
        updates = {}
        for name in table.columns:
            _check_writable(name, _PROTECTED_NODE_COLUMNS, "node")
            current = self._nodes[name] if name in self._nodes.columns else None
            column = _merged(len(self._nodes), positions, table[name], current, skip_missing=True)
            if name == "kind":
                column = as_kinds(column, "road", NODE_KINDS, len(self._nodes))
            updates[name] = column
        return self._derive(_with_columns(self._nodes, updates), self._edges, self._edge_positions)

    def add_nodes(self, frame: Any, id_column: Optional[str] = None, kind: str = "point") -> "SpatialNetwork":
        """Add nodes from a points input with IDs.

        Args:
            frame: A GeoDataFrame of points, a DataFrame with latitude and
                longitude columns, or a list of ``(lat, lon)`` tuples.
            id_column: Column holding the node IDs. The frame's index otherwise.
            kind: Kind of the new nodes.

        Raises:
            ValueError: If an ID is already a node of the network.
        """
        table = points_table(frame, id_column)
        clash = table.index.intersection(self._nodes.index)
        if len(clash):
            raise ValueError(f"{len(clash)} node IDs are already in the network, for example {list(clash[:5])}")
        table.insert(2, "kind", kind)
        nodes = _concat_tables([self._nodes, table])
        return self._rebuild(nodes, self._edges)

    # ----- edges

    def filter_edges(
        self, mask: Any = None, geometry: Any = None, predicate: str = "intersects"
    ) -> "SpatialNetwork":
        """Keep the selected edges. All nodes are kept, so some may be left without edges.

        Pass exactly one of ``mask`` or ``geometry``.

        Args:
            mask: Booleans in edge order.
            geometry: Area in EPSG:4326, a shapely geometry or a GeoDataFrame or
                GeoSeries in any CRS. Straight edges are tested as lines between
                their nodes.
            predicate: The edge's relation to the area: ``"intersects"``,
                ``"within"``, ``"crosses"``, or ``"touches"``.
        """
        if (mask is None) == (geometry is None):
            raise ValueError("Pass exactly one of mask or geometry")
        keep = self._edge_mask(mask) if mask is not None else self._edges_in(geometry, predicate)
        return self._keep_edges(keep)

    def remove_edges(self, edges: Any = None, mask: Any = None) -> "SpatialNetwork":
        """Remove edges. Nodes are kept.

        Pass exactly one of ``edges`` or ``mask``.

        Args:
            edges: ``(u, v, key)`` or ``(u, v)`` tuples, or a table with ``u``,
                ``v``, and optionally ``key`` columns. Without a key, every edge
                between the two nodes is removed. In an undirected network,
                ``(v, u)`` matches ``(u, v)``.
            mask: Booleans in edge order.

        Raises:
            KeyError: If an edge is not in the network.
        """
        if (edges is None) == (mask is None):
            raise ValueError("Pass exactly one of edges or mask")
        if mask is not None:
            selected = self._edge_mask(mask)
        else:
            selected = np.zeros(len(self._edges), dtype=bool)
            selected[self._edge_rows(*_edge_keys(edges))] = True
        return self._keep_edges(~selected)

    def set_edge_attribute(self, name: str, values: Any) -> "SpatialNetwork":
        """Set one edge attribute.

        Args:
            name: Column to set or create. ``u``, ``v``, and ``key`` cannot be set.
            values: A single value for every edge, values in edge order, or a
                Series indexed by ``(u, v, key)`` or a dict keyed by
                ``(u, v, key)`` tuples. With a Series or dict, other edges keep
                their value, or get a missing value in a new column.

        Raises:
            KeyError: If a key is not an edge of the network.
        """
        _check_writable(name, _PROTECTED_EDGE_COLUMNS, "edge")
        current = self._edges[name] if name in self._edges.columns else None

        def lookup(keys):
            u, v, key = _edge_keys(list(keys))
            if key is None:
                raise ValueError("Edge values must be keyed by (u, v, key)")
            return self._edge_rows(u, v, key)

        column = _assigned(values, len(self._edges), current, lookup)
        if name == "kind":
            column = as_kinds(column, "road", EDGE_KINDS, len(self._edges))
        return self._derive(self._nodes, _with_columns(self._edges, {name: column}), self._edge_positions)

    def update_edges(self, frame: pd.DataFrame) -> "SpatialNetwork":
        """Update several edge attributes from a table with ``u``, ``v``, and ``key`` columns.

        Values that are present overwrite; missing values leave the edge's value
        as it is. New columns are created, missing for other edges. In an
        undirected network, ``(v, u, key)`` matches ``(u, v, key)``.

        Raises:
            KeyError: If a row is not an edge of the network.
        """
        table = frame.reset_index() if set(_PROTECTED_EDGE_COLUMNS) <= set(frame.index.names) else frame
        missing = [c for c in _PROTECTED_EDGE_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"update_edges needs {missing} columns")
        rows = self._edge_rows(table["u"], table["v"], table["key"])
        updates = {}
        for name in table.columns:
            if name in _PROTECTED_EDGE_COLUMNS:
                continue
            current = self._edges[name] if name in self._edges.columns else None
            column = _merged(len(self._edges), rows, table[name], current, skip_missing=True)
            if name == "kind":
                column = as_kinds(column, "road", EDGE_KINDS, len(self._edges))
            updates[name] = column
        return self._derive(self._nodes, _with_columns(self._edges, updates), self._edge_positions)

    def add_weighted_edges(
        self,
        u: Sequence[Any],
        v: Sequence[Any],
        weights: Sequence[float],
        weight: str = "length",
        kind: str = "added",
        **attributes: Any,
    ) -> "SpatialNetwork":
        """Add edges with a weight, like networkx's ``add_weighted_edges_from`` but with arrays.

        Keys continue from the edges already between each pair of nodes. When
        ``weight`` is not ``"length"`` and no ``length`` is given, the length is
        computed along the given ``geometry``, or between the endpoints.

        Args:
            u, v: Endpoint node IDs.
            weights: The weight of each edge, stored in the ``weight`` column.
            weight: Column for ``weights``.
            kind: Kind of the new edges.
            **attributes: More columns, each a single value or one value per edge.

        Raises:
            KeyError: If an endpoint is not a node of the network.
        """
        u = list(u)
        v = list(v)
        weights = np.asarray(weights, dtype=np.float64)
        if not (len(u) == len(v) == len(weights)):
            raise ValueError("u, v, and weights must have the same length")
        new_u, new_v = self._positions(u), self._positions(v)
        old_u, old_v = self._edge_positions
        if not self._directed:
            new_u, new_v = np.minimum(new_u, new_v), np.maximum(new_u, new_v)
            old_u, old_v = np.minimum(old_u, old_v), np.maximum(old_u, old_v)
        keys = next_keys(old_u, old_v, self._edges["key"].to_numpy(), new_u, new_v)

        new = pd.DataFrame({"u": u, "v": v, "key": keys, weight: weights, "kind": kind})
        for name, value in attributes.items():
            new[name] = value
        if weight != "length" and "length" not in new.columns:
            new["length"] = np.nan
        edges = _concat_tables([self._edges, new], ignore_index=True)
        return self._rebuild(self._nodes, edges)

    def clip(self, geometry: Any, predicate: str = "intersects") -> "SpatialNetwork":
        """The edges matching ``predicate`` with the area, and their endpoints.

        Args:
            geometry: Area in EPSG:4326, a shapely geometry or a GeoDataFrame or
                GeoSeries in any CRS.
            predicate: ``"intersects"``, ``"within"``, ``"crosses"``, or ``"touches"``.
        """
        keep_edges = self._edges_in(geometry, predicate)
        pu, pv = self._edge_positions
        keep_nodes = np.zeros(len(self._nodes), dtype=bool)
        keep_nodes[pu[keep_edges]] = True
        keep_nodes[pv[keep_edges]] = True
        return self._keep_nodes(keep_nodes, keep_edges)

    def drop_geometry(self) -> "SpatialNetwork":
        """The network without edge geometry, to save memory when only the topology matters."""
        if "geometry" not in self._edges.columns:
            return self
        edges = self._edges.drop(columns="geometry")
        return self._derive(self._nodes, edges, self._edge_positions)

    def to_undirected(self, weight: Optional[str] = None) -> "SpatialNetwork":
        """An undirected network with one edge per pair of nodes.

        Reverse and parallel edges are merged into the one with the smallest
        weight, which keeps its attributes and geometry; every key becomes 0.

        Args:
            weight: Edge column to compare. The network's weight otherwise.
        """
        values = self._edges[weight or self._weight].to_numpy(dtype=np.float64)
        pu, pv = self._edge_positions
        low, high = np.minimum(pu, pv), np.maximum(pu, pv)
        order = np.lexsort((values, high, low))
        first = np.ones(len(order), dtype=bool)
        first[1:] = (low[order][1:] != low[order][:-1]) | (high[order][1:] != high[order][:-1])
        rows = np.sort(order[first])
        edges = self._edges.iloc[rows]
        edges = _with_columns(edges, {"key": np.zeros(len(rows), dtype=np.int16)})
        edges.index = pd.RangeIndex(len(edges))
        return self._derive(self._nodes, edges, (pu[rows], pv[rows]), directed=False)

    @classmethod
    def concat(cls, networks: Iterable["SpatialNetwork"]) -> "SpatialNetwork":
        """Combine networks into one.

        Nodes with the same ID are the same node, as where separately fetched
        areas meet; the first network's row wins. Edges with the same
        ``(u, v, key)`` are kept once. The weight is the first network's; the
        metadata is shared metadata, or the list of each network's metadata
        under ``"parts"``.

        Raises:
            ValueError: If the networks are not all directed or all undirected.
        """
        networks = list(networks)
        if not networks:
            raise ValueError("concat needs at least one network")
        directed = {n.directed for n in networks}
        if len(directed) > 1:
            raise ValueError("Networks must be all directed or all undirected")
        directed = directed.pop()
        nodes = _concat_tables([n._nodes for n in networks])
        nodes = nodes[~nodes.index.duplicated(keep="first")]
        edges = _concat_tables([n._edges for n in networks], ignore_index=True)
        pu = nodes.index.get_indexer(edges["u"])
        pv = nodes.index.get_indexer(edges["v"])
        if not directed:
            pu, pv = np.minimum(pu, pv), np.maximum(pu, pv)
        edges = edges[~repeated_rows(pu, pv, edges["key"].to_numpy())]
        metadata = [n._metadata for n in networks]
        merged = metadata[0] if all(m == metadata[0] for m in metadata) else {"parts": metadata}
        return cls(nodes, edges, directed=directed, weight=networks[0].weight, metadata=merged)

    # ----- selection helpers

    def _node_selection(self, ids: Any, mask: Any, geometry: Any, predicate: str) -> np.ndarray:
        if sum(x is not None for x in (ids, mask, geometry)) != 1:
            raise ValueError("Pass exactly one of ids, mask, or geometry")
        n = len(self._nodes)
        if ids is not None:
            selected = np.zeros(n, dtype=bool)
            selected[self._positions(ids)] = True
            return selected
        if mask is not None:
            if isinstance(mask, pd.Series):
                unknown = ~mask.index.isin(self._nodes.index)
                if unknown.any():
                    raise KeyError(f"{int(unknown.sum())} mask entries are not node IDs")
                return mask.reindex(self._nodes.index, fill_value=False).to_numpy(dtype=bool)
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (n,):
                raise ValueError(f"mask must have one value per node ({n}), got shape {mask.shape}")
            return mask
        if predicate not in _NODE_PREDICATES:
            raise ValueError(f"Node predicate must be one of {sorted(_NODE_PREDICATES)}, got {predicate!r}")
        area = _as_geometry(geometry)
        shapely.prepare(area)
        return _NODE_PREDICATES[predicate](area, self._nodes["x"].to_numpy(), self._nodes["y"].to_numpy())

    def _edge_mask(self, mask: Any) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self._edges),):
            raise ValueError(f"mask must have one value per edge ({len(self._edges)}), got shape {mask.shape}")
        return mask

    def _edges_in(self, geometry: Any, predicate: str) -> np.ndarray:
        if predicate not in _EDGE_PREDICATES:
            raise ValueError(f"Edge predicate must be one of {sorted(_EDGE_PREDICATES)}, got {predicate!r}")
        rows = self._edge_tree.query(_as_geometry(geometry), predicate=_EDGE_PREDICATES[predicate])
        keep = np.zeros(len(self._edges), dtype=bool)
        keep[rows] = True
        return keep

    def _edge_rows(self, u: Any, v: Any, key: Any = None) -> np.ndarray:
        """Edge rows for (u, v, key) lookups, or every key between u and v when key is None."""
        qu, qv = self._positions(u), self._positions(v)
        pu, pv = self._edge_positions
        if not self._directed:
            qu, qv = np.minimum(qu, qv), np.maximum(qu, qv)
            pu, pv = np.minimum(pu, pv), np.maximum(pu, pv)
        if key is None:
            edge_pairs = pd.MultiIndex.from_arrays([pu, pv])
            query = pd.MultiIndex.from_arrays([qu, qv])
            unknown = ~query.isin(edge_pairs)
            if unknown.any():
                raise KeyError(f"{int(unknown.sum())} node pairs have no edge, for example {_sample(u, v, unknown)}")
            return np.flatnonzero(edge_pairs.isin(query))
        index = pd.MultiIndex.from_arrays([pu, pv, self._edges["key"].to_numpy()])
        rows = index.get_indexer(pd.MultiIndex.from_arrays([qu, qv, np.asarray(key, dtype=np.int64)]))
        unknown = rows < 0
        if unknown.any():
            raise KeyError(f"{int(unknown.sum())} edges are not in the network, for example {_sample(u, v, unknown)}")
        return rows

    def _keep_nodes(self, keep_nodes: np.ndarray, keep_edges: Optional[np.ndarray] = None) -> "SpatialNetwork":
        pu, pv = self._edge_positions
        both_ends = keep_nodes[pu] & keep_nodes[pv]
        keep_edges = both_ends if keep_edges is None else keep_edges & both_ends
        new_position = np.cumsum(keep_nodes) - 1
        nodes = self._nodes[keep_nodes]
        edges = self._edges[keep_edges]
        edges.index = pd.RangeIndex(len(edges))
        n = len(nodes)
        positions = (
            as_positions(new_position[pu[keep_edges]], n),
            as_positions(new_position[pv[keep_edges]], n),
        )
        return self._derive(nodes, edges, positions)

    def _keep_edges(self, keep: np.ndarray) -> "SpatialNetwork":
        pu, pv = self._edge_positions
        edges = self._edges[keep]
        edges.index = pd.RangeIndex(len(edges))
        return self._derive(self._nodes, edges, (pu[keep], pv[keep]))


# ----- helpers


def _ids_or_missing(index: pd.Index, position: np.ndarray, found: np.ndarray, out_index: pd.Index) -> pd.Series:
    ids = pd.Series(index.take(np.where(found, position, 0)), index=out_index, name="node_id")
    if found.all():
        return ids
    if pd.api.types.is_integer_dtype(index.dtype):
        ids = ids.astype("Int64")
        ids[~found] = pd.NA
    else:
        ids = ids.astype(object)
        ids[~found] = None
    return ids


def _as_geometry(geometry: Any) -> shapely.Geometry:
    if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
        series = geometry.geometry if isinstance(geometry, gpd.GeoDataFrame) else geometry
        if series.crs is not None and not is_wgs84(series.crs):
            series = series.to_crs(WGS84)
        return series.union_all()
    if isinstance(geometry, shapely.Geometry):
        return geometry
    raise TypeError(f"Expected a shapely geometry, GeoSeries, or GeoDataFrame, got {type(geometry).__name__}")


def _check_writable(name: str, protected: Tuple[str, ...], what: str) -> None:
    if name in protected:
        raise ValueError(f"The {what} column {name!r} cannot be changed; build a new network instead")


def _edge_keys(edges: Any) -> Tuple[Any, Any, Any]:
    """(u, v, key) arrays from tuples or a table; key is None when not given."""
    if isinstance(edges, pd.DataFrame):
        return edges["u"], edges["v"], edges["key"] if "key" in edges.columns else None
    rows = [tuple(e) for e in edges]
    if not rows:
        return [], [], None
    lengths = {len(r) for r in rows}
    if lengths == {2}:
        return [r[0] for r in rows], [r[1] for r in rows], None
    if lengths == {3}:
        return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
    raise ValueError("Edges must all be (u, v) or all be (u, v, key) tuples")


def _sample(u: Any, v: Any, mask: np.ndarray) -> List[Tuple[Any, Any]]:
    pairs = list(zip(list(u), list(v)))
    return [p for p, m in zip(pairs, mask) if m][:5]


def _assigned(values: Any, n: int, current: Optional[pd.Series], lookup) -> Any:
    """A column of ``n`` values from a single value, values in order, or keyed values."""
    if isinstance(values, dict):
        values = pd.Series(values)
    if isinstance(values, pd.Series):
        keys = list(values.index) if values.index.nlevels > 1 else values.index
        return _merged(n, lookup(keys), values, current, skip_missing=False)
    if np.ndim(values) == 0:
        return np.full(n, values, dtype=object if isinstance(values, str) or values is None else None)
    if len(values) != n:
        raise ValueError(f"Expected {n} values, got {len(values)}")
    return values


def _merged(n: int, positions: np.ndarray, values: pd.Series, current: Optional[pd.Series], skip_missing: bool) -> Any:
    """``current`` with ``values`` written at ``positions``; the last write to a position wins."""
    new = pd.Series(values.to_numpy(), index=positions)
    if skip_missing:
        new = new[new.notna().to_numpy()]
    new = new[~new.index.duplicated(keep="last")]
    written = np.zeros(n, dtype=bool)
    written[new.index.to_numpy()] = True
    result = new.reindex(pd.RangeIndex(n))
    if current is None:
        return result.array
    result = result.where(written, pd.Series(current.to_numpy(), index=pd.RangeIndex(n)))
    if isinstance(current.dtype, pd.CategoricalDtype):
        return pd.Categorical(result.to_numpy())
    return _restore_dtype(result, current.dtype).array


def _restore_dtype(result: pd.Series, dtype: Any) -> pd.Series:
    """``result`` in the column's original dtype when nothing is lost, such as integers kept as integers."""
    try:
        converted = result.astype(dtype)
    except (TypeError, ValueError, OverflowError):
        return result
    same = (converted.astype(object) == result.astype(object)) | (converted.isna() & result.isna())
    return converted if bool(same.all()) else result


def _with_columns(table: pd.DataFrame, columns: Dict[str, Any]) -> pd.DataFrame:
    """``table`` with some columns replaced or added; the other columns are shared, not copied."""
    merged = {c: table[c] for c in table.columns}
    merged.update({name: (value.array if isinstance(value, pd.Series) else value) for name, value in columns.items()})
    return pd.DataFrame(merged, index=table.index, copy=False)


def _concat_tables(tables: List[pd.DataFrame], ignore_index: bool = False) -> pd.DataFrame:
    """Concatenate tables, keeping categorical and Arrow columns' types where a table lacks the column."""
    columns = list(dict.fromkeys(c for t in tables for c in t.columns))
    aligned = []
    for table in tables:
        missing = {}
        for c in columns:
            if c not in table.columns:
                template = next(t[c] for t in tables if c in t.columns)
                missing[c] = template.iloc[:0].reindex(pd.RangeIndex(len(table))).array
        if missing:
            table = _with_columns(table, missing)
        aligned.append(table)
    for c in columns:
        if all(isinstance(t[c].dtype, pd.CategoricalDtype) for t in aligned):
            categories = list(dict.fromkeys(x for t in aligned for x in t[c].cat.categories))
            aligned = [_with_columns(t, {c: t[c].cat.set_categories(categories)}) for t in aligned]
    return pd.concat([t[columns] for t in aligned], ignore_index=ignore_index)
