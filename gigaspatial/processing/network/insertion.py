"""Insertion: adding points of interest to a spatial network, joined to their nearest road or nodes."""

from functools import cached_property
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree

from gigaspatial.processing.geo import estimate_utm_crs_with_fallback
from gigaspatial.processing.network._arrays import next_keys
from gigaspatial.processing.network.editing import LOGGER, _concat_tables, _with_columns
from gigaspatial.processing.network.geodesy import (
    WGS84,
    chord_to_meters,
    haversine,
    line_lengths,
    meters_to_chord,
    unit_vectors,
)
from gigaspatial.processing.network.points import points_table

# Edges points are never joined to: the connections of earlier insertions.
_NOT_ROADS = ("connection", "point_link")
# Feet closer than this along a road share one split node.
_SAME_FOOT_M = 0.001


class Projected(NamedTuple):
    """Road edges in a UTM projection, where the nearest edge can be found by planar distance."""

    to_utm: Transformer
    rows: np.ndarray
    geometries: np.ndarray
    tree: shapely.STRtree


class InsertionMixin:
    """Insertion of points into ``SpatialNetwork``."""

    @cached_property
    def _projected_roads(self) -> Projected:
        crs = estimate_utm_crs_with_fallback(gpd.GeoDataFrame(geometry=[shapely.box(*self.bounds)], crs=WGS84))
        to_utm = Transformer.from_crs(WGS84, crs, always_xy=True)
        rows = np.flatnonzero(~self._edges["kind"].isin(_NOT_ROADS).to_numpy())
        geometries = shapely.transform(
            self._filled_geometries[rows], lambda c: np.column_stack(to_utm.transform(c[:, 0], c[:, 1]))
        )
        return Projected(to_utm, rows, geometries, shapely.STRtree(geometries))

    def insert_points(
        self,
        points: Any,
        id_column: Optional[str] = None,
        max_distance: Optional[float] = None,
        snap_to_node: float = 5.0,
        chain: bool = False,
        return_skipped: bool = False,
    ) -> Any:
        """Insert points of interest as nodes, each joined to its nearest road.

        Each point is joined to the nearest road edge at the foot of the
        perpendicular: the edge is split there, a node of kind ``split`` is
        added, and a ``connection`` edge joins the point (kind ``point``) to
        it. When the foot is within ``snap_to_node`` meters of the edge's end,
        the point connects to that node instead. Points whose feet fall on the
        same spot share one split node. Each road edge is split once, at all of
        its points, and the pieces' lengths add up exactly to the original
        length. In a directed network, both directions of a two-way road are
        split at the same nodes, and connections run both ways.

        Args:
            points: A GeoDataFrame of points, a DataFrame with latitude and
                longitude columns, or a list of ``(lat, lon)`` tuples. Their
                other columns become node attributes.
            id_column: Column holding the point IDs. The frame's index otherwise.
            max_distance: Meters. Points farther from any road are skipped,
                with a warning.
            snap_to_node: Meters. Connect to the edge's end node when the foot
                is this close to it.
            chain: Handle points nearest to the road first, and join a point
                to an already inserted point (a ``point_link`` edge) when that
                point is closer than any road.
            return_skipped: Also return the points that were skipped.

        Returns:
            The new network, or ``(network, skipped_points)``. Split nodes get
            negative integer IDs, continuing below the lowest existing one.
            When the network's weight is not ``length``, new connection edges
            get that weight equal to their length, and road pieces get their
            share of the road's weight.

        Raises:
            ValueError: If point IDs repeat or are already node IDs, or a
                point column is named ``x``, ``y``, or ``kind``.
        """
        table = points_table(points, id_column)
        clash = table.index.intersection(self._nodes.index)
        if len(clash):
            raise ValueError(f"{len(clash)} point IDs are already node IDs, for example {list(clash[:5])}")
        x, y = table["x"].to_numpy(), table["y"].to_numpy()
        roads = self._projected_roads
        if len(roads.rows) == 0:
            raise ValueError("The network has no road edges to insert points into")

        foot = _feet(roads, x, y)
        road_distance = haversine(y, x, foot.y, foot.x)
        target_node, link_to = self._snapped_targets(foot, snap_to_node)
        if chain:
            link_to = _chain_links(x, y, road_distance, max_distance)
        connected_by_road = link_to < 0
        link_distance = np.full(len(x), np.inf)
        linked = ~connected_by_road
        link_distance[linked] = haversine(y[linked], x[linked], y[link_to[linked]], x[link_to[linked]])
        distance = np.where(connected_by_road, road_distance, link_distance)
        if max_distance is None:
            inserted = np.ones(len(x), dtype=bool)
        else:
            inserted = _inserted(distance, link_to, road_distance, max_distance)
        skipped = np.flatnonzero(~inserted)
        if len(skipped):
            LOGGER.warning(f"{len(skipped)} points are farther than {max_distance} m from any road and were skipped")

        network = self._with_points(table, inserted, connected_by_road, foot, target_node, link_to, road_distance, link_distance)
        return _with_skipped(network, points, skipped, return_skipped)

    def connect_points(
        self,
        points: Any,
        id_column: Optional[str] = None,
        connect: str = "knn",
        k: int = 1,
        max_distance: Optional[float] = None,
        nodes: Optional[Any] = None,
        return_skipped: bool = False,
    ) -> Any:
        """Add points of interest as nodes, each joined directly to existing nodes.

        Each point becomes a node of kind ``point``, joined by straight-line
        edges to its ``k`` nearest nodes (``connect="knn"``) or to every node
        within ``max_distance`` meters (``connect="distance"``). Edges to nodes
        of kind ``point`` get kind ``point_link``; edges to other nodes get
        kind ``connection``. No existing edge is split. In a directed network,
        new edges run both ways.

        Args:
            points: A GeoDataFrame of points, a DataFrame with latitude and
                longitude columns, or a list of ``(lat, lon)`` tuples. Their
                other columns become node attributes.
            id_column: Column holding the point IDs. The frame's index otherwise.
            connect: ``"knn"`` joins each point to its ``k`` nearest candidate
                nodes; ``"distance"`` joins each point to every candidate node
                within ``max_distance`` meters.
            k: Number of neighbours for ``"knn"``.
            max_distance: Meters. Required for ``"distance"``; with ``"knn"``
                it drops longer links. Points with no candidate node within this
                distance are skipped, with a warning.
            nodes: Candidate target nodes: node IDs, a boolean mask in node
                order, or a boolean Series indexed by node ID. All existing
                nodes otherwise.
            return_skipped: Also return the points that were skipped.

        Returns:
            The new network, or ``(network, skipped_points)``. When the
            network's weight is not ``length``, new edges get that weight equal
            to their length.

        Raises:
            ValueError: If point IDs repeat or are already node IDs, a point
                column is named ``x``, ``y``, or ``kind``, the network has no
                candidate nodes, or ``connect`` or ``k`` is invalid.
            KeyError: If ``nodes`` names a node ID not in the network.
        """
        if connect == "knn":
            if int(k) < 1:
                raise ValueError(f"k must be at least 1, got {k}")
        elif connect == "distance":
            if max_distance is None:
                raise ValueError('connect="distance" needs max_distance')
        else:
            raise ValueError(f'connect must be "knn" or "distance", got {connect!r}')

        table = points_table(points, id_column)
        clash = table.index.intersection(self._nodes.index)
        if len(clash):
            raise ValueError(f"{len(clash)} point IDs are already node IDs, for example {list(clash[:5])}")
        candidates, tree = self._candidate_nodes(nodes)

        x, y = table["x"].to_numpy(), table["y"].to_numpy()
        node_x, node_y = self._nodes["x"].to_numpy(), self._nodes["y"].to_numpy()
        point_row, target_pos, length = _node_links(
            tree, candidates, x, y, node_x, node_y, connect, int(k), max_distance
        )
        if max_distance is None:
            inserted = np.ones(len(x), dtype=bool)
        else:
            inserted = np.zeros(len(x), dtype=bool)
            inserted[point_row] = True
        skipped = np.flatnonzero(~inserted)
        if len(skipped):
            LOGGER.warning(f"{len(skipped)} points have no node within {max_distance} m and were skipped")

        network = self._with_connected_points(table, inserted, point_row, target_pos, length)
        return _with_skipped(network, points, skipped, return_skipped)

    def _candidate_nodes(self, nodes: Optional[Any]) -> Tuple[np.ndarray, cKDTree]:
        """Positions of the candidate target nodes and a unit-sphere KD-tree over them."""
        if nodes is None:
            if len(self._nodes) == 0:
                raise ValueError("The network has no nodes to connect points to")
            return np.arange(len(self._nodes), dtype=np.int64), self._node_tree
        if isinstance(nodes, pd.Series) and pd.api.types.is_bool_dtype(nodes.dtype):
            selected = self._node_selection(None, nodes, None, "intersects")
        elif not isinstance(nodes, pd.Series) and pd.api.types.is_bool_dtype(np.asarray(nodes).dtype):
            selected = self._node_selection(None, nodes, None, "intersects")
        else:
            selected = self._node_selection(nodes, None, None, "intersects")
        candidates = np.flatnonzero(selected)
        if len(candidates) == 0:
            raise ValueError("The network has no nodes to connect points to")
        if len(candidates) == len(self._nodes):
            return candidates, self._node_tree
        xyz = unit_vectors(self._nodes["y"].to_numpy()[candidates], self._nodes["x"].to_numpy()[candidates])
        return candidates, cKDTree(xyz)

    def _with_connected_points(
        self,
        table: pd.DataFrame,
        inserted: np.ndarray,
        point_row: np.ndarray,
        target_pos: np.ndarray,
        length: np.ndarray,
    ) -> "SpatialNetwork":
        point_nodes = table[inserted].copy(deep=False)
        point_nodes.insert(2, "kind", "point")
        nodes = _concat_tables([self._nodes, point_nodes])

        ids = self._nodes.index
        point_ids = table.index
        target_is_point = self._nodes["kind"].to_numpy()[target_pos] == "point"
        kinds = np.where(target_is_point, "point_link", "connection")
        connections = pd.DataFrame(
            {
                "u": point_ids.take(point_row),
                "v": ids.take(target_pos),
                "key": np.zeros(len(point_row), dtype=np.int16),
                "length": length,
                "kind": kinds,
            }
        )
        if self._directed:
            connections = pd.concat(
                [connections, connections.rename(columns={"u": "v", "v": "u"})[connections.columns]], ignore_index=True
            )
        if self._weight != "length":
            connections[self._weight] = connections["length"]
        edges = _concat_tables([self._edges, connections], ignore_index=True)
        return type(self)(nodes, edges, directed=self._directed, weight=self._weight, metadata=self._metadata)

    def _snapped_targets(self, foot: "Foot", snap_to_node: float) -> Tuple[np.ndarray, np.ndarray]:
        """For each point, the end node it snaps to (or -1 to split its road), and no point links yet."""
        pu, pv = self._edge_positions
        xs, ys = self._nodes["x"].to_numpy(), self._nodes["y"].to_numpy()
        a, b = pu[foot.row], pv[foot.row]
        to_a = haversine(foot.y, foot.x, ys[a], xs[a])
        to_b = haversine(foot.y, foot.x, ys[b], xs[b])
        nearest_end = np.where(to_a <= to_b, a, b)
        snaps = np.minimum(to_a, to_b) <= snap_to_node
        return np.where(snaps, nearest_end, -1), np.full(len(foot.x), -1, dtype=np.int64)

    def _with_points(
        self,
        table: pd.DataFrame,
        inserted: np.ndarray,
        by_road: np.ndarray,
        foot: "Foot",
        target_node: np.ndarray,
        link_to: np.ndarray,
        road_distance: np.ndarray,
        link_distance: np.ndarray,
    ) -> "SpatialNetwork":
        ids = self._nodes.index
        n = len(ids)
        point_ids = table.index
        splitting = np.flatnonzero(inserted & by_road & (target_node < 0))
        snapped = np.flatnonzero(inserted & by_road & (target_node >= 0))
        chained = np.flatnonzero(inserted & ~by_road)

        twins = self._twin_rows(np.unique(foot.row[splitting]))
        road = foot.row[splitting].copy()
        for row, twin in twins.items():
            # A one-way road has no twin (-1) and keeps its own row.
            if 0 <= twin < row:
                road[road == row] = twin
        split_plan = self._split_plan(road, foot.x[splitting], foot.y[splitting])
        first_id = _lowest_negative_id(ids)
        split_ids = np.arange(first_id - 1, first_id - 1 - len(split_plan.x), -1)

        # Each splitting point connects to the split node of its foot.
        point_target = np.empty(len(point_ids), dtype=object)
        point_target[splitting] = split_ids[split_plan.node_of_point]
        point_target[snapped] = ids.take(target_node[snapped])
        point_target[chained] = point_ids.take(link_to[chained])

        pieces = self._pieces(split_plan, split_ids, twins)
        removed = np.zeros(len(self._edges), dtype=bool)
        for row in split_plan.roads:
            removed[row] = True
            if twins.get(row, -1) >= 0:
                removed[twins[row]] = True

        split_nodes = pd.DataFrame(
            {"x": split_plan.x, "y": split_plan.y, "kind": "split"}, index=pd.Index(split_ids, name="node_id")
        )
        point_nodes = table[inserted].copy(deep=False)
        point_nodes.insert(2, "kind", "point")
        nodes = _concat_tables([self._nodes, split_nodes, point_nodes])

        joined = np.concatenate([splitting, snapped, chained])
        kinds = ["connection"] * (len(splitting) + len(snapped)) + ["point_link"] * len(chained)
        lengths = np.concatenate([road_distance[splitting], self._snap_lengths(table, snapped, target_node), link_distance[chained]])
        connections = pd.DataFrame(
            {"u": point_ids.take(joined), "v": point_target[joined], "length": lengths, "kind": kinds}
        )
        if self._directed:
            connections = pd.concat(
                [connections, connections.rename(columns={"u": "v", "v": "u"})[connections.columns]], ignore_index=True
            )
        if self._weight != "length":
            connections[self._weight] = connections["length"]
        edges = _concat_tables([self._edges[~removed], pieces, connections], ignore_index=True)
        edges = self._fresh_keys(edges, int((~removed).sum()))
        return type(self)(nodes, edges, directed=self._directed, weight=self._weight, metadata=self._metadata)

    def _snap_lengths(self, table: pd.DataFrame, snapped: np.ndarray, target_node: np.ndarray) -> np.ndarray:
        xs, ys = self._nodes["x"].to_numpy(), self._nodes["y"].to_numpy()
        node = target_node[snapped]
        return haversine(table["y"].to_numpy()[snapped], table["x"].to_numpy()[snapped], ys[node], xs[node])

    def _twin_rows(self, rows: np.ndarray) -> Dict[int, int]:
        """For each road row, the row of the other direction of the same two-way road, or -1."""
        if not self._directed:
            return {int(r): -1 for r in rows}
        pu, pv = self._edge_positions
        n = len(self._nodes)
        keys = pu.astype(np.int64) * n + pv
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        wanted = pv[rows].astype(np.int64) * n + pu[rows]
        low = np.searchsorted(sorted_keys, wanted, side="left")
        high = np.searchsorted(sorted_keys, wanted, side="right")
        geoms = self._edges["geometry"].to_numpy(dtype=object) if "geometry" in self._edges.columns else None
        twins: Dict[int, int] = {}
        for row, start, stop in zip(rows, low, high):
            twin = -1
            for candidate in order[start:stop]:
                if geoms is None or (pd.isna(geoms[row]) and pd.isna(geoms[candidate])):
                    twin = int(candidate)
                elif not pd.isna(geoms[row]) and not pd.isna(geoms[candidate]):
                    if shapely.equals_exact(shapely.reverse(geoms[row]), geoms[candidate], tolerance=1e-9):
                        twin = int(candidate)
                if twin >= 0:
                    break
            twins[int(row)] = twin
            if twin >= 0:
                twins[twin] = int(row)
        return twins

    def _split_plan(self, road: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> "SplitPlan":
        """Where each road is split: one node per distinct foot, ordered along the road."""
        geoms = self._filled_geometries[road]
        position = shapely.line_locate_point(geoms, shapely.points(fx, fy))
        length = self._edges["length"].to_numpy()[road]
        line_degrees = shapely.length(geoms)
        along_m = np.where(line_degrees > 0, position / np.where(line_degrees > 0, line_degrees, 1) * length, 0.0)
        order = np.lexsort((along_m, road))
        same = np.zeros(len(order), dtype=bool)
        same[1:] = (road[order][1:] == road[order][:-1]) & (np.abs(np.diff(along_m[order])) < _SAME_FOOT_M)
        node_sorted = np.cumsum(~same) - 1
        node_of_point = np.empty(len(order), dtype=np.int64)
        node_of_point[order] = node_sorted
        firsts = order[~same]
        return SplitPlan(
            x=fx[firsts],
            y=fy[firsts],
            road_of_node=road[firsts],
            position=position[firsts],
            node_of_point=node_of_point,
            roads=np.unique(road),
        )

    def _pieces(self, plan: "SplitPlan", split_ids: np.ndarray, twins: Dict[int, int]) -> pd.DataFrame:
        """The pieces of every split road, and of its other direction."""
        ids = self._nodes.index
        pu, pv = self._edge_positions
        source: List[int] = []
        u: List[Any] = []
        v: List[Any] = []
        lines: List[np.ndarray] = []
        fractions: List[float] = []
        lengths = self._edges["length"].to_numpy()
        if len(plan.road_of_node) == 0:
            return self._edges.iloc[:0]
        boundaries = np.flatnonzero(np.r_[True, plan.road_of_node[1:] != plan.road_of_node[:-1]])
        for start, end in zip(boundaries, np.r_[boundaries[1:], len(plan.road_of_node)]):
            row = int(plan.road_of_node[start])
            coords = shapely.get_coordinates(self._filled_geometries[row])
            cuts = np.column_stack([plan.x[start:end], plan.y[start:end]])
            parts = _split_coords(coords, plan.position[start:end], cuts)
            part_lengths = line_lengths([shapely.LineString(p) for p in parts])
            total = part_lengths.sum()
            shares = part_lengths / total if total > 0 else np.full(len(parts), 1 / len(parts))
            chain_nodes = [ids[pu[row]], *split_ids[start:end], ids[pv[row]]]
            for j, part in enumerate(parts):
                source.append(row)
                u.append(chain_nodes[j])
                v.append(chain_nodes[j + 1])
                lines.append(part)
                fractions.append(shares[j])
            twin = twins.get(row, -1)
            if twin >= 0:
                for j, part in reversed(list(enumerate(parts))):
                    source.append(twin)
                    u.append(chain_nodes[j + 1])
                    v.append(chain_nodes[j])
                    lines.append(part[::-1])
                    fractions.append(shares[j])
        if not source:
            return self._edges.iloc[:0]
        source_rows = np.asarray(source)
        fraction = np.asarray(fractions)
        base = self._edges.iloc[source_rows]
        updates = {
            "u": u,
            "v": v,
            "key": np.zeros(len(source), dtype=np.int64),
            "length": lengths[source_rows] * fraction,
            "kind": np.full(len(source), "split", dtype=object),
            "geometry": np.array([shapely.LineString(p) for p in lines], dtype=object),
        }
        if self._weight != "length":
            updates[self._weight] = self._edges[self._weight].to_numpy(dtype=np.float64)[source_rows] * fraction
        pieces = _with_columns(base, updates)
        pieces.index = pd.RangeIndex(len(pieces))
        return pieces

    def _fresh_keys(self, edges: pd.DataFrame, kept: int) -> pd.DataFrame:
        """New edges continue the keys of edges already between the same nodes."""
        index = pd.Index(edges["u"]).append(pd.Index(edges["v"]))
        codes, _ = pd.factorize(index)
        a, b = codes[: len(edges)], codes[len(edges) :]
        if not self._directed:
            a, b = np.minimum(a, b), np.maximum(a, b)
        keys = np.zeros(len(edges), dtype=np.int64)
        keys[:kept] = edges["key"].to_numpy()[:kept]
        keys[kept:] = next_keys(a[:kept], b[:kept], keys[:kept], a[kept:], b[kept:])
        return _with_columns(edges, {"key": keys})


class Foot(NamedTuple):
    """Each point's foot on its nearest road edge, in EPSG:4326."""

    row: np.ndarray
    x: np.ndarray
    y: np.ndarray


class SplitPlan(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    road_of_node: np.ndarray
    position: np.ndarray
    node_of_point: np.ndarray
    roads: np.ndarray


def _feet(roads: Projected, x: np.ndarray, y: np.ndarray) -> Foot:
    px, py = roads.to_utm.transform(x, y)
    points = shapely.points(px, py)
    which, nearest = roads.tree.query_nearest(points, all_matches=False)
    order = np.argsort(which)
    nearest = nearest[order]
    lines = roads.geometries[nearest]
    foot = shapely.line_interpolate_point(lines, shapely.line_locate_point(lines, points))
    fx, fy = roads.to_utm.transform(shapely.get_x(foot), shapely.get_y(foot), direction="INVERSE")
    return Foot(roads.rows[nearest], np.asarray(fx), np.asarray(fy))


def _split_coords(coords: np.ndarray, positions: np.ndarray, cuts: np.ndarray) -> List[np.ndarray]:
    """A line's coordinates cut at sorted positions along it, through the cut points."""
    step = np.hypot(*np.diff(coords, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(step)])
    bounds = np.concatenate([[0.0], positions, [along[-1]]])
    ends = np.vstack([coords[:1], cuts, coords[-1:]])
    parts = []
    for j in range(len(bounds) - 1):
        inner = coords[(along > bounds[j]) & (along < bounds[j + 1])]
        parts.append(np.vstack([ends[j : j + 1], inner, ends[j + 1 : j + 2]]))
    return parts


def _chain_links(x: np.ndarray, y: np.ndarray, road_distance: np.ndarray, max_distance: Optional[float]) -> np.ndarray:
    """For each point, the earlier point it links to (nearer the road, and nearer than any road), or -1."""
    n = len(x)
    rank = np.empty(n, dtype=np.int64)
    rank[np.lexsort((np.arange(n), road_distance))] = np.arange(n)
    xyz = unit_vectors(y, x)
    tree = cKDTree(xyz)
    link = np.full(n, -1, dtype=np.int64)
    pending = np.arange(n)
    k = min(8, n)
    while len(pending) and n > 1:
        chord, j = tree.query(xyz[pending], k=k)
        chord, j = chord.reshape(len(pending), -1), j.reshape(len(pending), -1)
        meters = chord_to_meters(chord)
        valid = (j != pending[:, None]) & (rank[j] < rank[pending][:, None]) & (meters < road_distance[pending][:, None])
        masked = np.where(valid, meters, np.inf)
        best = np.argmin(masked, axis=1)
        found = np.isfinite(masked[np.arange(len(pending)), best])
        link[pending[found]] = j[found, best[found]]
        unresolved = ~found & (meters[:, -1] < road_distance[pending]) & (k < n)
        pending = pending[unresolved]
        k = min(2 * k, n)
    return link


def _inserted(distance: np.ndarray, link_to: np.ndarray, road_distance: np.ndarray, max_distance: float) -> np.ndarray:
    """Points whose connection is within max_distance and, when linked, link to an inserted point.

    A point links only to a point nearer the road, so going by road distance
    settles every target before the points that link to it.
    """
    inserted = distance <= max_distance
    linked = np.flatnonzero(link_to >= 0)
    for p in linked[np.argsort(road_distance[linked], kind="stable")]:
        if inserted[p] and not inserted[link_to[p]]:
            inserted[p] = False
    return inserted


def _lowest_negative_id(ids: pd.Index) -> int:
    """The lowest negative integer node ID, or 0, so split nodes can continue below it."""
    if pd.api.types.is_integer_dtype(ids.dtype):
        return min(0, int(ids.min())) if len(ids) else 0
    integers = [int(i) for i in ids if isinstance(i, (int, np.integer)) and not isinstance(i, bool) and i < 0]
    return min([0, *integers])


def _node_links(
    tree: cKDTree,
    candidates: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    connect: str,
    k: int,
    max_distance: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point rows, candidate target node positions, and great-circle lengths in meters."""
    n = len(x)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    xyz = unit_vectors(y, x)
    if connect == "knn":
        k_eff = min(k, len(candidates))
        _, j = tree.query(xyz, k=k_eff)
        j = np.asarray(j, dtype=np.int64).reshape(n, k_eff)
        point_row = np.repeat(np.arange(n, dtype=np.int64), k_eff)
        target_pos = candidates[j.ravel()]
    else:
        radius = float(meters_to_chord(max_distance)) * (1.0 + 1e-12) + 1e-15
        neighbours = tree.query_ball_point(xyz, radius)
        counts = np.array([len(nb) for nb in neighbours], dtype=np.int64)
        point_row = np.repeat(np.arange(n, dtype=np.int64), counts)
        picked = np.concatenate(neighbours).astype(np.int64) if counts.sum() else np.empty(0, dtype=np.int64)
        target_pos = candidates[picked]
    length = haversine(y[point_row], x[point_row], node_y[target_pos], node_x[target_pos])
    if max_distance is not None:
        keep = length <= max_distance
        point_row, target_pos, length = point_row[keep], target_pos[keep], length[keep]
    if connect == "distance" and len(point_row):
        order = np.lexsort((target_pos, length, point_row))
        point_row, target_pos, length = point_row[order], target_pos[order], length[order]
    return point_row, target_pos, length


def _with_skipped(network: Any, points: Any, skipped: np.ndarray, return_skipped: bool) -> Any:
    if not return_skipped:
        return network
    if isinstance(points, pd.DataFrame):
        return network, points.iloc[skipped]
    items = list(points)
    return network, [items[i] for i in skipped]
