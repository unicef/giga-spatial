"""Spatial networks: places as nodes with coordinates, joined by weighted edges."""

from functools import cached_property
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree

from gigaspatial.processing.network._arrays import as_positions, repeated_rows
from gigaspatial.processing.network.algorithms import AlgorithmsMixin
from gigaspatial.processing.network.editing import EditingMixin
from gigaspatial.processing.network.geodesy import (
    WGS84,
    haversine,
    line_lengths,
    meters_to_chord,
    unit_vectors,
)
from gigaspatial.processing.network.kinds import EDGE_KINDS, NODE_KINDS, as_kinds
from gigaspatial.processing.network.points import RESERVED_NODE_COLUMNS, is_wgs84, points_table

_EDGE_ID_COLUMNS = ("u", "v", "key")


class SpatialNetwork(EditingMixin, AlgorithmsMixin):
    """A spatial network held as a nodes table and an edges table.

    Nodes carry coordinates in EPSG:4326, and edges carry a length in meters
    or another weight. A road network fetched by ``OSMRoadsFetcher`` is one
    kind; points joined to their nearest neighbours are another.

    The nodes table is indexed by ``node_id`` and has the columns ``x``
    (longitude), ``y`` (latitude), and ``kind``, plus any attributes. The
    edges table has ``u``, ``v``, ``key``, ``length`` (meters), ``kind``, an
    optional ``geometry`` column of LineStrings, and any attributes. An edge
    is identified by ``(u, v, key)``.

    A network never changes after it is built: methods that edit it return a
    new network. The tables returned by ``nodes`` and ``edges`` are shallow
    copies; adding columns to them does not change the network, but do not
    modify their values in place.

    Args:
        nodes: Node table, indexed by node ID or with a ``node_id`` column,
            with ``x`` and ``y`` columns or Point geometries.
        edges: Edge table with ``u`` and ``v`` columns. A missing ``key`` is
            numbered per node pair, and a missing ``length`` is computed along
            the geometry, or between the endpoints when there is none.
        directed: Whether edges run from ``u`` to ``v`` only. A two-way road
            in a directed network is two rows, one per direction.
        crs: CRS of the input coordinates. They are converted to EPSG:4326.
            The CRS of a GeoDataFrame takes precedence for its geometries.
        weight: Default edge column for algorithms.
        metadata: Provenance, such as the fetch parameters.

    Raises:
        ValueError: If node IDs repeat, an edge references an unknown node,
            an edge ``(u, v, key)`` repeats, or the weight column is missing.
    """

    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        directed: bool = True,
        crs: Any = WGS84,
        weight: str = "length",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        nodes = _prepare_nodes(nodes, crs)
        edges, positions = _prepare_edges(edges, nodes, crs, bool(directed), weight)
        self._nodes = nodes
        self._edges = edges
        self._directed = bool(directed)
        self._weight = weight
        self._metadata = dict(metadata or {})
        self.__dict__["_edge_positions"] = positions

    @classmethod
    def _from_tables(
        cls,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        *,
        directed: bool,
        weight: str,
        metadata: Optional[Dict[str, Any]] = None,
        positions: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "SpatialNetwork":
        """Wrap tables that already satisfy the invariants, skipping validation."""
        network = cls.__new__(cls)
        network._nodes = nodes
        network._edges = edges
        network._directed = directed
        network._weight = weight
        network._metadata = dict(metadata or {})
        if positions is not None:
            network.__dict__["_edge_positions"] = positions
        return network

    # ----- constructors

    @classmethod
    def from_networkx(cls, G: nx.Graph, weight: str = "length") -> "SpatialNetwork":
        """Build a network from a networkx graph whose nodes have ``x`` and ``y``.

        osmnx graphs work directly. Whether the network is directed comes from
        the graph, and the CRS from ``G.graph["crs"]`` (EPSG:4326 if absent).
        Graphs without edge keys get key 0. The other graph attributes become
        the network's metadata.

        Args:
            G: Any networkx graph.
            weight: Default edge column for algorithms.

        Returns:
            The network.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError(f"Expected a networkx graph, got {type(G).__name__}")
        nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
        if G.number_of_nodes() and nodes.empty:
            nodes = pd.DataFrame(index=pd.Index(list(G.nodes)))
        if G.is_multigraph():
            rows = list(G.edges(keys=True, data=True))
        else:
            rows = [(u, v, 0, d) for u, v, d in G.edges(data=True)]
        edges = pd.DataFrame.from_records([r[3] for r in rows], index=pd.RangeIndex(len(rows)))
        clash = [c for c in _EDGE_ID_COLUMNS if c in edges.columns]
        if clash:
            raise ValueError(f"Edge attributes {clash} clash with the edge identifier columns")
        edges.insert(0, "u", [r[0] for r in rows])
        edges.insert(1, "v", [r[1] for r in rows])
        edges.insert(2, "key", [r[2] for r in rows])
        metadata = {k: v for k, v in G.graph.items() if k != "crs"}
        return cls(
            nodes,
            edges,
            directed=G.is_directed(),
            crs=G.graph.get("crs", WGS84),
            weight=weight,
            metadata=metadata,
        )

    @classmethod
    def from_points(
        cls,
        points: Any,
        connect: str = "knn",
        k: int = 5,
        max_distance: Optional[float] = None,
        id_column: Optional[str] = None,
        weight: str = "length",
        weight_func: Optional[Callable[[pd.DataFrame, pd.DataFrame, np.ndarray], Any]] = None,
    ) -> "SpatialNetwork":
        """Build an undirected network whose nodes are the points.

        Args:
            points: A GeoDataFrame of points, a DataFrame with latitude and
                longitude columns, or a list of ``(lat, lon)`` tuples or Points.
            connect: ``"knn"`` joins each point to its ``k`` nearest
                neighbours; ``"distance"`` joins every pair within
                ``max_distance`` meters.
            k: Number of neighbours for ``"knn"``.
            max_distance: Longest edge in meters. Required for ``"distance"``;
                with ``"knn"`` it drops longer neighbour links.
            id_column: Column holding node IDs. The frame's index otherwise.
            weight: Default weight column. Anything but ``"length"`` needs
                ``weight_func``.
            weight_func: Computes the ``weight`` column as
                ``weight_func(nodes_u, nodes_v, length)``, where ``nodes_u`` and
                ``nodes_v`` are the node rows at each end and ``length`` is in
                meters.

        Returns:
            The network. Nodes have kind ``"point"`` and edges ``"point_link"``.
        """
        nodes = points_table(points, id_column)
        nodes.insert(2, "kind", "point")
        lat = nodes["y"].to_numpy()
        lon = nodes["x"].to_numpy()
        if connect == "knn":
            if int(k) < 1:
                raise ValueError(f"k must be at least 1, got {k}")
            a, b = _knn_pairs(lat, lon, int(k))
        elif connect == "distance":
            if max_distance is None:
                raise ValueError('connect="distance" needs max_distance')
            pairs = cKDTree(unit_vectors(lat, lon)).query_pairs(
                float(meters_to_chord(max_distance)), output_type="ndarray"
            )
            a, b = pairs[:, 0], pairs[:, 1]
        else:
            raise ValueError(f'connect must be "knn" or "distance", got {connect!r}')
        length = haversine(lat[a], lon[a], lat[b], lon[b])
        if max_distance is not None:
            keep = length <= max_distance
            a, b, length = a[keep], b[keep], length[keep]
        order = np.lexsort((b, a))
        a, b, length = a[order], b[order], length[order]

        ids = nodes.index
        edges = pd.DataFrame(
            {
                "u": ids.take(a),
                "v": ids.take(b),
                "key": np.zeros(len(a), dtype=np.int16),
                "length": length,
                "kind": "point_link",
            }
        )
        if weight_func is not None:
            if weight == "length":
                raise ValueError("weight_func needs a weight name other than 'length'")
            values = np.asarray(weight_func(nodes.iloc[a], nodes.iloc[b], length), dtype=np.float64)
            if values.shape != (len(a),):
                raise ValueError(f"weight_func returned shape {values.shape}, expected ({len(a)},)")
            edges[weight] = values
        elif weight != "length":
            raise ValueError(f"weight={weight!r} needs weight_func to compute it")
        return cls(nodes, edges, directed=False, weight=weight)

    @classmethod
    def load(
        cls,
        path: Any,
        data_store: Optional[Any] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> "SpatialNetwork":
        """Load a network saved with :meth:`save`.

        List attributes such as ``osmid`` come back as one numpy array per
        edge, and text attributes as categoricals.

        Args:
            path: Folder the network was saved to.
            data_store: Data store to read from. The local file system otherwise.
            columns: Edge attribute columns to read, in addition to ``u``,
                ``v``, ``key``, ``length``, ``kind``, and the weight column.
                Include ``"geometry"`` to keep edge geometry. All columns
                otherwise.

        Raises:
            FileNotFoundError: If there is no saved network at ``path``.
            ValueError: If the folder was saved by a newer format version, or
                ``columns`` names a column that was not saved.
        """
        from gigaspatial.processing.network.storage import load_network

        return load_network(cls, path, data_store=data_store, columns=columns)

    def save(self, path: Any, data_store: Optional[Any] = None) -> None:
        """Save the network as a folder: ``nodes.parquet``, ``edges.parquet``, and ``network.json``.

        Edges are GeoParquet with WKB geometry, readable by geopandas, DuckDB,
        and QGIS; without edge geometry, ``edges.parquet`` is plain Parquet.
        Attributes that mix lists and single values, as osmnx produces, are
        normalised first (see ``normalize_attributes``); the network itself is
        not changed. Existing files in the folder are overwritten.

        Args:
            path: Folder to write.
            data_store: Data store to write to. The local file system otherwise.

        Raises:
            ValueError: If node IDs are neither integers nor text, or a column
                cannot be stored.
            TypeError: If the metadata is not JSON serializable.
        """
        from gigaspatial.processing.network.storage import save_network

        save_network(self, path, data_store=data_store)

    # ----- basic properties

    @property
    def nodes(self) -> pd.DataFrame:
        """The nodes table, indexed by node ID."""
        return self._nodes.copy(deep=False)

    @property
    def edges(self) -> pd.DataFrame:
        """The edges table."""
        return self._edges.copy(deep=False)

    @property
    def directed(self) -> bool:
        return self._directed

    @property
    def weight(self) -> str:
        """Default edge column for algorithms."""
        return self._weight

    @property
    def crs(self) -> str:
        """CRS of the stored coordinates, always EPSG:4326."""
        return WGS84

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    @property
    def node_index(self) -> pd.Index:
        """Node IDs by position: ``node_index[i]`` is the ID at position ``i``,
        and ``node_index.get_indexer(ids)`` gives positions."""
        return self._nodes.index

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return (
            f"SpatialNetwork(nodes={len(self._nodes):,}, edges={len(self._edges):,}, "
            f"directed={self._directed}, weight={self._weight!r})"
        )

    # ----- cached derived structures

    @cached_property
    def _edge_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        index = self._nodes.index
        return (
            as_positions(index.get_indexer(self._edges["u"]), len(index)),
            as_positions(index.get_indexer(self._edges["v"]), len(index)),
        )

    def _positions(self, ids: Any) -> np.ndarray:
        """Positions of node IDs; raises KeyError for IDs not in the network."""
        if np.ndim(ids) == 0:
            ids = [ids]
        positions = self._nodes.index.get_indexer(pd.Index(ids))
        missing = positions < 0
        if missing.any():
            sample = list(pd.Index(ids)[missing][:5])
            raise KeyError(f"{int(missing.sum())} node IDs are not in the network, for example {sample}")
        return positions

    @cached_property
    def nodes_gdf(self) -> gpd.GeoDataFrame:
        """Nodes as a GeoDataFrame of Points in EPSG:4326."""
        return gpd.GeoDataFrame(
            self._nodes,
            geometry=gpd.points_from_xy(self._nodes["x"], self._nodes["y"]),
            crs=WGS84,
        )

    @cached_property
    def edges_gdf(self) -> gpd.GeoDataFrame:
        """Edges as a GeoDataFrame in EPSG:4326, with straight lines where geometry is missing."""
        table = self._edges.drop(columns="geometry", errors="ignore")
        return gpd.GeoDataFrame(table, geometry=self._filled_geometries, crs=WGS84)

    @cached_property
    def _filled_geometries(self) -> np.ndarray:
        """Edge geometries with straight lines where they are missing."""
        return self._edge_geometries(fill=True)

    def _edge_geometries(self, fill: bool) -> np.ndarray:
        if "geometry" in self._edges.columns:
            geoms = self._edges["geometry"].to_numpy(dtype=object)
        else:
            geoms = np.full(len(self._edges), None, dtype=object)
        if not fill:
            return geoms
        missing = np.flatnonzero(pd.isna(geoms))
        if len(missing):
            pu, pv = self._edge_positions
            x = self._nodes["x"].to_numpy()
            y = self._nodes["y"].to_numpy()
            ends = np.stack(
                [
                    np.column_stack([x[pu[missing]], y[pu[missing]]]),
                    np.column_stack([x[pv[missing]], y[pv[missing]]]),
                ],
                axis=1,
            )
            geoms = geoms.copy()
            geoms[missing] = shapely.linestrings(ends)
        return geoms

    # ----- accessors

    def to_geodataframes(self, crs: Any = None) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Nodes and edges as GeoDataFrames, projected to ``crs`` if given."""
        nodes, edges = self.nodes_gdf, self.edges_gdf
        if crs is None:
            return nodes.copy(deep=False), edges.copy(deep=False)
        return nodes.to_crs(crs), edges.to_crs(crs)

    def node_coordinates(self, ids: Any = None, crs: Any = None) -> np.ndarray:
        """Node coordinates as an array of ``(y, x)`` rows.

        In EPSG:4326 that is ``(lat, lon)``. With ``crs``, the coordinates are
        projected and keep the ``(y, x)`` order.

        Args:
            ids: Node IDs, in the order wanted. All nodes otherwise.
            crs: CRS to project to.

        Raises:
            KeyError: If an ID is not in the network.
        """
        x = self._nodes["x"].to_numpy()
        y = self._nodes["y"].to_numpy()
        if ids is not None:
            positions = self._positions(ids)
            x, y = x[positions], y[positions]
        if crs is not None:
            x, y = Transformer.from_crs(WGS84, crs, always_xy=True).transform(x, y)
        return np.column_stack([y, x])

    def degree(self, directed: Optional[bool] = None) -> pd.Series:
        """Number of edges at each node.

        Args:
            directed: With ``True``, every edge row counts at both of its
                ends, as networkx counts a multigraph's degree. With ``False``,
                reverse and parallel edges between the same two nodes count
                once, so the degree is the number of roads meeting at a node.
                Defaults to how the network is directed.

        Returns:
            Degrees indexed by node ID. A self-loop counts twice.
        """
        directed = self._directed if directed is None else directed
        n = len(self._nodes)
        pu, pv = self._edge_positions
        if directed:
            counts = np.bincount(pu, minlength=n) + np.bincount(pv, minlength=n)
        else:
            low = np.minimum(pu, pv).astype(np.int64)
            high = np.maximum(pu, pv).astype(np.int64)
            pairs = np.unique(low * n + high)
            counts = np.bincount(pairs // n, minlength=n) + np.bincount(pairs % n, minlength=n)
        return pd.Series(counts, index=self._nodes.index, name="degree")

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """``(minx, miny, maxx, maxy)`` of the nodes and edge geometries, in EPSG:4326."""
        if len(self._nodes) == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        x = self._nodes["x"].to_numpy()
        y = self._nodes["y"].to_numpy()
        box = [x.min(), y.min(), x.max(), y.max()]
        geoms = self._edge_geometries(fill=False)
        present = ~pd.isna(geoms)
        if present.any():
            b = shapely.bounds(geoms[present])
            box = [
                min(box[0], np.nanmin(b[:, 0])),
                min(box[1], np.nanmin(b[:, 1])),
                max(box[2], np.nanmax(b[:, 2])),
                max(box[3], np.nanmax(b[:, 3])),
            ]
        return tuple(float(v) for v in box)

    def total_length(self, by: Optional[str] = None) -> Union[float, pd.Series]:
        """Sum of edge lengths in meters, optionally grouped by an edge column.

        In a directed network a two-way road is two rows, so it counts twice.
        """
        if by is None:
            return float(self._edges["length"].sum())
        return self._edges.groupby(by, observed=True)["length"].sum()

    def info(self) -> Dict[str, Any]:
        """Summary of the network."""
        geoms = self._edge_geometries(fill=False)
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "directed": self._directed,
            "weight": self._weight,
            "crs": WGS84,
            "bounds": self.bounds,
            "total_length_m": self.total_length(),
            "node_kinds": _counts(self._nodes["kind"]),
            "edge_kinds": _counts(self._edges["kind"]),
            "edges_with_geometry": int((~pd.isna(geoms)).sum()),
            "metadata": self.metadata,
        }

    def to_networkx(self, directed: Optional[bool] = None) -> nx.MultiGraph:
        """A new networkx graph with every node and edge attribute.

        Missing values are left out of the attribute dictionaries, so straight
        edges have no ``geometry``, as in osmnx. ``G.graph`` holds the metadata
        and ``crs``. The graph is built on every call and not cached.

        Args:
            directed: ``True`` gives a ``MultiDiGraph``; from an undirected
                network, every edge is added in both directions, with its
                geometry reversed for the reverse direction. ``False`` gives a
                ``MultiGraph``; from a directed network, a reverse edge with the
                same key replaces the forward one, as networkx's
                ``to_undirected`` does. Defaults to how the network is directed.

        Returns:
            ``nx.MultiDiGraph`` or ``nx.MultiGraph``.
        """
        directed = self._directed if directed is None else bool(directed)
        G = nx.MultiDiGraph() if directed else nx.MultiGraph()
        G.graph.update(self._metadata)
        G.graph["crs"] = WGS84

        G.add_nodes_from(zip(self._nodes.index.tolist(), _records(self._nodes)))

        attributes = self._edges.drop(columns=list(_EDGE_ID_COLUMNS))
        records = _records(attributes)
        u = self._edges["u"].tolist()
        v = self._edges["v"].tolist()
        keys = self._edges["key"].tolist()
        G.add_edges_from(zip(u, v, keys, records))
        if directed and not self._directed:
            reverse = [dict(r) for r in records]
            if "geometry" in attributes.columns:
                geoms = attributes["geometry"].to_numpy(dtype=object)
                present = np.flatnonzero(~pd.isna(geoms))
                for i, g in zip(present, shapely.reverse(geoms[present])):
                    reverse[i]["geometry"] = g
            G.add_edges_from(zip(v, u, keys, reverse))
        return G


# ----- table preparation


def _prepare_nodes(nodes: pd.DataFrame, crs: Any) -> pd.DataFrame:
    if not isinstance(nodes, pd.DataFrame):
        raise TypeError(f"nodes must be a DataFrame, got {type(nodes).__name__}")
    source_crs = crs
    if isinstance(nodes, gpd.GeoDataFrame):
        if nodes.crs is not None:
            source_crs = nodes.crs
        geometry = nodes.geometry
        table = pd.DataFrame(nodes.drop(columns=geometry.name))
        if "x" not in table.columns or "y" not in table.columns:
            if not (geometry.geom_type == "Point").all():
                raise ValueError("Node geometries must be Points")
            table["x"] = geometry.x.to_numpy()
            table["y"] = geometry.y.to_numpy()
    else:
        table = nodes.copy(deep=False)

    if "node_id" in table.columns:
        table = table.set_index("node_id")
    table.index = table.index.rename("node_id")
    if not table.index.is_unique:
        duplicated = list(table.index[table.index.duplicated()].unique()[:5])
        raise ValueError(f"Node IDs must be unique; repeated: {duplicated}")
    missing = [c for c in ("x", "y") if c not in table.columns]
    if missing:
        raise ValueError(f"Nodes need {missing} columns")

    x = table["x"].to_numpy(dtype=np.float64)
    y = table["y"].to_numpy(dtype=np.float64)
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Node coordinates must not be missing")
    if not is_wgs84(source_crs):
        x, y = Transformer.from_crs(source_crs, WGS84, always_xy=True).transform(x, y)
    kinds = as_kinds(table["kind"] if "kind" in table.columns else None, "road", NODE_KINDS, len(table))

    columns = {
        "x": np.asarray(x, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "kind": kinds,
    }
    columns.update({c: table[c] for c in table.columns if c not in RESERVED_NODE_COLUMNS})
    return pd.DataFrame(columns, index=table.index, copy=False)


def _prepare_edges(
    edges: pd.DataFrame, nodes: pd.DataFrame, crs: Any, directed: bool, weight: str
) -> Tuple[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    if not isinstance(edges, pd.DataFrame):
        raise TypeError(f"edges must be a DataFrame, got {type(edges).__name__}")
    geometry_crs = crs
    if isinstance(edges, gpd.GeoDataFrame):
        if edges.crs is not None:
            geometry_crs = edges.crs
        name = edges.geometry.name
        columns = {c: edges[c] for c in edges.columns if c not in (name, "geometry")}
        columns["geometry"] = edges.geometry.to_numpy()
        table = pd.DataFrame(columns, index=edges.index, copy=False)
    else:
        table = edges.copy(deep=False)
    table.index = pd.RangeIndex(len(table))
    missing = [c for c in ("u", "v") if c not in table.columns]
    if missing:
        raise ValueError(f"Edges need {missing} columns")

    index = nodes.index
    pu = index.get_indexer(table["u"])
    pv = index.get_indexer(table["v"])
    unknown = (pu < 0) | (pv < 0)
    if unknown.any():
        sample = table.loc[unknown, ["u", "v"]].head(5).values.tolist()
        raise ValueError(f"{int(unknown.sum())} edges reference nodes not in the network, for example {sample}")

    if "key" not in table.columns:
        table["key"] = table.groupby(["u", "v"], sort=False).cumcount()
    key = table["key"].to_numpy()
    if len(key) and (key.min() < 0 or key.max() > np.iinfo(np.int16).max):
        raise ValueError("Edge keys must be between 0 and 32767")
    key = key.astype(np.int16)
    if directed:
        repeated = repeated_rows(pu, pv, key)
    else:
        repeated = repeated_rows(np.minimum(pu, pv), np.maximum(pu, pv), key)
    if repeated.any():
        sample = table.loc[repeated, ["u", "v"]].head(5).values.tolist()
        raise ValueError(f"{int(repeated.sum())} edges repeat a (u, v, key), for example {sample}")

    if "geometry" in table.columns:
        geoms = table["geometry"].to_numpy(dtype=object).copy()
        absent = pd.isna(geoms)
        geoms[absent] = None
        present = np.flatnonzero(~absent)
        if len(present):
            if not (shapely.get_type_id(geoms[present]) == 1).all():
                raise ValueError("Edge geometries must be LineStrings")
            if not is_wgs84(geometry_crs):
                geoms[present] = gpd.GeoSeries(geoms[present], crs=geometry_crs).to_crs(WGS84).to_numpy()
        table["geometry"] = geoms
    else:
        geoms = None

    if "length" in table.columns:
        length = table["length"].to_numpy(dtype=np.float64).copy()
    else:
        length = np.full(len(table), np.nan)
    todo = np.flatnonzero(np.isnan(length))
    if len(todo):
        if geoms is not None:
            length[todo] = line_lengths(geoms[todo])
        straight = todo[np.isnan(length[todo])]
        x = nodes["x"].to_numpy()
        y = nodes["y"].to_numpy()
        length[straight] = haversine(y[pu[straight]], x[pu[straight]], y[pv[straight]], x[pv[straight]])

    kinds = as_kinds(table["kind"] if "kind" in table.columns else None, "road", EDGE_KINDS, len(table))
    columns = {
        "u": table["u"].to_numpy(),
        "v": table["v"].to_numpy(),
        "key": key,
        "length": length,
        "kind": kinds,
    }
    skip = (*_EDGE_ID_COLUMNS, "length", "kind", "geometry")
    columns.update({c: table[c] for c in table.columns if c not in skip})
    if geoms is not None:
        columns["geometry"] = geoms
    prepared = pd.DataFrame(columns, index=table.index, copy=False)

    if weight not in prepared.columns:
        raise ValueError(f"The weight column {weight!r} is not an edge column")
    if not pd.api.types.is_numeric_dtype(prepared[weight]):
        raise ValueError(f"The weight column {weight!r} must be numeric")
    return prepared, (as_positions(pu, len(index)), as_positions(pv, len(index)))


def _counts(kinds: pd.Series) -> Dict[str, int]:
    counts = kinds.value_counts()
    return {str(k): int(v) for k, v in counts[counts > 0].items()}


def _records(table: pd.DataFrame) -> list:
    """Row dictionaries without missing values.

    OSM attributes are mostly missing, so mostly-missing columns are added
    only where present instead of being written and deleted for every row.
    """
    n = len(table)
    missing = {c: table[c].isna().to_numpy() for c in table.columns}
    dense = [c for c in table.columns if missing[c].sum() <= n // 2]
    sparse = [c for c in table.columns if c not in dense]
    if dense:
        columns = [table[c].tolist() for c in dense]
        records = [dict(zip(dense, row)) for row in zip(*columns)]
    else:
        records = [{} for _ in range(n)]
    for c in dense:
        for i in np.flatnonzero(missing[c]):
            del records[i][c]
    for c in sparse:
        values = table[c].tolist()
        for i in np.flatnonzero(~missing[c]):
            records[i][c] = values[i]
    return records


# ----- points


def _knn_pairs(lat: np.ndarray, lon: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unordered pairs joining each point to its k nearest neighbours."""
    n = len(lat)
    if n < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    k = min(k, n - 1)
    xyz = unit_vectors(lat, lon)
    _, neighbours = cKDTree(xyz).query(xyz, k=k + 1)
    rows = np.broadcast_to(np.arange(n)[:, None], neighbours.shape)
    not_self = neighbours != rows
    keep = not_self & (np.cumsum(not_self, axis=1) <= k)
    a, b = rows[keep], neighbours[keep]
    pairs = np.unique(np.column_stack([np.minimum(a, b), np.maximum(a, b)]), axis=0)
    return pairs[:, 0], pairs[:, 1]
