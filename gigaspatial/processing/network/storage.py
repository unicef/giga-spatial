"""Saving and loading spatial networks as a folder of Parquet files (ADR 0005)."""

import datetime
import io
import json
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import CRS

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.processing.network.attributes import normalize_attributes

if TYPE_CHECKING:
    from gigaspatial.processing.network.core import SpatialNetwork

FORMAT = "gigaspatial.spatial_network"
FORMAT_VERSION = 1
NODES_FILE = "nodes.parquet"
EDGES_FILE = "edges.parquet"
METADATA_FILE = "network.json"
_ID_IS_INT = "node_id_is_int"
_REQUIRED_EDGE_COLUMNS = ("u", "v", "key", "length", "kind")


def _join(folder: Any, name: str) -> str:
    return str(PurePosixPath(str(folder)) / name)


def save_network(network: "SpatialNetwork", path: Any, data_store: Optional[DataStore] = None) -> None:
    data_store = data_store or LocalDataStore()
    nodes = network._nodes
    edges = network._edges
    node_ids, u, v, flags = _encode_ids(nodes.index, edges["u"], edges["v"])

    node_table = normalize_attributes(nodes, skip=("x", "y", "kind"))
    node_columns = {"node_id": node_ids}
    if flags is not None:
        node_columns[_ID_IS_INT] = flags
    node_columns.update({c: node_table[c] for c in node_table.columns})
    node_frame = pd.DataFrame(node_columns, copy=False)
    node_frame.index = pd.RangeIndex(len(node_frame))

    edge_table = normalize_attributes(edges, skip=(*_REQUIRED_EDGE_COLUMNS, "geometry"))
    edge_columns = {c: edge_table[c] for c in edge_table.columns}
    edge_columns["u"], edge_columns["v"] = u, v
    edge_frame = pd.DataFrame(edge_columns, index=edge_table.index, copy=False)

    metadata = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "directed": network.directed,
        "crs": network.crs,
        "weight": network.weight,
        "node_ids": "native" if flags is None else "mixed",
        "metadata": network._metadata,
        "library_version": _library_version(),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        metadata_text = json.dumps(metadata, indent=2)
    except TypeError as error:
        raise TypeError(f"Network metadata must be JSON serializable: {error}") from error

    nodes_bytes = _parquet_bytes(node_frame, "nodes")
    if "geometry" in edge_frame.columns:
        edges_bytes = _geoparquet_bytes(edge_frame)
    else:
        edges_bytes = _parquet_bytes(edge_frame, "edges")
    _write(data_store, _join(path, NODES_FILE), nodes_bytes)
    _write(data_store, _join(path, EDGES_FILE), edges_bytes)
    # Written last, so a folder with network.json always holds complete tables.
    with data_store.open(_join(path, METADATA_FILE), "w") as f:
        f.write(metadata_text)


def load_network(
    cls: type,
    path: Any,
    data_store: Optional[DataStore] = None,
    columns: Optional[Iterable[str]] = None,
) -> "SpatialNetwork":
    data_store = data_store or LocalDataStore()
    metadata_path = _join(path, METADATA_FILE)
    if not data_store.file_exists(metadata_path):
        raise FileNotFoundError(f"No saved spatial network at {path}: {METADATA_FILE} is missing")
    meta = json.loads(data_store.read_file(metadata_path, encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{metadata_path} does not describe a saved spatial network")
    version = meta.get("format_version")
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise ValueError(
            f"{path} was saved with spatial network format version {version}; this version of "
            f"giga-spatial reads up to version {FORMAT_VERSION}. Upgrade giga-spatial to load it."
        )

    with data_store.open(_join(path, NODES_FILE), "rb") as f:
        nodes = pq.read_table(f).to_pandas(types_mapper=_arrow_lists, ignore_metadata=True)
    edges = _read_edges(data_store, _join(path, EDGES_FILE), meta["weight"], columns)

    if meta.get("node_ids") == "mixed":
        nodes, edges = _decode_ids(nodes, edges)
    return cls(
        nodes,
        edges,
        directed=meta["directed"],
        crs=meta.get("crs", "EPSG:4326"),
        weight=meta["weight"],
        metadata=meta.get("metadata"),
    )


def _arrow_lists(arrow_type: pa.DataType) -> Optional[pd.ArrowDtype]:
    """Keep list columns as Arrow lists: compact, and each value reads as a Python list."""
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return pd.ArrowDtype(arrow_type)
    return None


def _read_edges(data_store: DataStore, path: str, weight: str, columns: Optional[Iterable[str]]) -> pd.DataFrame:
    with data_store.open(path, "rb") as f:
        schema = pq.ParquetFile(f).schema_arrow
        f.seek(0)
        names = schema.names
        if columns is None:
            wanted = names
        else:
            requested = [c for c in dict.fromkeys([*_REQUIRED_EDGE_COLUMNS, weight, *columns])]
            if "geometry" not in names and "geometry" in requested:
                requested.remove("geometry")
            unknown = [c for c in requested if c not in names]
            if unknown:
                raise ValueError(f"Edge columns {unknown} are not in the saved network")
            wanted = requested
        if "geometry" in wanted and _geometry_encoding(schema) != "WKB":
            table = gpd.read_parquet(f, columns=wanted)
            frame = pd.DataFrame({c: table[c] for c in table.columns if c != "geometry"}, copy=False)
            frame["geometry"] = table.geometry.to_numpy()
            return frame
        table = pq.read_table(f, columns=wanted)
    if "geometry" not in wanted:
        return table.to_pandas(types_mapper=_arrow_lists, ignore_metadata=True)
    wkb = table.column("geometry").to_numpy(zero_copy_only=False)
    frame = table.drop_columns(["geometry"]).to_pandas(types_mapper=_arrow_lists, ignore_metadata=True)
    frame["geometry"] = shapely.from_wkb(wkb)
    return frame


def _geometry_encoding(schema: pa.Schema) -> Optional[str]:
    geo = (schema.metadata or {}).get(b"geo")
    if geo is None:
        return None
    return json.loads(geo).get("columns", {}).get("geometry", {}).get("encoding")


def _write(data_store: DataStore, path: str, data: bytes) -> None:
    with data_store.open(path, "wb") as f:
        f.write(data)


def _parquet_bytes(frame: pd.DataFrame, what: str) -> bytes:
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as error:
        raise ValueError(f"Could not save the {what} table: {error}") from error
    return _table_bytes(table)


def _geoparquet_bytes(frame: pd.DataFrame) -> bytes:
    """GeoParquet 1.0: WKB geometry and the file-level ``geo`` metadata that GIS tools read."""
    geoms = frame["geometry"].to_numpy(dtype=object)
    attributes = pd.DataFrame({c: frame[c] for c in frame.columns if c != "geometry"}, copy=False)
    try:
        table = pa.Table.from_pandas(attributes, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as error:
        raise ValueError(f"Could not save the edges table: {error}") from error
    table = table.append_column("geometry", pa.array(shapely.to_wkb(geoms), type=pa.binary()))
    present = geoms[~pd.isna(geoms)]
    geo = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["LineString"] if len(present) else [],
                "crs": CRS.from_epsg(4326).to_json_dict(),
                "bbox": [float(v) for v in shapely.total_bounds(present)] if len(present) else [],
            }
        },
    }
    return _table_bytes(table, {b"geo": json.dumps(geo).encode()})


def _table_bytes(table: pa.Table, extra: Optional[dict] = None) -> bytes:
    # pandas metadata would name dtypes such as list<item: int64>[pyarrow], which pandas cannot read back.
    metadata = {k: v for k, v in (table.schema.metadata or {}).items() if k != b"pandas"}
    metadata.update(extra or {})
    buffer = io.BytesIO()
    pq.write_table(table.replace_schema_metadata(metadata), buffer, compression="zstd")
    return buffer.getvalue()


def _encode_ids(
    index: pd.Index, u: pd.Series, v: pd.Series
) -> Tuple[Any, Any, Any, Optional[np.ndarray]]:
    """Node IDs as Parquet can store them: native when all integers or all text, else text with a flag."""
    inferred = pd.api.types.infer_dtype(index, skipna=False)
    if inferred in ("integer", "string"):
        return index.to_numpy(), u.to_numpy(), v.to_numpy(), None
    values = index.to_numpy(dtype=object)
    is_int = np.fromiter(
        (isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)) for x in values),
        dtype=bool,
        count=len(values),
    )
    is_str = np.fromiter((isinstance(x, str) for x in values), dtype=bool, count=len(values))
    if not (is_int | is_str).all():
        bad = list(values[~(is_int | is_str)][:5])
        raise ValueError(f"Node IDs must be integers or text to be saved, got for example {bad}")
    text = values.astype(str)
    if not pd.Index(text).is_unique:
        raise ValueError("Node IDs cannot be saved: an integer ID and a text ID have the same digits, such as 5 and '5'")
    as_text = pd.Series(text, index=index)
    return text, as_text.loc[u].to_numpy(), as_text.loc[v].to_numpy(), is_int


def _decode_ids(nodes: pd.DataFrame, edges: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    text = nodes["node_id"].to_numpy(dtype=object)
    flags = nodes[_ID_IS_INT].to_numpy(dtype=bool)
    ids = text.copy()
    ids[flags] = [int(x) for x in text[flags]]
    lookup = pd.Index(text)
    nodes = nodes.drop(columns=[_ID_IS_INT])
    nodes["node_id"] = ids
    edges["u"] = ids[lookup.get_indexer(edges["u"])]
    edges["v"] = ids[lookup.get_indexer(edges["v"])]
    return nodes, edges


def _library_version() -> str:
    from gigaspatial import __version__

    return __version__
