"""
Road networks read from OpenStreetMap ``.osm.pbf`` files with pyosmium.

``build_road_network`` returns the network osmnx 2.0.2 builds for the same
ways, without osmnx or networkx: the same ways for each ``network_type``, the
same nodes, edges, keys, one-way rules, kept tags, lengths, and
``street_count``. The rules reproduced here are in
``docs/design/osm-roads-handler.md``, section 7.

The file is read once. pyosmium keeps a node location index while it reads,
and writes each kept way's line in C++, so Python touches each way but no
node coordinate; the edges are then cut and measured in numpy.
"""

import re
import tempfile
import time
from array import array
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import shapely

from gigaspatial.config import config as global_config
from gigaspatial.processing.network import SpatialNetwork
from gigaspatial.processing.network.geodesy import haversine

# osmnx 2.0.2 settings.useful_tags_way and useful_tags_node.
USEFUL_TAGS_WAY = (
    "access",
    "area",
    "bridge",
    "est_width",
    "highway",
    "junction",
    "landuse",
    "lanes",
    "maxspeed",
    "name",
    "oneway",
    "ref",
    "service",
    "tunnel",
    "width",
)
USEFUL_TAGS_NODE = ("highway", "junction", "railway", "ref")
# osmnx 2.0.2 graph._add_paths.
ONEWAY_VALUES = frozenset({"yes", "true", "1", "-1", "reverse", "T", "F"})
REVERSED_VALUES = frozenset({"-1", "reverse", "T"})
BIDIRECTIONAL_NETWORK_TYPES = frozenset({"walk"})

_PRIVATE = ("access", "private")
_NOT_AREA = ("area", "yes")
# osmnx 2.0.2 _overpass._get_network_filter: every preset needs a highway tag,
# then excludes ways whose key matches the pattern, as Overpass "!~" does: an
# unanchored regular expression, which a way without the key passes.
NETWORK_FILTERS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "drive": (
        _NOT_AREA,
        _PRIVATE,
        (
            "highway",
            "abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|no|path|"
            "pedestrian|planned|platform|proposed|raceway|razed|service|steps|track",
        ),
        ("motor_vehicle", "no"),
        ("motorcar", "no"),
        ("service", "alley|driveway|emergency_access|parking|parking_aisle|private"),
    ),
    "drive_service": (
        _NOT_AREA,
        _PRIVATE,
        (
            "highway",
            "abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|no|path|"
            "pedestrian|planned|platform|proposed|raceway|razed|steps|track",
        ),
        ("motor_vehicle", "no"),
        ("motorcar", "no"),
        ("service", "emergency_access|parking|parking_aisle|private"),
    ),
    "walk": (
        _NOT_AREA,
        _PRIVATE,
        ("highway", "abandoned|bus_guideway|construction|cycleway|motor|no|planned|platform|proposed|raceway|razed"),
        ("foot", "no"),
        ("service", "private"),
        ("sidewalk", "separate"),
        ("sidewalk:both", "separate"),
        ("sidewalk:left", "separate"),
        ("sidewalk:right", "separate"),
    ),
    "bike": (
        _NOT_AREA,
        _PRIVATE,
        (
            "highway",
            "abandoned|bus_guideway|construction|corridor|elevator|escalator|footway|motor|no|planned|platform|"
            "proposed|raceway|razed|steps",
        ),
        ("bicycle", "no"),
        ("service", "private"),
    ),
    "all_public": (
        _NOT_AREA,
        _PRIVATE,
        ("highway", "abandoned|construction|no|planned|platform|proposed|raceway|razed"),
        ("service", "private"),
    ),
    "all": (
        _NOT_AREA,
        ("highway", "abandoned|construction|no|planned|platform|proposed|raceway|razed"),
    ),
}

# The in-memory index takes about 16 bytes per node in the file; above this
# extract size (Nigeria is 680 MiB, about 2 GB of index) it moves to disk.
IN_MEMORY_INDEX_MAX_BYTES = 1 << 30


def way_filter(network_type: str = "drive_service", tags: Optional[Dict[str, Optional[Sequence[str]]]] = None):
    """
    The test a way's tags must pass, and the keys pyosmium preselects ways by.

    Args:
        network_type: An osmnx network type.
        tags: Allowed values per key, such as ``{"highway": ["primary"]}``,
            replacing ``network_type``. A way must match every key; ``None``
            as the values allows any value of that key.

    Returns:
        ``(passes, keys)``: ``passes(get)`` takes a tag lookup such as
        ``dict.get``; a way must carry one of ``keys`` to be tested at all.

    Raises:
        ValueError: If the network type is unknown or ``tags`` is empty.
    """
    if tags is not None:
        if not tags:
            raise ValueError("tags must name at least one key")
        allowed = [(key, None if values is None else frozenset(values)) for key, values in tags.items()]

        def passes(get: Callable[[str], Optional[str]]) -> bool:
            for key, values in allowed:
                value = get(key)
                if value is None or (values is not None and value not in values):
                    return False
            return True

        return passes, tuple(tags)

    if network_type not in NETWORK_FILTERS:
        raise ValueError(f"Unknown network_type {network_type!r}; use one of {sorted(NETWORK_FILTERS)}")
    excluded = [(key, re.compile(pattern).search) for key, pattern in NETWORK_FILTERS[network_type]]

    def passes(get: Callable[[str], Optional[str]]) -> bool:
        for key, search in excluded:
            value = get(key)
            if value is not None and search(value):
                return False
        return True

    return passes, ("highway",)


def node_index_for(size_bytes: int, requested: Optional[str] = None) -> str:
    """
    The pyosmium node location index for an extract of this size.

    ``flex_mem`` in memory up to ``IN_MEMORY_INDEX_MAX_BYTES``, and the
    disk-backed ``sparse_file_array`` above it.

    Raises:
        ValueError: For a dense index, which on 64-bit node IDs needs about
            100 GB even for a small country, or an unknown index.
    """
    name = requested or ("flex_mem" if size_bytes <= IN_MEMORY_INDEX_MAX_BYTES else "sparse_file_array")
    kind = name.split(",", 1)[0]
    if kind.startswith("dense"):
        raise ValueError(f"{kind} needs memory or disk for every possible node ID (about 100 GB); use flex_mem or sparse_file_array")
    known = ("flex_mem", "sparse_file_array", "sparse_mem_array", "sparse_mem_map")
    if kind not in known:
        raise ValueError(f"Unknown node_index {name!r}; use one of {list(known)}")
    return name


def build_road_network(
    path: Union[str, Path],
    *,
    network_type: str = "drive_service",
    tags: Optional[Dict[str, Optional[Sequence[str]]]] = None,
    simplify: bool = True,
    node_index: Optional[str] = None,
    extract: Optional[Dict[str, Any]] = None,
    logger: Any = None,
) -> SpatialNetwork:
    """
    The road network of a local ``.osm.pbf`` file, as osmnx builds it.

    Every connected part is kept: choosing the largest happens at load.

    Args:
        path: A local ``.osm.pbf`` file; pyosmium reads no file objects.
        network_type: An osmnx network type: ``"drive_service"``, ``"drive"``,
            ``"walk"``, ``"bike"``, ``"all_public"``, or ``"all"``.
        tags: Allowed values per OSM key, replacing ``network_type``.
        simplify: Join edges between junctions, as osmnx does.
        node_index: pyosmium node location index; chosen from the file's size otherwise.
        extract: What the file is, such as a Geofabrik extract's ``id``,
            ``date``, and ``url``. The file's name and header otherwise.
        logger: Logger for progress.

    Returns:
        A directed network in EPSG:4326 with lengths in meters. Its metadata
        records the extract, with its replication timestamp, under
        ``"extract"``, and the build options under ``"build"``.

    Raises:
        ImportError: If pyosmium is not installed.
        ValueError: If no way passes the filter.
    """
    logger = logger or global_config.get_logger("OSMRoadsHandler")
    path = Path(path)
    passes, keys = way_filter(network_type, tags)
    bidirectional = tags is None and network_type in BIDIRECTIONAL_NETWORK_TYPES
    index = node_index_for(path.stat().st_size, node_index)
    started = time.perf_counter()
    ways = _read(path, passes, keys, index)
    if not ways.ids:
        raise ValueError(f"No ways in {path.name} match {tags or network_type}")
    if ways.skipped:
        logger.warning(f"Skipped {ways.skipped:,} ways with nodes missing from {path.name}")
    stamp = ways.replication_timestamp
    described = {"id": path.name.split(".", 1)[0], "date": stamp[:10] if stamp else None, "url": None}
    built = {
        "extract": {**described, **(extract or {}), "replication_timestamp": stamp},
        "build": {
            "builder": "OSMRoadsHandler",
            "network_type": None if tags is not None else network_type,
            "tags": None if tags is None else {k: None if v is None else list(v) for k, v in tags.items()},
            "simplified": bool(simplify),
            "retain_all": True,
            "node_index": index.split(",", 1)[0],
            "pyosmium": _osmium_version(),
            "osmnx_rules": "2.0.2",
        },
    }
    network = _assemble(ways, bidirectional, built)
    raw_size = (len(network.nodes), len(network.edges))
    if simplify:
        network = network.simplify()
    logger.info(
        f"Built {network_type if tags is None else 'tagged'} roads of {path.name}: {raw_size[0]:,} nodes and "
        f"{raw_size[1]:,} edges"
        + (f", simplified to {len(network.nodes):,} and {len(network.edges):,}" if simplify else "")
        + f", in {time.perf_counter() - started:.1f} s"
    )
    return network


class _Ways:
    """What one read of a file keeps: the passing ways, their nodes' coordinates, and tagged nodes."""

    def __init__(self):
        # Typed arrays hold a node ID in 8 bytes, a Python list in about 40.
        self.ids = array("q")
        self.counts = array("q")
        self.refs = array("q")
        self.tags: List[Dict[str, str]] = []
        self.lines: List[bytes] = []
        self.node_tags: Dict[int, Dict[str, str]] = {}
        self.skipped = 0
        self.replication_timestamp: Optional[str] = None


def _read(path: Path, passes, keys: Tuple[str, ...], node_index: str) -> _Ways:
    osmium = import_osmium()
    ways = _Ways()
    factory = osmium.geom.WKBFactory()
    every_node = osmium.geom.ALL
    with tempfile.TemporaryDirectory(prefix="gigaspatial-osm-") as scratch:
        if node_index == "sparse_file_array":
            node_index = f"sparse_file_array,{Path(scratch) / 'nodes.idx'}"
        processor = (
            osmium.FileProcessor(str(path), osmium.osm.NODE | osmium.osm.WAY)
            .with_locations(node_index)
            .with_filter(osmium.filter.KeyFilter(*keys).enable_for(osmium.osm.WAY))
            .with_filter(osmium.filter.KeyFilter(*USEFUL_TAGS_NODE).enable_for(osmium.osm.NODE))
        )
        ways.replication_timestamp = processor.header.get("osmosis_replication_timestamp") or None
        ids, counts, refs, kept, lines = ways.ids, ways.counts, ways.refs, ways.tags, ways.lines
        for entity in processor:
            if entity.is_node():
                found = {k: v for k in USEFUL_TAGS_NODE if (v := entity.tags.get(k)) is not None}
                ways.node_tags[entity.id] = found
                continue
            get = entity.tags.get
            if not passes(get):
                continue
            nodes = entity.nodes
            if len(nodes) < 2:
                continue
            try:
                line = factory.create_linestring(nodes, every_node)
            except osmium.InvalidLocationError:
                ways.skipped += 1
                continue
            way_refs = [n.ref for n in nodes]
            ids.append(entity.id)
            counts.append(len(way_refs))
            refs.extend(way_refs)
            kept.append({k: v for k in USEFUL_TAGS_WAY if (v := get(k)) is not None})
            lines.append(line)
    return ways


def _assemble(ways: _Ways, bidirectional: bool, metadata: Dict[str, Any]) -> SpatialNetwork:
    n_ways = len(ways.ids)
    refs = np.frombuffer(ways.refs, dtype=np.int64).copy()
    way = np.repeat(np.arange(n_ways), np.frombuffer(ways.counts, dtype=np.int64))
    xy = shapely.get_coordinates(shapely.from_wkb(ways.lines))
    ways.refs, ways.lines = array("q"), []

    # osmnx drops consecutive repeats of a node within a way.
    keep = np.ones(len(refs), dtype=bool)
    keep[1:] = (refs[1:] != refs[:-1]) | (way[1:] != way[:-1])
    refs, way, xy = refs[keep], way[keep], xy[keep]

    oneway_tag = [t.get("oneway") for t in ways.tags]
    if bidirectional:
        one_way = np.zeros(n_ways, dtype=bool)
    else:
        one_way = np.array(
            [o in ONEWAY_VALUES or t.get("junction") == "roundabout" for o, t in zip(oneway_tag, ways.tags)], dtype=bool
        )
    flipped = one_way & np.array([o in REVERSED_VALUES for o in oneway_tag], dtype=bool)

    # A one-way way against its node order is read backwards.
    if flipped.any():
        counts = np.bincount(way, minlength=n_ways)
        starts = np.cumsum(counts) - counts
        offset = np.arange(len(refs)) - starts[way]
        source = np.where(flipped[way], starts[way] + counts[way] - 1 - offset, np.arange(len(refs)))
        refs, xy = refs[source], xy[source]

    node_ids, first = np.unique(refs, return_index=True)
    position = np.searchsorted(node_ids, refs)
    same_way = way[1:] == way[:-1]
    a, b, edge_way = position[:-1][same_way], position[1:][same_way], way[:-1][same_way]

    # osmnx adds each way's edges in node order, then for a two-way way the
    # same edges reversed; parallel edges are keyed in that order.
    back = ~one_way[edge_way]
    pu = np.concatenate([a, b[back]])
    pv = np.concatenate([b, a[back]])
    owner = np.concatenate([edge_way, edge_way[back]])
    is_reverse = np.concatenate([np.zeros(len(a), dtype=bool), np.ones(int(back.sum()), dtype=bool)])
    step = np.concatenate([np.arange(len(a)), np.flatnonzero(back)])
    order = np.lexsort((step, is_reverse, owner))
    pu, pv, owner, is_reverse = pu[order], pv[order], owner[order], is_reverse[order]
    key = _insertion_keys(pu, pv, len(node_ids))

    x, y = xy[first, 0], xy[first, 1]
    nodes = pd.DataFrame({"x": x, "y": y}, index=pd.Index(node_ids, name="node_id"))
    nodes["street_count"] = _street_counts(pu, pv, key, len(node_ids))
    for name, column in _node_tag_columns(node_ids, ways.node_tags).items():
        nodes[name] = column

    edges = {
        "u": node_ids[pu],
        "v": node_ids[pv],
        "key": key,
        "length": haversine(y[pu], x[pu], y[pv], x[pv]),
        "osmid": np.frombuffer(ways.ids, dtype=np.int64)[owner],
        "oneway": one_way[owner],
        "reversed": is_reverse,
    }
    for name in USEFUL_TAGS_WAY:
        if name == "oneway":
            continue
        values = [t.get(name) for t in ways.tags]
        if any(v is not None for v in values):
            per_way = pd.Categorical(values)
            edges[name] = pd.Categorical.from_codes(per_way.codes[owner], categories=per_way.categories)
    return SpatialNetwork(nodes, pd.DataFrame(edges, copy=False), directed=True, metadata=metadata)


def _insertion_keys(pu: np.ndarray, pv: np.ndarray, n_nodes: int) -> np.ndarray:
    """Edge keys as networkx gives them: 0, 1, ... per (u, v), in insertion order."""
    pair = pu.astype(np.int64) * n_nodes + pv
    order = np.argsort(pair, kind="stable")
    sorted_pair = pair[order]
    starts = np.r_[True, sorted_pair[1:] != sorted_pair[:-1]]
    first = np.maximum.accumulate(np.where(starts, np.arange(len(pair)), 0))
    key = np.empty(len(pair), dtype=np.int16)
    key[order] = np.arange(len(pair)) - first
    return key


def _street_counts(pu: np.ndarray, pv: np.ndarray, key: np.ndarray, n_nodes: int) -> np.ndarray:
    """osmnx's count_streets_per_node: undirected (u, v, key) edges per node, a self-loop counting 2 once."""
    loop = pu == pv
    a = np.minimum(pu, pv)[~loop]
    b = np.maximum(pu, pv)[~loop]
    k = key[~loop]
    order = np.lexsort((k, b, a))
    a, b, k = a[order], b[order], k[order]
    unique = np.r_[True, (a[1:] != a[:-1]) | (b[1:] != b[:-1]) | (k[1:] != k[:-1])]
    counts = np.bincount(a[unique], minlength=n_nodes) + np.bincount(b[unique], minlength=n_nodes)
    counts[np.unique(pu[loop])] += 2
    return counts.astype(np.int64)


def _node_tag_columns(node_ids: np.ndarray, node_tags: Dict[int, Dict[str, str]]) -> Dict[str, pd.Categorical]:
    tagged = np.array(sorted(node_tags), dtype=np.int64)
    rows = np.searchsorted(node_ids, tagged)
    inside = rows < len(node_ids)
    inside[inside] = node_ids[rows[inside]] == tagged[inside]
    columns = {}
    for name in USEFUL_TAGS_NODE:
        values = np.full(len(node_ids), None, dtype=object)
        for row, node in zip(rows[inside], tagged[inside]):
            value = node_tags[int(node)].get(name)
            if value is not None:
                values[row] = value
        if any(v is not None for v in values[rows[inside]]):
            columns[name] = pd.Categorical(values)
    return columns


def import_osmium():
    try:
        import osmium
    except ImportError as error:
        raise ImportError(
            "OSMRoadsHandler requires 'osmium' (pyosmium). Install it with: pip install 'giga-spatial[roads]'"
        ) from error
    return osmium


def _osmium_version() -> Optional[str]:
    try:
        return version("osmium")
    except PackageNotFoundError:
        return None
