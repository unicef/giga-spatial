import io
import json
from contextlib import contextmanager

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import shapely
from shapely.geometry import LineString

import gigaspatial
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.processing.network import SpatialNetwork
from gigaspatial.processing.network.attributes import normalize_attributes


class MemoryDataStore(DataStore):
    """A remote-style store: files are buffers, written when closed. Only open, read_file, and file_exists work."""

    def __init__(self):
        self.files = {}

    def read_file(self, path, encoding=None):
        data = self.files[path]
        return data.decode(encoding) if encoding else data

    def file_exists(self, path):
        return path in self.files

    @contextmanager
    def open(self, file, mode="r", encoding=None):
        if mode in ("w", "wb"):
            buffer = io.StringIO() if mode == "w" else io.BytesIO()
            yield buffer
            value = buffer.getvalue()
            self.files[file] = value.encode() if mode == "w" else value
        elif mode == "rb":
            yield io.BytesIO(self.files[file])
        else:
            yield io.StringIO(self.files[file].decode())

    def write_file(self, path, data, encoding=None):
        raise NotImplementedError

    def list_files(self, path):
        raise NotImplementedError

    def walk(self, top):
        raise NotImplementedError

    def is_file(self, path):
        raise NotImplementedError

    def is_dir(self, path):
        raise NotImplementedError

    def remove(self, path):
        raise NotImplementedError

    def rmdir(self, dir):
        raise NotImplementedError


def assert_same_network(a, b):
    assert (a.directed, a.weight, a.metadata) == (b.directed, b.weight, b.metadata)
    pd.testing.assert_index_equal(a.node_index, b.node_index)
    for column in ("x", "y", "kind"):
        assert a.nodes[column].tolist() == b.nodes[column].tolist()
    left = a.edges.set_index(["u", "v", "key"]).sort_index()
    right = b.edges.set_index(["u", "v", "key"]).sort_index()
    assert left.index.equals(right.index)
    assert left["length"].tolist() == right["length"].tolist()
    assert left["kind"].tolist() == right["kind"].tolist()
    if "geometry" in left.columns:
        lg, rg = left["geometry"].to_numpy(), right["geometry"].to_numpy()
        assert (pd.isna(lg) == pd.isna(rg)).all()
        present = ~pd.isna(lg)
        assert shapely.equals_exact(lg[present], rg[present], tolerance=0).all()


# ----- normalize_attributes


def test_normalize_keeps_typed_columns_and_categorises_text():
    table = pd.DataFrame({"length": [1.0, 2.0], "name": ["a", None], "lanes": ["2", 3]})
    out = normalize_attributes(table)
    assert out["length"] is table["length"] or out["length"].tolist() == [1.0, 2.0]
    assert isinstance(out["name"].dtype, pd.CategoricalDtype)
    assert out["name"].tolist()[0] == "a" and pd.isna(out["name"].tolist()[1])
    assert out["lanes"].tolist() == ["2", "3"]


def test_normalize_lists():
    table = pd.DataFrame(
        {
            "osmid": [[1, 2], 3],
            "highway": [["primary", "primary", "secondary"], "residential"],
            "oneway": [[True, False], True],
            "reversed": [[False, True], False],
            "maxspeed": [None, ["50", "60"]],
        }
    )
    out = normalize_attributes(table)
    assert out["osmid"].tolist() == [[1, 2], [3]]
    assert out["highway"].tolist() == ["primary;secondary", "residential"]
    assert out["oneway"].tolist() == [False, True]
    assert out["reversed"].tolist() == [True, False]
    assert pd.isna(out["maxspeed"].tolist()[0]) and out["maxspeed"].tolist()[1] == "50;60"


def test_normalize_int_lists_wraps_single_values():
    out = normalize_attributes(pd.DataFrame({"osmid": [5, 6]}), int_lists=["osmid"])
    assert out["osmid"].tolist() == [[5], [6]]


def test_normalize_keeps_integer_arrays_from_parquet():
    column = pd.Series([np.array([1, 2]), np.array([3])], dtype=object)
    out = normalize_attributes(pd.DataFrame({"osmid": column}))
    assert out["osmid"] is column or [list(v) for v in out["osmid"]] == [[1, 2], [3]]


# ----- save and load


def test_round_trip(kigali, tmp_path):
    kigali.save(tmp_path / "kigali")
    loaded = SpatialNetwork.load(tmp_path / "kigali")
    assert_same_network(kigali, loaded)
    original = kigali.edges.set_index(["u", "v", "key"]).sort_index()
    rebuilt = loaded.edges.set_index(["u", "v", "key"]).sort_index()
    assert [list(v) for v in rebuilt["osmid"]] == [list(v) for v in original["osmid"]]
    assert rebuilt["highway"].astype(str).tolist() == original["highway"].astype(str).tolist()
    assert loaded.nodes["street_count"].tolist() == kigali.nodes["street_count"].tolist()


def test_saved_files(kigali, tmp_path):
    kigali.save(tmp_path / "kigali")
    assert sorted(p.name for p in (tmp_path / "kigali").iterdir()) == ["edges.parquet", "network.json", "nodes.parquet"]
    assert b"geo" in pq.read_schema(tmp_path / "kigali" / "edges.parquet").metadata
    assert pq.read_schema(tmp_path / "kigali" / "nodes.parquet").names[:4] == ["node_id", "x", "y", "kind"]
    meta = json.loads((tmp_path / "kigali" / "network.json").read_text())
    assert meta["format"] == "gigaspatial.spatial_network" and meta["format_version"] == 1
    assert (meta["directed"], meta["crs"], meta["weight"], meta["node_ids"]) == (True, "EPSG:4326", "length", "native")
    assert meta["library_version"] == gigaspatial.__version__
    assert meta["metadata"] == kigali.metadata


def test_remote_style_data_store(kigali):
    store = MemoryDataStore()
    kigali.save("networks/kigali", data_store=store)
    assert set(store.files) == {f"networks/kigali/{n}" for n in ("nodes.parquet", "edges.parquet", "network.json")}
    assert_same_network(kigali, SpatialNetwork.load("networks/kigali", data_store=store))


def test_relative_path_in_local_data_store(kigali, tmp_path):
    store = LocalDataStore(tmp_path)
    kigali.save("roads/kigali", data_store=store)
    assert (tmp_path / "roads" / "kigali" / "network.json").is_file()
    assert len(SpatialNetwork.load("roads/kigali", data_store=store).edges) == 2802


def osmnx_like_graph():
    G = nx.MultiDiGraph(crs="EPSG:4326", simplified=True)
    G.add_node(1, x=30.0, y=-1.95)
    G.add_node(2, x=30.001, y=-1.95)
    G.add_node(3, x=30.001, y=-1.949)
    G.add_edge(1, 2, osmid=[11, 12], highway=["primary", "secondary"], oneway=[True, False], reversed=[False, True], lanes="2")
    G.add_edge(2, 3, osmid=13, highway="residential", oneway=True, reversed=False, lanes=3)
    return G


def test_osmnx_list_attributes_are_normalised_on_save(tmp_path):
    net = SpatialNetwork.from_networkx(osmnx_like_graph())
    net.save(tmp_path / "net")
    edges = SpatialNetwork.load(tmp_path / "net").edges.sort_values("u")
    assert [list(v) for v in edges["osmid"]] == [[11, 12], [13]]
    assert edges["highway"].astype(str).tolist() == ["primary;secondary", "residential"]
    assert edges["oneway"].tolist() == [False, True]
    assert edges["reversed"].tolist() == [True, False]
    assert edges["lanes"].astype(str).tolist() == ["2", "3"]
    assert net.edges.sort_values("u")["osmid"].tolist() == [[11, 12], 13]


def test_mixed_integer_and_text_node_ids(tmp_path):
    nodes = pd.DataFrame(
        {"x": [30.0, 30.001, 30.002], "y": [-1.95, -1.95, -1.95]},
        index=pd.Index([10, "school-1", -1], dtype=object),
    )
    edges = pd.DataFrame({"u": [10, -1], "v": ["school-1", 10]})
    net = SpatialNetwork(nodes, edges)
    net.save(tmp_path / "mixed")
    assert json.loads((tmp_path / "mixed" / "network.json").read_text())["node_ids"] == "mixed"
    loaded = SpatialNetwork.load(tmp_path / "mixed")
    assert loaded.node_index.tolist() == [10, "school-1", -1]
    assert [type(i) for i in loaded.node_index] == [int, str, int]
    assert loaded.edges[["u", "v"]].values.tolist() == [[10, "school-1"], [-1, 10]]


@pytest.mark.parametrize("ids, message", [([5, "5"], "same digits"), ([1.5, 2.5], "integers or text")])
def test_unsaveable_node_ids(tmp_path, ids, message):
    nodes = pd.DataFrame({"x": [30.0, 30.001], "y": [-1.95, -1.95]}, index=pd.Index(ids, dtype=object))
    net = SpatialNetwork(nodes, pd.DataFrame({"u": [ids[0]], "v": [ids[1]]}))
    with pytest.raises(ValueError, match=message):
        net.save(tmp_path / "bad")


def test_network_without_geometry(tmp_path):
    nodes = pd.DataFrame({"x": [30.0, 30.001], "y": [-1.95, -1.95]}, index=[1, 2])
    net = SpatialNetwork(nodes, pd.DataFrame({"u": [1], "v": [2]}), directed=False)
    net.save(tmp_path / "plain")
    assert b"geo" not in (pq.read_schema(tmp_path / "plain" / "edges.parquet").metadata or {})
    loaded = SpatialNetwork.load(tmp_path / "plain")
    assert not loaded.directed and "geometry" not in loaded.edges.columns


def test_load_selected_columns(kigali, tmp_path):
    kigali.save(tmp_path / "kigali")
    topology = SpatialNetwork.load(tmp_path / "kigali", columns=["highway"])
    assert list(topology.edges.columns) == ["u", "v", "key", "length", "kind", "highway"]
    with_geometry = SpatialNetwork.load(tmp_path / "kigali", columns=["geometry"])
    assert with_geometry.edges["geometry"].notna().sum() == kigali.edges["geometry"].notna().sum()
    with pytest.raises(ValueError, match="not in the saved network"):
        SpatialNetwork.load(tmp_path / "kigali", columns=["surface"])


def test_load_errors(kigali, tmp_path):
    with pytest.raises(FileNotFoundError, match="network.json"):
        SpatialNetwork.load(tmp_path / "missing")
    kigali.save(tmp_path / "kigali")
    meta_path = tmp_path / "kigali" / "network.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "format_version": 2}))
    with pytest.raises(ValueError, match="Upgrade giga-spatial"):
        SpatialNetwork.load(tmp_path / "kigali")
    meta_path.write_text(json.dumps({**meta, "format": "something.else"}))
    with pytest.raises(ValueError, match="does not describe"):
        SpatialNetwork.load(tmp_path / "kigali")


def test_unserialisable_metadata_writes_nothing(tmp_path):
    nodes = pd.DataFrame({"x": [30.0, 30.001], "y": [-1.95, -1.95]}, index=[1, 2])
    net = SpatialNetwork(nodes, pd.DataFrame({"u": [1], "v": [2]}), metadata={"when": object()})
    with pytest.raises(TypeError, match="JSON serializable"):
        net.save(tmp_path / "meta")
    assert not (tmp_path / "meta").exists()


def test_saving_again_overwrites(kigali, tmp_path):
    kigali.save(tmp_path / "net")
    line = LineString([(30.0, -1.95), (30.001, -1.95)])
    nodes = pd.DataFrame({"x": [30.0, 30.001], "y": [-1.95, -1.95]}, index=[1, 2])
    small = SpatialNetwork(nodes, pd.DataFrame({"u": [1], "v": [2], "geometry": [line]}))
    small.save(tmp_path / "net")
    assert len(SpatialNetwork.load(tmp_path / "net")) == 2
