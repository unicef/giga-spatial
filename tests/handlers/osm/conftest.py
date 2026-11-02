import datetime
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
import shapely

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.osm.roads import OSMRoadsConfig
from gigaspatial.processing.network import SpatialNetwork

DATA = Path(__file__).resolve().parents[2] / "data" / "osm_roads"
GEOFABRIK = "https://download.geofabrik.de"
LATEST_DATE = "2026-10-28"


class _Response:
    def __init__(self, url, status=200, body=b"", headers=None):
        self.url = url
        self.status_code = status
        self.content = body
        self.headers = {"content-length": str(len(body)), **(headers or {})}

    @property
    def text(self):
        return self.content.decode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} for {self.url}", response=self)

    def iter_content(self, chunk_size=None):
        size = chunk_size or len(self.content) or 1
        for start in range(0, len(self.content), size):
            yield self.content[start : start + size]

    def close(self):
        pass


class FakeGeofabrik:
    """Geofabrik's server: the index, ``-latest`` redirects, dated files and their ``.md5``.

    Every ``-latest`` URL redirects to the file of ``latest``; ``mirrored``
    extracts redirect to a mirror under the ``-latest`` name, as large ones do.
    """

    def __init__(self):
        self.files = {f"{GEOFABRIK}/index-v1.json": (DATA / "geofabrik_index_subset.json").read_bytes()}
        self.latest = {}
        self.mirrored = set()
        self.requests = []

    def serve(self, path, date, body=None):
        """Serve an extract's dated file, Kigali by default, with its checksum and replication state."""
        body = (DATA / "kigali.osm.pbf").read_bytes() if body is None else body
        stamp = datetime.date.fromisoformat(date).strftime("%y%m%d")
        dated = f"{GEOFABRIK}/{path}-{stamp}.osm.pbf"
        self.files[dated] = body
        self.files[f"{dated}.md5"] = f"{hashlib.md5(body).hexdigest()}  {dated.rsplit('/', 1)[1]}\n".encode()
        return dated

    def latest_of(self, path):
        return self.latest.get(path, LATEST_DATE)

    def head(self, url, allow_redirects=True, **kwargs):
        self.requests.append(("HEAD", url))
        if url.endswith("-latest.osm.pbf"):
            path = url[len(GEOFABRIK) + 1 : -len("-latest.osm.pbf")]
            if path in self.mirrored:
                location = f"https://mirror.example/{path.rsplit('/', 1)[1]}-latest.osm.pbf"
            else:
                location = f"{GEOFABRIK}/{path}-{datetime.date.fromisoformat(self.latest_of(path)):%y%m%d}.osm.pbf"
            return _Response(url, 307, headers={"location": location})
        return _Response(url, 200 if url in self.files else 404)

    def get(self, url, **kwargs):
        self.requests.append(("GET", url))
        if url.endswith("-updates/state.txt"):
            path = url[len(GEOFABRIK) + 1 : -len("-updates/state.txt")]
            body = f"sequenceNumber=4952\ntimestamp={self.latest_of(path)}T21\\:21\\:02Z\n".encode()
            return _Response(url, body=body)
        if url in self.files:
            return _Response(url, body=self.files[url])
        return _Response(url, 404)

    def asked(self, method=None):
        return [url for m, url in self.requests if method in (None, m)]


@pytest.fixture
def geofabrik(monkeypatch):
    server = FakeGeofabrik()
    monkeypatch.setattr(requests, "get", server.get)
    monkeypatch.setattr(requests, "head", server.head)
    return server


@pytest.fixture
def store(tmp_path):
    return LocalDataStore(tmp_path)


@pytest.fixture
def config(store, geofabrik):
    return OSMRoadsConfig(data_store=store, base_path=Path("osm/geofabrik"))


def _parts(value):
    return None if pd.isna(value) else frozenset(str(value).split(";"))


def assert_same_network(ours: SpatialNetwork, theirs: SpatialNetwork, exact_lengths: bool):
    """Same nodes and attributes, same (u, v, key) edges and attributes; text compared as sets of ';' parts."""
    on, tn = ours.nodes, theirs.nodes.loc[ours.nodes.index]
    assert set(on.index) == set(theirs.nodes.index)
    assert set(on.columns) == set(tn.columns)
    assert np.array_equal(on["x"], tn["x"]) and np.array_equal(on["y"], tn["y"])
    assert np.array_equal(on["street_count"], tn["street_count"])
    for name in set(on.columns) - {"x", "y", "kind", "street_count"}:
        assert on[name].astype(object).fillna("-").tolist() == tn[name].astype(object).fillna("-").tolist(), name

    oe = ours.edges.set_index(["u", "v", "key"])
    te = theirs.edges.set_index(["u", "v", "key"])
    assert oe.index.is_unique and set(oe.index) == set(te.index)
    te = te.loc[oe.index]
    assert set(oe.columns) == set(te.columns)
    if exact_lengths:
        assert np.array_equal(oe["length"], te["length"])
    else:
        assert oe["length"].to_numpy() == pytest.approx(te["length"].to_numpy(), abs=1e-9)
    for name in ("oneway", "reversed"):
        assert np.array_equal(oe[name], te[name]), name
    osmids = [frozenset(np.atleast_1d(v).tolist()) for v in oe["osmid"]]
    assert osmids == [frozenset(np.atleast_1d(v).tolist()) for v in te["osmid"]]
    if "geometry" in te.columns:
        ours_geometry, their_geometry = oe["geometry"].to_numpy(), te["geometry"].to_numpy()
        assert np.array_equal(pd.isna(ours_geometry), pd.isna(their_geometry))
        present = ~pd.isna(their_geometry)
        assert shapely.equals_exact(ours_geometry[present], their_geometry[present], 0).all()
    for name in set(oe.columns) - {"length", "kind", "oneway", "reversed", "osmid", "geometry"}:
        assert [_parts(v) for v in oe[name]] == [_parts(v) for v in te[name]], name


@pytest.fixture
def same_network():
    """``same_network(ours, theirs, exact_lengths)`` asserts two road networks are the same, as osmnx would build them."""
    return assert_same_network
