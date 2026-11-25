"""Offline and live tests for PeeringDBFetcher."""

from __future__ import annotations

import json

import geopandas as gpd
import pytest
import requests

import gigaspatial.fetchers.peeringdb as peeringdb
from gigaspatial.fetchers.peeringdb import PeeringDBFetcher

_FACILITY = {
    "id": 11839,
    "name": "MTN Centre Kigali",
    "country": "RW",
    "latitude": -1.940686,
    "longitude": 30.10344,
}
_UNLOCATED = {
    "id": 2159,
    "name": "Telecom House Kigali",
    "country": "RW",
    "latitude": None,
    "longitude": None,
}


class _Response:
    def __init__(self, status_code, body=b"", content_length=None):
        self.status_code = status_code
        self._body = body
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.closed = False

    def iter_content(self, chunk_size=65_536):
        if self._body:
            yield self._body

    def close(self):
        self.closed = True


def _json_response(records):
    body = json.dumps({"data": records}).encode()
    return _Response(200, body, content_length=len(body))


@pytest.fixture
def fetcher():
    return PeeringDBFetcher(api_key="", min_interval_seconds=0, max_retries=0)


def test_fetch_returns_the_patched_facility(fetcher, monkeypatch):
    monkeypatch.setattr(
        peeringdb.requests, "get", lambda *args, **kwargs: _json_response([_FACILITY])
    )

    frame = fetcher.fetch_facilities(country="RW")

    assert isinstance(frame, gpd.GeoDataFrame)
    assert frame.crs.to_epsg() == 4326
    assert len(frame) == 1
    assert frame.iloc[0]["name"] == "MTN Centre Kigali"
    assert frame.geometry.iloc[0].x == pytest.approx(30.10344)
    assert frame.geometry.iloc[0].y == pytest.approx(-1.940686)


def test_null_coordinates_stay_with_missing_geometry(fetcher, monkeypatch):
    monkeypatch.setattr(
        peeringdb.requests, "get", lambda *args, **kwargs: _json_response([_UNLOCATED])
    )

    frame = fetcher.fetch_facilities(country="RW")

    assert len(frame) == 1
    assert frame.iloc[0]["id"] == 2159
    assert frame.geometry.isna().iloc[0]


def test_country_and_extent_become_query_params(monkeypatch):
    seen = {}

    def fake_get(url, params, headers, timeout, stream):
        seen["params"] = params
        return _json_response([])

    monkeypatch.setattr(peeringdb.requests, "get", fake_get)
    PeeringDBFetcher(api_key="", min_interval_seconds=0).fetch_facilities(
        country="Rwanda",
        extent=(29.9, -2.1, 30.3, -1.8),
    )

    assert seen["params"] == {
        "depth": 0,
        "country": "RW",
        "longitude__gte": 29.9,
        "latitude__gte": -2.1,
        "longitude__lte": 30.3,
        "latitude__lte": -1.8,
    }


def test_api_key_sets_authorization_header(monkeypatch):
    seen = []

    def fake_get(url, params, headers, timeout, stream):
        seen.append(dict(headers))
        return _json_response([])

    monkeypatch.setattr(peeringdb.requests, "get", fake_get)
    PeeringDBFetcher(api_key="", min_interval_seconds=0).fetch_facilities()
    PeeringDBFetcher(api_key="test-key", min_interval_seconds=0).fetch_facilities()

    assert "Authorization" not in seen[0]
    assert seen[1]["Authorization"] == "Api-Key test-key"


def test_unknown_country_raises_before_the_request(fetcher, monkeypatch):
    def fake_get(*args, **kwargs):
        raise AssertionError("GET was called")

    monkeypatch.setattr(peeringdb.requests, "get", fake_get)

    with pytest.raises(ValueError, match="Notland"):
        fetcher.fetch_facilities(country="Notland")


def test_inverted_extent_raises_before_the_request(fetcher, monkeypatch):
    def fake_get(*args, **kwargs):
        raise AssertionError("GET was called")

    monkeypatch.setattr(peeringdb.requests, "get", fake_get)

    with pytest.raises(ValueError, match="minimum"):
        fetcher.fetch_facilities(extent=(30.0, -2.0, 29.0, -1.0))


def test_retries_429_after_the_interval(monkeypatch):
    slept = []
    monkeypatch.setattr(peeringdb.time, "sleep", slept.append)
    calls = {"n": 0}

    def fake_get(url, params, headers, timeout, stream):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(429)
        return _json_response([_FACILITY])

    monkeypatch.setattr(peeringdb.requests, "get", fake_get)
    frame = PeeringDBFetcher(
        api_key="", min_interval_seconds=2, max_retries=3
    ).fetch_facilities()

    assert calls["n"] == 2
    assert len(frame) == 1
    assert len(slept) == 1
    assert slept[0] == pytest.approx(2, abs=0.05)


def test_oversized_body_raises(fetcher, monkeypatch):
    monkeypatch.setattr(
        peeringdb.requests,
        "get",
        lambda *args, **kwargs: _Response(200, b"{}", content_length=50_000_001),
    )

    with pytest.raises(requests.RequestException, match="50000000"):
        fetcher.fetch_facilities()


def test_data_that_is_not_an_array_raises(fetcher, monkeypatch):
    body = b'{"data": {}}'
    monkeypatch.setattr(
        peeringdb.requests,
        "get",
        lambda *args, **kwargs: _Response(200, body, content_length=len(body)),
    )

    with pytest.raises(requests.RequestException, match="data array"):
        fetcher.fetch_facilities()


@pytest.mark.live
def test_live_fetch_of_rwanda():
    frame = PeeringDBFetcher().fetch_facilities(country="RW")

    assert isinstance(frame, gpd.GeoDataFrame)
    assert frame.crs.to_epsg() == 4326
    assert len(frame) > 0
    assert set(frame["country"]) == {"RW"}
