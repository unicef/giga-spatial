"""
The shared download helpers: they store what they fetch, raise on failure and
leave no partial file behind. All tests run offline against fake responses.
"""

import io
import zipfile

import geopandas as gpd
import pytest
import requests
from shapely.geometry import box

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.base import fetch


class _Response:
    def __init__(self, chunks=(b"data",), status=200, fail_after=None):
        self._chunks = list(chunks)
        self._status = status
        self._fail_after = fail_after
        self.headers = {"content-length": str(sum(len(c) for c in self._chunks))}
        self.closed = False

    def raise_for_status(self):
        if self._status >= 400:
            raise requests.HTTPError(f"{self._status} error")

    def iter_content(self, chunk_size=None):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise requests.ConnectionError("connection reset")
            yield chunk

    def close(self):
        self.closed = True


@pytest.fixture
def respond(monkeypatch):
    responses = []

    def _set(response):
        responses.append(response)
        monkeypatch.setattr(fetch.requests, "get", lambda url, **kwargs: response)
        return response

    return _set


def _zip(**members) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return archive.getvalue()


def test_stream_to_store_writes_the_body(respond, tmp_path):
    store = LocalDataStore()
    response = respond(_Response(chunks=[b"ab", b"cd"]))

    path = fetch.stream_to_store("https://x/f.bin", tmp_path / "f.bin", store)

    assert path.read_bytes() == b"abcd"
    assert response.closed


def test_stream_to_store_raises_on_http_error_and_stores_nothing(respond, tmp_path):
    respond(_Response(status=404))

    with pytest.raises(requests.HTTPError):
        fetch.stream_to_store("https://x/f.bin", tmp_path / "f.bin", LocalDataStore())

    assert not (tmp_path / "f.bin").exists()


def test_stream_to_store_removes_a_partial_file(respond, tmp_path):
    respond(_Response(chunks=[b"ab", b"cd"], fail_after=1))

    with pytest.raises(requests.ConnectionError):
        fetch.stream_to_store("https://x/f.bin", tmp_path / "f.bin", LocalDataStore())

    assert not (tmp_path / "f.bin").exists()


def test_stream_to_store_checks_the_md5(respond, tmp_path):
    respond(_Response(chunks=[b"ab", b"cd"]))
    good = "e2fc714c4727ee9395f324cd2e7f331f"

    fetch.stream_to_store("https://x/f.bin", tmp_path / "f.bin", LocalDataStore(), md5=good.upper())
    assert (tmp_path / "f.bin").read_bytes() == b"abcd"

    with pytest.raises(ValueError, match="checksum"):
        fetch.stream_to_store("https://x/g.bin", tmp_path / "g.bin", LocalDataStore(), md5="0" * 32)
    assert not (tmp_path / "g.bin").exists()


def test_stream_to_store_uses_the_given_session(tmp_path):
    calls = []

    class _Session:
        def get(self, url, **kwargs):
            calls.append(url)
            return _Response()

    fetch.stream_to_store(
        "https://x/f.bin", tmp_path / "f.bin", LocalDataStore(), session=_Session()
    )

    assert calls == ["https://x/f.bin"]


def test_extract_zip_stores_matching_members_flat(respond, tmp_path):
    respond(_Response(chunks=[_zip(**{"dir/a.tif": b"A", "b.txt": b"B"})]))

    paths = fetch.extract_zip_to_store(
        "https://x/a.zip", tmp_path / "out", LocalDataStore(), r".*\.tif$"
    )

    assert paths == [tmp_path / "out" / "a.tif"]
    assert paths[0].read_bytes() == b"A"
    assert not (tmp_path / "out" / "b.txt").exists()


def test_extract_zip_raises_when_no_member_matches(respond, tmp_path):
    respond(_Response(chunks=[_zip(**{"b.txt": b"B"})]))

    with pytest.raises(FileNotFoundError, match="matches"):
        fetch.extract_zip_to_store(
            "https://x/a.zip", tmp_path / "out", LocalDataStore(), r".*\.tif$"
        )


def test_extract_zip_raises_for_a_non_zip_body(respond, tmp_path):
    respond(_Response(chunks=[b"<html>error</html>"]))

    with pytest.raises(zipfile.BadZipFile):
        fetch.extract_zip_to_store("https://x/a.zip", tmp_path, LocalDataStore())


def test_tiles_intersecting_reprojects_the_geometry_to_the_grid():
    tiles = gpd.GeoDataFrame(
        {"tile_id": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    in_b = gpd.GeoSeries([box(1.2, 0.2, 1.4, 0.4)], crs="EPSG:4326").to_crs(3857)

    hits = fetch.tiles_intersecting(tiles, in_b.iloc[0], "EPSG:3857")

    assert hits["tile_id"].to_list() == ["b"]
