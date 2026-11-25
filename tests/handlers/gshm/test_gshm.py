"""
Tests for the GEM Global Seismic Hazard Map handler.

Run with:
    pytest tests/handlers/gshm/test_gshm.py -v
    pytest tests/handlers/gshm/test_gshm.py -m live -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
import requests
from shapely.geometry import Point, Polygon, box

from gigaspatial.handlers.gshm import ARCHIVE_URL, GSHMHandler
import gigaspatial.handlers.gshm as module

SMALL_SOURCE = box(0.2, 0.2, 0.8, 0.8)


@pytest.fixture
def handler(tmp_path):
    return GSHMHandler(base_path=tmp_path / "gshm", n_workers=1)


def _sample_bytes(points=None) -> bytes:
    if points is None:
        points = [(0.5, 0.5, 0.1), (5.0, 5.0, 0.9)]
    frame = gpd.GeoDataFrame(
        {
            "lon": [lon for lon, _, _ in points],
            "lat": [lat for _, lat, _ in points],
            "pga": [pga for _, _, pga in points],
        },
        geometry=[Point(lon, lat) for lon, lat, _ in points],
        crs="EPSG:4326",
    )
    fd, tmp = tempfile.mkstemp(suffix=".gpkg")
    os.close(fd)
    try:
        frame.to_file(tmp, driver="GPKG", layer="v2026")
        return Path(tmp).read_bytes()
    finally:
        os.remove(tmp)


@pytest.fixture
def stub_download(monkeypatch):
    """Write a two-point GeoPackage instead of fetching the archive."""
    fetched = []

    def fake_extract(url, dest_dir, store, pattern=None, **kwargs):
        fetched.append(url)
        path = Path(dest_dir) / "gem_gshm_v2026.1.gpkg"
        store.write_file(str(path), _sample_bytes())
        return [path]

    monkeypatch.setattr(module, "extract_zip_to_store", fake_extract)
    return fetched


def test_small_source_resolves_to_the_archive(handler):
    units = handler.config.get_relevant_data_units(SMALL_SOURCE)

    assert units == [{"url": ARCHIVE_URL}]


def test_stored_path_includes_the_version(handler):
    (unit,) = handler.config.get_relevant_data_units(SMALL_SOURCE)

    path = handler.config.get_data_unit_path(unit)
    assert path.parts[-2:] == ("v2026.1", "gem_gshm_v2026.1.gpkg")


def test_load_data_downloads_and_crops(handler, stub_download):
    data = handler.load_data(SMALL_SOURCE, crop_to_source=True, show_progress=False)

    assert stub_download == [ARCHIVE_URL]
    assert len(data) == 1
    assert data.iloc[0]["pga"] == pytest.approx(0.1)


def test_polygon_crop_drops_a_point_inside_the_bounds(handler, monkeypatch):
    """A point in the bounds but outside the polygon must not survive the crop."""
    inside = (0.3, 0.3, 0.1)
    inside_bounds_only = (0.5, 0.5, 0.4)
    outside = (5.0, 5.0, 0.9)
    triangle = Polygon([(0.2, 0.2), (0.55, 0.2), (0.2, 0.55)])

    assert triangle.contains(Point(inside[0], inside[1]))
    assert box(*triangle.bounds).contains(Point(*inside_bounds_only[:2]))
    assert not triangle.intersects(Point(*inside_bounds_only[:2]))

    def fake_extract(url, dest_dir, store, pattern=None, **kwargs):
        path = Path(dest_dir) / "gem_gshm_v2026.1.gpkg"
        store.write_file(
            str(path),
            _sample_bytes([inside, inside_bounds_only, outside]),
        )
        return [path]

    monkeypatch.setattr(module, "extract_zip_to_store", fake_extract)

    data = handler.load_data(triangle, crop_to_source=True, show_progress=False)

    assert list(data["pga"]) == pytest.approx([0.1])


def test_stored_units_are_not_downloaded_again(handler, stub_download):
    handler.load_data(SMALL_SOURCE, show_progress=False)
    handler.load_data(SMALL_SOURCE, show_progress=False)

    assert len(stub_download) == 1


def test_unknown_keyword_raises(handler):
    with pytest.raises(TypeError):
        handler.load_data(SMALL_SOURCE, not_an_option=True)


def _head(url: str) -> requests.Response:
    """Zenodo sometimes stalls a HEAD. Try again before treating that as absence."""
    last_error = None
    for _ in range(3):
        try:
            response = requests.head(url, allow_redirects=True, timeout=90)
        except requests.RequestException as error:
            last_error = error
            continue
        if response.status_code < 500:
            return response
        last_error = requests.RequestException(response.status_code)
    raise last_error


@pytest.mark.live
def test_resolved_units_exist_remotely(tmp_path):
    handler = GSHMHandler(base_path=tmp_path / "gshm")
    units = handler.config.get_relevant_data_units(SMALL_SOURCE)

    assert units
    for unit in units:
        response = _head(unit["url"])
        assert response.status_code == 200, unit["url"]
