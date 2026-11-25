"""Tests for the FLOPROS flood protection standards handler."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
import requests
from shapely.geometry import box

from gigaspatial.handlers.flopros import FLOPROSConfig, FLOPROSHandler
import gigaspatial.handlers.flopros as module

# Sits inside the Kigali polygon written by the download stub.
SMALL_SOURCE = box(30.2, -1.8, 30.8, -1.2)
SUPPLEMENT_URL = (
    "https://nhess.copernicus.org/articles/16/1049/2016/"
    "nhess-16-1049-2016-supplement.zip"
)
ARCHIVE_MEMBERS = [
    "nhess-16-1049-2016-supplement-title-page.pdf",
    "Scussolini_etal_Suppl_info/FLOPROS_Database_Design_&_Policy_layers_V1.xlsx",
    "Scussolini_etal_Suppl_info/FLOPROS_shp_V1/FLOPROS_shp_V1.shp",
    "Scussolini_etal_Suppl_info/FLOPROS_shp_V1/FLOPROS_shp_V1.shx",
    "Scussolini_etal_Suppl_info/FLOPROS_shp_V1/FLOPROS_shp_V1.dbf",
    "Scussolini_etal_Suppl_info/FLOPROS_shp_V1/FLOPROS_shp_V1.prj",
    "Scussolini_etal_Suppl_info/FLOPROS_shp_V1/FLOPROS_shp_V1.sbn",
    "Scussolini_etal_Suppl_info/FLOPROS_shp_V1/Readme.txt",
    "Scussolini_etal_Suppl_info/Supplementary information to the manuscript.docx",
]


@pytest.fixture
def handler(tmp_path):
    return FLOPROSHandler(base_path=tmp_path / "flopros", n_workers=1)


@pytest.fixture
def stub_download(monkeypatch):
    """Write a two-polygon shapefile instead of fetching the supplement."""
    fetched = []

    def fake_extract_zip_to_store(url, dest_dir, store, pattern=None, **kwargs):
        fetched.append(url)
        frame = gpd.GeoDataFrame(
            {
                "name": ["Kigali", "Elsewhere"],
                "geonunit": ["Rwanda", "Nowhere"],
                "type_en": ["Province", "Province"],
                "MerL_Riv": [100.0, 10.0],
            },
            geometry=[box(30, -2, 31, -1), box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        stored = []
        with tempfile.TemporaryDirectory() as tmp:
            shapefile = Path(tmp) / "FLOPROS_shp_V1.shp"
            frame.to_file(shapefile)
            for path in Path(tmp).iterdir():
                if path.suffix.lower() not in FLOPROSConfig.SHAPEFILE_SUFFIXES:
                    continue
                target = Path(dest_dir) / path.name
                store.write_file(str(target), path.read_bytes())
                stored.append(target)
        return stored

    monkeypatch.setattr(module, "extract_zip_to_store", fake_extract_zip_to_store)
    return fetched


def test_small_source_resolves_to_the_supplement(handler):
    units = handler.config.get_relevant_data_units(SMALL_SOURCE)

    assert units == [{"url": SUPPLEMENT_URL}]


def test_stored_path_is_the_shapefile(handler):
    (unit,) = handler.config.get_relevant_data_units(SMALL_SOURCE)
    path = handler.config.get_data_unit_path(unit)

    assert path.parts[-2:] == ("flopros", "FLOPROS_shp_V1.shp")


def test_archive_pattern_keeps_the_shapefile_parts():
    kept = [name for name in ARCHIVE_MEMBERS if re.match(FLOPROSConfig.MEMBER_PATTERN, name)]

    assert kept == [
        "Scussolini_etal_Suppl_info/FLOPROS_shp_V1/FLOPROS_shp_V1.shp",
        "Scussolini_etal_Suppl_info/FLOPROS_shp_V1/FLOPROS_shp_V1.shx",
        "Scussolini_etal_Suppl_info/FLOPROS_shp_V1/FLOPROS_shp_V1.dbf",
        "Scussolini_etal_Suppl_info/FLOPROS_shp_V1/FLOPROS_shp_V1.prj",
    ]


def test_shapefile_without_sidecars_is_not_available(handler):
    (unit,) = handler.config.get_relevant_data_units(SMALL_SOURCE)
    shapefile = handler.config.get_data_unit_path(unit)
    handler.data_store.write_file(str(shapefile), b"not a shapefile")

    assert handler.config.is_unit_available(unit, [shapefile]) is False


def test_null_geometry_does_not_skip_the_crop(handler):
    frame = gpd.GeoDataFrame(
        {"name": ["Kigali", "Elsewhere", "Missing"], "MerL_Riv": [100.0, 10.0, 0.0]},
        geometry=[box(30, -2, 31, -1), box(0, 0, 1, 1), None],
        crs="EPSG:4326",
    )

    cropped = handler.reader.crop_to_geometry(frame, SMALL_SOURCE, crs="EPSG:4326")

    assert list(cropped["name"]) == ["Kigali"]


def test_load_data_downloads_and_crops(handler, stub_download):
    data = handler.load_data(SMALL_SOURCE, crop_to_source=True, show_progress=False)

    assert stub_download == [SUPPLEMENT_URL]
    assert list(data["name"]) == ["Kigali"]
    assert list(data["MerL_Riv"]) == [100.0]


def test_stored_units_are_not_downloaded_again(handler, stub_download):
    handler.load_data(SMALL_SOURCE, show_progress=False)
    handler.load_data(SMALL_SOURCE, show_progress=False)

    assert len(stub_download) == 1


def test_unknown_keyword_raises(handler):
    with pytest.raises(TypeError):
        handler.load_data(SMALL_SOURCE, not_an_option=True)


@pytest.mark.live
def test_supplement_exists_remotely(tmp_path):
    handler = FLOPROSHandler(base_path=tmp_path / "flopros")
    units = handler.config.get_relevant_data_units(SMALL_SOURCE)

    assert units
    for unit in units:
        response = requests.get(unit["url"], stream=True, timeout=60)
        try:
            assert response.status_code == 200, unit["url"]
        finally:
            response.close()
