"""Tests for the Global Power Plant Database handler."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import requests
from pydantic import ValidationError
from shapely.geometry import box

from gigaspatial.handlers.wri.global_power_plant import GlobalPowerPlantHandler
import gigaspatial.handlers.wri.global_power_plant as module

# Contains Kajaki Hydroelectric Power Plant (32.322, 65.119).
SMALL_SOURCE = box(65.0, 32.0, 66.0, 33.0)
RELEASE_URL = (
    "https://datasets.wri.org/private-admin/dataset/"
    "53623dfd-3df6-4f15-a091-67457cdb571f/resource/"
    "66bcdacc-3d0e-46ad-9271-a5a76b1853d2/download/"
    "globalpowerplantdatabasev130.zip"
)


@pytest.fixture
def handler(tmp_path):
    return GlobalPowerPlantHandler(
        version="1.3.0",
        base_path=tmp_path / "global_power_plant_database",
        n_workers=1,
    )


@pytest.fixture
def stub_download(monkeypatch):
    """Write a small plant table instead of fetching the release zip."""
    fetched = []

    def fake_extract_zip_to_store(url, dest_dir, store, pattern=None, **kwargs):
        fetched.append(url)
        path = Path(dest_dir) / "global_power_plant_database.csv"
        rows = pd.DataFrame(
            {
                "name": [
                    "Kajaki Hydroelectric Power Plant Afghanistan",
                    "Elsewhere",
                ],
                "latitude": [32.322, 0.0],
                "longitude": [65.119, 0.0],
            }
        )
        store.write_file(str(path), rows.to_csv(index=False))
        return [path]

    monkeypatch.setattr(module, "extract_zip_to_store", fake_extract_zip_to_store)
    return fetched


def test_small_source_resolves_to_the_release(handler):
    units = handler.config.get_relevant_data_units(SMALL_SOURCE)

    assert units == [{"version": "1.3.0", "url": RELEASE_URL}]


def test_version_shows_in_the_stored_path(handler):
    (current,) = handler.config.get_relevant_data_units(SMALL_SOURCE)
    current_path = handler.config.get_data_unit_path(current)
    assert current_path.parts[-3:] == (
        "global_power_plant_database",
        "1.3.0",
        "global_power_plant_database.csv",
    )

    older = GlobalPowerPlantHandler(version="1.2.0", base_path=handler.config.base_path)
    (previous,) = older.config.get_relevant_data_units(SMALL_SOURCE)
    previous_path = older.config.get_data_unit_path(previous)

    assert "1.2.0" in previous_path.parts
    assert current_path != previous_path


def test_load_data_downloads_and_crops(handler, stub_download):
    data = handler.load_data(SMALL_SOURCE, crop_to_source=True, show_progress=False)

    assert stub_download == [RELEASE_URL]
    assert list(data["name"]) == ["Kajaki Hydroelectric Power Plant Afghanistan"]


def test_stored_units_are_not_downloaded_again(handler, stub_download):
    handler.load_data(SMALL_SOURCE, show_progress=False)
    handler.load_data(SMALL_SOURCE, show_progress=False)

    assert len(stub_download) == 1


def test_unknown_keyword_raises(handler):
    with pytest.raises(TypeError):
        handler.load_data(SMALL_SOURCE, not_an_option=True)


def test_unknown_version_raises():
    with pytest.raises(ValidationError):
        GlobalPowerPlantHandler(version="1.0.0")


@pytest.mark.live
def test_resolved_units_exist_remotely(tmp_path):
    handler = GlobalPowerPlantHandler(
        base_path=tmp_path / "global_power_plant_database"
    )
    units = handler.config.get_relevant_data_units(SMALL_SOURCE)

    assert units
    for unit in units:
        response = requests.get(unit["url"], stream=True, timeout=60)
        try:
            assert response.status_code == 200, unit["url"]
        finally:
            response.close()
