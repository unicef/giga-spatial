"""
Tests for country-level GloFAS historical downloads.

Offline tests use a stand-in catalogue and a stand-in EWDS job. The live
test reads the public catalogue and does not download a time series.

Run with:
    pytest tests/handlers/cems/test_glofas_historical.py -v
    pytest tests/handlers/cems/test_glofas_historical.py -m live -v
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import MultiPolygon, box

from gigaspatial.handlers.cems import glofas_historical as module
from gigaspatial.handlers.cems.glofas_historical import (
    GlofasHistoricalHandler,
    HistoricalCatalogue,
    dates_to_fetch,
    extraction_areas,
    group_dates,
    validate_cube,
)


DISCHARGE = "average_river_discharge_in_the_last_24_hours"
LATITUDE = np.array([-1.975, -1.925, -1.875])
LONGITUDE = np.array([29.025, 29.075, 29.125])


def collection(end: str, updated: str = "2026-11-14T00:00:00Z") -> dict:
    return {
        "updated": updated,
        "sci:doi": "10.24381/cds.a4fdd6b9",
        "extent": {"temporal": {"interval": [["1979-01-01T00:00:00Z", end]]}},
    }


def constraint_row(variable, timespan, product, days):
    return {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": [product],
        "timespan": [timespan],
        "variable": [variable],
        "year": sorted({f"{day.year:04d}" for day in days}),
        "month": sorted({f"{day.month:02d}" for day in days}),
        "day": sorted({f"{day.day:02d}" for day in days}),
    }


def catalogue_for(days_by_product, updated="2026-11-14T00:00:00Z", variable=DISCHARGE):
    rows = []
    all_days = []
    for product, days in days_by_product.items():
        rows.append(constraint_row(variable, "time_mean", product, days))
        all_days.extend(days)
    end = max(all_days).isoformat() + "T00:00:00Z"
    return HistoricalCatalogue(collection(end, updated), rows)


def write_cube(path, days, variable=DISCHARGE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((len(days), LATITUDE.size, LONGITUDE.size), dtype="float32")
    for index, day in enumerate(days):
        data[index] = float(day.day)
    dataset = xr.Dataset(
        {variable: (("time", "latitude", "longitude"), data)},
        coords={
            "time": pd.to_datetime(list(days)),
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        },
    )
    dataset[variable].attrs["units"] = "m3 s-1"
    dataset.to_netcdf(path)
    dataset.close()


def install_retrieve(monkeypatch):
    fetched = []

    def fake(self, request, dest, options):
        fetched.append(
            {
                "variable": list(request["variable"]),
                "product": list(request["product_type"]),
                "months": list(request["month"]),
                "days": list(request["day"]),
                "area": list(request["area"]),
            }
        )
        year = int(request["year"][0])
        days = [
            date(year, int(month), int(day))
            for month in request["month"]
            for day in request["day"]
        ]
        write_cube(dest, days, request["variable"][0])

    monkeypatch.setattr(module.GlofasHistoricalDownloader, "_retrieve", fake)
    return fetched


def build_handler(tmp_path, catalogue, monkeypatch):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    handler = GlofasHistoricalHandler(
        base_path=tmp_path / "glofas_historical",
        api_key="token",
        n_workers=1,
        system_version="version_4_0",
    )
    handler.config._catalogue = catalogue
    return handler


def test_dates_to_fetch_keeps_consolidated_and_refreshes_the_open_tail():
    stored = {"2020-01-01": "consolidated", "2020-06-01": "intermediate"}
    preferred = {
        date(2020, 1, 1): "consolidated",
        date(2020, 6, 1): "intermediate",
        date(2020, 6, 2): "intermediate",
    }
    pending = dates_to_fetch(
        stored,
        preferred,
        catalogue_updated="2026-11-14T00:00:00Z",
        stored_catalogue_updated="2026-11-14T00:00:00Z",
    )
    assert pending == [(date(2020, 6, 2), "intermediate")]

    upgraded = dates_to_fetch(
        {"2020-06-01": "intermediate"},
        {date(2020, 6, 1): "consolidated"},
        catalogue_updated="same",
        stored_catalogue_updated="same",
    )
    assert upgraded == [(date(2020, 6, 1), "consolidated")]

    refreshed = dates_to_fetch(
        {"2020-06-01": "intermediate"},
        {date(2020, 6, 1): "intermediate"},
        catalogue_updated="new",
        stored_catalogue_updated="old",
    )
    assert refreshed == [(date(2020, 6, 1), "intermediate")]


def test_group_dates_does_not_mix_months_with_different_days():
    groups = group_dates(
        [
            (date(2020, 1, 1), "consolidated"),
            (date(2020, 2, 2), "consolidated"),
        ]
    )
    assert len(groups) == 2
    assert groups[0]["months"] == [1]
    assert groups[0]["days"] == [1]
    assert groups[1]["months"] == [2]
    assert groups[1]["days"] == [2]


def test_extraction_areas_cover_islands_without_joining_distant_parts():
    one = extraction_areas(box(29, -2, 30, -1))
    assert len(one) == 1
    north, west, south, east = one[0]
    assert west < 29 and east > 30
    assert south < -2 and north > -1

    close = extraction_areas(
        MultiPolygon([box(29, -2, 30, -1), box(30.2, -2, 31, -1)])
    )
    assert len(close) == 1

    distant = extraction_areas(
        MultiPolygon([box(29, -2, 30, -1), box(120, 10, 121, 11)])
    )
    assert len(distant) == 2

    dateline = extraction_areas(
        MultiPolygon([box(175, -5, 180, 5), box(-180, -5, -175, 5)])
    )
    assert len(dateline) == 2
    assert dateline[0][3] < 0
    assert dateline[1][1] > 0


def test_validate_cube_rejects_duplicate_times_and_a_resampled_grid():
    time = pd.to_datetime(["2020-01-01", "2020-01-01"])
    duplicate = xr.Dataset(
        {"value": (("time", "latitude", "longitude"), np.ones((2, 2, 2)))},
        coords={
            "time": time,
            "latitude": [-1.975, -1.925],
            "longitude": [29.025, 29.075],
        },
    )
    area = (-0.949, 28.949, -2.051, 30.051)
    with pytest.raises(ValueError, match="duplicate"):
        validate_cube(
            duplicate,
            system_version="version_4_0",
            requested=[date(2020, 1, 1)],
            area=area,
        )

    coarse = xr.Dataset(
        {"value": (("time", "latitude", "longitude"), np.ones((1, 2, 2)))},
        coords={
            "time": pd.to_datetime(["2020-01-01"]),
            "latitude": [-1.9, -1.8],
            "longitude": [29.0, 29.1],
        },
    )
    with pytest.raises(ValueError, match="0.05"):
        validate_cube(
            coarse,
            system_version="version_4_0",
            requested=[date(2020, 1, 1)],
            area=area,
        )


def test_old_grib_name_is_rejected():
    with pytest.raises(ValueError, match="dis24"):
        GlofasHistoricalHandler().config.get_relevant_data_units("RWA", variable="dis24")


def test_a_geometry_is_not_a_country_key():
    handler = GlofasHistoricalHandler(api_key="token", n_workers=1)
    with pytest.raises(ValueError, match="country"):
        handler.config.get_search_key(box(0, 0, 1, 1), handler.config.options_type())


def test_incremental_update_downloads_only_new_days(tmp_path, monkeypatch, caplog):
    january = [date(2020, 1, 1), date(2020, 1, 2)]
    catalogue = catalogue_for({"consolidated": january})
    handler = build_handler(tmp_path, catalogue, monkeypatch)
    fetched = install_retrieve(monkeypatch)

    loaded = handler.load_data("RWA", variable=DISCHARGE)
    assert list(fetched[0]["days"]) == ["01", "02"]
    assert fetched[0]["area"][1] < 29 < 30 < fetched[0]["area"][3]
    assert loaded.sizes["time"] == 2
    assert float(loaded[DISCHARGE].isel(time=0).values.flat[0]) == 1.0

    metadata_path = next(Path(tmp_path).rglob("metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["country_iso3"] == "RWA"
    assert metadata["country_name"] == "Rwanda"
    assert metadata["variable"] == DISCHARGE
    assert metadata["units"] == "m3 s-1"
    assert metadata["spatial_resolution_degrees"] == pytest.approx(0.05)
    assert metadata["temporal_resolution"] == "daily"
    assert metadata["latest_stored_observation"] == "2020-01-02"
    assert metadata["source_latest_observation"] == "2020-01-02"
    assert metadata["last_successful_download_utc"].endswith("Z")
    assert metadata["last_successful_download_utc"] != metadata["latest_stored_observation"]
    assert metadata["dataset"] == "cems-glofas-historical"

    fetched.clear()
    caplog.set_level(logging.INFO)
    assert handler.ensure_data_available("RWA", variable=DISCHARGE) is True
    assert fetched == []
    assert "already up to date" in caplog.text

    later = catalogue_for({"consolidated": january + [date(2020, 1, 4)]})
    handler.config._catalogue = later
    handler.load_data("RWA", variable=DISCHARGE)
    assert fetched[-1]["days"] == ["04"]
    reloaded = handler.reader.load("RWA", variable=DISCHARGE)
    values = {
        pd.Timestamp(stamp).date(): float(reloaded[DISCHARGE].sel(time=stamp).values.flat[0])
        for stamp in reloaded["time"].values
    }
    assert values[date(2020, 1, 1)] == 1.0
    assert values[date(2020, 1, 2)] == 2.0
    assert values[date(2020, 1, 4)] == 4.0
    assert len(values) == 3


def test_a_failed_request_keeps_days_already_stored(tmp_path, monkeypatch):
    rows = [
        constraint_row(DISCHARGE, "time_mean", "consolidated", [date(2020, 1, 1)]),
        constraint_row(DISCHARGE, "time_mean", "consolidated", [date(2020, 2, 2)]),
    ]
    catalogue = HistoricalCatalogue(collection("2020-02-02T00:00:00Z"), rows)
    handler = build_handler(tmp_path, catalogue, monkeypatch)
    calls = {"n": 0}

    def fake(self, request, dest, options):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("EWDS unavailable")
        year = int(request["year"][0])
        days = [
            date(year, int(month), int(day))
            for month in request["month"]
            for day in request["day"]
        ]
        write_cube(dest, days)

    monkeypatch.setattr(module.GlofasHistoricalDownloader, "_retrieve", fake)
    assert handler.ensure_data_available("RWA", variable=DISCHARGE) is False
    metadata = json.loads(next(Path(tmp_path).rglob("metadata.json")).read_text(encoding="utf-8"))
    assert list(metadata["dates"]) == ["2020-01-01"]

    fetched = []

    def recover(self, request, dest, options):
        fetched.append(list(request["day"]))
        year = int(request["year"][0])
        days = [
            date(year, int(month), int(day))
            for month in request["month"]
            for day in request["day"]
        ]
        write_cube(dest, days)

    monkeypatch.setattr(module.GlofasHistoricalDownloader, "_retrieve", recover)
    assert handler.ensure_data_available("RWA", variable=DISCHARGE) is True
    assert fetched == [["02"]]
    metadata = json.loads(next(Path(tmp_path).rglob("metadata.json")).read_text(encoding="utf-8"))
    assert sorted(metadata["dates"]) == ["2020-01-01", "2020-02-02"]


def test_missing_api_key_does_not_start_a_download(tmp_path, monkeypatch):
    catalogue = catalogue_for({"consolidated": [date(2020, 1, 1)]})
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    handler = GlofasHistoricalHandler(
        base_path=tmp_path / "glofas_historical", api_key="", n_workers=1
    )
    handler.config._catalogue = catalogue
    fetched = install_retrieve(monkeypatch)
    assert handler.ensure_data_available("RWA", variable=DISCHARGE) is False
    assert fetched == []
    assert list(Path(tmp_path).rglob("*.nc")) == []


def test_all_four_variables_resolve_for_a_country(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    handler = GlofasHistoricalHandler(base_path=tmp_path, api_key="token", n_workers=1)
    units = handler.config.get_relevant_data_units("Rwanda")
    assert {unit["variable"] for unit in units} == set(module.VARIABLES)
    assert {unit["country_iso3"] for unit in units} == {"RWA"}


def test_auxiliary_subset_keeps_native_cell_values(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    seen = []

    def fake_stream(url, path, store, **kwargs):
        seen.append(url)
        latitude = np.arange(-2.475, -0.4, 0.05)
        longitude = np.arange(28.525, 30.6, 0.05)
        data = np.full((latitude.size, longitude.size), 7.0, dtype="float32")
        dataset = xr.Dataset(
            {"uparea": (("latitude", "longitude"), data)},
            coords={"latitude": latitude, "longitude": longitude},
        )
        dataset["uparea"].attrs["units"] = "m2"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_netcdf(path)
        dataset.close()
        return path

    monkeypatch.setattr(module, "stream_to_store", fake_stream)
    handler = GlofasHistoricalHandler(
        base_path=tmp_path / "glofas_historical", api_key="token", n_workers=1
    )
    loaded = handler.load_data("RWA", variable="upstream_area")
    assert "uparea_glofas_v4_0.nc" in seen[0]
    assert set(np.unique(loaded["uparea"].values)) == {7.0}
    assert float(np.diff(loaded["longitude"].values).flat[0]) == pytest.approx(0.05)


def test_version_5_has_no_published_auxiliary_file(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    handler = GlofasHistoricalHandler(
        base_path=tmp_path, api_key="token", n_workers=1, system_version="version_5_0"
    )
    caplog.set_level(logging.WARNING)
    with pytest.raises(ValueError, match="No GloFAS historical variables"):
        handler.config.get_relevant_data_units("RWA", variable="upstream_area")
    assert "version_5_0" in caplog.text


@pytest.mark.live
def test_live_catalogue_lists_the_four_variables_and_their_timespans():
    catalogue = module.fetch_catalogue()
    assert catalogue.temporal_start == date(1979, 1, 1)
    assert catalogue.temporal_end >= date(2024, 1, 1)
    discharge = catalogue.availability("version_4_0", DISCHARGE)
    runoff = catalogue.availability("version_4_0", "runoff_water_equivalent")
    snow = catalogue.availability("version_4_0", "snow_depth_water_equivalent")
    soil = catalogue.availability("version_4_0", "soil_wetness_index")
    assert discharge.timespan == "time_mean"
    assert runoff.timespan == "time_mean"
    assert snow.timespan == "instantaneous"
    assert soil.timespan == "instantaneous"
    assert discharge.latest >= date(2024, 1, 1)
    assert min(discharge.preferred) == date(1979, 1, 1)
