"""
Tests for country-level GloFAS historical queries.

Offline tests use a stand-in catalogue and a stand-in EWDS job. The live
test reads the public catalogue and does not download a time series.

Run with:
    pytest tests/fetchers/test_glofas_historical.py -v
    pytest tests/fetchers/test_glofas_historical.py -m live -v
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import MultiPolygon, box

from gigaspatial.fetchers.cems import glofas_historical as module
from gigaspatial.fetchers.cems.glofas_historical import (
    GlofasHistoricalFetcher,
    HistoricalCatalogue,
    extraction_areas,
    group_dates,
    selected_days,
    validate_cube,
)
from gigaspatial.handlers.cems.flood_hazard import CEMSFloodHazardHandler


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


def write_cube(path, days, variable=DISCHARGE, area=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if area is None:
        latitude = LATITUDE
        longitude = LONGITUDE
    else:
        north, west, south, east = area
        lat_mid = round(((south + north) / 2) / 0.05) * 0.05
        lon_mid = round(((west + east) / 2) / 0.05) * 0.05
        latitude = np.array([lat_mid - 0.05, lat_mid, lat_mid + 0.05])
        longitude = np.array([lon_mid - 0.05, lon_mid, lon_mid + 0.05])
    data = np.zeros((len(days), latitude.size, longitude.size), dtype="float32")
    for index, day in enumerate(days):
        data[index] = float(day.day)
    dataset = xr.Dataset(
        {variable: (("time", "latitude", "longitude"), data)},
        coords={
            "time": pd.to_datetime(list(days)),
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    dataset[variable].attrs["units"] = "m3 s-1"
    dataset.to_netcdf(path)
    dataset.close()


def install_retrieve(monkeypatch):
    fetched = []

    def fake(self, request, dest):
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
        write_cube(dest, days, request["variable"][0], area=request["area"])

    monkeypatch.setattr(module.GlofasHistoricalFetcher, "_retrieve", fake)
    return fetched


def build_fetcher(catalogue, monkeypatch, **kwargs):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    kwargs.setdefault("api_key", "token")
    kwargs.setdefault("system_version", "version_4_0")
    fetcher = GlofasHistoricalFetcher(**kwargs)
    fetcher._catalogue = catalogue
    return fetcher


def test_selected_days_keeps_the_window_and_the_preferred_product():
    preferred = {
        date(2020, 1, 1): "consolidated",
        date(2020, 6, 1): "intermediate",
        date(2020, 6, 2): "intermediate",
    }
    assert selected_days(preferred, date(2020, 6, 1), date(2020, 6, 2)) == [
        (date(2020, 6, 1), "intermediate"),
        (date(2020, 6, 2), "intermediate"),
    ]
    catalogue = catalogue_for(
        {
            "intermediate": [date(2020, 6, 1)],
            "consolidated": [date(2020, 6, 1)],
        }
    )
    availability = catalogue.availability("version_4_0", DISCHARGE)
    assert availability.preferred[date(2020, 6, 1)] == "consolidated"


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

    close = extraction_areas(MultiPolygon([box(29, -2, 30, -1), box(30.2, -2, 31, -1)]))
    assert len(close) == 1

    distant = extraction_areas(MultiPolygon([box(29, -2, 30, -1), box(120, 10, 121, 11)]))
    assert len(distant) == 2

    dateline = extraction_areas(MultiPolygon([box(175, -5, 180, 5), box(-180, -5, -175, 5)]))
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
        GlofasHistoricalFetcher(api_key="token").fetch("RWA", variable="dis24")


def test_a_geometry_is_not_a_country_key():
    fetcher = GlofasHistoricalFetcher(api_key="token")
    with pytest.raises(ValueError, match="country"):
        fetcher.fetch(box(0, 0, 1, 1), variable=DISCHARGE)


def test_a_query_returns_the_requested_window_and_queries_again(monkeypatch):
    january = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 4)]
    catalogue = catalogue_for({"consolidated": january})
    fetcher = build_fetcher(catalogue, monkeypatch)
    fetched = install_retrieve(monkeypatch)

    loaded = fetcher.fetch("RWA", variable=DISCHARGE, start="2020-01-01", end="2020-01-02")
    assert list(fetched[0]["days"]) == ["01", "02"]
    assert fetched[0]["product"] == ["consolidated"]
    assert fetched[0]["area"][1] < 29 < 30 < fetched[0]["area"][3]
    assert loaded.sizes["time"] == 2
    assert float(loaded[DISCHARGE].isel(time=0).values.flat[0]) == 1.0
    assert loaded.attrs["dataset"] == "cems-glofas-historical"
    assert loaded.attrs["country_iso3"] == "RWA"
    assert loaded[DISCHARGE].attrs["units"] == "m3 s-1"
    assert loaded.attrs["spatial_resolution_degrees"] == pytest.approx(0.05)

    loaded = fetcher.fetch("RWA", variable=DISCHARGE, start="2020-01-04", end="2020-01-04")
    assert fetched[-1]["days"] == ["04"]
    assert loaded.sizes["time"] == 1
    assert len(fetched) == 2


def test_a_failed_request_returns_nothing(monkeypatch):
    rows = [
        constraint_row(DISCHARGE, "time_mean", "consolidated", [date(2020, 1, 1)]),
        constraint_row(DISCHARGE, "time_mean", "consolidated", [date(2020, 2, 2)]),
    ]
    catalogue = HistoricalCatalogue(collection("2020-02-02T00:00:00Z"), rows)
    fetcher = build_fetcher(catalogue, monkeypatch)
    calls = {"n": 0}

    def fake(self, request, dest):
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

    monkeypatch.setattr(module.GlofasHistoricalFetcher, "_retrieve", fake)
    with pytest.raises(RuntimeError, match="EWDS unavailable"):
        fetcher.fetch("RWA", variable=DISCHARGE, start="2020-01-01", end="2020-02-02")
    assert calls["n"] == 2


def test_missing_api_key_does_not_start_a_query(monkeypatch):
    catalogue = catalogue_for({"consolidated": [date(2020, 1, 1)]})
    fetcher = build_fetcher(catalogue, monkeypatch, api_key="")
    fetched = install_retrieve(monkeypatch)
    with pytest.raises(RuntimeError, match="EWDS_API_KEY"):
        fetcher.fetch("RWA", variable=DISCHARGE, start="2020-01-01", end="2020-01-01")
    assert fetched == []


def test_all_four_variables_are_queried_for_a_country(monkeypatch):
    day = [date(2020, 1, 1)]
    rows = [
        constraint_row(variable, "time_mean", "consolidated", day) for variable in module.VARIABLES
    ]
    catalogue = HistoricalCatalogue(collection("2020-01-01T00:00:00Z"), rows)
    fetcher = build_fetcher(catalogue, monkeypatch)
    fetched = install_retrieve(monkeypatch)
    loaded = fetcher.fetch("Rwanda", start="2020-01-01", end="2020-01-01")
    assert set(loaded) == set(module.VARIABLES)
    assert {call["variable"][0] for call in fetched} == set(module.VARIABLES)


def test_auxiliary_subset_keeps_native_cell_values(monkeypatch):
    seen = []

    def fake_download(self, url, dest, session=None):
        seen.append(url)
        latitude = np.arange(-2.475, -0.4, 0.05)
        longitude = np.arange(28.525, 30.6, 0.05)
        data = np.full((latitude.size, longitude.size), 7.0, dtype="float32")
        dataset = xr.Dataset(
            {"uparea": (("latitude", "longitude"), data)},
            coords={"latitude": latitude, "longitude": longitude},
        )
        dataset["uparea"].attrs["units"] = "m2"
        dataset.to_netcdf(dest)
        dataset.close()

    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    monkeypatch.setattr(module.GlofasHistoricalFetcher, "_download_to_path", fake_download)
    loaded = GlofasHistoricalFetcher(api_key="token").fetch("RWA", variable="upstream_area")
    assert "uparea_glofas_v4_0.nc" in seen[0]
    assert set(np.unique(loaded["uparea"].values)) == {7.0}
    assert float(np.diff(loaded["longitude"].values).flat[0]) == pytest.approx(0.05)


def test_version_5_has_no_published_auxiliary_file(monkeypatch, caplog):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    fetcher = GlofasHistoricalFetcher(api_key="token", system_version="version_5_0")
    caplog.set_level(logging.WARNING, logger=fetcher.logger.name)
    with pytest.raises(ValueError, match="No GloFAS historical variables"):
        fetcher.fetch("RWA", variable="upstream_area")
    assert "version_5_0" in caplog.text


def test_islands_are_queried_as_separate_rectangles(monkeypatch):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: MultiPolygon(
            [box(29, -2, 30, -1), box(120, 10, 121, 11)]
        ),
    )
    catalogue = catalogue_for({"consolidated": [date(2020, 1, 1)]})
    fetcher = GlofasHistoricalFetcher(api_key="token")
    fetcher._catalogue = catalogue
    fetched = install_retrieve(monkeypatch)
    loaded = fetcher.fetch("RWA", variable=DISCHARGE, start="2020-01-01", end="2020-01-01")
    assert len(fetched) == 2
    assert fetched[0]["area"] != fetched[1]["area"]
    assert set(loaded) and all(item.sizes["time"] == 1 for item in loaded.values())


def test_fetcher_is_registered_and_the_handler_is_gone():
    from gigaspatial import GlofasHistoricalFetcher as root
    from gigaspatial.fetchers import GlofasHistoricalFetcher as exported
    import gigaspatial.handlers as handlers

    assert root is GlofasHistoricalFetcher
    assert exported is GlofasHistoricalFetcher
    assert "GlofasHistoricalHandler" not in handlers.__all__
    assert CEMSFloodHazardHandler.__name__ == "CEMSFloodHazardHandler"


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
