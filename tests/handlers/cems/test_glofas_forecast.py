"""
Tests for country-level GloFAS forecast and auxiliary downloads.

Offline tests use a stand-in catalogue and a stand-in EWDS job. Live tests
read the public catalogue and the auxiliary file headers. A real forecast
download runs only when ``EWDS_API_KEY`` is set.

Run with:
    pytest tests/handlers/cems/test_glofas_forecast.py -v
    pytest tests/handlers/cems/test_glofas_forecast.py -m live -o addopts= -v
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import MultiPolygon, box

from gigaspatial.handlers.boundaries import AdminBoundaries
from gigaspatial.handlers.cems import glofas_forecast as module
from gigaspatial.handlers.cems.flood_hazard import CEMSFloodHazardHandler
from gigaspatial.handlers.cems.glofas_forecast import (
    AUXILIARY,
    RETURN_PERIODS,
    ForecastCatalogue,
    GlofasForecastHandler,
    decode_payload,
    expected_resolution,
    extraction_areas,
    runs_to_fetch,
    subset_native,
    v4_auxiliary_applies,
)


DISCHARGE = "river_discharge_in_the_last_24_hours"
SNOW = "snow_depth_water_equivalent"
RUNOFF = "runoff_water_equivalent"
SOIL = "soil_wetness_index"
LATITUDE = np.array([-1.975, -1.925])
LONGITUDE = np.array([29.025, 29.075])
DAY_A = date(2024, 3, 1)
DAY_B = date(2024, 3, 2)
DAY_C = date(2024, 3, 4)


def collection(updated="2026-11-16T00:00:00Z") -> dict:
    return {
        "updated": updated,
        "sci:doi": "10.24381/cds.ff1aef77",
        "extent": {"temporal": {"interval": [["2019-11-05T00:00:00Z", "2026-11-16T00:00:00Z"]]}},
    }


def constraint_row(variable, model, product, days, leads=("24", "48"), system=("operational",)):
    return {
        "system_version": list(system),
        "hydrological_model": [model],
        "product_type": [product],
        "variable": [variable],
        "year": sorted({f"{day.year:04d}" for day in days}),
        "month": sorted({f"{day.month:02d}" for day in days}),
        "day": sorted({f"{day.day:02d}" for day in days}),
        "leadtime_hour": list(leads),
    }


def catalogue_for(rows, updated="2026-11-16T00:00:00Z"):
    return ForecastCatalogue(collection(updated), rows)


def standard_rows(days):
    return [
        constraint_row(DISCHARGE, "lisflood", "control_forecast", days),
        constraint_row(DISCHARGE, "lisflood", "ensemble_perturbed_forecasts", days),
        constraint_row(SNOW, "lisflood", "control_forecast", [date(2024, 2, 28)]),
        constraint_row(RUNOFF, "lisflood", "control_forecast", [date(2024, 2, 29)]),
        constraint_row(SOIL, "lisflood", "control_forecast", [date(2024, 2, 28)]),
        constraint_row(
            DISCHARGE,
            "htessel_lisflood",
            "control_forecast",
            [date(2019, 11, 5)],
            system=("operational",),
        ),
    ]


def write_forecast(path, day, leads=(24, 48), members=(0,), fill=1.0, step=0.05):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    latitude = LATITUDE if step == 0.05 else np.array([-1.95, -1.85])
    longitude = LONGITUDE if step == 0.05 else np.array([29.05, 29.15])
    data = np.full((1, len(members), len(leads), latitude.size, longitude.size), fill, dtype="float32")
    valid = [np.datetime64(day.isoformat()) + np.timedelta64(int(hour), "h") for hour in leads]
    dataset = xr.Dataset(
        {"dis24": (("time", "number", "step", "latitude", "longitude"), data)},
        coords={
            "time": [np.datetime64(day.isoformat(), "ns")],
            "number": np.array(members, dtype="int32"),
            "step": np.array(leads, dtype="int32"),
            "valid_time": ("step", np.array(valid, dtype="datetime64[ns]")),
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    dataset["dis24"].attrs["units"] = "m3 s-1"
    dataset.to_netcdf(path)
    dataset.close()


def install_retrieve(monkeypatch, generation):
    fetched = []

    def fake(self, request, dest, options):
        day = date(int(request["year"][0]), int(request["month"][0]), int(request["day"][0]))
        leads = tuple(int(hour) for hour in request["leadtime_hour"])
        product = request["product_type"][0]
        members = (1, 2, 3) if product == "ensemble_perturbed_forecasts" else (0,)
        fetched.append({"day": day, "leads": leads, "product": product, "area": list(request["area"])})
        write_forecast(dest, day, leads, members, fill=generation["n"] * 10 + day.day)

    monkeypatch.setattr(module.GlofasForecastDownloader, "_retrieve", fake)
    return fetched


def build_handler(tmp_path, catalogue, monkeypatch, **kwargs):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    handler = GlofasForecastHandler(
        base_path=tmp_path / "forecast",
        auxiliary_base_path=tmp_path / "auxiliary",
        api_key="token",
        n_workers=1,
        system_version=kwargs.pop("system_version", "operational"),
        **kwargs,
    )
    handler.config._catalogue = catalogue
    return handler


def test_runs_to_fetch_keeps_complete_runs_and_refreshes_the_open_tail():
    leads = (24, 48)
    expected = {DAY_A: leads, DAY_B: leads}
    stored = {
        DAY_A.isoformat(): {"lead_hours": [24, 48], "open": False, "catalogue_updated": "old"},
        DAY_B.isoformat(): {"lead_hours": [24, 48], "open": True, "catalogue_updated": "old"},
    }
    assert runs_to_fetch(stored, expected, catalogue_updated="old", latest=DAY_B) == []
    refreshed = runs_to_fetch(stored, expected, catalogue_updated="new", latest=DAY_B)
    assert refreshed == [DAY_B]
    short = {
        DAY_A.isoformat(): {"lead_hours": [24], "open": False, "catalogue_updated": "same"},
    }
    assert runs_to_fetch(short, {DAY_A: leads}, catalogue_updated="same", latest=DAY_A) == []
    assert runs_to_fetch(short, {DAY_A: leads}, catalogue_updated="later", latest=DAY_A) == [DAY_A]


def test_variable_constraints_keep_separate_start_dates():
    catalogue = catalogue_for(standard_rows([DAY_A]))
    discharge = catalogue.availability("operational", DISCHARGE)
    snow = catalogue.availability("operational", SNOW)
    runoff = catalogue.availability("operational", RUNOFF)
    soil = catalogue.availability("operational", SOIL)
    assert min(discharge.days("htessel_lisflood", "control_forecast")) == date(2019, 11, 5)
    assert min(discharge.days("lisflood", "control_forecast")) == DAY_A
    assert min(snow.days("lisflood", "control_forecast")) == date(2024, 2, 28)
    assert min(runoff.days("lisflood", "control_forecast")) == date(2024, 2, 29)
    assert min(soil.days("lisflood", "control_forecast")) == date(2024, 2, 28)
    assert snow.days("lisflood", "control_forecast")[date(2024, 2, 28)] == (24, 48)
    with pytest.raises(ValueError, match="version_2_1"):
        catalogue.availability("version_2_1", SNOW)


def test_expected_resolution_follows_the_published_grid():
    assert expected_resolution("operational", date(2023, 7, 26)) == 0.05
    assert expected_resolution("operational", date(2023, 7, 25)) == 0.1
    assert expected_resolution("version_2_1", date(2021, 5, 26)) == 0.1
    assert expected_resolution("version_3_1", date(2023, 7, 26)) == 0.1


def test_v4_auxiliary_is_not_paired_with_legacy_forecasts():
    assert v4_auxiliary_applies("operational") is True
    assert v4_auxiliary_applies("operational", date(2023, 7, 26)) is True
    assert v4_auxiliary_applies("operational", date(2023, 7, 25)) is False
    assert v4_auxiliary_applies("version_2_1", date(2024, 1, 1)) is False
    assert v4_auxiliary_applies("version_3_1", date(2023, 7, 26)) is False


def test_extraction_areas_cover_islands_without_joining_distant_parts():
    one = extraction_areas(box(29, -2, 30, -1))
    assert len(one) == 1
    north, west, south, east = one[0]
    assert west < 29 and east > 30
    assert south < -2 and north > -1

    distant = extraction_areas(MultiPolygon([box(29, -2, 30, -1), box(120, 10, 121, 11)]))
    assert len(distant) == 2

    dateline = extraction_areas(MultiPolygon([box(175, -5, 180, 5), box(-180, -5, -175, 5)]))
    assert len(dateline) == 2
    assert dateline[0][3] < 0
    assert dateline[1][1] > 0


def test_country_resolution_uses_admin_boundaries_for_islands(tmp_path, monkeypatch):
    geometry = MultiPolygon([box(29, -2, 30, -1), box(120, 10, 121, 11)])

    class _Boundaries:
        def to_geoms(self):
            return [geometry]

    monkeypatch.setattr(AdminBoundaries, "create", classmethod(lambda cls, country_code, **kwargs: _Boundaries()))
    catalogue = catalogue_for(standard_rows([DAY_A]))
    handler = GlofasForecastHandler(
        base_path=tmp_path / "forecast",
        auxiliary_base_path=tmp_path / "auxiliary",
        api_key="token",
        n_workers=1,
    )
    handler.config._catalogue = catalogue
    units = handler.config.get_relevant_data_units(
        "Rwanda",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
    )
    assert handler.config.get_search_key("Rwanda", handler.config.options_type()) == "RWA"
    assert len(units) == 2
    assert {unit["country_iso3"] for unit in units} == {"RWA"}
    wests = sorted(unit["area"][1] for unit in units)
    assert wests[0] < 30 < 100 < wests[1]


def test_a_geometry_is_not_a_country_key():
    handler = GlofasForecastHandler(api_key="token", n_workers=1)
    with pytest.raises(ValueError, match="country"):
        handler.config.get_search_key(box(0, 0, 1, 1), handler.config.options_type())


def test_old_grib_name_is_rejected():
    with pytest.raises(ValueError, match="dis24"):
        GlofasForecastHandler().config.options_type(variable="dis24")


def test_subset_keeps_native_cells_and_categorical_codes():
    latitude = np.array([89.975, 89.925, 89.875])
    longitude = np.array([-179.975, -179.925, -179.875])
    values = np.array([[1, 2, 3], [9, 5, 4], [8, 7, 255]], dtype="uint8")
    dataset = xr.Dataset(
        {"ldd": (("lat", "lon"), values)},
        coords={"lat": latitude, "lon": longitude},
    )
    dataset["ldd"].attrs["_FillValue"] = np.uint8(255)
    dataset["crs"] = xr.DataArray(np.array(b"", dtype="|S1"), attrs={"grid_mapping_name": "latitude_longitude"})
    subset = subset_native(dataset, (89.95, -179.95, 89.90, -179.90))
    module._require_categorical(subset)
    assert subset["ldd"].dtype == np.uint8
    assert subset["lat"].values.tolist() == [89.925]
    assert subset["lon"].values.tolist() == [-179.925]
    assert int(subset["ldd"].values.item()) == 5
    assert subset["lat"].attrs.get("standard_name", None) is None or "lat" in subset.dims


def test_all_return_periods_are_separate_auxiliary_datasets():
    periods = {
        spec["return_period_years"]
        for spec in AUXILIARY.values()
        if spec["return_period_years"] is not None
    }
    assert periods == {float(period) for period in RETURN_PERIODS}
    assert AUXILIARY["flood_threshold_mu"]["distribution_parameter"] == "mu"
    assert AUXILIARY["flood_threshold_sigma"]["distribution_parameter"] == "sigma"
    assert AUXILIARY["ldd"]["categorical"] is True
    names = {spec["filename"] for spec in AUXILIARY.values()}
    assert len(names) == len(AUXILIARY)


def test_legacy_version_does_not_receive_v4_auxiliary(tmp_path, monkeypatch, caplog):
    rows = [
        constraint_row(
            DISCHARGE,
            "htessel_lisflood",
            "control_forecast",
            [date(2021, 5, 26)],
            system=("version_2_1",),
        )
    ]
    handler = build_handler(tmp_path, catalogue_for(rows), monkeypatch, system_version="version_2_1")
    caplog.set_level(logging.WARNING)
    units = handler.config.get_relevant_data_units("RWA", include_auxiliary=True)
    assert {unit["kind"] for unit in units} == {"forecast"}
    assert "version_2_1" in caplog.text
    with pytest.raises(ValueError, match="version_2_1"):
        handler.config.get_relevant_data_units("RWA", variable="ldd")


def test_forecast_preserves_members_steps_and_original_names(tmp_path, monkeypatch):
    catalogue = catalogue_for(standard_rows([DAY_A, DAY_B]))
    handler = build_handler(tmp_path, catalogue, monkeypatch)
    generation = {"n": 1}
    fetched = install_retrieve(monkeypatch, generation)
    loaded = handler.load_data(
        "RWA",
        variable=DISCHARGE,
        product_type="ensemble_perturbed_forecasts",
        hydrological_model="lisflood",
        show_progress=False,
    )
    assert fetched[0]["leads"] == (24, 48)
    assert fetched[0]["area"][1] < 29 < 30 < fetched[0]["area"][3]
    assert "dis24" in loaded.data_vars
    assert DISCHARGE not in loaded.data_vars
    assert loaded.sizes["number"] == 3
    assert loaded.sizes["step"] == 2
    assert loaded.sizes["time"] == 2
    assert "valid_time" in loaded.coords
    assert loaded["dis24"].attrs["units"] == "m3 s-1"
    assert float(np.diff(loaded["longitude"].values)[0]) == pytest.approx(0.05)
    assert set(np.round(loaded["longitude"].values, 3)).issubset({29.025, 29.075})


def test_control_forecast_keeps_a_size_one_member_dimension(tmp_path, monkeypatch):
    handler = build_handler(tmp_path, catalogue_for(standard_rows([DAY_A])), monkeypatch)
    install_retrieve(monkeypatch, {"n": 1})
    loaded = handler.load_data(
        "RWA",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
        show_progress=False,
    )
    assert loaded.sizes["number"] == 1
    assert int(loaded["number"].values[0]) == 0


def test_netcdf_zip_and_grib_dispatch(monkeypatch):
    dataset = xr.Dataset(
        {"dis24": (("time", "number", "step", "latitude", "longitude"), np.ones((1, 2, 2, 2, 2)))},
        coords={
            "time": [np.datetime64("2024-03-01", "ns")],
            "number": [1, 2],
            "step": [24, 48],
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        },
    )
    handle = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    handle.close()
    netcdf_path = Path(handle.name)
    dataset.to_netcdf(netcdf_path)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("data.nc", netcdf_path.read_bytes())
    netcdf_path.unlink()
    opened = decode_payload(archive.getvalue())
    assert opened.sizes["number"] == 2
    assert opened.sizes["step"] == 2
    opened.close()

    def fake_grib(raw):
        assert raw[:4] == b"GRIB"
        return dataset.copy()

    monkeypatch.setattr(module, "open_grib_bytes", fake_grib)
    grib = decode_payload(b"GRIB fake message")
    assert grib.sizes["number"] == 2
    assert "step" in grib.dims
    grib.close()


def test_resampled_forecast_is_rejected(tmp_path, monkeypatch):
    handler = build_handler(tmp_path, catalogue_for(standard_rows([DAY_A])), monkeypatch)

    def coarse(self, request, dest, options):
        write_forecast(dest, DAY_A, step=0.1)

    monkeypatch.setattr(module.GlofasForecastDownloader, "_retrieve", coarse)
    assert (
        handler.ensure_data_available(
            "RWA",
            variable=DISCHARGE,
            product_type="control_forecast",
            hydrological_model="lisflood",
            show_progress=False,
        )
        is False
    )
    assert list(Path(tmp_path).rglob("*.nc")) == []


def test_incremental_download_refreshes_only_the_open_tail(tmp_path, monkeypatch, caplog):
    catalogue = catalogue_for(standard_rows([DAY_A, DAY_B]))
    handler = build_handler(tmp_path, catalogue, monkeypatch)
    generation = {"n": 1}
    fetched = install_retrieve(monkeypatch, generation)
    handler.load_data(
        "RWA",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
        show_progress=False,
    )
    assert [item["day"] for item in fetched] == [DAY_A, DAY_B]

    fetched.clear()
    caplog.set_level(logging.INFO)
    assert (
        handler.ensure_data_available(
            "RWA",
            variable=DISCHARGE,
            product_type="control_forecast",
            hydrological_model="lisflood",
            show_progress=False,
        )
        is True
    )
    assert fetched == []
    assert "already up to date" in caplog.text

    generation["n"] = 2
    later = catalogue_for(standard_rows([DAY_A, DAY_B, DAY_C]), updated="2026-11-17T00:00:00Z")
    handler.config._catalogue = later
    handler.load_data(
        "RWA",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
        show_progress=False,
    )
    assert [item["day"] for item in fetched] == [DAY_B, DAY_C]
    reloaded = handler.reader.load(
        "RWA",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
    )
    values = {
        pd.Timestamp(stamp).date(): float(reloaded["dis24"].sel(time=stamp).values.flat[0])
        for stamp in reloaded["time"].values
    }
    assert values[DAY_A] == 11
    assert values[DAY_B] == 22
    assert values[DAY_C] == 24


def test_a_failed_request_keeps_runs_already_stored(tmp_path, monkeypatch):
    handler = build_handler(tmp_path, catalogue_for(standard_rows([DAY_A, DAY_C])), monkeypatch)
    calls = {"n": 0}

    def fake(self, request, dest, options):
        calls["n"] += 1
        day = date(int(request["year"][0]), int(request["month"][0]), int(request["day"][0]))
        if calls["n"] == 2:
            raise RuntimeError("EWDS unavailable")
        write_forecast(dest, day, fill=float(day.day))

    monkeypatch.setattr(module.GlofasForecastDownloader, "_retrieve", fake)
    assert (
        handler.ensure_data_available(
            "RWA",
            variable=DISCHARGE,
            product_type="control_forecast",
            hydrological_model="lisflood",
            show_progress=False,
        )
        is False
    )
    metadata = json.loads(next(Path(tmp_path).rglob("metadata.json")).read_text(encoding="utf-8"))
    assert list(metadata["runs"]) == [DAY_A.isoformat()]

    def recover(self, request, dest, options):
        day = date(int(request["year"][0]), int(request["month"][0]), int(request["day"][0]))
        write_forecast(dest, day, fill=float(day.day))

    monkeypatch.setattr(module.GlofasForecastDownloader, "_retrieve", recover)
    assert (
        handler.ensure_data_available(
            "RWA",
            variable=DISCHARGE,
            product_type="control_forecast",
            hydrological_model="lisflood",
            show_progress=False,
        )
        is True
    )
    metadata = json.loads(next(Path(tmp_path).rglob("metadata.json")).read_text(encoding="utf-8"))
    assert sorted(metadata["runs"]) == [DAY_A.isoformat(), DAY_C.isoformat()]
    stored = handler.reader.load(
        "RWA",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
    )
    assert float(stored["dis24"].sel(time=np.datetime64(DAY_A.isoformat())).values.flat[0]) == 1


def test_request_retries_a_transient_status():
    class Response:
        def __init__(self, status):
            self.status_code = status
            self.text = "busy"

    class Session:
        def __init__(self):
            self.calls = 0

        def request(self, method, url, **kwargs):
            self.calls += 1
            if self.calls < 3:
                return Response(503)
            return Response(200)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    session = Session()
    response = module._request(session, "GET", "https://example.test")
    assert response.status_code == 200
    assert session.calls == 3
    monkeypatch.undo()


def test_an_interrupted_replace_does_not_publish_a_partial_file(tmp_path, monkeypatch):
    handler = build_handler(tmp_path, catalogue_for(standard_rows([DAY_A])), monkeypatch)
    install_retrieve(monkeypatch, {"n": 1})
    real_replace = module.os.replace
    calls = {"n": 0}

    def explode(source, target):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", explode)
    assert (
        handler.ensure_data_available(
            "RWA",
            variable=DISCHARGE,
            product_type="control_forecast",
            hydrological_model="lisflood",
            show_progress=False,
        )
        is False
    )
    assert list(Path(tmp_path).rglob("*.nc")) == []
    assert (
        handler.ensure_data_available(
            "RWA",
            variable=DISCHARGE,
            product_type="control_forecast",
            hydrological_model="lisflood",
            show_progress=False,
        )
        is True
    )
    assert len(list(Path(tmp_path).rglob("*.nc"))) == 1
    assert list(Path(tmp_path).rglob("*.partial")) == []


def test_a_stored_run_stays_readable_when_metadata_is_missing(tmp_path, monkeypatch):
    handler = build_handler(tmp_path, catalogue_for(standard_rows([DAY_A])), monkeypatch)
    fetched = install_retrieve(monkeypatch, {"n": 1})
    loaded = handler.load_data(
        "RWA",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
        show_progress=False,
    )
    for path in (tmp_path / "forecast").rglob("metadata.json"):
        path.unlink()
    again = handler.load_data(
        "RWA",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
        show_progress=False,
    )
    assert len(fetched) == 1
    assert float(again["dis24"].values.flat[0]) == float(loaded["dis24"].values.flat[0])


def test_missing_api_key_does_not_start_a_forecast_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    handler = GlofasForecastHandler(
        base_path=tmp_path / "forecast",
        auxiliary_base_path=tmp_path / "auxiliary",
        api_key="",
        n_workers=1,
    )
    handler.config._catalogue = catalogue_for(standard_rows([DAY_A]))
    fetched = install_retrieve(monkeypatch, {"n": 1})
    assert (
        handler.ensure_data_available(
            "RWA",
            variable=DISCHARGE,
            product_type="control_forecast",
            show_progress=False,
        )
        is False
    )
    assert fetched == []
    assert list(Path(tmp_path).rglob("*.nc")) == []


def test_auxiliary_extract_preserves_ldd_and_is_not_downloaded_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    downloads = []
    fingerprints = {"upstream_area": "10|old", "ldd": "11|old", "elevation": "12|old"}
    for period in RETURN_PERIODS:
        fingerprints[f"flood_threshold_rl_{period}"] = f"{period}|old"
    fingerprints["flood_threshold_mu"] = "mu|old"
    fingerprints["flood_threshold_sigma"] = "sigma|old"

    def fake_fingerprint(self, name):
        length, modified = fingerprints[name].split("|")
        return {
            "content_length": length,
            "last_modified": modified,
            "content_type": "application/x-netcdf",
            "url": module.AUXILIARY[name]["url"],
        }

    def fake_stream(url, path, store, **kwargs):
        downloads.append(url)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        latitude = np.arange(-2.475, -0.4, 0.05)
        longitude = np.arange(28.525, 30.6, 0.05)
        shape = (latitude.size, longitude.size)
        if path.name.startswith("ldd"):
            data = np.full(shape, 7, dtype="uint8")
            data[shape[0] // 2, shape[1] // 2] = np.uint8(255)
            dataset = xr.Dataset({"ldd": (("lat", "lon"), data)}, coords={"lat": latitude, "lon": longitude})
        elif "flood_threshold" in path.name:
            dataset = xr.Dataset(
                {"threshold": (("latitude", "longitude"), np.full(shape, 3.5, dtype="float64"))},
                coords={"latitude": latitude, "longitude": longitude},
            )
        else:
            name = "uparea" if "uparea" in path.name else "elevation"
            dataset = xr.Dataset(
                {name: (("latitude", "longitude"), np.full(shape, 4.0, dtype="float32"))},
                coords={"latitude": latitude, "longitude": longitude},
            )
            dataset[name].attrs["units"] = "m2" if name == "uparea" else "m"
        dataset.to_netcdf(path)
        dataset.close()

    monkeypatch.setattr(module.GlofasForecastConfig, "auxiliary_fingerprint", fake_fingerprint)
    monkeypatch.setattr(module, "stream_to_store", fake_stream)
    handler = GlofasForecastHandler(
        base_path=tmp_path / "forecast",
        auxiliary_base_path=tmp_path / "auxiliary",
        api_key="",
        n_workers=1,
    )
    loaded = handler.load_data("RWA", variable="ldd", show_progress=False)
    assert loaded["ldd"].dtype == np.uint8
    assert set(np.unique(loaded["ldd"].values)).issubset({7, 255})
    assert 255 in set(np.unique(loaded["ldd"].values))
    assert float(np.diff(sorted(loaded["lon"].values))[0]) == pytest.approx(0.05)
    first = len(downloads)
    handler.load_data("RWA", variable="ldd", show_progress=False)
    assert len(downloads) == first

    fingerprints["ldd"] = "99|new"
    handler.config._fingerprints.clear()
    handler.load_data("RWA", variable="ldd", show_progress=False)
    assert len(downloads) == first + 1

    for name in module.AUXILIARY:
        handler.load_data("RWA", variable=name, show_progress=False)
    stored_names = {path.parent.parent.name for path in Path(tmp_path, "auxiliary").rglob("static.nc")}
    assert "flood_threshold_rl_1.5" in stored_names
    assert "flood_threshold_rl_500.0" in stored_names
    assert "flood_threshold_mu" in stored_names
    assert "flood_threshold_sigma" in stored_names
    assert "upstream_area" in stored_names
    assert "elevation" in stored_names
    metadata = json.loads(
        next(path for path in Path(tmp_path, "auxiliary").rglob("metadata.json") if "ldd" in path.as_posix()).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["auxiliary_version"] == "4.0"
    assert metadata["applies_from"] == "2023-07-26"
    assert metadata["categorical"] is True
    assert "version_3_1" in metadata["applies_to"]


def test_forecast_and_auxiliary_paths_stay_separate(tmp_path, monkeypatch):
    handler = build_handler(tmp_path, catalogue_for(standard_rows([DAY_A])), monkeypatch)
    install_retrieve(monkeypatch, {"n": 1})
    monkeypatch.setattr(
        module.GlofasForecastConfig,
        "auxiliary_fingerprint",
        lambda self, name: {
            "content_length": "1",
            "last_modified": "old",
            "content_type": "application/x-netcdf",
            "url": "https://example.test/" + name,
        },
    )

    def fake_stream(url, path, store, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        latitude = np.arange(-2.475, -0.4, 0.05)
        longitude = np.arange(28.525, 30.6, 0.05)
        data = np.ones((latitude.size, longitude.size), dtype="float32")
        dataset = xr.Dataset(
            {"uparea": (("latitude", "longitude"), data)},
            coords={"latitude": latitude, "longitude": longitude},
        )
        dataset.to_netcdf(path)
        dataset.close()

    monkeypatch.setattr(module, "stream_to_store", fake_stream)
    handler.load_data(
        "RWA",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
        include_auxiliary=False,
        show_progress=False,
    )
    handler.load_data("RWA", variable="upstream_area", show_progress=False)
    forecast_root = tmp_path / "forecast"
    auxiliary_root = tmp_path / "auxiliary"
    forecast_files = list(forecast_root.rglob("*.nc"))
    auxiliary_files = list(auxiliary_root.rglob("static.nc"))
    assert forecast_files
    assert auxiliary_files
    assert all(path.is_relative_to(forecast_root) for path in forecast_files)
    assert all(path.is_relative_to(auxiliary_root) for path in auxiliary_files)


def test_handler_is_registered_and_flood_hazard_stays_separate():
    from gigaspatial.handlers.cems import GlofasForecastHandler as exported

    assert exported is GlofasForecastHandler
    assert CEMSFloodHazardHandler.__module__ == "gigaspatial.handlers.cems.flood_hazard"
    with pytest.raises(ModuleNotFoundError):
        __import__("gigaspatial.handlers.glofas")


def test_provider_errors_do_not_echo_the_api_key():
    class Response:
        status_code = 401
        text = "token secret-token rejected"

    assert "secret-token" not in module._provider_message(Response(), "secret-token")


@pytest.mark.live
def test_live_catalogue_covers_each_variable_on_its_own_dates():
    catalogue = module.fetch_catalogue()
    assert catalogue.temporal_start == date(2019, 11, 5)
    assert catalogue.doi == "10.24381/cds.ff1aef77"
    discharge = catalogue.availability("operational", DISCHARGE)
    snow = catalogue.availability("operational", SNOW)
    runoff = catalogue.availability("operational", RUNOFF)
    soil = catalogue.availability("operational", SOIL)
    assert min(discharge.days("htessel_lisflood", "control_forecast")) == date(2019, 11, 5)
    assert min(discharge.days("lisflood", "control_forecast")) == date(2021, 5, 26)
    assert min(snow.days("lisflood", "control_forecast")) == date(2024, 2, 28)
    assert min(soil.days("lisflood", "control_forecast")) == date(2024, 2, 28)
    assert min(runoff.days("lisflood", "control_forecast")) == date(2024, 2, 29)
    leads = discharge.days("lisflood", "control_forecast")[date(2024, 3, 1)]
    assert leads == tuple(range(24, 744, 24))
    legacy = catalogue.availability("version_3_1", DISCHARGE)
    assert min(legacy.days("lisflood", "control_forecast")) == date(2023, 7, 26)
    with pytest.raises(ValueError):
        catalogue.availability("version_2_1", SNOW)


@pytest.mark.live
def test_live_auxiliary_headers_identify_every_v4_file():
    session = module.requests.Session()
    seen = set()
    for name, spec in AUXILIARY.items():
        response = module._request(session, "HEAD", spec["url"], timeout=60, allow_redirects=True)
        assert response.status_code == 200
        assert "netcdf" in response.headers.get("Content-Type", "").lower()
        assert int(response.headers.get("Content-Length", "0")) > 1000
        seen.add(spec["filename"])
    assert "uparea_glofas_v4_0.nc" in seen
    assert "elevation_glofas_v4_0.nc" in seen
    assert "ldd_glofas_v4_0.nc" in seen
    assert "flood_threshold_glofas_v4_rl_1.5.nc" in seen
    assert "flood_threshold_glofas_v4_rl_500.0.nc" in seen
    assert "flood_threshold_glofas_v4_mu.nc" in seen
    assert "flood_threshold_glofas_v4_sigma.nc" in seen


@pytest.mark.live
def test_live_auxiliary_country_extract(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    handler = GlofasForecastHandler(
        base_path=tmp_path / "forecast",
        auxiliary_base_path=tmp_path / "auxiliary",
        api_key="",
        n_workers=1,
    )
    for name in ("ldd", "upstream_area", "elevation", "flood_threshold_rl_1.5"):
        loaded = handler.load_data("RWA", variable=name, show_progress=False)
        assert loaded.sizes
        lat_name = "latitude" if "latitude" in loaded.coords else "lat"
        lon_name = "longitude" if "longitude" in loaded.coords else "lon"
        latitude = np.asarray(loaded[lat_name].values, dtype=float)
        longitude = np.asarray(loaded[lon_name].values, dtype=float)
        assert latitude.size and longitude.size
        assert float(np.nanmax(latitude)) <= 0
        assert float(np.nanmin(longitude)) >= 28
        step = float(np.abs(np.diff(np.sort(latitude))).flat[0]) if latitude.size > 1 else 0.05
        assert step == pytest.approx(0.05)
    ldd = handler.reader.load("RWA", variable="ldd")
    assert ldd["ldd"].dtype == np.uint8
    assert set(np.unique(ldd["ldd"].values.astype(int))).issubset(set(range(1, 10)) | {255})
    elevation = handler.reader.load("RWA", variable="elevation")
    upstream = handler.reader.load("RWA", variable="upstream_area")
    threshold = handler.reader.load("RWA", variable="flood_threshold_rl_1.5")
    assert elevation["elevation"].attrs["units"] == "m"
    assert upstream["uparea"].attrs["units"] == "m2"
    assert threshold["rl_1.5"].dtype == np.float64
    again = []
    monkeypatch.setattr(module, "stream_to_store", lambda *args, **kwargs: again.append(args))
    assert handler.ensure_data_available("RWA", variable="ldd", show_progress=False) is True
    assert again == []


@pytest.mark.live
def test_live_forecast_download_for_one_country_day(tmp_path, monkeypatch):
    from gigaspatial.config import config as global_config

    if not global_config.EWDS_API_KEY:
        pytest.skip("EWDS_API_KEY is not set")
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: box(29, -2, 30, -1),
    )
    catalogue = module.fetch_catalogue()
    latest = max(catalogue.availability("operational", DISCHARGE).days("lisflood", "control_forecast"))
    day = date.fromordinal(latest.toordinal() - 2)
    handler = GlofasForecastHandler(
        base_path=tmp_path / "forecast",
        auxiliary_base_path=tmp_path / "auxiliary",
        n_workers=1,
    )
    loaded = handler.load_data(
        "RWA",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
        reference_start=day,
        reference_end=day,
        show_progress=False,
    )
    assert loaded.sizes["step"] == 30
    lat_name = "latitude" if "latitude" in loaded.coords else "lat"
    assert float(np.abs(np.diff(np.sort(loaded[lat_name].values))).flat[0]) == pytest.approx(0.05)
    calls = {"n": 0}

    def forbid(self, request, dest, options):
        calls["n"] += 1
        raise AssertionError("unchanged forecast was requested again")

    monkeypatch.setattr(module.GlofasForecastDownloader, "_retrieve", forbid)
    assert (
        handler.ensure_data_available(
            "RWA",
            variable=DISCHARGE,
            product_type="control_forecast",
            hydrological_model="lisflood",
            reference_start=day,
            reference_end=day,
            show_progress=False,
        )
        is True
    )
    assert calls["n"] == 0
