"""
Tests for the GloFAS forecast fetcher.

Offline tests use a stand-in catalogue and a stand-in EWDS job. Live tests
read the public catalogue and the auxiliary files. A real forecast query
runs only when ``EWDS_API_KEY`` is set.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import MultiPolygon, box

from gigaspatial.fetchers.cems import glofas_forecast as module
from gigaspatial.fetchers.cems.glofas_forecast import (
    AUXILIARY,
    RETURN_PERIODS,
    ForecastCatalogue,
    GlofasForecastFetcher,
    GlofasForecastResolveOptions,
    align_static_maps,
    decode_payload,
    expected_resolution,
    extraction_areas,
    subset_native,
    v4_auxiliary_applies,
)
from gigaspatial.handlers.cems.flood_hazard import CEMSFloodHazardHandler


DISCHARGE = "river_discharge_in_the_last_24_hours"
SNOW = "snow_depth_water_equivalent"
RUNOFF = "runoff_water_equivalent"
SOIL = "soil_wetness_index"
LATITUDE = np.array([-1.975, -1.925])
LONGITUDE = np.array([29.025, 29.075])
DAY_A = date(2024, 3, 1)
DAY_B = date(2024, 3, 2)


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
        constraint_row(DISCHARGE, "htessel_lisflood", "control_forecast", [date(2019, 11, 5)]),
    ]


def write_forecast(path, day, leads=(24, 48), members=(0,), fill=1.0, step=0.05, latitude=None, longitude=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    latitude = LATITUDE if latitude is None else latitude
    longitude = LONGITUDE if longitude is None else longitude
    if step != 0.05 and latitude is LATITUDE:
        latitude = np.array([-1.95, -1.85])
        longitude = np.array([29.05, 29.15])
    data = np.full((1, len(members), len(leads), len(latitude), len(longitude)), fill, dtype="float32")
    valid = [np.datetime64(day.isoformat(), "ns") + np.timedelta64(int(hour), "h") for hour in leads]
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


def patch_country(monkeypatch, geometry=box(29, -2, 30, -1)):
    monkeypatch.setattr(
        module,
        "source_geometry",
        lambda source, crs="EPSG:4326", options=None: geometry,
    )


def fetcher_for(catalogue, monkeypatch, **kwargs):
    patch_country(monkeypatch)
    kwargs.setdefault("api_key", "token")
    fetcher = GlofasForecastFetcher(**kwargs)
    fetcher._catalogue = catalogue
    return fetcher


def install_retrieve(monkeypatch):
    fetched = []

    def fake(self, request, dest):
        day = date(int(request["year"][0]), int(request["month"][0]), int(request["day"][0]))
        leads = tuple(int(hour) for hour in request["leadtime_hour"])
        product = request["product_type"][0]
        members = (1, 2, 3) if product == "ensemble_perturbed_forecasts" else (0,)
        north, west, south, east = request["area"]
        latitude = np.array([(south + north) / 2 - 0.025, (south + north) / 2 + 0.025])
        longitude = np.array([(west + east) / 2 - 0.025, (west + east) / 2 + 0.025])
        fetched.append({"day": day, "leads": leads, "product": product, "area": list(request["area"])})
        write_forecast(dest, day, leads, members, latitude=latitude, longitude=longitude)

    monkeypatch.setattr(module.GlofasForecastFetcher, "_retrieve", fake)
    return fetched


def test_variable_constraints_keep_separate_start_dates():
    catalogue = catalogue_for(standard_rows([DAY_A]))
    discharge = catalogue.availability("operational", DISCHARGE)
    assert min(discharge.days("htessel_lisflood", "control_forecast")) == date(2019, 11, 5)
    assert min(discharge.days("lisflood", "control_forecast")) == DAY_A
    assert min(catalogue.availability("operational", SNOW).days("lisflood", "control_forecast")) == date(2024, 2, 28)
    with pytest.raises(ValueError, match="version_2_1"):
        catalogue.availability("version_2_1", SNOW)


def test_expected_resolution_follows_the_published_grid():
    assert expected_resolution("operational", date(2023, 7, 26)) == 0.05
    assert expected_resolution("operational", date(2023, 7, 25)) == 0.1
    assert expected_resolution("version_2_1", date(2021, 5, 26)) == 0.1


def test_v4_auxiliary_is_not_paired_with_legacy_forecasts():
    assert v4_auxiliary_applies("operational", date(2023, 7, 26)) is True
    assert v4_auxiliary_applies("operational", date(2023, 7, 25)) is False
    assert v4_auxiliary_applies("version_2_1", date(2024, 1, 1)) is False


def test_extraction_areas_cover_islands_without_joining_distant_parts():
    distant = extraction_areas(MultiPolygon([box(29, -2, 30, -1), box(120, 10, 121, 11)]))
    assert len(distant) == 2
    dateline = extraction_areas(MultiPolygon([box(175, -5, 180, 5), box(-180, -5, -175, 5)]))
    assert len(dateline) == 2
    assert dateline[0][3] < 0
    assert dateline[1][1] > 0


def test_old_grib_name_is_rejected():
    with pytest.raises(ValueError, match="dis24"):
        GlofasForecastResolveOptions(variable="dis24")


def test_a_geometry_is_not_a_country():
    fetcher = GlofasForecastFetcher(api_key="token")
    with pytest.raises(ValueError, match="country"):
        fetcher.fetch(box(0, 0, 1, 1))


def test_islands_are_queried_as_separate_rectangles(monkeypatch):
    patch_country(
        monkeypatch,
        MultiPolygon([box(29, -2, 30, -1), box(120, 10, 121, 11)]),
    )
    fetcher = GlofasForecastFetcher(api_key="token")
    fetcher._catalogue = catalogue_for(standard_rows([DAY_A]))
    fetched = install_retrieve(monkeypatch)
    fetcher.fetch(
        "Rwanda",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
    )
    assert len(fetched) == 2
    wests = sorted(area[1] for area in (item["area"] for item in fetched))
    assert wests[0] < 30 < 100 < wests[1]


def test_forecast_preserves_members_steps_and_original_names(monkeypatch):
    fetcher = fetcher_for(catalogue_for(standard_rows([DAY_A, DAY_B])), monkeypatch)
    fetched = install_retrieve(monkeypatch)
    loaded = fetcher.fetch(
        "RWA",
        variable=DISCHARGE,
        product_type="ensemble_perturbed_forecasts",
        hydrological_model="lisflood",
    )
    assert [item["day"] for item in fetched] == [DAY_A, DAY_B]
    assert "dis24" in loaded.data_vars
    assert loaded.sizes["number"] == 3
    assert loaded.sizes["step"] == 2
    assert loaded.sizes["time"] == 2
    assert loaded["dis24"].attrs["units"] == "m3 s-1"


def test_resampled_forecast_is_rejected(monkeypatch):
    fetcher = fetcher_for(catalogue_for(standard_rows([DAY_A])), monkeypatch)

    def coarse(self, request, dest):
        write_forecast(dest, DAY_A, step=0.1)

    monkeypatch.setattr(module.GlofasForecastFetcher, "_retrieve", coarse)
    with pytest.raises(ValueError, match="not resampled"):
        fetcher.fetch(
            "RWA",
            variable=DISCHARGE,
            product_type="control_forecast",
            hydrological_model="lisflood",
        )


def test_missing_api_key_does_not_start_a_forecast_query(monkeypatch):
    fetcher = fetcher_for(catalogue_for(standard_rows([DAY_A])), monkeypatch, api_key="")
    called = []
    monkeypatch.setattr(
        module.GlofasForecastFetcher,
        "_retrieve",
        lambda self, request, dest: called.append(request),
    )
    with pytest.raises(RuntimeError, match="EWDS_API_KEY"):
        fetcher.fetch(
            "RWA",
            variable=DISCHARGE,
            product_type="control_forecast",
            hydrological_model="lisflood",
        )
    assert called == []


def test_legacy_version_does_not_retrieve_v4_auxiliary(monkeypatch, caplog):
    rows = [
        constraint_row(
            DISCHARGE,
            "htessel_lisflood",
            "control_forecast",
            [date(2021, 5, 26)],
            system=("version_2_1",),
        )
    ]
    fetcher = fetcher_for(catalogue_for(rows), monkeypatch, system_version="version_2_1")

    def fake(self, request, dest):
        north, west, south, east = request["area"]
        latitude = np.array([(south + north) / 2 - 0.05, (south + north) / 2 + 0.05])
        longitude = np.array([(west + east) / 2 - 0.05, (west + east) / 2 + 0.05])
        write_forecast(
            dest,
            date(2021, 5, 26),
            leads=(24, 48),
            step=0.1,
            latitude=latitude,
            longitude=longitude,
        )

    monkeypatch.setattr(module.GlofasForecastFetcher, "_retrieve", fake)
    downloads = []
    monkeypatch.setattr(
        module.GlofasForecastFetcher,
        "_download_to_path",
        lambda self, url, dest, session=None: downloads.append(url),
    )
    caplog.set_level(logging.WARNING, logger=fetcher.logger.name)
    loaded = fetcher.fetch("RWA", include_auxiliary=True, product_type="control_forecast")
    assert "version_2_1" in caplog.text
    assert downloads == []
    assert "dis24" in loaded.data_vars
    with pytest.raises(ValueError, match="version_2_1"):
        fetcher.fetch("RWA", variable="ldd")


def test_request_retries_a_transient_status(monkeypatch):
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

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    response = module._request(Session(), "GET", "https://example.test")
    assert response.status_code == 200


def test_provider_errors_do_not_echo_the_api_key():
    class Response:
        status_code = 401
        text = "token secret-key was rejected"

    message = module._provider_message(Response(), "secret-key")
    assert "secret-key" not in message
    assert "***" in message


def test_netcdf_zip_dispatch():
    import tempfile

    dataset = xr.Dataset({"dis24": (("number", "step"), np.ones((2, 2)))})
    handle = Path(tempfile.mkdtemp()) / "data.nc"
    dataset.to_netcdf(handle)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("data.nc", handle.read_bytes())
    opened = decode_payload(archive.getvalue())
    assert opened.sizes["number"] == 2
    opened.close()
    dataset.close()


def test_subset_keeps_native_cells_and_categorical_codes():
    latitude = np.array([89.975, 89.925, 89.875])
    longitude = np.array([-179.975, -179.925, -179.875])
    values = np.array([[1, 2, 3], [9, 5, 4], [8, 7, 255]], dtype="uint8")
    dataset = xr.Dataset(
        {"ldd": (("lat", "lon"), values)},
        coords={"lat": latitude, "lon": longitude},
    )
    subset = subset_native(dataset, (89.95, -179.95, 89.90, -179.90))
    module._require_categorical(subset)
    assert subset["ldd"].dtype == np.uint8
    assert int(subset["ldd"].values.item()) == 5


def test_all_return_periods_are_separate_auxiliary_datasets():
    periods = {
        spec["return_period_years"]
        for spec in AUXILIARY.values()
        if spec["return_period_years"] is not None
    }
    assert periods == {float(period) for period in RETURN_PERIODS}
    assert "uparea_glofas_v4_0.nc" in {spec["filename"] for spec in AUXILIARY.values()}


def test_combined_query_aligns_discharge_with_thresholds_and_upstream_area(monkeypatch):
    fetcher = fetcher_for(catalogue_for(standard_rows([DAY_A])), monkeypatch)
    install_retrieve(monkeypatch)

    def fake_download(self, url, dest, session=None):
        filename = url.rsplit("/", 1)[-1]
        if filename.startswith("ldd"):
            values = np.ones((2, 2), dtype="uint8")
            dataset = xr.Dataset(
                {"ldd": (("latitude", "longitude"), values)},
                coords={"latitude": LATITUDE, "longitude": LONGITUDE},
            )
        else:
            fill = 1.0e6 if "uparea" in filename else 5.0
            dataset = xr.Dataset(
                {"band": (("latitude", "longitude"), np.full((2, 2), fill, dtype="float32"))},
                coords={"latitude": LATITUDE, "longitude": LONGITUDE},
            )
        dataset.to_netcdf(dest)
        dataset.close()

    monkeypatch.setattr(module.GlofasForecastFetcher, "_download_to_path", fake_download)
    monkeypatch.setattr(
        module.GlofasForecastFetcher,
        "_retrieve",
        lambda self, request, dest: write_forecast(dest, DAY_A, leads=(24, 48), fill=12.0),
    )
    loaded = fetcher.fetch(
        "RWA",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
        include_auxiliary=True,
    )
    discharge = loaded[DISCHARGE]
    assert discharge["upstream_area"].values[0, 0] == pytest.approx(1.0e6)
    assert float(discharge["exceeded_return_period_years"].values.flat[0]) == 500.0
    assert "upstream_area" in loaded


def test_v4_discharge_aligns_with_reversed_auxiliary_latitude():
    discharge = np.array([[[[[10.0, 3.0], [module.AUXILIARY_FILL, 20.0]]]]], dtype="float32")
    forecast = xr.Dataset(
        {"dis24": (("time", "number", "step", "latitude", "longitude"), discharge)},
        coords={
            "time": [np.datetime64("2024-03-01", "ns")],
            "number": np.array([0], dtype="int32"),
            "step": np.array([24], dtype="int32"),
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        },
    )
    upstream = np.array([[3.0e6, module.AUXILIARY_FILL], [1.0e6, 2.0e6]], dtype="float32")
    low = np.full((2, 2), 5.0, dtype="float32")
    high = np.full((2, 2), 15.0, dtype="float32")

    def static(name, values, latitude):
        return xr.Dataset(
            {name: (("latitude", "longitude"), values)},
            coords={"latitude": latitude, "longitude": LONGITUDE},
        )

    aligned = align_static_maps(
        forecast,
        {
            "upstream_area": static("uparea", upstream, LATITUDE[::-1]),
            "flood_threshold_rl_1.5": static("threshold", low, LATITUDE[::-1]),
            "flood_threshold_rl_2.0": static("threshold", high, LATITUDE[::-1]),
        },
    )
    assert aligned["upstream_area"].values[0, 0] == pytest.approx(1.0e6)
    period = aligned["exceeded_return_period_years"].values[0, 0, 0]
    assert period[0, 0] == pytest.approx(1.5)
    assert period[0, 1] == pytest.approx(0.0)
    assert np.isnan(period[1, 0])
    assert np.isnan(period[1, 1])


def test_auxiliary_cells_off_the_forecast_grid_are_rejected():
    forecast = xr.Dataset(
        {"dis24": (("latitude", "longitude"), np.ones((2, 2), dtype="float32"))},
        coords={"latitude": LATITUDE, "longitude": LONGITUDE},
    )
    shifted = xr.Dataset(
        {"uparea": (("latitude", "longitude"), np.ones((2, 2), dtype="float32"))},
        coords={"latitude": LATITUDE, "longitude": LONGITUDE + 0.04},
    )
    with pytest.raises(ValueError, match="were not interpolated"):
        align_static_maps(forecast, {"upstream_area": shifted})


def test_fetcher_is_registered_and_the_handler_is_gone():
    from gigaspatial import GlofasForecastFetcher as root
    from gigaspatial.fetchers import GlofasForecastFetcher as exported
    import gigaspatial.handlers as handlers

    assert root is GlofasForecastFetcher
    assert exported is GlofasForecastFetcher
    assert "GlofasForecastHandler" not in handlers.__all__
    assert CEMSFloodHazardHandler.__name__ == "CEMSFloodHazardHandler"


@pytest.mark.live
def test_live_catalogue_covers_discharge():
    catalogue = module.fetch_catalogue()
    discharge = catalogue.availability("operational", DISCHARGE)
    assert discharge.days("lisflood", "control_forecast")


@pytest.mark.live
def test_live_auxiliary_country_extract(monkeypatch):
    patch_country(monkeypatch)
    fetcher = GlofasForecastFetcher(api_key="")
    loaded = fetcher.fetch("RWA", variable="upstream_area")
    latitude = np.asarray(loaded["latitude"].values if "latitude" in loaded.coords else loaded["lat"].values)
    assert latitude.size
    assert float(np.nanmax(latitude)) <= 0


@pytest.mark.live
def test_live_forecast_query_for_one_country_day(monkeypatch):
    from gigaspatial.config import config as global_config

    if not global_config.EWDS_API_KEY:
        pytest.skip("EWDS_API_KEY is not set")
    patch_country(monkeypatch)
    catalogue = module.fetch_catalogue()
    latest = max(catalogue.availability("operational", DISCHARGE).days("lisflood", "control_forecast"))
    day = date.fromordinal(latest.toordinal() - 2)
    loaded = GlofasForecastFetcher().fetch(
        "RWA",
        variable=DISCHARGE,
        product_type="control_forecast",
        hydrological_model="lisflood",
        reference_start=day,
        reference_end=day,
    )
    assert loaded.sizes["step"] >= 1
    latitude = loaded["latitude"] if "latitude" in loaded.coords else loaded["lat"]
    assert float(np.abs(np.diff(np.sort(latitude.values))).flat[0]) == pytest.approx(0.05)
