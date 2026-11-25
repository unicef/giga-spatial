"""Offline tests for the ERA5 and ERA5-Land fetchers."""

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import MultiPolygon, box

from gigaspatial.fetchers.climate.era5 import ERA5, fetch_era5
from gigaspatial.fetchers.climate.era5land import ERA5_LAND, fetch_era5land
from gigaspatial.fetchers.climate.shared import (
    ClimateAuthError,
    ClimateAvailabilityError,
    ClimateLimitError,
    ClimateRequestError,
    clip,
    countries_intersecting,
    latest_available,
    merge_climate,
    open_climate,
    period_available,
    plan_requests,
    redact,
    request_areas,
    resolve_credentials,
    select_variables,
    standardize,
)

HOUR = datetime(2025, 1, 1, tzinfo=timezone.utc)
AREA = box(0, 0, 0.02, 0.02)


class FakeCDS:
    """Writes a tiny NetCDF for whatever variables the request asks for."""

    def __init__(self, latest=HOUR, fail_times=0, reject_batches=False, reject_names=()):
        self.latest = latest
        self.fail_times = fail_times
        self.reject_batches = reject_batches
        self.reject_names = set(reject_names)
        self.calls = []

    def latest_available(self, dataset_id):
        return self.latest

    def retrieve(self, dataset_id, request, target):
        self.calls.append((dataset_id, dict(request), target))
        names = list(request["variable"])
        if self.fail_times:
            self.fail_times -= 1
            target.write_bytes(b"partial")
            raise TimeoutError("timed out talking to CDS")
        if self.reject_batches and len(names) > 1:
            raise ClimateRequestError("400 combination is not compatible")
        if set(names) & self.reject_names:
            raise ClimateRequestError(f"400 invalid variable {names[0]}")
        values = np.full((1, 2, 2), 280.0)
        short = {"2m_temperature": "2t", "total_precipitation": "tp"}
        data = {}
        for name in names:
            stored = short[name] if request["data_format"] == "grib" and name in short else name
            data[stored] = (("time", "latitude", "longitude"), values.copy())
        dataset = xr.Dataset(
            data,
            coords={
                "time": [np.datetime64("2025-01-01T00")],
                "latitude": np.array([0.0, 0.25]),
                "longitude": np.array([0.0, 0.25]),
            },
        )
        dataset.to_netcdf(target)


def _fetch(function, variables, retriever=None, **kwargs):
    kwargs.setdefault("time", ("2025-01-01T00:00", "2025-01-01T00:00"))
    kwargs.setdefault("retriever", retriever or FakeCDS())
    return function(AREA, variables, **kwargs)


def test_public_imports():
    import gigaspatial
    import gigaspatial.fetchers

    assert gigaspatial.fetch_era5 is fetch_era5
    assert gigaspatial.fetch_era5land is fetch_era5land
    assert gigaspatial.fetchers.fetch_era5 is fetch_era5
    assert gigaspatial.fetchers.fetch_era5land is fetch_era5land


def test_catalogues_are_independent_and_complete():
    era5_ids = {item.id for item in ERA5.variables}
    land_ids = {item.id for item in ERA5_LAND.variables}

    assert len(era5_ids) == 262
    assert len(land_ids) == 60
    assert era5_ids != land_ids
    assert ERA5.by_id()["2m_temperature"].units == "K"
    assert ERA5.by_id()["2m_temperature"].description == "2m temperature"
    assert ERA5.by_id()["total_precipitation"].units == "m"
    assert ERA5.by_id()["total_precipitation"].accumulated is True
    assert ERA5.by_id()["vertical_integral_of_temperature"].units is None
    assert ERA5_LAND.by_id()["soil_type"].units == "code"
    assert ERA5_LAND.by_id()["land_sea_mask"].time_invariant is True
    assert "product_type" not in ERA5_LAND.__dataclass_fields__ or ERA5_LAND.default_product is None
    assert ERA5.start_year == 1940
    assert ERA5_LAND.start_year == 1950
    assert ERA5.resolution == 0.25
    assert ERA5_LAND.resolution == 0.1


def test_variable_selection_modes():
    one = select_variables(ERA5, "2m_temperature")
    many = select_variables(ERA5, ["2m_temperature", "total_precipitation", "2m_temperature"])
    everything = select_variables(ERA5, "all")

    assert [item.id for item in one] == ["2m_temperature"]
    assert [item.id for item in many] == ["2m_temperature", "total_precipitation"]
    assert len(everything) == 262
    with pytest.raises(ValueError, match="unknown CDS variable"):
        select_variables(ERA5, "not_a_real_variable")
    with pytest.raises(ValueError, match="unknown CDS variable"):
        select_variables(ERA5_LAND, "mean_wave_period")


def test_request_construction_differs_by_dataset():
    hour = [HOUR]
    era5_requests = plan_requests(
        ERA5, select_variables(ERA5, ["2m_temperature"]), hour, [(1.0, 0.0, 0.0, 1.0)], "netcdf"
    )
    land_requests = plan_requests(
        ERA5_LAND,
        select_variables(ERA5_LAND, ["2m_temperature", "total_precipitation"]),
        hour,
        [(1.0, 0.0, 0.0, 1.0)],
        "grib",
    )

    assert era5_requests[0]["product_type"] == ["reanalysis"]
    assert era5_requests[0]["data_format"] == "netcdf"
    assert "grid" not in era5_requests[0]
    assert "format" not in era5_requests[0]
    assert era5_requests[0]["area"] == [1.0, 0.0, 0.0, 1.0]
    assert "product_type" not in land_requests[0]
    assert land_requests[0]["data_format"] == "grib"
    assert land_requests[0]["variable"] == ["2m_temperature", "total_precipitation"]
    with pytest.raises(ValueError, match="no product_type"):
        plan_requests(
            ERA5_LAND,
            select_variables(ERA5_LAND, "2m_temperature"),
            hour,
            [(1.0, 0.0, 0.0, 1.0)],
            "netcdf",
            product_type="reanalysis",
        )


def test_all_variables_are_split_and_invariants_are_separate():
    hours = [HOUR]
    area = (0.25, 0.0, 0.0, 0.25)
    era5_requests = plan_requests(ERA5, ERA5.variables, hours, [area], "netcdf")
    land_requests = plan_requests(ERA5_LAND, ERA5_LAND.variables, hours, [area], "netcdf")
    era5_names = [name for request in era5_requests for name in request["variable"]]
    land_names = [name for request in land_requests for name in request["variable"]]

    assert era5_names == [item.id for item in ERA5.variables]
    assert set(land_names) == {item.id for item in ERA5_LAND.variables}
    assert all(len(request["variable"]) <= 10 for request in era5_requests)
    invariant = [request for request in land_requests if request["variable"] == ["land_sea_mask"]]
    assert len(invariant) == 1
    assert invariant[0]["time"] == ["00:00"]


def test_iso3_uses_admin_boundaries(monkeypatch):
    def create(country_code=None, **kwargs):
        entity = SimpleNamespace(country_iso="ITA", name="Italy", name_en="Italy", boundary_id="ITA")
        return SimpleNamespace(entities=[entity], to_geoms=lambda: [AREA])

    monkeypatch.setattr(
        "gigaspatial.handlers.boundaries.AdminBoundaries.create",
        staticmethod(create),
    )
    dataset = fetch_era5("ITA", ["2m_temperature"], time=("2025-01-01T00:00", "2025-01-01T00:00"), retriever=FakeCDS())

    assert dataset.attrs["area"] == "Italy (ITA)"
    assert dataset["2m_temperature"].shape == (1, 1, 1)


def test_country_geometry_and_geojson_are_clipped():
    geometry = {"type": "Polygon", "coordinates": [[[0, 0], [0.02, 0], [0.02, 0.02], [0, 0.02], [0, 0]]]}
    dataset = fetch_era5(geometry, ["2m_temperature"], time=("2025-01-01T00:00", "2025-01-01T00:00"), retriever=FakeCDS())

    assert dataset.attrs["area"] == "geometry"
    assert dataset["2m_temperature"].shape == (1, 1, 1)
    assert float(dataset["2m_temperature"].values.item()) == 280.0


def test_border_straddling_geometry_keeps_every_country_and_separate_boxes():
    import geopandas as gpd

    boundaries = gpd.GeoDataFrame(
        {"country_iso": ["ITA", "FRA"], "geometry": [box(0, 0, 1, 1), box(0.5, 0.5, 2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    crossing = box(0.6, 0.6, 0.8, 0.8)
    distant = MultiPolygon([box(179.5, 0, 180, 0.5), box(-180, 0, -179.5, 0.5)])

    assert countries_intersecting(crossing, boundaries) == ["FRA", "ITA"]
    assert len(request_areas(distant, 0.25)) == 2


def test_iso3_without_a_boundary_table_returns_its_own_code(monkeypatch):
    def create(country_code=None, **kwargs):
        entity = SimpleNamespace(country_iso="ITA", name="Italy", name_en="Italy", boundary_id="ITA")
        return SimpleNamespace(entities=[entity], to_geoms=lambda: [AREA])

    monkeypatch.setattr(
        "gigaspatial.handlers.boundaries.AdminBoundaries.create",
        staticmethod(create),
    )
    assert countries_intersecting("ITA", None) == ["ITA"]


def test_bounding_box_then_exact_clip_leaves_interior_missing():
    retriever = FakeCDS()
    dataset = _fetch(fetch_era5, ["2m_temperature"], retriever)
    requested = retriever.calls[0][1]["area"]

    assert requested[0] >= 0.02 and requested[2] <= 0
    assert requested[1] <= 0 and requested[3] >= 0.02
    assert dataset["2m_temperature"].isnull().any() or dataset["2m_temperature"].shape == (1, 1, 1)
    window = xr.Dataset(
        {"2m_temperature": (("latitude", "longitude"), np.array([[1.0, 2.0, 3.0]]))},
        coords={"latitude": [0.0], "longitude": [0.0, 0.25, 0.5]},
    )
    clipped = clip(window, MultiPolygon([box(-0.02, -0.02, 0.02, 0.02), box(0.48, -0.02, 0.52, 0.02)]), 0.25)
    assert np.isnan(clipped["2m_temperature"].values[0, 1])
    assert clipped["2m_temperature"].values[0, 0] == 1.0
    assert clipped["2m_temperature"].values[0, 2] == 3.0


def test_temporal_filters_and_historical_coverage():
    from gigaspatial.fetchers.climate.shared import selection_hours

    month = selection_hours(ERA5, time="2025-01")
    one_hour = selection_hours(ERA5, year=2025, month=1, day=1, hour=0)
    assert len(month) == 31 * 24
    assert one_hour == [HOUR]
    with pytest.raises(ValueError, match="1940"):
        selection_hours(ERA5, year=1939, month=1, day=1, hour=0)
    with pytest.raises(ValueError, match="1950"):
        selection_hours(ERA5_LAND, year=1949)
    with pytest.raises(ValueError, match="not both"):
        selection_hours(ERA5, time="2025-01", hour=0)
    with pytest.raises(ValueError, match="full_history"):
        selection_hours(ERA5, full_history=True, year=2025)
    with pytest.raises(ClimateAvailabilityError):
        _fetch(fetch_era5, ["2m_temperature"], FakeCDS(datetime(2024, 12, 31, tzinfo=timezone.utc)))


def test_netcdf_and_grib_standardize_to_the_same_structure(monkeypatch):
    netcdf = _fetch(fetch_era5, ["2m_temperature", "total_precipitation"], data_format="netcdf")
    retriever = FakeCDS()

    def open_downloaded(path, data_format=None):
        return xr.load_dataset(path)

    monkeypatch.setattr("gigaspatial.fetchers.climate.shared.open_climate", open_downloaded)
    grib = _fetch(fetch_era5, ["2m_temperature", "total_precipitation"], retriever, data_format="grib")

    assert retriever.calls[0][1]["data_format"] == "grib"
    assert netcdf.attrs["data_format"] == "netcdf"
    assert grib.attrs["data_format"] == "grib"
    assert list(netcdf.data_vars) == list(grib.data_vars)
    assert netcdf["2m_temperature"].attrs["units"] == grib["2m_temperature"].attrs["units"] == "K"
    assert grib["total_precipitation"].attrs["units"] == "m"
    assert grib["total_precipitation"].attrs["accumulated"] is True
    np.testing.assert_allclose(netcdf["2m_temperature"].values, grib["2m_temperature"].values)


def test_open_climate_reads_netcdf_and_refuses_grib_without_cfgrib(tmp_path, monkeypatch):
    path = tmp_path / "sample.nc"
    xr.Dataset({"2m_temperature": (("latitude", "longitude"), np.ones((1, 1)))}).to_netcdf(path)
    opened = open_climate(path)
    assert "2m_temperature" in opened

    grib = tmp_path / "sample.grib"
    grib.write_bytes(b"GRIB" + b"\x00" * 16)

    def fail_open(*args, **kwargs):
        raise OSError("ecCodes is not installed")

    monkeypatch.setattr(xr, "open_dataset", fail_open)
    with pytest.raises(ImportError, match="cfgrib"):
        open_climate(grib, "grib")


def test_metadata_is_attached():
    dataset = _fetch(fetch_era5land, ["2m_temperature", "total_precipitation"])

    assert dataset.attrs["dataset_id"] == "reanalysis-era5-land"
    assert dataset.attrs["dataset_name"] == "ERA5-Land"
    assert dataset.attrs["source_url"].startswith("https://cds.climate.copernicus.eu/")
    assert dataset.attrs["variables"] == "2m_temperature,total_precipitation"
    assert "2m temperature" in dataset["2m_temperature"].attrs["long_name"]
    assert dataset.attrs["temporal_resolution"] == "1 hour"
    assert dataset.attrs["spatial_resolution"] == "0.1 degrees"
    assert dataset.attrs["start"].startswith("2025-01-01")
    assert dataset.attrs["retrieved_at"]
    assert dataset.attrs["latest_available"] == "2025-01-01T00:00:00Z"
    assert "Native CDS grid" in dataset.attrs["processing"]


def test_missing_recent_period_and_catalogue_end():
    published = datetime(2024, 11, 16, 23, tzinfo=timezone.utc)
    status = period_available(
        "reanalysis-era5-single-levels",
        "2024-11-16T00:00",
        "2024-11-17T23:00",
        latest=published,
    )
    assert status["available"] is False
    assert status["missing_hours"][0] == "2024-11-17T00:00:00Z"
    assert status["missing_hours"][-1] == "2024-11-17T23:00:00Z"

    def getter(url):
        dataset_id = url.rsplit("/", 1)[-1]
        end = "2026-11-10T00:00:00+00:00" if "land" in dataset_id else "2026-11-12T00:00:00+00:00"
        return {"extent": {"temporal": {"interval": [["1940-01-01T00:00:00+00:00", end]]}}}

    era5_end = latest_available("reanalysis-era5-single-levels", getter=getter)
    land_end = latest_available("reanalysis-era5-land", getter=getter)
    assert era5_end != land_end
    with pytest.raises(ClimateAvailabilityError, match="no temporal extent"):
        latest_available("reanalysis-era5-single-levels", getter=lambda url: {})


def test_merge_keeps_the_first_finite_value():
    first = xr.Dataset(
        {"total_precipitation": (("time",), np.array([0.1]))},
        coords={"time": [np.datetime64("2025-01-01T00")]},
    )
    second = xr.Dataset(
        {"total_precipitation": (("time",), np.array([5.0]))},
        coords={"time": [np.datetime64("2025-01-01T00")]},
    )
    merged = merge_climate([first, second])
    assert float(merged["total_precipitation"].values.item()) == 0.1


def test_memory_guard_rejects_a_large_request_before_download():
    retriever = FakeCDS()
    with pytest.raises(ClimateLimitError, match="raise max_bytes"):
        fetch_era5(
            box(-10, -10, 10, 10),
            "all",
            time="2020-01",
            retriever=retriever,
            max_bytes=1024,
            check_availability=False,
        )
    assert retriever.calls == []


def test_authentication_error(monkeypatch):
    monkeypatch.setattr(
        "gigaspatial.fetchers.climate.shared.resolve_credentials",
        lambda api_url=None, api_key=None: ("https://cds.climate.copernicus.eu/api", ""),
    )
    with pytest.raises(ClimateAuthError, match="CDS_API_KEY"):
        fetch_era5(AREA, ["2m_temperature"], time="2025-01-01")


def test_credentials_are_redacted(monkeypatch, tmp_path):
    secret = "hidden-cds-key"
    path = tmp_path / ".cdsapirc"
    path.write_text("url: https://cds.climate.copernicus.eu/api\nkey: hidden-cds-key\n", encoding="utf-8")
    monkeypatch.setattr("gigaspatial.fetchers.climate.shared.Path.home", lambda: tmp_path)
    monkeypatch.delenv("CDS_API_KEY", raising=False)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.setattr("gigaspatial.config.config.CDS_API_KEY", "", raising=False)
    url, key = resolve_credentials()
    assert url.startswith("https://cds.climate.copernicus.eu")
    assert key == secret
    assert secret not in redact(f"failed key={secret}", secret)


def test_retry_then_success(monkeypatch):
    sleeps = []
    monkeypatch.setattr("gigaspatial.fetchers.climate.shared.time.sleep", lambda seconds: sleeps.append(seconds))
    retriever = FakeCDS(fail_times=2)
    dataset = _fetch(fetch_era5, ["2m_temperature"], retriever, max_retries=3)
    assert dataset["2m_temperature"].shape[0] == 1
    assert len(retriever.calls) == 3
    assert sleeps == [1, 2]


def test_interrupted_download_is_removed_before_the_next_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr("gigaspatial.fetchers.climate.shared.time.sleep", lambda seconds: None)
    seen = []

    class PartialThenOk(FakeCDS):
        def retrieve(self, dataset_id, request, target):
            seen.append(target.exists() and target.stat().st_size)
            if len(seen) == 1:
                target.write_bytes(b"partial-download")
                raise ConnectionError("connection reset")
            super().retrieve(dataset_id, request, target)

    dataset = _fetch(fetch_era5, ["2m_temperature"], PartialThenOk(), max_retries=2)
    assert float(dataset["2m_temperature"].values.item()) == 280.0
    assert seen[1] == 0


def test_unavailable_cds_and_rejected_variable(monkeypatch):
    monkeypatch.setattr("gigaspatial.fetchers.climate.shared.time.sleep", lambda seconds: None)

    class Down:
        def latest_available(self, dataset_id):
            return HOUR

        def retrieve(self, dataset_id, request, target):
            raise ConnectionError("503 upstream unavailable")

    with pytest.raises(ClimateRequestError, match="2m_temperature"):
        _fetch(fetch_era5, ["2m_temperature"], Down(), max_retries=2)
    with pytest.raises(ClimateRequestError, match="mean_wave_period"):
        _fetch(fetch_era5, ["mean_wave_period"], FakeCDS(reject_names={"mean_wave_period"}))


def test_incompatible_batch_is_split_instead_of_dropping_a_variable():
    retriever = FakeCDS(reject_batches=True)
    dataset = _fetch(fetch_era5, ["2m_temperature", "total_precipitation"], retriever)
    assert set(dataset.data_vars) == {"2m_temperature", "total_precipitation"}
    assert len(retriever.calls) == 3


def test_standardize_wraps_longitudes_without_resampling():
    dataset = xr.Dataset(
        {"2m_temperature": (("latitude", "longitude"), np.array([[1.0, 2.0]]))},
        coords={"latitude": [0.0], "longitude": [350.0, 10.0]},
    )
    dataset["2m_temperature"].attrs["provider"] = "kept"
    standardized = standardize(dataset, ERA5, area_label="geometry", data_format="netcdf", bounds=(-10, 0, 10, 0))
    assert list(standardized["longitude"].values) == [-10.0, 10.0]
    assert standardized["2m_temperature"].attrs["provider"] == "kept"
    assert standardized["2m_temperature"].attrs["units"] == "K"
