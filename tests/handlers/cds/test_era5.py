"""Offline tests for the ERA5 and ERA5-Land handlers."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, box

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.cds import era5 as era5_module
from gigaspatial.handlers.cds.era5 import (
    CDSIntegrityError,
    CDSRequestError,
    CDSResolveOptions,
    ERA5Handler,
    _dask_available,
    build_cds_request,
    clip_cube,
    dataset_chunks,
    duplicate_timestamps,
    fetch_latest_available,
    grid_for_area,
    infer_units,
    intersecting_mask,
    latest_from_collection,
    load_variables,
    merge_cubes,
    merge_records,
    metadata_of,
    missing_timestamps,
    open_downloaded,
    plan_hours,
    plan_slices,
    read_cube,
    redact,
    request_areas,
    resolve_credentials,
    retrieve_with_retry,
    select_variables,
    selection_hours,
    slice_hours,
    slices_from_hours,
    synthetic_cube,
    variables_from_form,
    wrap_longitudes,
    write_cube,
)
from gigaspatial.handlers.cds.era5land import ERA5LandHandler

UTC = timezone.utc
COUNTRY = box(0, 0, 0.02, 0.02)


class FakeCDS:
    """In-memory CDS client. Writes a native grid NetCDF for each request."""

    def __init__(self, latest, resolution=0.25):
        self.latest = latest
        self.resolution = resolution
        self.calls = []
        self.latest_calls = []
        self.remaining_failures = 0
        self.error = None
        self.drop_last_hour = False
        self.fail_days = set()

    def latest_available(self, dataset_id):
        self.latest_calls.append(dataset_id)
        if isinstance(self.latest, dict):
            return self.latest[dataset_id]
        return self.latest

    def retrieve(self, dataset_id, request, target):
        self.calls.append({"dataset": dataset_id, "request": json.loads(json.dumps(request))})
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise TimeoutError("timed out talking to CDS")
        if self.error is not None:
            raise self.error
        if set(request["day"]) & self.fail_days:
            target.write_bytes(b"partial")
            raise TimeoutError("connection reset while CDS was writing the file")
        from gigaspatial.handlers.cds.era5 import TimeSlice

        time_slice = TimeSlice(
            request["year"][0],
            request["month"][0],
            tuple(request["day"]),
            tuple(request["time"]),
        )
        hours = slice_hours(time_slice)
        if self.drop_last_hour and len(hours) > 1:
            hours = hours[:-1]
        cube = grid_for_area(
            request["area"],
            self.resolution,
            hours,
            request["variable"][0],
            fill=5.0,
        )
        write_cube(target, cube)


def _boundary(country_code):
    code = {"Italy": "ITA", "ITA": "ITA", "France": "FRA", "FRA": "FRA"}.get(
        country_code, country_code
    )
    entity = SimpleNamespace(
        country_iso=code,
        name=code,
        name_en={"ITA": "Italy", "FRA": "France"}.get(code, code),
        boundary_id=code,
    )
    return SimpleNamespace(entities=[entity], to_geoms=lambda: [COUNTRY])


@pytest.fixture
def boundaries(monkeypatch):
    from gigaspatial.handlers import boundaries as boundaries_module

    calls = []

    def create(country_code=None, **kwargs):
        calls.append(country_code)
        return _boundary(country_code)

    monkeypatch.setattr(boundaries_module.AdminBoundaries, "create", staticmethod(create))
    return calls


def _handler(cls, tmp_path, retriever):
    return cls(
        data_store=LocalDataStore(),
        base_path=tmp_path / cls.name,
        retriever=retriever,
        n_workers=1,
        api_key="test-key",
    )


def _options(**kwargs):
    return CDSResolveOptions(**kwargs)


def _hour(year, month, day, hour):
    return datetime(year, month, day, hour, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def test_bundled_catalogues_are_dataset_specific_and_complete():
    era5 = load_variables("reanalysis-era5-single-levels")
    land = load_variables("reanalysis-era5-land")

    assert len(era5) == 262
    assert len(land) == 60
    assert {item.id for item in era5}.isdisjoint({"snow_cover", "total_evaporation"})
    assert "mean_wave_direction" in {item.id for item in era5}
    assert "mean_wave_direction" not in {item.id for item in land}
    assert "snow_cover" in {item.id for item in land}
    temperature = next(item for item in era5 if item.id == "2m_temperature")
    rain = next(item for item in era5 if item.id == "total_precipitation")
    assert temperature.units == "K"
    assert temperature.accumulated is False
    assert rain.units == "m"
    assert rain.accumulated is True
    assert rain.description == "Total precipitation"


def test_variable_selection_one_many_and_all():
    catalogue = load_variables("reanalysis-era5-land")

    one = select_variables(catalogue, ("2m_temperature",))
    many = select_variables(catalogue, ("total_precipitation", "2m_temperature", "2m_temperature"))
    everything = select_variables(catalogue, ("__all__",))

    assert [item.id for item in one] == ["2m_temperature"]
    assert [item.id for item in many] == ["total_precipitation", "2m_temperature"]
    assert len(everything) == len(catalogue)
    with pytest.raises(ValueError, match="unknown CDS variable"):
        select_variables(catalogue, ("not_a_real_variable",))
    with pytest.raises(ValueError, match="not selected"):
        select_variables(catalogue, ())


def test_units_are_not_shared_when_the_name_is_not_in_the_ecmwf_table():
    units, accumulated = infer_units("vertical_integral_of_temperature")
    assert units is None
    assert accumulated is False


def test_discovery_adds_a_variable_without_changing_the_handler(monkeypatch):
    bundled = [{"id": "2m_temperature", "description": "2m temperature", "groups": ["Temperature"]}]
    discovered = bundled + [
        {"id": "new_land_field", "description": "New land field", "groups": ["Temperature"]}
    ]

    merged = merge_records(bundled, discovered)
    variables = [era5_module._variable(item) for item in merged]

    assert [item.id for item in variables] == ["2m_temperature", "new_land_field"]
    assert variables[1].description == "New land field"

    def getter(url):
        if url.endswith("/form.json"):
            return [
                {
                    "name": "variable",
                    "details": {
                        "groups": [
                            {
                                "label": "Temperature",
                                "values": ["2m_temperature", "brand_new"],
                                "labels": {
                                    "2m_temperature": "2m temperature",
                                    "brand_new": "Brand new",
                                },
                            }
                        ]
                    },
                }
            ]
        return {"links": [{"rel": "form", "href": url + "/form.json"}]}

    live = era5_module.discover_records("reanalysis-era5-land", "https://cds.example/api", getter)
    assert {item["id"] for item in live} == {"2m_temperature", "brand_new"}
    offline = load_variables("reanalysis-era5-land", refresh=False, getter=getter)
    assert "brand_new" not in {item.id for item in offline}


def test_form_parser_reads_cds_labels():
    records = variables_from_form(
        [
            {
                "name": "variable",
                "details": {
                    "groups": [
                        {
                            "label": "Popular",
                            "values": ["2m_temperature"],
                            "labels": {"2m_temperature": "2m temperature"},
                        }
                    ]
                },
            }
        ]
    )
    assert records == [
        {"id": "2m_temperature", "description": "2m temperature", "groups": ["Popular"]}
    ]


# ---------------------------------------------------------------------------
# Time, request, availability
# ---------------------------------------------------------------------------


def test_temporal_filters_and_full_history_and_dataset_starts():
    one_day = selection_hours(
        _options(variables="2m_temperature", year=2024, month=11, day=15, hour=0),
        start_year=1940,
        latest=None,
    )
    assert one_day == [_hour(2024, 11, 15, 0)]

    leap = selection_hours(
        _options(variables="2m_temperature", year=2024, month=2),
        start_year=1940,
        latest=None,
    )
    assert leap[-1] == _hour(2024, 2, 29, 23)
    with pytest.raises(ValueError, match="does not include any hour"):
        selection_hours(
            _options(variables="2m_temperature", year=2023, month=2, day=29),
            start_year=1940,
            latest=None,
        )

    window = selection_hours(
        _options(variables="2m_temperature", start="2024-11-15", end="2024-11-16", hour=12),
        start_year=1940,
        latest=None,
    )
    assert window == [_hour(2024, 11, 15, 12), _hour(2024, 11, 16, 12)]

    history = selection_hours(
        _options(variables="all", full_history=True),
        start_year=1940,
        latest=_hour(1940, 1, 1, 2),
    )
    assert history[0] == _hour(1940, 1, 1, 0)
    assert history[-1] == _hour(1940, 1, 1, 2)
    assert len(history) == 3

    with pytest.raises(ValueError, match="full_history cannot be combined"):
        selection_hours(
            _options(variables="2m_temperature", full_history=True, year=2020),
            start_year=1940,
            latest=_hour(2020, 1, 1, 0),
        )
    with pytest.raises(ValueError, match="not downloaded unless"):
        selection_hours(_options(variables="2m_temperature"), start_year=1940, latest=None)
    with pytest.raises(ValueError, match="starts in 1950"):
        selection_hours(
            _options(variables="2m_temperature", year=1949),
            start_year=1950,
            latest=None,
        )


def test_slices_are_a_cartesian_product_and_do_not_invent_hours():
    hours = [
        _hour(2024, 11, 15, 0),
        _hour(2024, 11, 15, 1),
        _hour(2024, 11, 16, 0),
    ]
    slices = slices_from_hours(hours)
    expanded = [hour for time_slice in slices for hour in slice_hours(time_slice)]
    assert sorted(expanded) == sorted(hours)
    assert len(slices) == 2


def test_request_uses_the_current_cds_form_and_not_a_resampled_grid():
    unit = {
        "variable": "2m_temperature",
        "year": "2024",
        "month": "11",
        "days": ("15",),
        "times": ("00:00",),
        "data_format": "netcdf",
        "product_type": "reanalysis",
    }
    era5 = build_cds_request(unit, (0.25, 0.0, 0.0, 0.25), include_product_type=True)
    land = build_cds_request(unit, (0.1, 0.0, 0.0, 0.1), include_product_type=False)

    assert era5["product_type"] == ["reanalysis"]
    assert "product_type" not in land
    assert era5["area"] == [0.25, 0.0, 0.0, 0.25]
    assert era5["data_format"] == "netcdf"
    assert era5["download_format"] == "unarchived"
    assert "grid" not in era5
    assert "format" not in era5
    assert land["variable"] == ["2m_temperature"]


def test_catalogue_end_is_read_per_dataset_and_missing_end_is_an_error():
    era5 = latest_from_collection(
        {"extent": {"temporal": {"interval": [["1940-01-01T00:00:00+00:00", "2024-11-16T00:00:00+00:00"]]}}},
        "reanalysis-era5-single-levels",
    )
    land = latest_from_collection(
        {"extent": {"temporal": {"interval": [["1950-01-01T00:00:00+00:00", "2024-11-12T00:00:00+00:00"]]}}},
        "reanalysis-era5-land",
    )

    assert era5 == _hour(2024, 11, 16, 0)
    assert land == _hour(2024, 11, 12, 0)
    assert era5 != land
    with pytest.raises(Exception, match="does not publish an end"):
        latest_from_collection(
            {"extent": {"temporal": {"interval": [["1950-01-01T00:00:00+00:00", None]]}}},
            "reanalysis-era5-land",
        )


def test_fetch_latest_uses_the_dataset_collection(monkeypatch):
    seen = []

    def getter(url):
        seen.append(url)
        dataset_id = url.rstrip("/").split("/")[-1]
        end = "2024-11-16T00:00:00+00:00" if dataset_id.endswith("single-levels") else "2024-11-10T00:00:00+00:00"
        return {"extent": {"temporal": {"interval": [["1940-01-01T00:00:00+00:00", end]]}}}

    monkeypatch.setattr(
        "gigaspatial.handlers.cds.era5._get_json",
        getter,
    )
    era5 = fetch_latest_available("reanalysis-era5-single-levels", "https://cds.example/api", getter)
    land = fetch_latest_available("reanalysis-era5-land", "https://cds.example/api", getter)

    assert era5 == _hour(2024, 11, 16, 0)
    assert land == _hour(2024, 11, 10, 0)
    assert seen[0].endswith("reanalysis-era5-single-levels")
    assert seen[1].endswith("reanalysis-era5-land")


def test_credentials_and_redaction(tmp_path, monkeypatch):
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.delenv("CDS_API_KEY", raising=False)
    cdsapirc = tmp_path / ".cdsapirc"
    cdsapirc.write_text("url: https://cds.example/api\nkey: file-secret\n", encoding="utf-8")

    url, key = resolve_credentials(cdsapirc=cdsapirc)
    explicit_url, explicit_key = resolve_credentials(
        "https://override.example/api", "explicit-secret", cdsapirc=cdsapirc
    )

    assert (url, key) == ("https://cds.example/api", "file-secret")
    assert explicit_key == "explicit-secret"
    assert explicit_url == "https://override.example/api"
    assert "file-secret" not in redact("failed key=file-secret", "file-secret")
    assert redact("token=abc") == "token=***"


def test_retry_then_succeed_and_do_not_retry_a_client_error(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("gigaspatial.handlers.cds.era5.time.sleep", lambda seconds: sleeps.append(seconds))
    target = tmp_path / "download.nc"
    state = {"tries": 0}

    def flaky(dataset_id, request, path):
        state["tries"] += 1
        if state["tries"] < 3:
            path.write_bytes(b"partial")
            raise TimeoutError("timed out")
        path.write_bytes(b"ok")

    retrieve_with_retry(SimpleNamespace(retrieve=flaky), "reanalysis-era5-single-levels", {}, target, max_retries=3)
    assert state["tries"] == 3
    assert target.read_bytes() == b"ok"
    assert sleeps == [1, 2]

    def rejected(dataset_id, request, path):
        state["tries"] += 1
        raise CDSRequestError("400 invalid request", retryable=False)

    state["tries"] = 0
    with pytest.raises(CDSRequestError, match="invalid request"):
        retrieve_with_retry(
            SimpleNamespace(retrieve=rejected),
            "reanalysis-era5-single-levels",
            {},
            tmp_path / "nope.nc",
            max_retries=3,
        )
    assert state["tries"] == 1


# ---------------------------------------------------------------------------
# Geometry, clip, merge, files
# ---------------------------------------------------------------------------


def test_country_parts_across_the_antimeridian_stay_separate_boxes():
    geometry = MultiPolygon([box(170, -1, 175, 1), box(-175, -1, -170, 1)])
    areas = request_areas(geometry, 0.25)

    assert len(areas) == 2
    for north, west, south, east in areas:
        assert north == 1.0 or north == 1.25 or east > west
        assert east - west < 20


def test_clip_keeps_native_spacing_and_does_not_use_the_bbox_as_the_result():
    longitudes = np.array([0.0, 0.25, 0.5])
    latitudes = np.array([0.0])
    values = np.array([[[1.0, 0.0, 3.0]]])
    cube = synthetic_cube(
        [_hour(2024, 11, 15, 0)],
        latitudes,
        longitudes,
        "total_precipitation",
        fill=0,
    )
    cube.data["total_precipitation"] = values
    geometry = MultiPolygon([box(-0.05, -0.05, 0.05, 0.05), box(0.45, -0.05, 0.55, 0.05)])

    clipped = clip_cube(cube, geometry, 0.25)

    assert list(clipped.longitudes) == [0.0, 0.25, 0.5]
    assert np.diff(clipped.longitudes).tolist() == [0.25, 0.25]
    row = clipped.data["total_precipitation"][0, 0]
    assert row[0] == 1.0
    assert np.isnan(row[1])
    assert row[2] == 3.0


def test_longitudes_above_180_are_wrapped_without_changing_values():
    wrapped, arrays = wrap_longitudes(
        np.array([10.0, 350.0]),
        [np.array([[[1.0, 2.0]]])],
    )
    assert wrapped.tolist() == [-10.0, 10.0]
    assert arrays[0][0, 0].tolist() == [2.0, 1.0]


def test_merge_does_not_sum_accumulated_values_or_duplicate_timestamps():
    first = synthetic_cube([_hour(2024, 11, 15, 0)], [0.0], [0.0], "total_precipitation", fill=5)
    second = synthetic_cube([_hour(2024, 11, 15, 0)], [0.0], [0.0], "total_precipitation", fill=9)
    later = synthetic_cube([_hour(2024, 11, 15, 1)], [0.0], [0.0], "total_precipitation", fill=1)
    merged = merge_cubes([first, second, later])

    assert merged.data["total_precipitation"][0, 0, 0] == 5
    assert merged.data["total_precipitation"][1, 0, 0] == 1
    assert duplicate_timestamps(merged.times) == []
    assert missing_timestamps(
        [_hour(2024, 11, 15, 0), _hour(2024, 11, 15, 1), _hour(2024, 11, 15, 2)],
        merged.times,
    ) == [_hour(2024, 11, 15, 2)]


def test_netcdf_round_trip_and_zip_and_grib_dispatch(tmp_path, monkeypatch):
    cube = synthetic_cube([_hour(2024, 11, 15, 0), _hour(2024, 11, 15, 1)], [0.0, 0.25], [10.0], "2m_temperature", fill=280)
    cube.variable_attrs["2m_temperature"] = {"units": "K", "description": "2m temperature"}
    cube.attrs["dataset_id"] = "reanalysis-era5-single-levels"
    path = tmp_path / "sample.nc"
    write_cube(path, cube)
    loaded = read_cube(path)

    assert loaded.data["2m_temperature"].shape == (2, 2, 1)
    assert loaded.variable_attrs["2m_temperature"]["units"] == "K"
    assert loaded.attrs["dataset_id"] == "reanalysis-era5-single-levels"
    np.testing.assert_allclose(loaded.latitudes, [0.0, 0.25])

    archive = tmp_path / "download.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(path, "sample.nc")
    unzipped = open_downloaded(archive)
    assert unzipped.data["2m_temperature"].shape == (2, 2, 1)

    grib = tmp_path / "sample.grib"
    grib.write_bytes(b"GRIB-not-really")
    seen = []

    def fake_grib(path):
        seen.append(Path_name(path))
        return cube

    monkeypatch.setattr("gigaspatial.handlers.cds.era5.open_grib", fake_grib)
    assert open_downloaded(grib) is cube
    assert seen == [str(grib)]


def Path_name(path):
    return str(path)


def test_large_native_grid_is_masked_without_changing_spacing():
    longitudes = np.arange(200) * 0.25
    latitudes = np.arange(200) * 0.25
    mask = intersecting_mask(longitudes, latitudes, 0.25, box(1, 1, 1.2, 1.2))
    kept = longitudes[mask.any(axis=0)]

    assert mask.shape == (200, 200)
    assert kept.size < longitudes.size
    if kept.size > 1:
        assert np.allclose(np.diff(kept), 0.25)
    cube = synthetic_cube(
        [_hour(2024, 1, day, hour) for day in (1, 2) for hour in range(24)],
        latitudes[:40],
        longitudes[:40],
        "2m_temperature",
        fill=1,
    )
    assert cube.data["2m_temperature"].shape == (48, 40, 40)
    assert dataset_chunks() == ({"time": 24} if _dask_available() else None)


def test_dask_chunks_follow_whether_dask_imports(monkeypatch):
    monkeypatch.setattr("gigaspatial.handlers.cds.era5._dask_available", lambda: True)
    assert dataset_chunks() == {"time": 24}
    monkeypatch.setattr("gigaspatial.handlers.cds.era5._dask_available", lambda: False)
    assert dataset_chunks() is None


def test_plan_downloads_only_the_new_day_and_is_idempotent():
    record = {
        "time_invariant": False,
        "last_downloaded_timestamp": "2024-11-15T23:00:00Z",
        "missing_periods": [],
        "pending_periods": [],
    }
    latest = _hour(2024, 11, 16, 23)
    hours = plan_hours(record, latest)
    assert hours[0] == _hour(2024, 11, 16, 0)
    assert hours[-1] == _hour(2024, 11, 16, 23)
    assert len(hours) == 24

    record["last_downloaded_timestamp"] = "2024-11-16T23:00:00Z"
    assert plan_hours(record, latest) == []

    partial = dict(record)
    partial["last_downloaded_timestamp"] = "2024-11-15T23:00:00Z"
    midnight = plan_slices(partial, _hour(2024, 11, 16, 0))
    assert slice_hours(midnight[0]) == [_hour(2024, 11, 16, 0)]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def test_handlers_are_registered_under_their_names():
    import gigaspatial
    from gigaspatial.handlers import ERA5Config, ERA5LandConfig

    assert gigaspatial.ERA5Handler is ERA5Handler
    assert gigaspatial.ERA5LandHandler is ERA5LandHandler
    assert ERA5Handler.name == "cds-era5"
    assert ERA5LandHandler.name == "cds-era5land"
    assert ERA5Config.DATASET_ID == "reanalysis-era5-single-levels"
    assert ERA5LandConfig.DATASET_ID == "reanalysis-era5-land"
    assert ERA5Config.START_YEAR == 1940
    assert ERA5LandConfig.START_YEAR == 1950
    assert ERA5Config.RESOLUTION == 0.25
    assert ERA5LandConfig.RESOLUTION == 0.1


def test_country_resolution_uses_admin_boundaries_and_not_a_private_matcher(boundaries, tmp_path):
    handler = _handler(ERA5Handler, tmp_path, FakeCDS(_hour(2024, 11, 16, 0)))
    key = handler.config.get_search_key("Italy", _options(variables="2m_temperature", year=2024, month=11, day=15))

    assert boundaries == ["Italy"]
    assert key.country_id == "ITA"
    assert key.country_name == "Italy"
    assert key.boundary_id == "ITA"
    geometry = handler.config.data_store.read_file(key.geometry_path, encoding="utf-8")
    assert "POLYGON" in geometry
    north, west, south, east = key.areas[0]
    assert west <= 0 <= east
    assert south <= 0 <= north
    assert (north - south) < 1


def test_a_geometry_source_does_not_resolve_a_country(boundaries, tmp_path):
    handler = _handler(ERA5Handler, tmp_path, FakeCDS(_hour(2024, 11, 16, 0)))
    key = handler.config.get_search_key(COUNTRY, _options(variables="2m_temperature", year=2024, month=11, day=15))

    assert boundaries == []
    assert key.country_id.startswith("geom_")
    assert key.country_name is None


def test_download_clips_to_the_country_and_records_metadata(boundaries, tmp_path):
    fake = FakeCDS(_hour(2024, 11, 16, 23))
    handler = _handler(ERA5Handler, tmp_path, fake)
    data = handler.load_data(
        "Italy",
        variables=("2m_temperature", "total_precipitation"),
        start="2024-11-15",
        end="2024-11-15",
        hour=0,
        show_progress=False,
    )
    meta = metadata_of(data)

    assert len(fake.calls) == 2
    assert {call["request"]["variable"][0] for call in fake.calls} == {
        "2m_temperature",
        "total_precipitation",
    }
    assert "grid" not in fake.calls[0]["request"]
    assert fake.calls[0]["request"]["product_type"] == ["reanalysis"]
    assert meta["dataset_id"] == "reanalysis-era5-single-levels"
    assert meta["handler_name"] == "cds-era5"
    assert meta["country_id"] == "ITA"
    assert meta["country_name"] == "Italy"
    assert meta["boundary_id"] == "ITA"
    assert meta["units"]["2m_temperature"] == "K"
    assert meta["units"]["total_precipitation"] == "m"
    assert meta["variable_descriptions"]["2m_temperature"] == "2m temperature"
    assert meta["temporal_resolution"] == "1 hour"
    assert meta["spatial_resolution"] == "0.25 degrees"
    assert meta["source_dataset_url"].endswith("reanalysis-era5-single-levels?tab=overview")
    assert meta["start_date"] == "2024-11-15T00:00:00Z"
    assert meta["end_date"] == "2024-11-15T00:00:00Z"
    assert meta["crs"] == "EPSG:4326"
    assert "Native CDS grid" in meta["processing"]
    assert meta["download_timestamp"]
    values = _values(data, "2m_temperature")
    assert np.isfinite(values).any()
    assert 0.25 not in np.round(_coord(data, "longitude"), 6)

    status = json.loads((tmp_path / "cds-era5" / "ITA" / "_status.json").read_text(encoding="utf-8"))
    assert status["variables"]["2m_temperature"]["last_downloaded_timestamp"] == "2024-11-15T00:00:00Z"
    assert status["variables"]["total_precipitation"]["accumulated"] is True
    assert status["variables"]["2m_temperature"]["accumulated"] is False

    calls_before = len(fake.calls)
    handler.load_data(
        "Italy",
        variables="2m_temperature",
        start="2024-11-15",
        end="2024-11-15",
        hour=0,
        show_progress=False,
    )
    assert len(fake.calls) == calls_before


def test_incremental_update_is_per_dataset_and_idempotent(boundaries, tmp_path):
    era5_latest = _hour(2024, 11, 16, 23)
    land_latest = _hour(2024, 11, 12, 0)
    era5_fake = FakeCDS({"reanalysis-era5-single-levels": era5_latest})
    land_fake = FakeCDS(
        {"reanalysis-era5-land": land_latest},
        resolution=0.1,
    )
    era5 = _handler(ERA5Handler, tmp_path, era5_fake)
    land = _handler(ERA5LandHandler, tmp_path, land_fake)
    kwargs = dict(
        variables="2m_temperature",
        start="2024-11-15",
        end="2024-11-15",
        show_progress=False,
    )
    era5.load_data("Italy", **kwargs)
    land.load_data("Italy", **kwargs)
    era5_before = len(era5_fake.calls)
    land_before = len(land_fake.calls)

    era5_report = era5.update("Italy", force=True)
    land_report = land.update("ITA", force=True)

    assert len(era5_fake.calls) == era5_before + 1
    new_request = era5_fake.calls[-1]["request"]
    assert new_request["day"] == ["16"]
    assert new_request["time"][0] == "00:00"
    assert new_request["time"][-1] == "23:00"
    assert "15" not in new_request["day"]
    assert era5_fake.latest_calls == ["reanalysis-era5-single-levels"]
    assert land_fake.calls == land_fake.calls[:land_before]
    assert len(land_fake.calls) == land_before
    assert land_fake.latest_calls == ["reanalysis-era5-land"]
    assert land_report["dataset_id"] == "reanalysis-era5-land"
    assert era5_report["countries"][0]["variables"]["2m_temperature"]["last_downloaded_timestamp"] == "2024-11-16T23:00:00Z"
    assert land_report["countries"][0]["variables"]["2m_temperature"]["status"] == "current"
    assert land_report["countries"][0]["latest_available_cds_timestamp"] == "2024-11-12T00:00:00Z"

    calls_after = len(era5_fake.calls)
    again = era5.update("Italy", force=True)
    assert len(era5_fake.calls) == calls_after
    assert again["countries"][0]["downloaded"] == []
    loaded = era5.load_data("Italy", variables="2m_temperature", start="2024-11-15", end="2024-11-16", show_progress=False)
    times = _times(loaded)
    assert len(times) == len(set(times))
    assert len(times) == 48


def test_failed_update_keeps_the_stored_day_and_records_the_gap(boundaries, tmp_path):
    fake = FakeCDS(_hour(2024, 11, 16, 23))
    handler = _handler(ERA5Handler, tmp_path, fake)
    handler.load_data(
        "Italy",
        variables="total_precipitation",
        start="2024-11-15",
        end="2024-11-15",
        show_progress=False,
    )
    stored = list((tmp_path / "cds-era5" / "ITA").rglob("*.nc"))
    assert stored
    original = stored[0].read_bytes()
    fake.fail_days.add("16")

    report = handler.update(force=True)
    reloaded = stored[0].read_bytes()
    status = json.loads((tmp_path / "cds-era5" / "ITA" / "_status.json").read_text(encoding="utf-8"))
    record = status["variables"]["total_precipitation"]

    assert reloaded == original
    assert record["last_downloaded_timestamp"] == "2024-11-15T23:00:00Z"
    assert record["missing_periods"]
    assert record["retries"] >= 1
    assert record["error"]
    assert report["countries"][0]["status"] == "error"
    assert not list((tmp_path / "cds-era5" / "ITA").rglob("*16*.ok"))


def test_incomplete_hour_is_not_stored(boundaries, tmp_path):
    fake = FakeCDS(_hour(2024, 11, 15, 23), )
    fake.drop_last_hour = True
    handler = _handler(ERA5Handler, tmp_path, fake)

    with pytest.raises(CDSIntegrityError, match="missing"):
        handler.downloader.download_data_unit(
            _one_unit(handler),
            handler.downloader.options_type(show_progress=False),
        )
    assert list((tmp_path / "cds-era5").rglob("*.ok")) == []


def test_all_variables_resolve_without_downloading(boundaries, tmp_path):
    handler = _handler(ERA5LandHandler, tmp_path, FakeCDS(_hour(2024, 1, 1, 0), resolution=0.1))
    units = handler.config.get_relevant_data_units(
        "Italy", variables="all", year=2024, month=1, day=1, hour=0
    )
    names = [unit["variable"] for unit in units]
    assert len(names) == 60
    assert len(set(names)) == 60
    assert "product_type" not in build_cds_request(units[0], units[0]["areas"][0], include_product_type=False)


def test_unknown_config_field_and_secret_is_not_in_repr(tmp_path):
    with pytest.raises(TypeError):
        ERA5Handler(base_pth=tmp_path)
    config = _handler(ERA5Handler, tmp_path, FakeCDS(_hour(2024, 1, 1, 0))).config
    config.api_key = "super-secret-key"
    assert "super-secret-key" not in repr(config)


def test_update_frequency_skips_a_recent_check(boundaries, tmp_path):
    fake = FakeCDS(_hour(2024, 11, 16, 23))
    handler = _handler(ERA5Handler, tmp_path, fake)
    handler.load_data(
        "Italy",
        variables="2m_temperature",
        start="2024-11-15",
        end="2024-11-15",
        hour=0,
        show_progress=False,
    )
    handler.config.update_frequency_hours = 24
    first = handler.update("Italy", force=True)
    checks = len(fake.latest_calls)
    second = handler.update("Italy")

    assert first["countries"][0]["status"] == "updated"
    assert second["countries"][0]["status"] == "skipped"
    assert len(fake.latest_calls) == checks


def test_command_line_updates_each_handler(monkeypatch):
    from gigaspatial.handlers.cds import era5 as cli

    seen = []

    class _Handler:
        def __init__(self):
            self.name = "stub"

        def update(self, source=None, force=False, variables=None):
            seen.append((source, force))
            return {"handler_name": "stub", "dataset_id": "x", "countries": [{"status": "updated"}]}

    monkeypatch.setitem(cli.HANDLERS, "cds-era5", _Handler)
    monkeypatch.setitem(cli.HANDLERS, "cds-era5land", _Handler)
    assert cli.main(["update", "--handler", "cds-era5", "--country", "ITA", "--force"]) == 0
    assert seen == [("ITA", True)]


def _one_unit(handler):
    return handler.config.get_relevant_data_units(
        "Italy",
        variables="2m_temperature",
        start="2024-11-15",
        end="2024-11-15",
    )[0]


def _values(data, name):
    if hasattr(data, "data_vars"):
        return np.asarray(data[name].values)
    return data.data[name]


def _coord(data, name):
    if hasattr(data, "coords"):
        return np.asarray(data[name].values)
    return data.longitudes if name == "longitude" else data.latitudes


def _times(data):
    if hasattr(data, "coords"):
        return [str(value) for value in data["time"].values]
    return [str(value) for value in data.times]
