"""ECMWF open-data catalogue, downloads, and wind fields."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gigaspatial.fetchers.ecmwf.catalogue import (
    field_urls,
    forecast_steps,
    gust_param,
    index_url,
    merge_ranges,
    parse_index,
    select_messages,
    stream_for,
    track_url,
    variable_names,
)
from gigaspatial.fetchers.ecmwf.open_data import EcmwfOpenDataFetcher
from gigaspatial.fetchers.ecmwf.wind import (
    add_wind_speed,
    pair_storm_fields,
    track_boxes,
)


RUN = date(2026, 11, 26)
ROOT = "https://data.ecmwf.int/forecasts"


def test_forecast_horizons_match_the_open_data_schedule():
    ifs_main = forecast_steps("ifs", 0)
    assert ifs_main[0] == 0 and ifs_main[-1] == 360
    assert 144 in ifs_main and 147 not in ifs_main and 150 in ifs_main
    assert forecast_steps("ifs", 12) == ifs_main
    assert forecast_steps("ifs", 6)[-1] == 90
    assert forecast_steps("ifs", 18) == forecast_steps("ifs", 6)

    ens_main = forecast_steps("ifs-ens", 12)
    assert ens_main == ifs_main
    assert forecast_steps("ifs-ens", 6)[-1] == 144
    assert 90 in forecast_steps("ifs-ens", 18)

    for hour in (0, 6, 12, 18):
        aifs = forecast_steps("aifs-single", hour)
        assert aifs[0] == 0 and aifs[-1] == 360 and 3 not in aifs
        assert forecast_steps("aifs-ens", hour) == aifs


def test_track_file_step_follows_each_system():
    assert track_url(RUN, 0, "ifs").endswith("20261126000000-360h-oper-tf.bufr")
    assert track_url(RUN, 18, "ifs-ens").endswith(
        "20261126180000-144h-enfo-tf.bufr"
    )
    assert track_url(RUN, 6, "aifs-single").endswith(
        "20261126060000-360h-oper-tf.bufr"
    )
    assert track_url(RUN, 12, "aifs-ens").endswith(
        "20261126120000-360h-enfo-tf.bufr"
    )


def test_field_urls_cover_resolution_and_ensemble_files():
    (deterministic,) = field_urls(RUN, 0, "ifs", "0p25", 24)
    assert deterministic == (
        f"{ROOT}/20261126/00z/ifs/0p25/oper/20261126000000-24h-oper-fc.grib2"
    )
    surface, pressure = field_urls(RUN, 0, "ifs", "0.1", 24)
    assert surface.endswith("-oper-fc-sfc.grib2")
    assert pressure.endswith("-oper-fc-pl.grib2")

    (ensemble,) = field_urls(RUN, 0, "ifs-ens", "0p25", 24)
    assert ensemble.endswith("/enfo/20261126000000-24h-enfo-ef.grib2")

    control, perturbed = field_urls(RUN, 0, "aifs-ens", "0p25", 24)
    assert control.endswith("aifs-ens/0p25/enfo/20261126000000-24h-enfo-cf.grib2")
    assert perturbed.endswith("aifs-ens/0p25/enfo/20261126000000-24h-enfo-pf.grib2")


def test_early_06_utc_deterministic_runs_use_the_scda_stream():
    assert stream_for("ifs", date(2026, 5, 11), 6) == "scda"
    url = field_urls(date(2026, 5, 11), 6, "ifs", "0p25", 0)[0]
    assert "/scda/" in url and url.endswith("-scda-fc.grib2")
    assert stream_for("ifs", date(2026, 5, 12), 6) == "oper"


def test_aifs_rejects_the_tenth_degree_grid():
    with pytest.raises(ValueError, match="0.25"):
        field_urls(RUN, 0, "aifs-single", "0p1", 0)
    with pytest.raises(ValueError, match="90"):
        field_urls(RUN, 6, "ifs", "0p25", 96)


def test_documented_variables_include_the_wind_components():
    for system in ("ifs", "ifs-ens", "aifs-single", "aifs-ens"):
        names = variable_names(system)
        assert "10u" in names and "10v" in names
    assert "10fg" in variable_names("ifs")
    assert "10fg" not in variable_names("aifs-single")
    assert gust_param(90) == "10fg"
    assert gust_param(96) == "10fg3"


def test_index_selection_keeps_requested_messages_and_merges_ranges():
    text = "\n".join(
        [
            '{"param":"10u","levtype":"sfc","step":"24","_offset":0,"_length":10}',
            '{"param":"2t","levtype":"sfc","step":"24","_offset":10,"_length":5}',
            '{"param":"10u","levtype":"sfc","step":"24","number":"1","_offset":10,"_length":5}',
            '{"param":"t","levtype":"pl","step":"48","_offset":20,"_length":8}',
        ]
    )
    rows = select_messages(parse_index(text), param=["10u"], step=24, levtype="sfc")
    assert len(rows) == 2
    assert merge_ranges(rows) == ((0, 15),)
    assert index_url("https://example.test/file.grib2").endswith("file.index")


class _Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def close(self):
        return None


class _Session:
    def __init__(self, index=b"", body=b"GRIB"):
        self.calls = []
        self.index = index
        self.body = body

    def get(self, url, timeout=None, headers=None):
        self.calls.append((url, headers))
        if url.endswith(".index"):
            return _Response(self.index)
        return _Response(self.body)


def test_omitting_param_downloads_each_step_file_whole():
    session = _Session()
    fetcher = EcmwfOpenDataFetcher(session=session)
    raw = fetcher.download_fields(
        system="ifs", date=RUN, time=0, step=24, resol="0p25"
    )
    assert raw == b"GRIB"
    assert len(session.calls) == 1
    url, headers = session.calls[0]
    assert url.endswith("20261126000000-24h-oper-fc.grib2")
    assert headers is None


def test_a_parameter_list_uses_the_index_byte_ranges():
    index = (
        '{"param":"10u","levtype":"sfc","step":"24","_offset":0,"_length":4}\n'
        '{"param":"msl","levtype":"sfc","step":"24","_offset":4,"_length":4}\n'
    )
    session = _Session(index=index.encode(), body=b"u10!")
    fetcher = EcmwfOpenDataFetcher(session=session)
    raw = fetcher.download_fields(
        system="ifs",
        date="2026-11-26",
        time=0,
        step=24,
        resol="0p25",
        param=["10u"],
        levtype="sfc",
    )
    assert raw == b"u10!"
    urls = [url for url, _ in session.calls]
    assert urls[0].endswith(".index")
    assert session.calls[1][1]["Range"] == "bytes=0-3"


def test_wind_speed_is_the_magnitude_of_the_components():
    dataset = xr.Dataset({"10u": 3.0, "10v": 4.0})
    speed = add_wind_speed(dataset)["wind_speed"]
    assert speed.item() == pytest.approx(5.0)
    assert speed.attrs["units"] == "m s-1"


def test_each_storm_is_cropped_to_its_own_track_with_gust():
    dataset = xr.Dataset(
        {
            "10u": (("latitude", "longitude"), [[0, 3, 0], [0, 3, 0], [0, 0, 0]]),
            "10v": (("latitude", "longitude"), [[0, 4, 0], [0, 4, 0], [0, 0, 0]]),
            "10fg": (("latitude", "longitude"), [[1, 9, 1], [1, 9, 1], [1, 1, 1]]),
        },
        coords={"latitude": [10.0, 9.0, 8.0], "longitude": [-1.0, 0.0, 1.0]},
    )
    tracks = pd.DataFrame(
        {
            "storm_id": ["NEAR", "FAR"],
            "member": [1, 1],
            "system": ["ifs-ens", "ifs-ens"],
            "latitude": [9.0, 8.0],
            "longitude": [0.0, 1.0],
        }
    )
    near, far = pair_storm_fields(tracks, dataset, margin_degrees=0.4)
    assert near.storm_id == "NEAR"
    assert float(np.squeeze(near.fields.wind_speed.values)) == pytest.approx(5.0)
    assert float(np.squeeze(near.fields.gust.values)) == pytest.approx(9.0)
    assert list(near.fields.latitude.values) == [9.0]
    assert list(near.fields.longitude.values) == [0.0]
    assert float(np.squeeze(far.fields.gust.values)) == pytest.approx(1.0)
    assert near.area.bounds == pytest.approx((-0.4, 8.6, 0.4, 9.4))
    assert near.fields.wind_speed.sizes == near.fields.gust.sizes


def test_a_track_across_the_antimeridian_is_two_boxes():
    boxes = track_boxes([170.0, -170.0], [10.0, 12.0], margin_degrees=0)
    assert len(boxes) == 2
    assert boxes[0].bounds[0] == pytest.approx(170.0)
    assert boxes[1].bounds[2] == pytest.approx(-170.0)
    assert np.isfinite(boxes[0].bounds).all()
