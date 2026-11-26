"""ECMWF tropical-cyclone track handler."""

import pandas as pd
import pytest
from shapely.geometry import box

from gigaspatial.fetchers.ecmwf.tropical_cyclone import EcmwfTropicalCycloneFetcher
from gigaspatial.handlers.ecmwf import tropical_cyclone as module
from gigaspatial.handlers.ecmwf.tropical_cyclone import (
    EcmwfTropicalCycloneHandler,
    centre_system,
    filenames_from_listing,
    parse_diss_coordinate,
    parse_track_filename,
)

LISTING = """
<a href="/file/20261015000000/A_JSXX04ECMF150000_C_ECMP_20261015000000_tropical_cyclone_track_SIMON_-108p5degW_19p2degN_bufr4.bin">SIMON</a>
<a href="/file/20261015000000/A_JSXX01ECEP150000_C_ECMP_20261015000000_tropical_cyclone_track_36W_159p4degE_10p8degN_bufr4.bin">36W</a>
"""
SIMON = (
    "A_JSXX04ECMF150000_C_ECMP_20261015000000_"
    "tropical_cyclone_track_SIMON_-108p5degW_19p2degN_bufr4.bin"
)


def test_diss_names_identify_the_ifs_systems_and_the_named_position():
    names = filenames_from_listing(LISTING)
    assert centre_system(names[0]) == "ifs-ens"
    assert centre_system(names[1]) == "ifs"
    simon = parse_track_filename(SIMON)
    assert simon["storm_id"] == "SIMON"
    assert simon["longitude"] == pytest.approx(-108.5)
    assert simon["latitude"] == pytest.approx(19.2)
    southern = parse_track_filename(
        "tropical_cyclone_track_70S_50p2degE_-1p5degS_bufr4.bin"
    )
    assert southern["storm_id"] == "70S"
    assert southern["longitude"] == pytest.approx(50.2)
    assert southern["latitude"] == pytest.approx(-1.5)
    assert parse_diss_coordinate("10degN") == 10


def test_a_cycle_stores_one_file_per_forecasting_system(tmp_path):
    handler = EcmwfTropicalCycloneHandler(
        date="20261015", time=0, base_path=tmp_path, n_workers=1
    )
    units = handler.config.get_relevant_data_units(box(-1, -1, 1, 1))
    assert [unit["system"] for unit in units] == [
        "ifs",
        "ifs-ens",
        "aifs-single",
        "aifs-ens",
    ]
    path = handler.config.get_data_unit_path(units[0])
    assert path.parts[-4:] == ("2026-10-15", "00", "ifs", "tracks.geojson")


def test_essential_files_are_filtered_to_the_requested_system(tmp_path, monkeypatch):
    fetched = []

    def fake_get(url, headers=None, timeout=60):
        fetched.append(url)
        if url.endswith("/"):
            return LISTING.encode()
        if "SIMON" in url:
            return b"not-a-bufr"
        return None

    monkeypatch.setattr(module, "http_get", fake_get)
    handler = EcmwfTropicalCycloneHandler(
        date="2026-10-15", time=0, system="ifs", base_path=tmp_path, n_workers=1
    )
    frame = handler.load_data(box(-109, 19, -108, 20), crop_to_source=True)
    assert list(frame.storm_id) == ["SIMON"]
    assert frame.geometry.iloc[0].x == pytest.approx(-108.5)
    assert any(url.endswith(SIMON) for url in fetched)
    assert not any("ECEP" in url for url in fetched)

    outside = handler.load_data(box(0, 0, 1, 1), crop_to_source=True)
    assert len(outside) == 0


def test_aifs_tracks_fall_back_to_the_open_data_bufr(tmp_path, monkeypatch):
    fetched = []

    def fake_get(url, headers=None, timeout=60):
        fetched.append(url)
        if url.endswith("/"):
            return LISTING.encode()
        if url.endswith("-tf.bufr"):
            return b"OPEN"
        return None

    def fake_decode(content):
        if content != b"OPEN":
            return None
        return pd.DataFrame(
            [
                {
                    "storm_id": "NAMED",
                    "member": 1,
                    "step": 0,
                    "latitude": 12.0,
                    "longitude": 50.0,
                    "pressure": 100000.0,
                    "maximum_wind": 18.0,
                }
            ]
        )

    monkeypatch.setattr(module, "http_get", fake_get)
    monkeypatch.setattr(module, "decode_bufr", fake_decode)
    handler = EcmwfTropicalCycloneHandler(
        date="2026-10-15",
        time=18,
        system="aifs-single",
        base_path=tmp_path,
        n_workers=1,
    )
    frame = handler.load_data(box(49, 11, 51, 13), crop_to_source=True)
    assert list(frame.storm_id) == ["NAMED"]
    assert list(frame.source) == ["open-data"]
    assert list(frame.system) == ["aifs-single"]
    track = next(url for url in fetched if url.endswith(".bufr"))
    assert "/aifs-single/0p25/oper/" in track
    assert track.endswith("20261015180000-360h-oper-tf.bufr")


def test_the_track_fetcher_returns_the_essential_position(monkeypatch):
    def fake_get(url, headers=None, timeout=60):
        if url.endswith("/"):
            return LISTING.encode()
        if "SIMON" in url:
            return b"not-a-bufr"
        return None

    monkeypatch.setattr(module, "http_get", fake_get)
    frame = EcmwfTropicalCycloneFetcher().fetch(
        box(-109, 19, -108, 20),
        system="ifs",
        date="2026-10-15",
        time=0,
    )
    assert list(frame.storm_id) == ["SIMON"]
    assert list(frame.source) == ["essential"]


def test_a_missing_date_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="date"):
        EcmwfTropicalCycloneHandler(base_path=tmp_path)
