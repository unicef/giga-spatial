import datetime
from pathlib import Path

import pytest
import requests

from gigaspatial.handlers.osm_roads import (
    OSMRoadsDownloader,
    OSMRoadsHandler,
    OSMRoadsResolveOptions,
    replication_timestamp,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "osm_roads"
GEOFABRIK = "https://download.geofabrik.de"
LATEST_DATE = "2026-10-28"
RWANDA = f"{GEOFABRIK}/africa/rwanda-latest.osm.pbf"
KIGALI_DATE = "2026-10-26"
TODAY = datetime.datetime.now(datetime.timezone.utc).date()


def days_ago(n):
    return (TODAY - datetime.timedelta(days=n)).isoformat()


def store_extract(tmp_path, date, path="africa/rwanda"):
    stored = tmp_path / "osm/geofabrik" / date / f"{path}.osm.pbf"
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_bytes((DATA / "kigali.osm.pbf").read_bytes())
    return stored


@pytest.fixture
def handler(store, geofabrik):
    return OSMRoadsHandler(data_store=store, base_path=Path("osm/geofabrik"))


# ----- the date of a unit


def test_units_carry_the_latest_date_and_dated_url(config, geofabrik):
    (unit,) = config.get_relevant_data_units("RWA")
    assert unit["date"] == LATEST_DATE
    assert unit["url"] == f"{GEOFABRIK}/africa/rwanda-261028.osm.pbf"
    assert config.get_data_unit_path(unit) == Path("osm/geofabrik/2026-10-28/africa/rwanda.osm.pbf")
    assert geofabrik.asked("HEAD") == [RWANDA]


def test_mirrored_extracts_take_the_date_of_their_replication_state(config, geofabrik):
    geofabrik.mirrored.add("africa/rwanda")
    geofabrik.latest["africa/rwanda"] = "2026-10-27"
    (unit,) = config.get_relevant_data_units("RWA")
    assert unit["date"] == "2026-10-27" and unit["url"].endswith("/africa/rwanda-261027.osm.pbf")
    assert f"{GEOFABRIK}/africa/rwanda-updates/state.txt" in geofabrik.asked("GET")


def test_the_newest_stored_date_is_used_offline(config, geofabrik, tmp_path):
    store_extract(tmp_path, "2026-10-01")
    store_extract(tmp_path, KIGALI_DATE)
    (tmp_path / "osm/geofabrik/notes").mkdir()
    store_extract(tmp_path, "2026-10-27", path="africa/uganda")
    assert config.stored_dates("africa/rwanda") == ["2026-10-01", KIGALI_DATE]
    (unit,) = config.get_relevant_data_units("RWA")
    assert unit["date"] == KIGALI_DATE
    assert geofabrik.asked("HEAD") == []


def test_refresh_asks_for_the_latest(config, geofabrik, tmp_path):
    store_extract(tmp_path, KIGALI_DATE)
    (unit,) = config.get_relevant_data_units("RWA", refresh=True)
    assert unit["date"] == LATEST_DATE and geofabrik.asked("HEAD") == [RWANDA]


def test_max_age_days(config, geofabrik, tmp_path):
    store_extract(tmp_path, days_ago(3))
    geofabrik.latest["africa/rwanda"] = TODAY.isoformat()
    assert config.get_relevant_data_units("RWA", max_age_days=7)[0]["date"] == days_ago(3)
    assert geofabrik.asked("HEAD") == []
    assert config.get_relevant_data_units("RWA", max_age_days=1)[0]["date"] == TODAY.isoformat()


def test_max_age_days_keeps_the_stored_extract_when_geofabrik_is_down(config, geofabrik, tmp_path, monkeypatch):
    store_extract(tmp_path, days_ago(30))
    warnings = []
    monkeypatch.setattr(config.logger, "warning", warnings.append)

    def down(url, **kwargs):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(requests, "head", down)
    assert config.get_relevant_data_units("RWA", max_age_days=7)[0]["date"] == days_ago(30)
    assert any("Cannot reach Geofabrik" in w for w in warnings)
    with pytest.raises(requests.ConnectionError):
        config.get_relevant_data_units("RWA", refresh=True)


def test_a_pinned_date(config, geofabrik, tmp_path):
    (unit,) = config.get_relevant_data_units("RWA", date="2026-10-01")
    assert unit["date"] == "2026-10-01" and unit["url"].endswith("/africa/rwanda-261001.osm.pbf")
    assert geofabrik.asked("HEAD") == []
    assert OSMRoadsResolveOptions(date=datetime.date(2026, 10, 1)).date == "2026-10-01"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        OSMRoadsResolveOptions(date="01/10/2026")
    with pytest.raises(ValueError, match="pins"):
        OSMRoadsResolveOptions(date="2026-10-01", refresh=True)


# ----- downloads


def test_download_stores_the_dated_file_by_date(handler, geofabrik, tmp_path):
    geofabrik.latest["africa/rwanda"] = KIGALI_DATE
    dated = geofabrik.serve("africa/rwanda", KIGALI_DATE)
    assert handler.ensure_data_available("RWA")
    stored = tmp_path / "osm/geofabrik/2026-10-26/africa/rwanda.osm.pbf"
    assert stored.read_bytes() == (DATA / "kigali.osm.pbf").read_bytes()
    assert geofabrik.asked("GET")[-2:] == [f"{dated}.md5", dated]

    geofabrik.requests.clear()
    fresh = OSMRoadsHandler(data_store=handler.data_store, base_path=Path("osm/geofabrik"))
    assert fresh.ensure_data_available("RWA")
    assert geofabrik.asked() == []


def test_regions_are_stored_under_their_country(handler, geofabrik, tmp_path):
    regions = ["centro-oeste", "nordeste", "norte", "sudeste", "sul"]
    for region in regions:
        geofabrik.serve(f"south-america/brazil/{region}", LATEST_DATE)
    assert handler.ensure_data_available("BRA", regions=True)
    folder = tmp_path / "osm/geofabrik" / LATEST_DATE / "south-america/brazil"
    assert sorted(p.name for p in folder.iterdir()) == [f"{r}.osm.pbf" for r in regions]


def test_a_checksum_mismatch_stores_nothing(handler, geofabrik, tmp_path):
    dated = geofabrik.serve("africa/rwanda", LATEST_DATE)
    geofabrik.files[f"{dated}.md5"] = b"0" * 32 + b"  rwanda-261028.osm.pbf\n"
    assert not handler.ensure_data_available("RWA")
    assert not (tmp_path / "osm/geofabrik" / LATEST_DATE / "africa/rwanda.osm.pbf").exists()
    (unit,) = handler.config.get_relevant_data_units("RWA")
    with pytest.raises(ValueError, match="checksum"):
        handler.downloader.download_data_unit(unit, handler.downloader.options_type())


def test_an_expired_date_says_so(handler, geofabrik):
    assert not handler.ensure_data_available("RWA", date="2026-10-10")
    (unit,) = handler.config.get_relevant_data_units("RWA", date="2026-10-10")
    with pytest.raises(FileNotFoundError, match="no longer keeps rwanda of 2026-10-10"):
        OSMRoadsDownloader(config=handler.config).download_data_unit(unit, handler.downloader.options_type())


# ----- the replication timestamp, offline


def test_replication_timestamp_of_a_stored_extract(handler, geofabrik, tmp_path):
    pytest.importorskip("osmium")
    geofabrik.latest["africa/rwanda"] = KIGALI_DATE
    geofabrik.serve("africa/rwanda", KIGALI_DATE)
    handler.ensure_data_available("RWA")
    (unit,) = handler.config.get_relevant_data_units("RWA")
    stamp = replication_timestamp(handler.config.get_data_unit_path(unit), handler.data_store)
    assert stamp == datetime.datetime(2026, 10, 26, 21, 24, 46, tzinfo=datetime.timezone.utc)
    assert stamp.date().isoformat() == unit["date"]


def test_replication_timestamp_is_none_without_one(tmp_path):
    osmium = pytest.importorskip("osmium")
    bare = tmp_path / "bare.osm.pbf"
    with osmium.SimpleWriter(str(bare)) as writer:
        writer.add_node(osmium.osm.mutable.Node(id=1, location=(30.06, -1.95)))
    assert replication_timestamp(bare) is None
