import datetime
import json
import types
from pathlib import Path

import pytest
from shapely.geometry import Point, box

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.osm_roads import (
    CODE_FIXES,
    MULTI_COUNTRY_EXTRACTS,
    GeofabrikIndex,
    OSMRoadsConfig,
    OSMRoadsHandler,
    OSMRoadsResolveOptions,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "osm_roads"
SUBSET = DATA / "geofabrik_index_subset.json"
INDEX_URL = "https://download.geofabrik.de/index-v1.json"
GAZA = box(34.3, 31.3, 34.5, 31.5)
SINGAPORE = box(103.7, 1.25, 103.95, 1.45)


@pytest.fixture
def boundaries(monkeypatch):
    """Country boundaries for the countries without an extract of their own; records the lookups."""
    from gigaspatial.handlers import boundaries as module

    shapes = {"PS": GAZA, "SG": SINGAPORE, "MY": box(101.0, 3.0, 102.0, 4.0)}
    asked = []

    def create(country_code=None, **kwargs):
        asked.append(country_code)
        code = {"PSE": "PS", "SGP": "SG", "MYS": "MY"}.get(country_code, country_code)
        return types.SimpleNamespace(to_geoms=lambda: [shapes[code]])

    monkeypatch.setattr(module.AdminBoundaries, "create", staticmethod(create))
    return asked


@pytest.fixture(scope="module")
def index():
    return GeofabrikIndex(json.loads(SUBSET.read_text()))


def ids(units):
    return [u["id"] for u in units]


# ----- storing the index


def test_index_is_downloaded_once_and_stored(config, geofabrik, tmp_path):
    config.geofabrik_index()
    config.geofabrik_index()
    assert geofabrik.asked() == [INDEX_URL]
    assert (tmp_path / "osm/geofabrik/index-v1.json").is_file()
    stamp = (tmp_path / "osm/geofabrik/index-v1.json.downloaded").read_text()
    assert datetime.datetime.fromisoformat(stamp).tzinfo is not None
    again = OSMRoadsConfig(data_store=LocalDataStore(tmp_path), base_path=Path("osm/geofabrik"))
    assert "rwanda" in again.geofabrik_index().ids and geofabrik.asked() == [INDEX_URL]


def test_refresh_and_max_age(config, geofabrik, tmp_path):
    def downloads():
        return geofabrik.asked().count(INDEX_URL)

    config.geofabrik_index()
    config.geofabrik_index(OSMRoadsResolveOptions(refresh=True))
    assert downloads() == 2
    config.geofabrik_index(OSMRoadsResolveOptions(max_age_days=30))
    assert downloads() == 2
    old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=40)
    (tmp_path / "osm/geofabrik/index-v1.json.downloaded").write_text(old.isoformat())
    config.geofabrik_index(OSMRoadsResolveOptions(max_age_days=30))
    assert downloads() == 3


# ----- the index


def test_own_extracts_and_code_fixes(index):
    assert index.own_extract("RW") == "rwanda"
    assert index.own_extract("MY") is None
    assert index.own_extract("PS") is None
    assert index.own_extract("VU") == "vanuatu"
    assert index.own_extract("PF") == "polynesie-francaise"


def test_regions(index):
    assert index.regions("BR", "brazil") == ["centro-oeste", "nordeste", "norte", "sudeste", "sul"]
    assert index.regions("RW", "rwanda") == []
    assert index.regions("PS", None) == []


def test_cover(index):
    geometries = dict(zip(index.ids, index.geometries))
    assert index.cover(Point(30.06, -1.95).buffer(0.02)) == ["rwanda"]
    across = geometries["rwanda"].intersection(geometries["uganda"]).representative_point().buffer(0.05)
    assert sorted(index.cover(across)) == ["rwanda", "uganda"]
    regions = geometries["nordeste"].intersection(geometries["norte"]).representative_point().buffer(0.05)
    assert sorted(index.cover(regions)) == ["nordeste", "norte"]
    assert index.cover(GAZA) == ["israel-and-palestine"]


def test_continents_are_left_out_of_covers(index):
    with pytest.raises(ValueError, match='extracts=\\["africa"\\]'):
        index.cover(Point(7.5, 9.0).buffer(0.5))


def test_unit_shape(index):
    assert index.unit("nordeste", matches_country=True) == {
        "id": "nordeste",
        "url": "https://download.geofabrik.de/south-america/brazil/nordeste-latest.osm.pbf",
        "path": "south-america/brazil/nordeste",
        "matches_country": True,
    }


# ----- the curated lists, against the whole index


@pytest.fixture(scope="module")
def whole_index():
    return json.loads((DATA / "geofabrik_index_nogeom.json").read_text())


def test_every_multi_code_extract_is_listed(whole_index):
    multi = {f["properties"]["id"] for f in whole_index["features"] if len(f["properties"].get("iso3166-1:alpha2") or []) > 1}
    assert multi <= MULTI_COUNTRY_EXTRACTS


def test_curated_extracts_exist_with_the_codes_they_fix(whole_index):
    by_id = {f["properties"]["id"]: f["properties"] for f in whole_index["features"]}
    assert MULTI_COUNTRY_EXTRACTS <= set(by_id)
    for extract in CODE_FIXES:
        assert by_id[extract].get("iso3166-1:alpha2") in (["VU"], ["MH"]), extract


def test_no_country_has_two_own_extracts(whole_index):
    index = GeofabrikIndex(whole_index)
    counts = {}
    for extract, codes in zip(index.ids, index.codes):
        if len(codes) == 1 and extract not in MULTI_COUNTRY_EXTRACTS:
            counts.setdefault(codes[0], []).append(extract)
    assert {code: extracts for code, extracts in counts.items() if len(extracts) > 1} == {}
    assert index.own_extract("VU") == "vanuatu" and index.own_extract("MH") == "marshall-islands"


# ----- choosing extracts for a source


def test_country_with_its_own_extract(config, boundaries):
    for source in ("RWA", "Rwanda", "rw"):
        units = config.get_relevant_data_units(source)
        assert ids(units) == ["rwanda"] and units[0]["matches_country"]
    assert boundaries == []


def test_country_regions(config, boundaries):
    units = config.get_relevant_data_units("BRA", regions=True)
    assert ids(units) == ["centro-oeste", "nordeste", "norte", "sudeste", "sul"]
    assert all(u["matches_country"] for u in units)
    assert ids(config.get_relevant_data_units("BRA")) == ["brazil"]
    assert ids(config.get_relevant_data_units("RWA", regions=True)) == ["rwanda"]


def test_countries_inside_multi_country_extracts_are_cropped(config, boundaries):
    for source, extract in (("PSE", "israel-and-palestine"), ("SGP", "malaysia-singapore-brunei"), ("MYS", "malaysia-singapore-brunei")):
        units = config.get_relevant_data_units(source)
        assert ids(units) == [extract] and not units[0]["matches_country"]
    assert boundaries == ["PS", "SG", "MY"]


def test_areas_and_points(config):
    units = config.get_relevant_data_units(box(30.05, -1.96, 30.07, -1.94))
    assert ids(units) == ["rwanda"] and not units[0]["matches_country"]
    assert ids(config.get_relevant_data_units([(-1.95, 30.06)], buffer=500)) == ["rwanda"]


def test_named_extracts(config):
    units = config.get_relevant_data_units("RWA", extracts=["africa"])
    assert ids(units) == ["africa"] and not units[0]["matches_country"]
    units = config.get_relevant_data_units("BRA", extracts=["nordeste"])
    assert units[0]["matches_country"]
    with pytest.raises(ValueError, match="Unknown Geofabrik extracts"):
        config.get_relevant_data_units("RWA", extracts=["atlantis"])


def test_extracts_option_is_hashable():
    options = OSMRoadsResolveOptions(extracts=["rwanda", "uganda"])
    assert options.extracts == ("rwanda", "uganda") and hash(options)


def test_unknown_country_and_handler_entry(tmp_path, geofabrik):
    handler = OSMRoadsHandler(data_store=LocalDataStore(tmp_path), base_path=Path("osm/geofabrik"))
    assert ids(handler.config.get_relevant_data_units("Rwanda")) == ["rwanda"]
    with pytest.raises(ValueError, match="Unknown country"):
        handler.config.get_relevant_data_units("Atlantis")
    with pytest.raises(TypeError):
        handler.config.get_relevant_data_units("Rwanda", network_typo="drive")
