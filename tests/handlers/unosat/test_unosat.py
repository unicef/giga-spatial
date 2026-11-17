"""
Tests for the UNOSAT country catalogue.

Offline tests use stand-in catalogue rows. The live test reads the public
country and disaster-type lists and does not download product files.

Run with:
    pytest tests/handlers/unosat/test_unosat.py -v
    pytest tests/handlers/unosat/test_unosat.py -m live -o addopts= -v
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import requests
from shapely.geometry import box

from gigaspatial.handlers import unosat as module
from gigaspatial.handlers.unosat import (
    UnosatHandler,
    build_product_record,
    collect_window,
    extract_dates,
    list_resources,
    match_unosat_areas,
    parse_glide,
)


AREAS = [
    "Cape Verde",
    "Congo",
    "Congo (the Democratic Republic of the)",
    "Korea (the Republic of)",
    "Macedonia (the former Yugoslav Republic of)",
    "Mozambique",
    "Turkey",
    "Viet Nam",
]


def summary(product_id=3811, glide="TC20240313MOZ", created="2024-03-27T10:38:04Z", kind="Tropical Cyclone"):
    return {
        "product_id": product_id,
        "title": "Satellite detected water extent over Maputo province, Mozambique, as of 23 March 2024",
        "glide": glide,
        "created_at": created,
        "activation_type": kind,
        "area_event_name": "Mozambique",
        "latitude": -26.0,
        "longitude": 32.4,
        "image_file": f"/static/unosat_filesystem/{product_id}/preview.jpg",
    }


def detail(product_id=3811, glide="TC20240313MOZ", pdf="report.pdf", shp=True, excel="/unosat_filesystem/3811/None"):
    event = {
        "id": product_id,
        "title": summary(product_id, glide)["title"],
        "description": (
            "Status: Flooded agricultural areas observed\n"
            "Potentially affected structures observed in Vohemar District as of 31 March 2024."
        ),
        "sources": "Imagery Date: 23 Mar. 2024\nAcquisition date: 01 April 2024\nCopyright: ESA",
        "pdf_name": pdf,
        "excel_table": excel,
        "glide": glide,
        "created_at": "2024-03-27T10:38:04Z",
        "shp_link": f"/static/unosat_filesystem/{product_id}/flood_shp.zip" if shp else None,
        "kml_link": None,
        "gdp_link": None,
        "image_file": f"/static/unosat_filesystem/{product_id}/preview.jpg",
        "wmap_link": "https://experience.arcgis.com/experience/example",
        "wms_link": "",
    }
    return {"map_event": event, "latitude": -26.0, "longitude": 32.4, "related_products": []}


def install(monkeypatch, summaries, details):
    monkeypatch.setattr(
        module,
        "fetch_reference",
        lambda session, api_url: {"countries": AREAS, "activation_types": ["Floods", "Earthquake"], "products": []},
    )
    monkeypatch.setattr(module, "discover_products", lambda *args, **kwargs: summaries)
    monkeypatch.setattr(
        module,
        "fetch_detail",
        lambda session, api_url, product_id: details[product_id],
    )
    calls = []

    def fake_stream(url, path, store, **kwargs):
        calls.append(url)
        if url.endswith(".pdf"):
            payload = b"%PDF-1.4\n"
        elif url.endswith(".zip"):
            payload = b"PK\x03\x04fake"
        else:
            payload = b"\xff\xd8\xffpreview"
        store.write_file(str(path), payload)
        return Path(path)

    monkeypatch.setattr(module, "stream_to_store", fake_stream)
    monkeypatch.setattr(module.time, "sleep", lambda *args, **kwargs: None)
    return calls


def test_country_name_and_iso3_match_unosat_names():
    assert match_unosat_areas("MOZ", module.pycountry.countries.get(alpha_3="MOZ"), AREAS) == ["Mozambique"]
    assert match_unosat_areas("VNM", module.pycountry.countries.get(alpha_3="VNM"), AREAS) == ["Viet Nam"]
    assert match_unosat_areas("CPV", module.pycountry.countries.get(alpha_3="CPV"), AREAS) == ["Cape Verde"]
    assert match_unosat_areas("TUR", module.pycountry.countries.get(alpha_3="TUR"), AREAS) == ["Turkey"]
    assert match_unosat_areas("MKD", module.pycountry.countries.get(alpha_3="MKD"), AREAS) == [
        "Macedonia (the former Yugoslav Republic of)"
    ]
    assert match_unosat_areas("COD", module.pycountry.countries.get(alpha_3="COD"), AREAS) == [
        "Congo (the Democratic Republic of the)"
    ]
    assert match_unosat_areas("COG", module.pycountry.countries.get(alpha_3="COG"), AREAS) == ["Congo"]
    assert match_unosat_areas("KOR", module.pycountry.countries.get(alpha_3="KOR"), AREAS) == [
        "Korea (the Republic of)"
    ]


def test_a_geometry_is_not_a_country():
    handler = UnosatHandler(base_path=Path("unused"), n_workers=1)
    with pytest.raises(ValueError, match="country"):
        handler.config.get_search_key(box(0, 0, 1, 1), handler.config.options_type())


def test_glide_date_is_the_event_date_and_publication_is_not():
    parsed = parse_glide("TC20240313MOZ")
    assert parsed["event_date"] == "2024-03-13"
    assert parsed["event_id"] == "TC20240313MOZ"
    assert parse_glide("No glide") is None
    assert parse_glide("None") is None
    assert parse_glide("FL20261399BTN") is None

    extracted = extract_dates(
        "Water extent as of 23 March 2024",
        "Observed as of 31 March 2024.",
        "Imagery Date: 23 Mar. 2024\nAcquisition date: 01 April 2024",
    )
    assert extracted["observation_date"] == "2024-03-23"
    assert extracted["imagery_acquisition_date"] == "2024-03-23"
    assert extracted["imagery_acquisition_dates"] == ["2024-03-23", "2024-04-01"]
    assert extracted["imagery_acquisition_provenance"]["source"] == "sources"
    assert extracted["observation_provenance"]["confidence"] == "medium"
    assert "23 March 2024" in extracted["observation_provenance"]["evidence"]

    later_first = extract_dates(
        "Water extent as of 31 March 2024",
        "Earlier pass as of 23 March 2024.",
        "Imagery Date: 01 April 2024\nAcquisition date: 23 Mar. 2024",
    )
    assert later_first["observation_date"] == "2024-03-23"
    assert later_first["observation_provenance"]["evidence"] == "23 March 2024."
    assert later_first["imagery_acquisition_date"] == "2024-03-23"
    assert later_first["imagery_acquisition_provenance"]["evidence"] == "23 Mar. 2024"

    record = build_product_record(
        summary(),
        detail(),
        country_iso3="MOZ",
        country_name="Mozambique",
        unosat_areas=["Mozambique"],
        include_previews=True,
    )
    assert record["event_start_date"] == "2024-03-13"
    assert record["publication_date"] == "2024-03-27"
    assert record["event_start_date"] != record["publication_date"]
    assert record["imagery_acquisition_date"] == "2024-03-23"
    assert record["disaster_type"] == "Tropical Cyclone"
    assert record["product_status"] == "Flooded agricultural areas observed"
    assert record["administrative_area"] == "Vohemar District"
    assert record["event_end_date"] is None
    assert record["last_updated_date"] is None
    assert record["data_license"] is None
    assert "experience.arcgis.com" not in record["download_urls"][0]


def test_missing_glide_does_not_invent_an_event_or_a_date():
    record = build_product_record(
        summary(glide=None, created=None, kind="Complex Emergency"),
        detail(glide=None, pdf=None, shp=False),
        country_iso3="MOZ",
        country_name="Mozambique",
        unosat_areas=["Mozambique"],
        include_previews=False,
    )
    record_detail = detail(glide=None, pdf=None, shp=False)
    record_detail["map_event"]["created_at"] = None
    record_detail["map_event"]["title"] = "Assessment"
    record_detail["map_event"]["description"] = None
    record_detail["map_event"]["sources"] = None
    bare = summary(glide=None, created=None, kind="Complex Emergency")
    bare["title"] = "Assessment"
    record = build_product_record(
        bare,
        record_detail,
        country_iso3="MOZ",
        country_name="Mozambique",
        unosat_areas=["Mozambique"],
        include_previews=False,
    )
    assert record["event_id"] is None
    assert record["event_start_date"] is None
    assert record["publication_date"] is None
    assert record["observation_date"] is None
    assert record["disaster_type"] == "Complex Emergency"
    assert record["resources"] == []


def test_none_links_are_not_files_and_duplicate_urls_are_dropped():
    event = detail()["map_event"]
    event["shp_link"] = event["shp_link"]
    event["kml_link"] = event["shp_link"]
    resources = list_resources(event, include_previews=True, api_url="https://unosat.org")
    urls = [item["url"] for item in resources]
    assert len(urls) == len(set(urls))
    assert all(not url.endswith("/None") for url in urls)
    assert {item["format"] for item in resources} >= {"pdf", "shp", "preview-image"}

    embedded = dict(event)
    embedded["excel_table"] = (
        "/unosat_filesystem/2718/http://unosat-maps.web.cern.ch/unosat-maps/MZ/TC20190423MOZ/table.zip"
    )
    embedded["shp_link"] = None
    embedded["kml_link"] = None
    embedded["gdp_link"] = None
    embedded["pdf_name"] = None
    embedded["image_file"] = None
    external = list_resources(embedded, include_previews=False, api_url="https://unosat.org")
    assert external[0]["url"] == "http://unosat-maps.web.cern.ch/unosat-maps/MZ/TC20190423MOZ/table.zip"
    assert external[0]["format"] == "zip"


def test_a_full_window_is_split_so_products_are_not_dropped(monkeypatch):
    calls = []

    def fake(session, api_url, areas, start, end):
        calls.append((start, end))
        if (start, end) == (date(2024, 1, 1), date(2024, 12, 31)):
            return [{"map_event": {"id": 1}}, {"map_event": {"id": 2}}]
        if start < date(2024, 7, 2):
            return [{"map_event": {"id": 1}}]
        return [{"map_event": {"id": 2}}]

    monkeypatch.setattr(module, "fetch_window", fake)
    rows = collect_window(None, "https://unosat.org", ["Mozambique"], date(2024, 1, 1), date(2024, 12, 31), 2)
    ids = [(row.get("map_event") or row)["id"] for row in rows]
    assert ids == [1, 2]
    assert len(calls) == 3


def test_country_catalogue_groups_an_event_and_orders_by_event_date(tmp_path, monkeypatch):
    calls = install(
        monkeypatch,
        [
            summary(3811, "TC20240313MOZ", "2024-03-27T10:38:04Z", "Tropical Cyclone"),
            summary(100, "FL20190115MOZ", "2019-01-20T00:00:00Z", "Floods"),
        ],
        {
            3811: detail(3811, "TC20240313MOZ"),
            100: detail(100, "FL20190115MOZ", pdf="old.pdf", shp=False),
        },
    )
    handler = UnosatHandler(base_path=tmp_path / "unosat", n_workers=1)
    catalogue = handler.load_data("Mozambique")
    assert catalogue["country_iso3"] == "MOZ"
    assert [event["event_id"] for event in catalogue["events"]] == ["TC20240313MOZ", "FL20190115MOZ"]
    newest = catalogue["events"][0]
    assert newest["disaster_type"] == "Tropical Cyclone"
    assert newest["event_start_date"] == "2024-03-13"
    assert newest["publication_date"] == "2024-03-27"
    assert newest["order_date_kind"] == "event"
    assert newest["product_count"] == 1
    assert "pdf" in newest["available_formats"]
    assert catalogue["last_successful_update"]
    assert catalogue["last_checked"]
    assert Path(tmp_path, "unosat", "MOZ", "events", "TC20240313MOZ", "metadata.json").is_file()
    assert any(url.endswith("report.pdf") for url in calls)

    again = install(monkeypatch, [
        summary(3811, "TC20240313MOZ", "2024-03-27T10:38:04Z", "Tropical Cyclone"),
        summary(100, "FL20190115MOZ", "2019-01-20T00:00:00Z", "Floods"),
    ], {
        3811: detail(3811, "TC20240313MOZ"),
        100: detail(100, "FL20190115MOZ", pdf="old.pdf", shp=False),
    })
    second = UnosatHandler(base_path=tmp_path / "unosat", n_workers=1)
    second.load_data("MOZ")
    assert again == []

    added = install(monkeypatch, [
        summary(3811, "TC20240313MOZ", "2024-03-27T10:38:04Z", "Tropical Cyclone"),
        summary(100, "FL20190115MOZ", "2019-01-20T00:00:00Z", "Floods"),
        summary(200, "EQ20210101MOZ", "2021-01-02T00:00:00Z", "Earthquake"),
    ], {
        3811: detail(3811, "TC20240313MOZ"),
        100: detail(100, "FL20190115MOZ", pdf="old.pdf", shp=False),
        200: detail(200, "EQ20210101MOZ", pdf="quake.pdf", shp=False),
    })
    third = UnosatHandler(base_path=tmp_path / "unosat", n_workers=1)
    updated = third.load_data("Mozambique")
    assert [url for url in added if url.endswith(".pdf")] == [
        "https://unosat.org/static/unosat_filesystem/200/quake.pdf"
    ]
    assert {event["disaster_type"] for event in updated["events"]} >= {
        "Earthquake",
        "Floods",
        "Tropical Cyclone",
    }


def test_a_failed_download_is_retried_and_then_recorded(tmp_path, monkeypatch):
    install(monkeypatch, [summary()], {3811: detail()})
    attempts = {"n": 0}

    def flaky(url, path, store, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise requests.ConnectionError("temporary")
        store.write_file(str(path), b"%PDF-1.4\n" if url.endswith(".pdf") else b"PK\x03\x04")
        return Path(path)

    monkeypatch.setattr(module, "stream_to_store", flaky)
    handler = UnosatHandler(base_path=tmp_path / "unosat", n_workers=1)
    catalogue = handler.load_data("MOZ")
    assert attempts["n"] >= 3
    assert catalogue["events"]

    install(monkeypatch, [summary()], {3811: detail()})

    def broken(url, path, store, **kwargs):
        error = requests.HTTPError("500")
        error.response = type("Response", (), {"status_code": 500})()
        raise error

    monkeypatch.setattr(module, "stream_to_store", broken)
    handler = UnosatHandler(base_path=tmp_path / "other", n_workers=1)
    assert handler.ensure_data_available("MOZ") is False
    assert list(Path(tmp_path, "other").rglob("*.pdf")) == []


@pytest.mark.live
def test_live_catalogue_lists_countries_and_disaster_types():
    reference = module.fetch_reference(module._session(), module.DEFAULT_API_URL)
    assert "Mozambique" in reference["countries"]
    types = set(reference["activation_types"])
    for name in (
        "Floods",
        "Flash Flood",
        "Drought",
        "Earthquake",
        "Tropical Cyclone",
        "Landslides",
        "Wild Fire",
        "Tsunami",
        "Volcano",
        "Complex Emergency",
    ):
        assert name in types
