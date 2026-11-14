"""
UNOSAT historical disaster products for one country.

The public catalogue is the products page, https://unosat.org/products/.
That page is a web application. Its data calls, confirmed from the page
bundle and from the live service, are:

- ``GET /our_products`` returns the newest products, the country names UNOSAT
  uses, and every activation type currently published.
- ``POST /our_products/`` with ``date_from``, ``date_to`` and ``region``
  returns the products for those country names in that window. A country
  filter without dates returns nothing. The response is one JSON list, and
  the page paginates that list in the browser. This handler therefore asks
  year by year, and splits a year that comes back at the size guard.
- ``GET /our_products/{id}`` returns the product text, the GLIDE number, and
  the file links.

Files live under ``/static/unosat_filesystem/{id}/``. A link whose file name
is ``None`` is not a file. Web map and WMS links are recorded and not saved
as data. ``created_at`` is when UNOSAT published the product, not when the
disaster happened. The event date is taken from a GLIDE number when that
number contains a real date.

Disaster types are not a fixed list. Whatever ``activation_type`` the
catalogue returns is kept, including floods, storms, earthquakes, conflict
emergencies and the other categories UNOSAT publishes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import threading
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import pycountry
import requests
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from gigaspatial.config import config as global_config
from gigaspatial.handlers.base import (
    BaseHandler,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    DownloadOptions,
    ReadOptions,
    ResolveOptions,
)
from gigaspatial.handlers.base.fetch import stream_to_store
from gigaspatial.handlers.base.source import SourceKind, classify_source


CATALOGUE_PAGE = "https://unosat.org/products/"
DEFAULT_API_URL = "https://unosat.org"
USER_AGENT = "giga-spatial/1.0"
ATTRIBUTION = "United Nations Satellite Centre (UNOSAT), UNITAR"
# A window that returns this many products is split. The live country
# queries observed so far are smaller than this. The split is a guard so a
# truncated response cannot hide the rest of a year.
WINDOW_SPLIT_AT = 180
EARLIEST_PAGE_YEAR = 2000
OLDEST_YEAR = 1900
RETRY_PAUSES = (0, 1, 2)
RETRY_STATUS = {429, 500, 502, 503, 504}
_CATALOGUE_LOCK = threading.Lock()

GLIDE_RE = re.compile(r"^([A-Za-z]{2})(\d{8})([A-Za-z]{3})$")
DATE_IN_TEXT_RE = re.compile(
    r"\b(\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4})\b"
)
ACQUISITION_RE = re.compile(
    r"(?:Acquisition date|Imagery Date|Image date)\s*:?\s*([^\r\n;]+)",
    re.IGNORECASE,
)
AS_OF_RE = re.compile(r"\bas of\s+([^,;\r\n]+)", re.IGNORECASE)
STATUS_RE = re.compile(r"Status:\s*([^\r\n]+)", re.IGNORECASE)
ADMIN_RE = re.compile(
    r"\b([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3})\s+"
    r"(District|Province|Region|Commune|Governorate|State|Municipality)\b"
)

# UNOSAT's area list does not always use the current ISO country name.
AREA_ALIASES = {
    "CPV": ("Cape Verde",),
    "TUR": ("Turkey", "Türkiye"),
    "MKD": ("Macedonia (the former Yugoslav Republic of)", "North Macedonia"),
    "SWZ": ("Eswatini", "Swaziland"),
    "CIV": ("Côte D'Ivoire", "Cote d'Ivoire"),
    "PSE": ("Occupied Palestinian Territory", "West Bank"),
    "CZE": ("Czech Republic", "Czechia"),
}


@dataclasses.dataclass(frozen=True)
class UnosatResolveOptions(ResolveOptions):
    """Resolve options for one UNOSAT country load."""

    include_previews: bool = True


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class UnosatConfig(BaseHandlerConfig):
    """
    Configuration for a country's UNOSAT products.

    Attributes:
        base_path: Directory of the stored catalogue and files.
        api_url: Site root that serves ``/our_products``.
    """

    options_type = UnosatResolveOptions
    units_match_country = True

    base_path: Path = Field(
        default_factory=lambda: global_config.get_path("unosat", "bronze")
    )
    api_url: str = DEFAULT_API_URL

    def __post_init__(self):
        super().__post_init__()
        self.api_url = self.api_url.rstrip("/")

    def get_search_key(self, source, options: UnosatResolveOptions) -> str:
        """Return the ISO3 code. Products are stored one country at a time."""
        if classify_source(source) is not SourceKind.COUNTRY:
            raise ValueError(
                "UNOSAT products are stored by country. "
                "Pass a country name or ISO code, for example 'Mozambique'."
            )
        try:
            return pycountry.countries.lookup(str(source)).alpha_3
        except LookupError as exc:
            raise ValueError(f"Unknown country: {source!r}.") from exc

    def get_relevant_data_units_by_key(
        self, search_key: str, options: UnosatResolveOptions
    ) -> List[dict]:
        """One unit per UNOSAT product for the country's published area names."""
        country = pycountry.countries.get(alpha_3=search_key)
        session = _session()
        reference = fetch_reference(session, self.api_url)
        areas = match_unosat_areas(search_key, country, reference["countries"])
        summaries = discover_products(session, self.api_url, areas, date.today())
        units = []
        records = []
        for summary in summaries:
            detail, detail_error = _detail_or_none(
                session, self.api_url, summary["product_id"], self.logger
            )
            record = build_product_record(
                summary,
                detail,
                country_iso3=search_key,
                country_name=country.name,
                unosat_areas=areas,
                include_previews=options.include_previews,
                api_url=self.api_url,
            )
            record["detail_error"] = detail_error
            stored = _read_json(self.data_store, self._product_dir(
                {"country_iso3": search_key, "product_id": summary["product_id"]}
            ) / "metadata.json")
            if (
                stored
                and stored.get("detail_fingerprint")
                and stored.get("detail_fingerprint") == record.get("detail_fingerprint")
            ):
                record = stored
            records.append(record)
            units.append(
                {
                    "country_iso3": search_key,
                    "country_name": country.name,
                    "unosat_areas": tuple(areas),
                    "product_id": summary["product_id"],
                    "include_previews": options.include_previews,
                    "summary_fingerprint": record["summary_fingerprint"],
                    "detail_fingerprint": record["detail_fingerprint"],
                    "detail_error": detail_error,
                    "record": record,
                }
            )
        self._write_catalogue(
            search_key, country.name, areas, records, downloaded=False, api_url=self.api_url
        )
        return units

    def get_data_unit_path(self, unit: dict, options=None) -> Path:
        """Metadata file for one product."""
        return self._product_dir(unit) / "metadata.json"

    def get_stored_paths(self, unit: dict, options=None) -> List[Path]:
        """Catalogue, product metadata, and every file kept for the product."""
        paths = [self._catalogue_path(unit), self.get_data_unit_path(unit)]
        meta = _read_json(self.data_store, self.get_data_unit_path(unit)) or {}
        for resource in meta.get("resources") or []:
            local = resource.get("local_path")
            if local:
                paths.append(Path(local))
        return paths

    def is_unit_available(self, unit: dict, paths) -> bool:
        """True when the stored product still matches the catalogue and its files exist."""
        meta = _read_json(self.data_store, self.get_data_unit_path(unit)) or {}
        if not meta or not _files_present(self.data_store, meta):
            return False
        if unit.get("detail_error") and meta.get("detail_fingerprint"):
            return meta.get("summary_fingerprint") == unit.get("summary_fingerprint")
        if not unit.get("detail_fingerprint"):
            return False
        return meta.get("detail_fingerprint") == unit.get("detail_fingerprint")

    def _country_dir(self, iso3: str) -> Path:
        return self.base_path / iso3

    def _catalogue_path(self, unit: dict) -> Path:
        return self._country_dir(unit["country_iso3"]) / "catalogue.json"

    def _product_dir(self, unit: dict) -> Path:
        return self._country_dir(unit["country_iso3"]) / "products" / str(unit["product_id"])

    def _write_catalogue(
        self, iso3, country_name, areas, records, downloaded: bool, api_url: str
    ) -> dict:
        path = self._country_dir(iso3) / "catalogue.json"
        previous = _read_json(self.data_store, path) or {}
        now = _utc_now()
        catalogue = assemble_catalogue(
            iso3,
            country_name,
            areas,
            records,
            retrieved_at=now,
            last_checked=now,
            last_successful_update=(
                now if downloaded else previous.get("last_successful_update")
            ),
            api_url=api_url,
        )
        _write_json(self.data_store, path, catalogue)
        _write_event_files(self.data_store, self._country_dir(iso3), catalogue)
        return catalogue


class UnosatDownloader(BaseHandlerDownloader):
    """Downloads UNOSAT files that are missing or no longer match the catalogue."""

    options_type = DownloadOptions
    config_type: ClassVar[type] = UnosatConfig

    def download_data_unit(self, unit: dict, options: DownloadOptions) -> List[Path]:
        """
        Store one product's original files.

        A file that is already stored for the same URL and still validates is
        not downloaded again. A replaced file is kept beside the new one.
        """
        record = dict(unit["record"])
        if record.get("detail_error") or not record.get("detail_fingerprint"):
            detail, error = _detail_or_none(
                _session(), self.config.api_url, unit["product_id"], self.logger
            )
            if detail is None:
                raise RuntimeError(
                    f"UNOSAT product {unit['product_id']} details were not available: {error}"
                )
            record = build_product_record(
                _summary_from_record(record),
                detail,
                country_iso3=unit["country_iso3"],
                country_name=unit["country_name"],
                unosat_areas=list(unit["unosat_areas"]),
                include_previews=unit["include_previews"],
                api_url=self.config.api_url,
            )
        previous = _read_json(self.data_store, self.config.get_data_unit_path(unit)) or {}
        resources = _sync_resources(
            self.data_store,
            self.config._product_dir(unit) / "original",
            record.get("resources") or [],
            previous.get("resources") or [],
            self.logger,
        )
        record["resources"] = resources
        record["retrieval_timestamp"] = _utc_now()
        _write_json(self.data_store, self.config.get_data_unit_path(unit), record)
        self._refresh_catalogue(unit, record)
        return self.config.get_stored_paths(unit)

    def _refresh_catalogue(self, unit: dict, record: dict) -> None:
        """Replace this product in the country catalogue and mark a successful update."""
        with _CATALOGUE_LOCK:
            path = self.config._catalogue_path(unit)
            current = _read_json(self.data_store, path) or {}
            records = []
            replaced = False
            for existing in current.get("products") or []:
                if existing.get("product_id") == record["product_id"]:
                    records.append(record)
                    replaced = True
                else:
                    records.append(existing)
            if not replaced:
                records.append(record)
            self.config._write_catalogue(
                unit["country_iso3"],
                unit["country_name"],
                list(unit["unosat_areas"]),
                records,
                downloaded=True,
                api_url=self.config.api_url,
            )


class UnosatReader(BaseHandlerReader):
    """Loads the stored country catalogue."""

    options_type = ReadOptions
    config_type: ClassVar[type] = UnosatConfig

    def load_from_paths(self, source_data_path: List[str], options: ReadOptions):
        """
        Return the country catalogue.

        Events are ordered by event date, newest first. An event with no
        event date is ordered by publication date and marked as such.
        """
        catalogues = []
        seen = set()
        for raw in source_data_path:
            country_dir = _country_dir_from_path(Path(raw))
            if country_dir is None or country_dir in seen:
                continue
            seen.add(country_dir)
            catalogue = _read_json(self.data_store, country_dir / "catalogue.json")
            if catalogue:
                catalogues.append(catalogue)
        if not catalogues:
            raise ValueError("No stored UNOSAT catalogue was found.")
        if len(catalogues) == 1:
            return catalogues[0]
        return {item["country_iso3"]: item for item in catalogues}


class UnosatHandler(BaseHandler):
    """
    Handler for UNOSAT disaster products at country level.

    One call checks the live catalogue and downloads files that are missing
    or whose product metadata changed:

        UnosatHandler().load_data("Mozambique")
        UnosatHandler().load_data("MOZ")

    The result is the country catalogue, not a single map.
    """

    def create_config(self, data_store, logger, **kwargs) -> UnosatConfig:
        kwargs.setdefault("n_workers", 2)
        return UnosatConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(self, config, data_store, logger, **kwargs) -> UnosatDownloader:
        return UnosatDownloader(config=config, data_store=data_store, logger=logger)

    def create_reader(self, config, data_store, logger, **kwargs) -> UnosatReader:
        return UnosatReader(config=config, data_store=data_store, logger=logger)


def fetch_reference(session, api_url: str) -> dict:
    """Country names and activation types from the public catalogue."""
    response = _request(session, "GET", f"{api_url.rstrip('/')}/our_products", timeout=60)
    payload = response.json()
    return {
        "countries": list(payload.get("countries") or []),
        "activation_types": list(payload.get("activation_types") or []),
        "products": payload.get("products") or [],
    }


def discover_products(session, api_url: str, areas: Sequence[str], today: date) -> List[dict]:
    """Every product UNOSAT publishes for these area names, oldest page included."""
    found: Dict[int, dict] = {}

    def absorb(rows: Sequence[dict]) -> None:
        for row in rows:
            summary = summarise_product(row)
            if summary is not None:
                found[summary["product_id"]] = summary

    for year in range(today.year, EARLIEST_PAGE_YEAR - 1, -1):
        end = today if year == today.year else date(year, 12, 31)
        absorb(
            collect_window(session, api_url, areas, date(year, 1, 1), end, WINDOW_SPLIT_AT)
        )
    cursor = date(EARLIEST_PAGE_YEAR - 1, 12, 31)
    while cursor.year >= OLDEST_YEAR:
        start = date(max(OLDEST_YEAR, cursor.year - 9), 1, 1)
        rows = collect_window(session, api_url, areas, start, cursor, WINDOW_SPLIT_AT)
        absorb(rows)
        if not rows:
            break
        cursor = start - timedelta(days=1)
    return [found[key] for key in sorted(found)]


def collect_window(session, api_url, areas, start: date, end: date, split_at: int) -> List[dict]:
    """Products in one date window. A full window is split in half."""
    rows = fetch_window(session, api_url, areas, start, end)
    if len(rows) < split_at or (end - start).days <= 31:
        return rows
    mid = start + timedelta(days=(end - start).days // 2)
    left = collect_window(session, api_url, areas, start, mid, split_at)
    right = collect_window(session, api_url, areas, mid + timedelta(days=1), end, split_at)
    return _dedupe_rows(left + right)


def fetch_window(session, api_url, areas, start: date, end: date) -> List[dict]:
    """One catalogue request. Dates are required for a country filter to match."""
    response = _request(
        session,
        "POST",
        f"{api_url.rstrip('/')}/our_products/",
        json={
            "date_from": start.isoformat(),
            "date_to": end.isoformat(),
            "region": list(areas),
            "title": "",
        },
        timeout=180,
    )
    payload = response.json()
    if isinstance(payload, dict):
        payload = payload.get("products") or []
    if not isinstance(payload, list):
        raise ValueError("UNOSAT catalogue response was not a product list.")
    return payload


def fetch_detail(session, api_url: str, product_id: int) -> dict:
    """Full metadata and file links for one product."""
    response = _request(
        session,
        "GET",
        f"{api_url.rstrip('/')}/our_products/{product_id}",
        timeout=60,
    )
    payload = response.json()
    if "map_event" not in payload:
        raise ValueError(f"UNOSAT product {product_id} has no map_event.")
    return payload


def summarise_product(row: dict) -> Optional[dict]:
    """Fields present on a catalogue row. Rows without an id are skipped."""
    event = row.get("map_event") or {}
    product_id = event.get("id")
    if product_id is None:
        return None
    return {
        "product_id": int(product_id),
        "title": event.get("title"),
        "glide": _blank_to_none(event.get("glide")),
        "created_at": event.get("created_at"),
        "activation_type": row.get("activation_type"),
        "area_event_name": row.get("area_event_name"),
        "latitude": row.get("latitude"),
        "longitude": row.get("longitude"),
        "image_file": event.get("image_file"),
    }


def match_unosat_areas(iso3: str, country, unosat_names: Sequence[str]) -> List[str]:
    """UNOSAT area names that mean this country. Raises when none match."""
    wanted = {_normalise(name) for name in _country_labels(iso3, country)}
    hits = [name for name in unosat_names if _normalise(name) in wanted]
    if not hits:
        raise ValueError(
            f"No UNOSAT area name matches {country.name} ({iso3}). "
            "The catalogue country list does not include it."
        )
    return hits


def parse_glide(value: Optional[str]) -> Optional[dict]:
    """
    A GLIDE number such as ``TC20240313MOZ``.

    The eight digits are the event date. ``No glide`` and ``None`` are not
    event identifiers.
    """
    if not value:
        return None
    match = GLIDE_RE.fullmatch(value.strip())
    if match is None:
        return None
    try:
        event_date = datetime.strptime(match.group(2), "%Y%m%d").date()
    except ValueError:
        return None
    return {
        "event_id": value.strip(),
        "hazard_code": match.group(1).upper(),
        "event_date": event_date.isoformat(),
        "glide_country_code": match.group(3).upper(),
    }


def extract_dates(title: Optional[str], description: Optional[str], sources: Optional[str]) -> dict:
    """
    Observation and acquisition dates found in the product text.

    Publication time is not read here. A date is stored only with the phrase
    it came from.
    """
    acquisitions = []
    for evidence in ACQUISITION_RE.findall(sources or ""):
        for found in _dates_in_phrase(evidence):
            acquisitions.append((found, evidence.strip()))
    observations = []
    for text, source_name in ((title, "title"), (description, "description")):
        for evidence in AS_OF_RE.findall(text or ""):
            for found in _dates_in_phrase(evidence):
                observations.append((found, source_name, evidence.strip()))
    acquisitions.sort(key=lambda item: item[0])
    observations.sort(key=lambda item: item[0])
    acquisition_dates = _unique(item[0] for item in acquisitions)
    observation_dates = _unique(item[0] for item in observations)
    return {
        "imagery_acquisition_dates": acquisition_dates,
        "imagery_acquisition_date": acquisition_dates[0] if acquisition_dates else None,
        "imagery_acquisition_provenance": (
            {
                "source": "sources",
                "confidence": "high",
                "evidence": acquisitions[0][1],
            }
            if acquisitions
            else None
        ),
        "observation_date": observation_dates[0] if observation_dates else None,
        "observation_provenance": (
            {
                "source": observations[0][1],
                "confidence": "medium",
                "evidence": observations[0][2],
            }
            if observations
            else None
        ),
    }


def build_product_record(
    summary: dict,
    detail: Optional[dict],
    *,
    country_iso3: str,
    country_name: str,
    unosat_areas: Sequence[str],
    include_previews: bool,
    api_url: str = DEFAULT_API_URL,
) -> dict:
    """One product, with event dates kept apart from the publication date."""
    event = (detail or {}).get("map_event") or {}
    glide_value = _blank_to_none(event.get("glide") or summary.get("glide"))
    glide = parse_glide(glide_value)
    title = event.get("title") or summary.get("title")
    description = event.get("description")
    sources = event.get("sources")
    extracted = extract_dates(title, description, sources)
    publication = _date_from_timestamp(event.get("created_at") or summary.get("created_at"))
    resources = []
    if detail is not None:
        resources = list_resources(
            event,
            include_previews=include_previews,
            api_url=api_url,
        )
    provenance = []
    if glide is not None:
        provenance.append(
            {
                "field": "event_start_date",
                "value": glide["event_date"],
                "source": "glide",
                "confidence": "high",
                "evidence": glide["event_id"],
            }
        )
    if publication is not None:
        provenance.append(
            {
                "field": "publication_date",
                "value": publication,
                "source": "unosat.created_at",
                "confidence": "high",
                "evidence": event.get("created_at") or summary.get("created_at"),
            }
        )
    if extracted["imagery_acquisition_provenance"]:
        provenance.append(
            {
                "field": "imagery_acquisition_date",
                "value": extracted["imagery_acquisition_date"],
                **extracted["imagery_acquisition_provenance"],
            }
        )
    if extracted["observation_provenance"]:
        provenance.append(
            {
                "field": "observation_date",
                "value": extracted["observation_date"],
                **extracted["observation_provenance"],
            }
        )
    status_match = STATUS_RE.search(description or "")
    record = {
        "event_id": glide["event_id"] if glide else None,
        "product_id": summary["product_id"],
        "country": country_name,
        "country_iso3": country_iso3,
        "unosat_areas": list(unosat_areas),
        "event_name": title,
        "disaster_type": summary.get("activation_type") or event.get("activation_type"),
        "disaster_subtype": None,
        "glide_hazard_code": glide["hazard_code"] if glide else None,
        "event_start_date": glide["event_date"] if glide else None,
        "event_end_date": None,
        "observation_date": extracted["observation_date"],
        "imagery_acquisition_date": extracted["imagery_acquisition_date"],
        "imagery_acquisition_dates": extracted["imagery_acquisition_dates"],
        "publication_date": publication,
        "last_updated_date": None,
        "GLIDE_number": glide["event_id"] if glide else None,
        "geographical_location": summary.get("area_event_name"),
        "administrative_area": _administrative_area(description),
        "product_title": title,
        "product_description": description,
        "product_status": status_match.group(1).strip() if status_match else None,
        "source_url": f"{CATALOGUE_PAGE}{summary['product_id']}",
        "download_urls": [item["url"] for item in resources],
        "service_urls": _service_urls(event),
        "available_formats": sorted({item["format"] for item in resources}),
        "data_license": None,
        "attribution": ATTRIBUTION,
        "retrieval_timestamp": None,
        "latitude": (detail or {}).get("latitude", summary.get("latitude")),
        "longitude": (detail or {}).get("longitude", summary.get("longitude")),
        "related_product_ids": [
            item.get("id")
            for item in (detail or {}).get("related_products") or []
            if isinstance(item, dict) and item.get("id") is not None
        ],
        "date_provenance": provenance,
        "resources": resources,
        "raw_created_at": event.get("created_at") or summary.get("created_at"),
    }
    record["summary_fingerprint"] = _fingerprint(
        {
            "product_id": record["product_id"],
            "title": summary.get("title"),
            "glide": summary.get("glide"),
            "created_at": summary.get("created_at"),
            "activation_type": summary.get("activation_type"),
            "area_event_name": summary.get("area_event_name"),
        }
    )
    record["detail_fingerprint"] = (
        None
        if detail is None
        else _fingerprint(
            {
                "summary": record["summary_fingerprint"],
                "description": description,
                "sources": sources,
                "resources": [item["url"] for item in resources],
                "related_product_ids": record["related_product_ids"],
            }
        )
    )
    return record


def list_resources(event: dict, *, include_previews: bool, api_url: str) -> List[dict]:
    """Downloadable files. Empty and ``None`` links are omitted."""
    product_id = event.get("id")
    resources = []
    pdf_name = event.get("pdf_name")
    if _file_name_ok(pdf_name):
        resources.append(
            _resource(
                urljoin(api_url, f"/static/unosat_filesystem/{product_id}/{pdf_name}"),
                pdf_name,
                "pdf",
            )
        )
    for field, fmt in (
        ("shp_link", "shp"),
        ("kml_link", "kml"),
        ("gdp_link", "gdb"),
        ("excel_table", "table"),
    ):
        link = event.get(field)
        if not _link_ok(link):
            continue
        url = _absolute_url(api_url, link)
        name = url.rstrip("/").split("/")[-1]
        resources.append(_resource(url, name, _format_for(name, fmt)))
    if include_previews and _link_ok(event.get("image_file")):
        url = _absolute_url(api_url, event["image_file"])
        name = url.rstrip("/").split("/")[-1]
        resources.append(_resource(url, name, "preview-image"))
    unique = []
    seen = set()
    for item in resources:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        unique.append(item)
    return unique


def assemble_catalogue(
    iso3: str,
    country_name: str,
    areas: Sequence[str],
    records: Sequence[dict],
    *,
    retrieved_at: str,
    last_checked: str,
    last_successful_update: Optional[str],
    api_url: str = DEFAULT_API_URL,
) -> dict:
    """Country catalogue grouped by GLIDE number, newest event first."""
    grouped: Dict[str, List[dict]] = {}
    ungrouped = []
    for record in records:
        event_id = record.get("event_id")
        if event_id:
            grouped.setdefault(event_id, []).append(record)
        else:
            ungrouped.append(record)
    events = [_event_record(event_id, products) for event_id, products in grouped.items()]
    events = _sort_events(events)
    publications = [record.get("publication_date") for record in records if record.get("publication_date")]
    return {
        "country_iso3": iso3,
        "country_name": country_name,
        "unosat_areas": list(areas),
        "source": CATALOGUE_PAGE,
        "api_url": api_url,
        "attribution": ATTRIBUTION,
        "data_license": None,
        "retrieved_at": retrieved_at,
        "last_checked": last_checked,
        "last_successful_update": last_successful_update,
        "product_count": len(records),
        "coverage": {
            "oldest_publication": min(publications) if publications else None,
            "newest_publication": max(publications) if publications else None,
        },
        "events": events,
        "products_without_event": [_public_product(record) for record in ungrouped],
        "products": list(records),
    }


def _event_record(event_id: str, products: Sequence[dict]) -> dict:
    ordered = sorted(products, key=lambda item: item.get("publication_date") or "")
    latest = ordered[-1]
    start_dates = [item.get("event_start_date") for item in products if item.get("event_start_date")]
    publications = [item.get("publication_date") for item in products if item.get("publication_date")]
    event_start = min(start_dates) if start_dates else None
    publication = max(publications) if publications else None
    order_date = event_start or publication
    if event_start:
        order_kind = "event"
    elif publication:
        order_kind = "publication"
    else:
        order_kind = "unknown"
    resources = []
    for product in products:
        resources.extend(product.get("resources") or [])
    areas = []
    for product in products:
        area = product.get("administrative_area")
        if area and area not in areas:
            areas.append(area)
    return {
        "event_id": event_id,
        "event_name": latest.get("event_name"),
        "disaster_type": latest.get("disaster_type"),
        "country": latest.get("country"),
        "country_iso3": latest.get("country_iso3"),
        "geographical_location": latest.get("geographical_location"),
        "administrative_area": "; ".join(areas) if areas else None,
        "event_start_date": event_start,
        "event_end_date": None,
        "publication_date": publication,
        "latest_assessment_date": publication,
        "order_date": order_date,
        "order_date_kind": order_kind,
        "product_count": len(products),
        "resources": resources,
        "available_formats": sorted({item.get("format") for item in resources if item.get("format")}),
        "product_ids": [item["product_id"] for item in products],
        "GLIDE_number": event_id,
    }


def _sort_events(events: Sequence[dict]) -> List[dict]:
    dated = [event for event in events if event.get("order_date")]
    undated = [event for event in events if not event.get("order_date")]
    dated.sort(key=lambda event: event["order_date"], reverse=True)
    return dated + undated


def _public_product(record: dict) -> dict:
    return {
        "product_id": record.get("product_id"),
        "event_id": None,
        "product_title": record.get("product_title"),
        "disaster_type": record.get("disaster_type"),
        "publication_date": record.get("publication_date"),
        "order_date": record.get("publication_date"),
        "order_date_kind": "publication" if record.get("publication_date") else "unknown",
        "resources": record.get("resources") or [],
    }


def _sync_resources(store, directory: Path, resources: Sequence[dict], previous: Sequence[dict], logger) -> List[dict]:
    previous_by_url = {item.get("url"): item for item in previous}
    stored = []
    for resource in resources:
        url = resource["url"]
        destination = directory / _safe_name(resource["filename"])
        earlier = previous_by_url.get(url)
        if earlier and earlier.get("local_path") and store.file_exists(earlier["local_path"]):
            if _valid_file(store, earlier["local_path"], earlier.get("format")):
                resource = dict(resource)
                resource["local_path"] = earlier["local_path"]
                resource["skipped"] = True
                stored.append(resource)
                continue
        if store.file_exists(str(destination)):
            _keep_previous_file(store, destination)
        _stream_with_retry(url, destination, store, logger)
        if not _valid_file(store, str(destination), resource.get("format")):
            if store.file_exists(str(destination)):
                store.remove(str(destination))
            raise ValueError(f"Downloaded file is not a valid {resource.get('format')}: {url}")
        resource = dict(resource)
        resource["local_path"] = str(destination)
        resource["downloaded_at"] = _utc_now()
        resource["skipped"] = False
        stored.append(resource)
    return stored


def _stream_with_retry(url: str, destination: Path, store, logger) -> None:
    last = None
    for pause in RETRY_PAUSES:
        if pause:
            time.sleep(pause)
        try:
            stream_to_store(url, destination, store, show_progress=False, timeout=120)
            return
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in RETRY_STATUS:
                raise
            last = exc
            logger.warning("UNOSAT download failed for %s: %s", url, exc)
        except requests.RequestException as exc:
            last = exc
            logger.warning("UNOSAT download failed for %s: %s", url, exc)
    raise last


def _valid_file(store, path: str, fmt: Optional[str]) -> bool:
    if not store.file_exists(path):
        return False
    raw = store.read_file(path)
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw or raw[:15].lstrip().lower().startswith((b"<!doctype", b"<html")):
        return False
    if fmt == "pdf":
        return raw.startswith(b"%PDF")
    if fmt in {"shp", "gdb", "xlsx", "zip"}:
        return raw.startswith(b"PK")
    if fmt == "xls":
        return raw.startswith(b"\xd0\xcf\x11\xe0")
    if fmt == "kml":
        return b"<kml" in raw[:500].lower() or raw.startswith(b"<?xml")
    return True


def _keep_previous_file(store, destination: Path) -> None:
    raw = store.read_file(str(destination))
    stamp = _utc_now().replace(":", "").replace("-", "")
    previous = destination.with_name(f"{destination.stem}.previous-{stamp}{destination.suffix}")
    store.write_file(str(previous), raw)


def _files_present(store, meta: dict) -> bool:
    resources = meta.get("resources") or []
    if not resources:
        return bool(meta.get("detail_fingerprint"))
    return all(
        item.get("local_path") and store.file_exists(item["local_path"])
        for item in resources
    )


def _write_event_files(store, country_dir: Path, catalogue: dict) -> None:
    for event in catalogue.get("events") or []:
        event_id = event.get("event_id")
        if not event_id:
            continue
        _write_json(store, country_dir / "events" / _safe_name(event_id) / "metadata.json", event)


def _detail_or_none(session, api_url: str, product_id: int, logger):
    try:
        return fetch_detail(session, api_url, product_id), None
    except (requests.RequestException, ValueError) as exc:
        logger.warning("UNOSAT product %s details failed: %s", product_id, exc)
        return None, str(exc)


def _summary_from_record(record: dict) -> dict:
    return {
        "product_id": record["product_id"],
        "title": record.get("product_title"),
        "glide": record.get("GLIDE_number"),
        "created_at": record.get("raw_created_at"),
        "activation_type": record.get("disaster_type"),
        "area_event_name": record.get("geographical_location"),
        "latitude": record.get("latitude"),
        "longitude": record.get("longitude"),
    }


def _country_labels(iso3: str, country) -> List[str]:
    labels = [country.name, iso3]
    for attr in ("official_name", "common_name"):
        value = getattr(country, attr, None)
        if value:
            labels.append(value)
    labels.extend(AREA_ALIASES.get(iso3, ()))
    return labels


def _normalise(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = [word for word in text.split() if word != "the"]
    return " ".join(words)


def _dates_in_phrase(phrase: str) -> List[str]:
    found = []
    for token in DATE_IN_TEXT_RE.findall(phrase):
        parsed = _parse_written_date(token)
        if parsed is not None:
            found.append(parsed)
    return found


def _parse_written_date(token: str) -> Optional[str]:
    cleaned = token.replace(",", " ")
    cleaned = re.sub(r"([A-Za-z]{3})\.", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _date_from_timestamp(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _administrative_area(description: Optional[str]) -> Optional[str]:
    if not description:
        return None
    found = []
    for name, kind in ADMIN_RE.findall(description):
        label = f"{name} {kind}"
        if label not in found:
            found.append(label)
    if not found:
        return None
    return "; ".join(found)


def _service_urls(event: dict) -> List[dict]:
    urls = []
    for field in ("wmap_link", "wms_link"):
        link = event.get(field)
        if link and str(link).strip():
            urls.append({"field": field, "url": link})
    return urls


def _resource(url: str, filename: str, fmt: str) -> dict:
    return {"url": url, "filename": filename, "format": fmt, "local_path": None}


def _format_for(filename: str, default: str) -> str:
    suffix = Path(filename).suffix.lower()
    return {
        ".pdf": "pdf",
        ".zip": default if default in {"shp", "gdb"} else "zip",
        ".kml": "kml",
        ".kmz": "kmz",
        ".geojson": "geojson",
        ".json": "geojson",
        ".tif": "geotiff",
        ".tiff": "geotiff",
        ".csv": "csv",
        ".xls": "xls",
        ".xlsx": "xlsx",
        ".jpg": "preview-image",
        ".jpeg": "preview-image",
        ".png": "preview-image",
    }.get(suffix, default)


def _absolute_url(api_url: str, link: str) -> str:
    """A site-relative path, or an absolute URL embedded in a bad catalogue path."""
    text = str(link).strip()
    if text.lower().startswith(("http://", "https://")):
        return text
    embedded = re.search(r"https?://\S+", text)
    if embedded:
        return embedded.group(0)
    return urljoin(api_url, text)


def _link_ok(value) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return False
    return _file_name_ok(text.rstrip("/").split("/")[-1])


def _file_name_ok(value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "none", "null"}


def _blank_to_none(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "no glide"}:
        return None
    return text


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "file"


def _fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _dedupe_rows(rows: Sequence[dict]) -> List[dict]:
    found = {}
    for row in rows:
        summary = summarise_product(row)
        if summary is not None:
            found[summary["product_id"]] = row
    return list(found.values())


def _unique(values: Iterable[str]) -> List[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _country_dir_from_path(path: Path) -> Optional[Path]:
    parts = path.parts
    if "products" in parts:
        return Path(*parts[: parts.index("products")])
    if "events" in parts:
        return Path(*parts[: parts.index("events")])
    if path.name == "catalogue.json":
        return path.parent
    return None


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def _request(session, method: str, url: str, timeout: float = 60, **kwargs):
    last = None
    for pause in RETRY_PAUSES:
        if pause:
            time.sleep(pause)
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last = exc
            continue
        if response.status_code in RETRY_STATUS:
            last = requests.HTTPError(f"{response.status_code} for {url}", response=response)
            continue
        response.raise_for_status()
        return response
    raise last


def _read_json(store, path: Path) -> Optional[dict]:
    if not store.file_exists(str(path)):
        return None
    return json.loads(store.read_file(str(path), encoding="utf-8"))


def _write_json(store, path: Path, payload: dict) -> None:
    store.write_file(
        str(path),
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
