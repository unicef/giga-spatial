"""
Geofabrik's OpenStreetMap extracts: the index, which extracts cover a source, and their dated files.

Facts about Geofabrik's server and index (October 2026) are in section 3 of
``docs/design/osm-roads-handler.md``; the choice of extracts is ADR 0007.
"""

import datetime
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import numpy as np
import requests
import shapely

from gigaspatial.handlers.base.fetch import TIMEOUT

GEOFABRIK_URL = "https://download.geofabrik.de"
LATEST = "-latest.osm.pbf"
EXTRACT_SUFFIX = ".osm.pbf"
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATED_NAME = re.compile(r"-(\d{6})\.osm\.pbf$")

# Extracts holding more than one country, which a country source must be cropped from.
# Several carry one ISO code or none in Geofabrik's index; tested against a stored copy of it.
MULTI_COUNTRY_EXTRACTS = frozenset(
    {
        "alps",
        "american-oceania",
        "britain-and-ireland",
        "dach",
        "gcc-states",
        "haiti-and-domrep",
        "ireland-and-northern-ireland",
        "israel-and-palestine",
        "malaysia-singapore-brunei",
        "sea",
        "senegal-and-gambia",
        "south-africa-and-lesotho",
    }
)
# ISO codes the index gets wrong (October 2026): these territories carry Vanuatu's or the Marshall Islands' code.
CODE_FIXES = {
    "ile-de-clipperton": (),
    "pitcairn-islands": ("PN",),
    "polynesie-francaise": ("PF",),
    "tokelau": ("TK",),
    "wallis-et-futuna": ("WF",),
}


class GeofabrikIndex:
    """
    Geofabrik's extracts, with their polygons and ISO codes (corrected by ``CODE_FIXES``).

    A continent is a root extract without an ISO code, such as ``africa``;
    Russia and Antarctica are roots with a code and count as countries.
    """

    def __init__(self, collection: Dict[str, Any]):
        features = collection["features"]
        properties = [f["properties"] for f in features]
        self.ids: List[str] = [p["id"] for p in properties]
        self._position = {extract: i for i, extract in enumerate(self.ids)}
        self.parents = [p.get("parent") for p in properties]
        self.codes = [CODE_FIXES.get(p["id"], tuple(p.get("iso3166-1:alpha2") or ())) for p in properties]
        self.subdivisions = [tuple(p.get("iso3166-2") or ()) for p in properties]
        self.urls = [p["urls"]["pbf"] for p in properties]
        self.geometries = np.array(
            [shapely.geometry.shape(f["geometry"]) if f.get("geometry") else None for f in features], dtype=object
        )
        self.continents = np.array([parent is None and not codes for parent, codes in zip(self.parents, self.codes)])

    def own_extract(self, country: str) -> Optional[str]:
        """The country's own extract: the only one carrying just its code, and not holding other countries."""
        own = [
            extract
            for extract, codes in zip(self.ids, self.codes)
            if codes == (country,) and extract not in MULTI_COUNTRY_EXTRACTS
        ]
        return own[0] if len(own) == 1 else None

    def regions(self, country: str, own: Optional[str]) -> List[str]:
        """The sub-region extracts of a country's own extract.

        These are its children, or for a country whose regions are filed
        elsewhere (the US states sit under ``north-america``), the extracts
        coded with its ISO 3166-2 subdivisions.
        """
        if own is None:
            return []
        children = [extract for extract, parent in zip(self.ids, self.parents) if parent == own]
        if children:
            return children
        return [
            extract
            for extract, subdivisions in zip(self.ids, self.subdivisions)
            if subdivisions and all(code.startswith(f"{country}-") for code in subdivisions)
        ]

    def require(self, extracts: Iterable[str]) -> List[str]:
        """The named extracts, which must exist."""
        extracts = list(extracts)
        unknown = [extract for extract in extracts if extract not in self._position]
        if unknown:
            raise ValueError(f"Unknown Geofabrik extracts: {unknown}")
        return extracts

    def cover(self, area: shapely.Geometry) -> List[str]:
        """The smallest extracts that together cover an area, continents left out.

        Extracts are taken from the smallest up while part of the area is
        still uncovered; any extract the others already cover is then dropped.
        """
        present = np.array([g is not None for g in self.geometries])
        if not present.any():
            raise ValueError("This copy of Geofabrik's index has no polygons")
        candidates = np.flatnonzero(present & ~self.continents)
        touching = candidates[shapely.intersects(self.geometries[candidates], area)]
        order = touching[np.argsort(shapely.area(self.geometries[touching]), kind="stable")]
        chosen: List[int] = []
        uncovered = area
        for i in order:
            if uncovered.is_empty or uncovered.area <= area.area * 1e-9:
                break
            if self.geometries[i].intersects(uncovered):
                chosen.append(int(i))
                uncovered = uncovered.difference(self.geometries[i])
        for i in sorted(chosen, key=lambda j: -self.geometries[j].area):
            others = [self.geometries[j] for j in chosen if j != i]
            if others and area.intersection(self.geometries[i]).difference(shapely.union_all(others)).area <= area.area * 1e-9:
                chosen.remove(i)
        if not chosen:
            raise ValueError(
                "No country or regional Geofabrik extract covers the area; "
                'name a continent with extracts=["africa"] to use one'
            )
        return [self.ids[i] for i in chosen]

    def unit(self, extract: str, matches_country: bool) -> Dict[str, Any]:
        url = self.urls[self._position[extract]]
        return {"id": extract, "url": url, "path": extract_path(url), "matches_country": matches_country}


def extract_path(url: str) -> str:
    """An extract's path in Geofabrik's folders, such as ``africa/rwanda``."""
    path = urlparse(url).path.lstrip("/")
    return path[: -len(LATEST)] if path.endswith(LATEST) else path


def dated_url(latest_url: str, date: str) -> str:
    """The dated file for a ``-latest`` URL: ``rwanda-latest.osm.pbf`` of 2026-10-27 is ``rwanda-261027.osm.pbf``."""
    stamp = datetime.date.fromisoformat(date).strftime("%y%m%d")
    return f"{latest_url[: -len(LATEST)]}-{stamp}.osm.pbf"


def latest_date(url: str) -> str:
    """
    The date of Geofabrik's latest file for an extract, from its ``-latest`` URL, as ``YYYY-MM-DD``.

    ``-latest`` redirects to the dated file, except for large extracts that
    redirect to a mirror under the same name; their date is that of the
    extract's replication state.

    Raises:
        requests.RequestException: If Geofabrik cannot be reached.
        ValueError: If neither gives a date.
    """
    response = requests.head(url, allow_redirects=False, timeout=TIMEOUT)
    response.raise_for_status()
    match = _DATED_NAME.search(response.headers.get("location", ""))
    if match:
        return datetime.datetime.strptime(match.group(1), "%y%m%d").date().isoformat()
    state = requests.get(url[: -len(LATEST)] + "-updates/state.txt", timeout=TIMEOUT)
    state.raise_for_status()
    match = re.search(r"^timestamp=(\d{4}-\d{2}-\d{2})", state.text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot tell the date of {url}: no dated redirect or replication timestamp")
    return match.group(1)
