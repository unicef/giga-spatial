# OSM extracts are chosen by polygon, not by ISO code

`OSMRoadsHandler` chooses the smallest Geofabrik extract whose polygon covers the source's area, using the polygons in Geofabrik's `index-v1.json`. A country's ISO code is only a fast path, and a country source skips the crop only when the extract is that country's own file: exactly that one code in the index, and not on a curated list of extracts that hold several countries. The index's codes cannot be trusted alone. In October 2026, five French and Pacific territories were coded `VU`, Saudi Arabia, Singapore, and Brunei had no extract of their own, and `malaysia-singapore-brunei` carried only Malaysia's code.

## Considered options

- **Look extracts up by ISO code only.** Rejected for the errors and gaps above.
- **Treat every country source as matching its file (`units_match_country = True`).** Rejected: a country inside a multi-country or regional extract would come back with its neighbours' roads.
- **Compare the extract's area with the country's to detect multi-country files.** Rejected: Geofabrik's polygons run past national borders by varying amounts, so no ratio separates the cases reliably.

## Consequences

- A country's own file and its sub-region files are returned uncropped, with the small leftovers across borders that ADR 0001 accepts for datasets published per country. Passing the country's boundary geometry as the source gives an exact crop.
- The curated list lives in the module and is tested against a stored copy of the index, so a new multi-country extract shows up as a failing test.
- Continent extracts (Africa is 7.4 GiB) are used only when named.
