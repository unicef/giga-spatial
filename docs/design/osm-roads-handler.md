# Design: the OSM roads handler

This is the agreed design for `OSMRoadsHandler`, which builds country-scale road networks from Geofabrik's OpenStreetMap extracts. It returns the same `SpatialNetwork` as `OSMRoadsFetcher`, which queries Overpass and suits cities, districts, and areas around schools. It records the decisions from the design round of 28–29 October 2026 and splits the work into sub-tasks. Each sub-task starts by reading this document, `GLOSSARY.md`, the ADRs listed below, `docs/design/spatial-network.md`, and `.agents/reference/handler-contract.md`.

Read first:

- `GLOSSARY.md` and `CONTEXT.md` (local files, gitignored): extract, road network, handler, source, data store.
- [ADR 0001](../adr/0001-search-key-and-search-geometry.md): search key and search geometry; small leftovers for datasets published per country.
- [ADR 0003](../adr/0003-public-import-paths.md): public import paths and lazy loading.
- [ADR 0005](../adr/0005-spatial-network-storage.md): how a saved network is laid out.
- [ADR 0006](../adr/0006-osm-extract-storage.md): extracts stored by date, uncropped networks beside them, cropping at load.
- [ADR 0007](../adr/0007-osm-extract-choice.md): extracts chosen by polygon, with a curated list of multi-country extracts.

This page is not in the docs site navigation.

## 1. Why a handler

- Overpass is a shared service. A country-scale fetch takes many queries (osmnx splits areas above 2,500 km²), can take hours, runs into fair-use limits (roughly 10,000 queries and 1 GB a day), and needs osmnx's networkx graph in memory: about 2.3 GB for unsimplified Rwanda. Every public Overpass server was unreachable on 26–27 October 2026.
- An extract is a file kept in a data store and reused, which the glossary calls a handler. Users need both small areas (the fetcher) and whole countries or provinces (this handler).
- The handler reads extracts with pyosmium and builds the node and edge tables directly, without osmnx or networkx. On Rwanda, pyosmium read every road way with coordinates in 8.4 s, and `simplify` takes 9 s; the route through OSM XML and osmnx took 300 s.

## 2. Names, modules, and dependencies

| Item | Decision |
|---|---|
| Classes | `OSMRoadsConfig`, `OSMRoadsDownloader`, `OSMRoadsReader`, `OSMRoadsHandler` in `gigaspatial/handlers/osm_roads.py` |
| Public names | `from gigaspatial import OSMRoadsHandler`; config, downloader, and reader from `gigaspatial.handlers` (ADR 0003) |
| Dependency | `osmium` (pyosmium) added to `giga-spatial[roads]`, imported lazily with the usual `ImportError` naming the extra. Wheels exist for Python 3.10 to 3.14 on macOS, Linux, and Windows |
| Data type | a new `DATA_TYPES` key `osm`, so `config.get_path("osm", "bronze")` works |
| Suffix | `.pbf` added to `DATA_FILE_SUFFIXES` in `gigaspatial/handlers/base/source.py`; today `"x.osm.pbf"` is classified as a country |

The handler follows the handler contract: the config's only required hook is `get_relevant_data_units_by_key`; the downloader implements `download_data_unit`; the reader implements `load_from_paths`; options are frozen dataclasses with no `**kwargs`.

## 3. Geofabrik facts (October 2026)

- `https://download.geofabrik.de/index-v1.json` (3.8 MB) lists 554 extracts, each with `id`, `parent`, `name`, `urls.pbf`, and a MultiPolygon. 191 carry `iso3166-1:alpha2` codes; 71 carry `iso3166-2`. A lighter `index-v1-nogeom.json` exists.
- Every country split into sub-regions also has a country file (China 33 regions, France 27, Spain 18, Germany 16, India 6, Brazil 5, and others). The US is a country file `us` plus 53 state extracts under `north-america`.
- Multi-country extracts: `gcc-states`, `haiti-and-domrep`, `israel-and-palestine`, and `senegal-and-gambia` carry several codes; `malaysia-singapore-brunei` and `ireland-and-northern-ireland` carry one; `south-africa-and-lesotho`, `dach`, `alps`, `britain-and-ireland`, and `sea` carry none.
- Code errors and gaps: `polynesie-francaise`, `wallis-et-futuna`, `tokelau`, `american-oceania`, and `ile-de-clipperton` are coded `VU`; `pitcairn-islands` is coded `MH`. Saudi Arabia, Singapore, Brunei, and about 33 other countries exist only inside larger extracts.
- Sizes: Rwanda 65 MiB, Ethiopia 134 MiB, Kenya 335 MiB, South Africa 404 MiB, Mexico 618 MiB, Nigeria 680 MiB, India 1.6 GiB, Indonesia 1.7 GiB, Brazil 2.0 GiB, Germany 4.6 GiB, Africa 7.4 GiB.
- `-latest.osm.pbf` redirects (307) to a dated file such as `rwanda-261027.osm.pbf`; some large files redirect to a mirror. `.md5` files exist for both names. Daily dated files are kept about a week, then monthly and yearly snapshots. Files update daily around 21:00 CET. The PBF header carries the replication timestamp, so a stored file's age can be read offline.

## 4. Choosing extracts (ADR 0007)

- The index is downloaded once and stored at `osm/geofabrik/index-v1.json`, refreshed with the same `refresh` and `max_age_days` options as extracts.
- For a source's area, choose the smallest extract whose polygon covers it. A source that needs more than one extract, such as an area crossing a border, loads each and combines them with `concat`.
- A country's ISO code is a fast path: when exactly one extract carries that one code and the extract is not on the multi-country list, it is the country's own file.
- The curated list of multi-country extracts lives in the module and is tested against a stored copy of the index.
- Continent extracts are used only when named with `extracts=`.
- `regions=False` (default): a country source uses the country file. `regions=True`: it uses all of the country's sub-region files, each built and saved on its own and combined at load. `extracts=[...]` names Geofabrik extract IDs directly and overrides the choice.

## 5. Storage (ADR 0006)

```
osm/geofabrik/index-v1.json
osm/geofabrik/2026-10-27/africa/rwanda.osm.pbf
osm/geofabrik/2026-10-27/africa/rwanda.networks/drive_service-simplified/
osm/geofabrik/2026-10-27/south-america/brazil.osm.pbf
osm/geofabrik/2026-10-27/south-america/brazil/nordeste.osm.pbf
osm/geofabrik/2026-10-27/south-america/brazil/nordeste.networks/drive_service-simplified/
```

- The date folder is the extract's own date as an ISO date; below it, the path mirrors Geofabrik's URL path.
- The data unit is `{id, url, date}`; its stored path is the `.pbf`. The base's availability check covers the `.pbf` only.
- Network folders are a cache the reader derives from the `.pbf` path, one per variant: the network type or a short hash of the tag filter, and `simplified` or `raw`. A missing network is built from the extract and saved; it never triggers a download.
- A stored network always covers its whole extract, with every connected part. It is never cropped.
- Loading uses the newest stored date holding the extract, unless `date=` pins one. Old dates are never deleted automatically.

## 6. Downloads

- Resolve `-latest` to the dated file name, stream it into the data store with `fetch.stream_to_store`, and verify it against Geofabrik's `.md5`.
- `refresh=False` (default) reuses the stored extract; `refresh=True` downloads today's file; `max_age_days=None` (default) never refreshes automatically, and a number refreshes extracts older than that.
- The network's metadata records the extract's ID, date, URL, and replication timestamp. Reproducing an old result needs the stored extract, since Geofabrik deletes daily files after about a week; the docs say so.

## 7. Building a network from an extract

The same network as osmnx for every `network_type`, tested against osmnx on the Kigali test extract.

1. Read the ways without coordinates (0.4 s for Rwanda): way IDs, node references, and the kept tags. Apply the filter in Python.
2. Read only the referenced nodes' coordinates, into numpy arrays.
3. Cut ways into edges and compute lengths in numpy; build the `SpatialNetwork` tables directly; run `simplify` for the simplified variant.

osmnx rules to reproduce exactly (osmnx 2.0.2):

- Every preset requires `highway`, excludes `area=yes`, and, except `all`, excludes `access=private`. The presets' excluded `highway` values and extra conditions are in `osmnx/_overpass.py` (`drive` 75–82, `drive_service` 85–92, `walk` 99–106, `bike` 110–115, `all_public` 119–123, `all` 127–130). Their conditions are Overpass `!~` patterns: unanchored regular expressions, and a way without the key passes (so `walk`'s `motor` also drops `motorway`).
- `tags=` replaces Overpass's `custom_filter`: a dict of key to allowed values, such as `{"highway": ["primary", "secondary"]}`.
- One-way: `oneway` in `{"yes", "true", "1", "-1", "reverse", "T", "F"}`, or `junction=roundabout`; `walk` is two-way everywhere. Reversed: `oneway` in `{"-1", "reverse", "T"}`, and the node order is reversed. Two-way ways get both directions, the reverse edges with `reversed=True`.
- Kept tags: osmnx's `useful_tags_way` (`access`, `area`, `bridge`, `est_width`, `highway`, `junction`, `landuse`, `lanes`, `maxspeed`, `name`, `oneway`, `ref`, `service`, `tunnel`, `width`) and `useful_tags_node` (`highway`, `junction`, `railway`, `ref`). Consecutive duplicate nodes are dropped.
- Lengths: great-circle per segment before simplification, summed by `simplify`. `street_count` per node as osmnx's `count_streets_per_node`.
- The node coordinate index: pyosmium's in-memory `flex_mem` up to about 1 GiB of extract (about 16 bytes per node in the file, about 2 GB for Nigeria), and the disk-backed `sparse_file_array` above it. `node_index=` overrides the choice. Dense indexes are never used: on Rwanda's 64-bit node IDs they need about 115 GB.
- Attributes are normalised with `normalize_attributes(edges, int_lists=("osmid",))`, as in the fetcher.

## 8. Loading for a source

`OSMRoadsHandler.load_data(source)` returns a `SpatialNetwork`; `crop_to_source` defaults to `True`. For each chosen extract, load or build its network, then:

1. Clip to the source's area, with the fetcher's rules: a country's boundary, a geometry as given, a frame's union, or points as hull, bounding box, or buffers widened by `buffer` meters; nodes inside the area are kept, and with `truncate_by_edge`, edges with one end inside. **No crop** when the source is a country and the extract is its own file or one of its sub-region files (ADR 0007), accepting small leftovers across borders as ADR 0001 does. A country's boundary geometry as the source gives an exact crop.
2. Keep the largest connected part, unless `retain_all=True`.
3. Combine several extracts with `concat`.

- The reader overrides `crop_to_geometry` to clip the network; the base's version returns unknown types uncropped without a warning. `units_match_country` stays `False`, because the no-crop decision depends on the extract.
- pyosmium cannot read Python file objects. With a remote data store, the reader copies the extract to a local temporary file, reads it, and deletes it; a `LocalDataStore` path is read in place. Built networks are saved back to the data store.
- Path sources: a `.pbf` path loads that extract through the network cache; a network folder path loads that network as it is.

Parameters, matching the fetcher where they overlap: `network_type="drive_service"`, `tags=None`, `simplify=True`, `retain_all=False`, `truncate_by_edge=False`, `buffer=2000`, `area="hull"`, `regions=False`, `extracts=None`, `date=None`, `refresh=False`, `max_age_days=None`, `node_index=None`.

## 9. Performance targets

| Operation | Target |
|---|---|
| Build Rwanda's network from its extract | under 30 s (osmnx route: 300 s) |
| Nigeria (680 MiB) on a 16 GB machine | fits in memory with the default index |
| Load a stored network | under 1 s for Rwanda (ADR 0005 load: 0.48 s) |
| Extracts over 1 GiB | use the disk-backed index automatically |

## 10. Tests and fixtures

- A stored subset of `index-v1.json` with the countries the tests need (Rwanda, Brazil and its regions, the multi-country extracts, a mis-coded territory), plus a test that checks the curated multi-country list against a full stored index.
- A small Kigali `.pbf` cut from the Rwanda extract with pyosmium (a few hundred KB), and osmnx's network for the same file recorded beside it, built through `graph_from_xml` once by a script in `benchmarks/`.
- Downloads mocked offline; one `live` test downloads Rwanda's extract.

## 11. Sub-tasks

Each sub-task is its own chat and pull request on `feat/osm-roads-handler`, stacked on `feat/spatial-network` until PR #18 merges.

| # | Sub-task | Delivers |
|---|---|---|
| 1 | Scaffolding | `osmium` in `[roads]`; `.pbf` in `DATA_FILE_SUFFIXES`; `DATA_TYPES["osm"]`; the module skeleton; the index subset and Kigali `.pbf` fixtures, with the scripts that make them |
| 2 | Choosing extracts | storing the index; smallest covering extract by polygon with the ISO fast path; the own-country rule and curated multi-country list; `regions=` and `extracts=`; continents only when named |
| 3 | Downloads and storage | `-latest` to the dated file; streaming and checksum; the date-first layout; `refresh`, `max_age_days`, `date=`; the offline replication timestamp |
| 4 | Building a network | the three-pass read; osmnx's filters, one-way rules, kept tags, lengths, and `street_count`; `simplify`; the node index by size; the equivalence test on the Kigali extract; Rwanda under 30 s |
| 5 | Loading for a source | network folders per variant, built once and reused; clipping, the largest part, `concat`; the no-crop rule; local copies for remote stores; the reader's crop |
| 6 | Handler and docs | wiring `OSMRoadsHandler`, import registration, a live test, a country-scale section in the road networks guide, the API reference, and the changelog |

## 12. Deferred

- One entry point that picks the fetcher or the handler by area size.
- Other OSM datasets from the same stored extracts, such as buildings or points of interest.
- Updating stored extracts with Geofabrik's daily diffs (`pyosmium-up-to-date`) instead of downloading them again.
