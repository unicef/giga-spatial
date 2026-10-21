# Handler quartet contract

Shared by the `judge-handler-base` and `build-handler` skills. Re-read the code every session. This file is a checklist. If a method has moved, changed meaning, or lost the subclass that illustrated it, trust the code and say so. Do not score a later refactor against a leak that no longer exists.

## Scope

The file-pipeline quartet only:

- `BaseHandlerConfig` turns a source into a search key, the search key into data units, and units into stored paths. It owns the private unit cache.
- `BaseHandlerDownloader` fetches units into the data store.
- `BaseHandlerReader` loads stored paths and may crop to the source.
- `BaseHandler` builds the three, sorts the caller's kwargs, and runs `ensure_data_available`, then `load_data`.

They live in the `gigaspatial/handlers/base/` package:

| Module | Holds |
| --- | --- |
| `config.py` | `BaseHandlerConfig` |
| `downloader.py` | `BaseHandlerDownloader`, the thread pool and the per-unit error contract |
| `reader.py` | `BaseHandlerReader`, loading and cropping |
| `handler.py` | `BaseHandler`, wiring and the public entries |
| `source.py` | `classify_source`, `is_path_source`: the one source classifier |
| `resolution.py` | `UnitResolution` and `UnitCache` |
| `options.py` | `ResolveOptions`, `DownloadOptions`, `ReadOptions`, `RasterReadOptions`, `sort_options` |
| `fetch.py` | `stream_to_store`, `extract_zip_to_store`, `tiles_intersecting` |

The base is internal ([ADR 0002](../../docs/adr/0002-handler-base-in-the-api-reference.md)): handlers in this repository are its only subclasses, and renames carry no aliases.

Out of scope: fetchers, GEE, image downloaders, BigQuery clients, and `AdminBoundaries`. `AdminBoundaries` is a helper the config calls for a country source. It is not a dataset handler.

## Invariants

1. **A search key and a search geometry are different things.** `get_search_key(source, options)` returns whatever the unit lookup understands: an ISO code, a filter dict, a geometry, or `None` for a global dataset. Units are looked up from the key alone. The search geometry is always a Shapely geometry in `config.crs`, built by `extract_search_geometry`, and it is the only thing cropping uses. A key that is a geometry must be the search geometry.
2. **Anything that changes which file gets fetched is part of the unit.** `download_data_unit` and the path hooks never need an option to know what to fetch. Example: Google Open Buildings puts `data_type` (polygons or points) in the unit dict. The same holds for stored paths: a config field or resolve option that picks the file (year, resolution, product, data type) shows in the stored path, since two handlers may share a base path and data store.
3. **Kwargs stop at the public entry.** `load_data`, `download_and_load`, `ensure_data_available`, `get_available_data_info`, `reader.load`, `reader.resolve_source_paths`, `downloader.download`, `downloader.download_data_units`, `config.get_relevant_data_units`, `config.get_data_unit_paths` and `config.get_search_geometry` call `sort_options`. An unknown key raises `TypeError`. Below the entry, each component receives only its own frozen options object (its `options_type`). No hook takes `**kwargs`; `tests/handlers/base/test_handler_conformance.py` enforces that.
4. **`get_stored_paths(unit, options) -> list[Path]` returns the stored files** of a unit: what ends up in the data store, not the remote archive. The default wraps `get_data_unit_path`. `is_unit_available(unit, paths)` decides whether a unit needs a download; the default requires every path to exist.
5. **The unit cache is private to the base.** `UnitCache` is keyed by the source key plus the resolve options, which is why resolve options must be hashable. It stores a `UnitResolution(units, search_key)` and the search geometry separately. Handlers change resolution only through `get_search_key` and `get_relevant_data_units_by_key`, never by overriding `get_relevant_data_units` or touching the cache.
6. **`download_data_unit(unit, options)` returns the stored paths, or raises.** It runs in `config.n_workers` threads, so it must not share mutable state unguarded. The base re-raises programming errors (`TypeError`, `AttributeError`, `NameError`), logs any other failure for that unit, carries on with the rest, and logs a "N of M failed" summary. `ensure_data_available` returns `False` only when files are still missing after the attempt.
7. **Persistence goes through `DataStore`.** Existence checks and dataset bytes use `data_store`. A downloader may stage in a temp directory, then write through the store; the `fetch.py` helpers do that and remove partial files on failure. The handler, config, downloader and reader share one store for a run.
8. **The base earns its place.** Behavior stays on the base only when deleting it would copy the same complexity into several handlers, and at least two adapters make the seam real. One adapter is a hypothetical seam.

## Cropping

`crop_to_source=True` crops to the search geometry, with declared exceptions:

- Path sources are returned as loaded, logged at debug level.
- Country sources are not cropped on configs with `units_match_country = True` (WorldPop, OpenCellID, Google-MS combined, MS Buildings, HDX, RWI): their units are already country files.
- A source with no geometry, such as an HDX filter dict, raises `ValueError`.

## Not required

Do not fail a base or a handler for lacking these. Ask whether more than one adapter needs the same hook. If only one does, it stays on that handler.

- Pydantic on every config. The base is a stdlib dataclass. Option classes in Pydantic-dataclass modules use `@dataclasses.dataclass(frozen=True)`.
- `.npy` or centroid caches. The Google-MS combined centroid cache is that handler's concern.
- Streaming or chunked reads on the base. Deferred until two readers need the same hook.

## Method map

Re-verify each of these. Note the override, the return, and whether it still takes only its options object.

### `BaseHandlerConfig`

- `options_type`, `units_match_country`. Class variables. `options_type` is a frozen `ResolveOptions` subclass.
- `get_relevant_data_units(source, force_recompute=False, **kwargs)`. Public entry. Sorts kwargs, then `_resolve_units`: cache, `get_search_key`, `get_relevant_data_units_by_key`. Not an override point.
- `get_search_key(source, options)`. Hook. Default: the search geometry.
- `get_relevant_data_units_by_key(search_key, options)`. Abstract hook.
- `get_data_unit_path(unit, options=None)`. Hook for one file per unit. `options` is `None` outside resolution.
- `get_stored_paths(unit, options=None)`. Hook for units stored as several files.
- `is_unit_available(unit, paths)`. Hook. WorldPop overrides it, since files in a school-age zip are known only after extraction.
- `get_data_unit_paths(units, **kwargs)`. Public entry. Flattens `get_stored_paths` over units; accepts a DataFrame of units.
- `extract_search_geometry(source, options=None)`. Hook that must return a Shapely geometry in `self.crs`, or raise `ValueError`. Reads the source in its own CRS, else `options.crs`, else EPSG:4326.
- `get_search_geometry(source, **kwargs)`, `get_cached_search_geometry(source, **kwargs)`, `clear_unit_cache()`. Concrete.
- `crs`. Property, default `EPSG:4326`.

### `BaseHandlerDownloader`

- `options_type`. A frozen `DownloadOptions` subclass; `show_progress` is the base option.
- `config_type`. Class variable, the handler's config class. `__init__(config=None, data_store=None, logger=None)` takes a config, a dict of its fields, or `None` for its defaults; downloaders and readers are public building blocks, built directly as well as by the handler (`srtm_manager.py`). The conformance test requires it on every handler component.
- `download_data_unit(unit, options)`. Abstract hook. See invariant 6.
- `download_data_units(units, **kwargs)`. Public entry. Thread pool sized `min(n_workers, len(units))`, flattens returned paths.
- `download(source, **kwargs)`. Public entry. Sorts into resolve and download options, resolves, downloads.

### `BaseHandlerReader`

- `options_type`. A frozen `ReadOptions` subclass; raster readers use `RasterReadOptions`.
- `config_type`. As on the downloader.
- `load_from_paths(paths, options)`. Abstract hook. Receives paths as strings.
- `load(source, crop_to_source=False, **kwargs)`. Public entry. Resolve paths, `load_from_paths`, optional crop (see Cropping).
- `resolve_source_paths(source, **kwargs)`. Public entry. Stored paths of the source's units, or the given paths for a path source. Callers outside the reader (Google-MS combined, the POI and zonal generators, `buildings_engine.py`) use it and rely on `Path.stem` for stored paths.
- `resolve_by_paths(paths)`. Concrete. Wraps one path in a list.
- `_load_raster_data`, `_load_tabular_data`, `crop_to_geometry`. Concrete helpers. `crop_to_geometry` still takes free kwargs (`crs`, clip options); it is a helper, not a seam.

### `BaseHandler`

- `__init__(config=None, downloader=None, reader=None, data_store=None, logger=None, **config_kwargs)`. Config fields go to `create_config`; with `config=` they raise `TypeError`. A handler `__init__` that takes `**kwargs` must pass them to `super().__init__`, and its `create_config` to the config. Configs set `extra="forbid"`, and an unknown field becomes `TypeError`. A handler with no named config fields needs no `__init__` (SRTM, Google Open Buildings). A handler that names fields in its signature (GHSL `product`, WorldPop `year`) stores them on `self` for `create_config` and passes only `**kwargs` on: named arguments always carry a value, even a default, so passing them to `super().__init__` would raise next to `config=`.
- `create_config`, `create_downloader`, `create_reader`. Abstract factories. Thin overrides are expected.
- `ensure_data_available(source, force_download=False, **kwargs)`. Public entry, returns `bool`. Path sources are checked for existence, never downloaded. Other sources resolve units (`force_download` also recomputes them), download the units `is_unit_available` rejects, then re-check.
- `load_data(source, crop_to_source=False, ensure_available=True, **kwargs)`. Public entry. Raises `RuntimeError` when availability fails.
- `download_and_load(source, crop_to_source=False, force_download=False, **kwargs)`. Public entry.
- `get_available_data_info(source, **kwargs)`. Public entry. Unknown kwargs raise before its error-swallowing `try`.
- `select_options(kwargs, *components)`. For handler methods and generators that share one kwargs dict with other consumers. Defaults to all three components; pass `handler.config` for resolve options only.

## Registrations

A handler module is not done until every row below holds. Rows with a check fail a test when missed; the others fail silently.

| Registration | Where | Checked by |
| --- | --- | --- |
| Module listed in `HANDLER_MODULES` | `tests/handlers/base/test_handler_conformance.py` | Nothing; a missing module skips every conformance rule |
| The four classes in the `attach` map and the `TYPE_CHECKING` block | `gigaspatial/handlers/__init__.py` | `tests/test_public_api.py` |
| The handler class in the `attach` map and the `TYPE_CHECKING` block | `gigaspatial/__init__.py` | `tests/test_public_api.py` |
| A `DATA_TYPES` key for the `base_path` default | `gigaspatial/config.py` | Import fails with `KeyError` |
| Credentials, when needed: a setting, a sample line, a table row | `gigaspatial/config.py`, `.env_sample`, `docs/getting-started/configuration.md` | Nothing |
| A section with the handler's public methods | `docs/api/handlers.md` | Nothing |
| An entry under `Added` for the next release | `CHANGELOG.md` | Nothing |
| Offline tests, and a `live`-marked test that the resolved units exist remotely | `tests/handlers/<module path>/` | Nothing |

## Finding subclasses

Search for classes that inherit the four bases. Do not treat the list below as complete. `tests/handlers/base/test_handler_conformance.py` walks the modules in `HANDLER_MODULES`; a handler module missing from that list is a finding.

- `gigaspatial/handlers/ghsl.py`
- `gigaspatial/handlers/google_open_buildings.py`
- `gigaspatial/handlers/microsoft_global_buildings.py`
- `gigaspatial/handlers/google_ms_combined_buildings.py`
- `gigaspatial/handlers/worldpop.py`
- `gigaspatial/handlers/ookla_speedtest.py`
- `gigaspatial/handlers/opencellid.py`
- `gigaspatial/handlers/hdx.py`
- `gigaspatial/handlers/rwi.py` (via HDX)
- `gigaspatial/handlers/nasa/srtm.py`
- `gigaspatial/handlers/wri/aqueduct_flood.py`

## Leaks to watch for

The leaks this contract was first written against (a geometry hook returning country codes and filter dicts, `**kwargs` forwarded into downloaders that drop them, a bare list written into the unit cache, a positional `ensure_available`) were closed by the 2026-10 refactor. Each now has a test in `tests/handlers/base/`. Look for their return in a new handler or a base change:

- **Fake abstraction.** A hook whose return has two meanings, such as `extract_search_geometry` returning something that is not a geometry, or a search key that is a geometry other than the search geometry.
- **Contract break, options.** A hook that grows `**kwargs`, reads an option it does not declare, or an option class that is not frozen or not hashable.
- **Contract break, unit.** A download or path hook that needs an option to know which file it fetches (invariant 2).
- **Contract break, cache.** Any override of `get_relevant_data_units` or access to `_units`.
- **Swallowed failure.** A `download_data_unit` that logs and returns `None` instead of raising, which hides the failure from the base's summary.
- **One-off on the base.** `_load_raster_data` and `_load_tabular_data` sit on every reader while raster and table handlers each use one. Worth a finding only while both stay on the shared reader with no seam between them.
- **Global side effects.** A handler that changes process-wide state, such as the default SSL context or warning filters, to make one request work. Scope it to that request (see GHSL `verify_tiles_ssl`).
- **Contract break, stored path.** A config field or resolve option that picks the file but is missing from the stored path, so two handlers overwrite or reuse each other's files (invariant 2).
- **Leftover boilerplate.** Code the base already does. Not a contract break, but it makes a handler look like it needs code it does not. Two shapes:
  - A downloader or reader `__init__` that turns a dict or `None` into a config. Declare `config_type` instead; the base builds it.
  - An override whose body only calls `super()` with the same signature and defaults. An override that changes a default is not boilerplate: Ookla `reader.load` defaults `source` to `None`, Aqueduct `load_data` defaults `crop_to_source` to `True`.
