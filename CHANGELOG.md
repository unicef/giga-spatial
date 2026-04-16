# Changelog

All notable changes to this project will be documented in this file.

## [v0.9.3] - 2026-04-16

### Changed

-   **DataStore-Aware Reading API (`gigaspatial/core/io/readers.py`)**
    -   Refactored `read_json`, `read_dataset`, and `read_datasets` to prioritize the file path as the first argument, making `data_store` an optional second argument that defaults to `LocalDataStore()` if not provided. This simplifies usage for local file operations: `df = read_dataset("data.shp")`.
    -   Updated `read_gzipped_json_or_csv` to also support an optional `data_store`.
    -   Systematically updated all downstream callers in `BaseHandler`, `AdminBoundaries`, `EntityTable`, `OpenCellIDReader`, and `HDXReader` to conform to the new argument order.

-   **`map_wp_pop`: Automatic Project-Specific Statistics (`gigaspatial/generators/`)**
    -   Implemented automatic statistic selection for the WorldPop population mapping method in both `PoiViewGenerator` and `ZonalViewGenerator`:
        -   **`degree_of_urbanization`**: Automatically uses `"median"` (for categorical class mapping).
        -   **`pop` / `age_structures`**: Automatically uses `"sum"` (for population count mapping).
    -   This ensures consistent and mathematically correct aggregation defaults across different WorldPop datasets.

-   **`TransmissionNode` Schema Update (`gigaspatial/core/schemas/transmission_node.py`)**
    -   Added `is_logical_node` (Optional[bool]) field to track whether a site hosts active transmission equipment.
    -   Implemented `_normalize_transmission_medium` in `TransmissionNodeProcessor` to standardize physical medium values (e.g., mapping "fibre" to "fiber").
    -   Integrated `BACKHAUL_ALIAS_MAP` from the `CellTower` schema to ensure consistent normalization of shared media types (Fiber, Microwave, Satellite) across infrastructure entities.

### Fixed

-   **`EntityProcessor._drop_duplicates`: Robust handling of unhashable columns (`gigaspatial/processing/entity_processor.py`)**
    -   Resolved a `TypeError` that occurred when deduplicating DataFrames containing unhashable types (e.g., `set`, `list`, `dict`) in non-geometry columns.
    -   Updated the deduplication pipeline to dynamically inspect `object` dtype columns and exclude those containing unhashable values from the comparison subset.
    -   Added defensive handling to return the DataFrame unchanged if no comparable (hashable) columns are identified, preventing crashes in `pd.DataFrame.drop_duplicates`.

-   **Centralized GEE Dependency Management (`gigaspatial/handlers/gee/`)**
    -   Refactored `GEEConfig` and `GEEProfiler` to utilize a single shared Earth Engine availability check, centralizing the handling of `earthengine-api` and `geemap` optional dependencies.
    -   Implemented robust `ImportError` protection for GEE-specific methods and improved type hint safety across the module, ensuring the core library remains functional without GEE installed.

-   **HDX Resource Matching: Prevent false positives for country codes**
    -   Modified `HDXConfig._match_pattern` in `gigaspatial/handlers/hdx.py` to support regex-based token matching, ensuring that country identifiers (e.g., ISO-2/ISO-3 codes) are matched as distinct components delimited by `_`, `-`, `/`, or `.` rather than arbitrary substrings. This prevents incorrect matching where a short code exists within a longer string or dataset identifier.
    -   Added `token_match` parameter to `HDXConfig.get_dataset_resources` to toggle this behavior.
    -   Enabled `token_match=True` by default in `RWIConfig.get_relevant_data_units` (`gigaspatial/handlers/rwi.py`) to resolve ambiguous resource matching across multi-country datasets.

### Documentation

-   **README overhaul (`README.md`)**
    -   Rewrote the **Quick Start** section to use correct, runnable code. The prior example referenced a non-existent `POIViewGenerator` class (correct name is `PoiViewGenerator`), used an undefined `points` variable, and duplicated the GHSL mapping call. Updated to use `GigaSchoolLocationFetcher` → `PoiViewGenerator` → enrichment chain, matching the actual v0.9.x API. Added a second grid-based example using `H3ViewGenerator`.
    -   Added optional dependency table covering `azure-storage-blob`, `snowflake-connector-python`, and `earthengine-api`/`geemap` with links to the new Configuration Guide.
    -   Replaced the vague **Key Features** prose with a concise, class-name-anchored bullet list (`TifProcessor`, `H3ViewGenerator`, `WPPopulationHandler`, etc.).
    -   Replaced the **Core Concepts** freeform list with a structured Markdown table mapping concept names to their concrete Python classes.
    -   Added a **Supported Datasets** table covering all nine source categories (Buildings, Population & Settlements, Network & Connectivity, POI, Humanitarian, Earth Observation, Giga, Admin Boundaries, Relative Wealth).
    -   Removed the redundant **View Generators** section (content consolidated into Core Concepts table).
    -   Updated installation link to point to the new [Configuration Guide](https://unicef.github.io/giga-spatial/getting-started/configuration/).

-   **New Configuration Guide (`docs/getting-started/configuration.md`)**
    -   Created a comprehensive guide serving as the master reference for all library settings.
    -   **Path Management**: Documents the medallion architecture (`bronze` / `silver` / `gold` tiers) and how `ROOT_DATA_DIR` drives the tier hierarchy. Explains the `config.set_path()` and `config.get_path()` API and environment variable override precedence.
    -   **Storage Backends**: Dedicated setup sections for Local filesystem, Azure Data Lake Storage (ADLS, including connection string vs. SAS token authentication), and Snowflake internal stages (including SPCS mode).
    -   **API Integrations**: Covers Google Earth Engine (service account and Application Default Credentials), OpenStreetMap (Overpass endpoint config), and Ookla (tile cache configuration).
    -   **Environment Variable Master Table**: Single reference table for all 30+ supported environment variables, grouped by subsystem with descriptions and example values.
    -   Cross-referenced from the updated Installation Guide.

-   **Updated Installation Guide (`docs/getting-started/installation.md`)**
    -   Modernized "Next Steps" section to explicitly direct users to the new Configuration Guide before the Quick Start guide, reflecting the prerequisite relationship.
    -   Fixed a missing `---` separator that left the section header floating.

-   **New User Guide How-to Articles (`docs/user-guide/`)**
    -   Extracted the five core workflows from production Jupyter notebooks in the repository root and converted them into structured narrative guides. Each guide explains the architectural "Why" (handler vs. generator responsibilities) before the "How" (code).
    -   **`school-proximity.md`**: `GigaSchoolLocationFetcher` + `PoiViewGenerator.find_nearest_buildings()` — building proximity analysis at national scale. Derived from `optimize_building_mapping.ipynb`.
    -   **`infrastructure-normalization.md`**: `read_dataset` + `TransmissionNodeTable` + `EntityProcessor` — normalizing partner KMZ/GPKG fiber data into the Giga schema. Derived from `process_infra_ken.ipynb`.
    -   **`settlement-characterization.md`**: `GHSLDataHandler(product="GHS_SMOD")` + `PoiViewGenerator.map_smod()` — automated urban/rural stratification using the GHSL Settlement Model. Derived from GHSL mapping notebooks.
    -   **`population-accessibility.md`**: `WPPopulationHandler` + `PoiViewGenerator.map_wp_pop()` — population catchment estimation within a configurable buffer radius. Derived from `test_worldpop.ipynb`.
    -   **`zonal-statistics-grids.md`**: `TifProcessor` + `H3ViewGenerator.map_rasters()` — aggregating raster data (elevation, nightlights) onto H3 hexagonal grids for global comparative analysis. Corrects the non-existent `H3Hexagons.agg_raster()` pattern in favor of the actual `H3ViewGenerator.map_rasters()` API. Derived from `test_tifprocessor.ipynb`.

-   **`mkdocs.yml`: Navigation updated**
    -   Added `Configuration: getting-started/configuration.md` to the Getting Started section.
    -   Added all five new User Guide articles to the User Guide navigation section, replacing the previously commented-out placeholder entries.

### Dependencies

-   Added `pyarrow>=17.0.0` to core dependencies to support optimized tabular data processing and standard Parquet I/O.

## [v0.9.2] - 2026-03-30

### Added

-   **UNICEF SDG Data Fetcher (`gigaspatial/handlers/unicef/sdg.py`)** - New stateless fetcher for the UNICEF SDMX Data Warehouse, providing access to 700+ child welfare indicators (mortality, education, nutrition, immunisation, WASH, SDGs).
    -   `UnicefDataFetcher` - Pydantic dataclass fetcher wrapping the `unicefdata` Python package. Returns indicator data directly as a `pd.DataFrame` with no download/caching lifecycle, consistent with the GigaSpatial fetcher pattern (e.g., `GigaSchoolLocationFetcher`).
    -   Supports indicator filtering by `countries` (ISO3, validated via `pycountry`), `year` / year range, `sex` disaggregation, `latest` / `mrv` recency filters, and `circa` approximate-year matching.
    -   `fetch()` - Core method returning a flat `pd.DataFrame` of SDMX indicator observations.
    -   `fetch_as_geodataframe()` - Spatially enriches fetched indicator data by left-joining to an admin boundary `GeoDataFrame` on ISO3 country code, enabling direct integration with GigaSpatial's view generator workflows.
    -   `search_indicators()` / `list_categories()` - Static convenience wrappers exposing `unicefdata` discovery utilities without requiring fetcher instantiation.
    -   `unicefdata` added as a new optional dependency under the `[unicef]` extra (`pip install "giga-spatial[unicef]"`).

-   **`OSMLocationFetcher.fetch_by_osmid`: Fetch a single OSM element by ID**
    -   Added `fetch_by_osmid(osmid, element_type, include_metadata)` to `gigaspatial/handlers/osm.py`, enabling direct lookup of any OSM element (node, way, or relation) by its numeric OSM ID via the Overpass API `<type>(id:<osmid>)` filter.
    -   `element_type` accepts `"node"` (default), `"way"`, or `"relation"`, routing to the correct Overpass output mode (`center` for nodes/relations, `geom` for ways) and the appropriate internal processor (`_process_node_relation` or `_process_way`).
    -   Returns a processed `Dict` with the same field structure as `fetch_locations` results (`source_id`, `name`, `name_en`, `type`, `geometry`, `latitude`, `longitude`, and metadata fields when `include_metadata=True`), ensuring interoperability with downstream GigaSpatial workflows.
    -   Includes a graceful fallback for elements that do not match any configured `location_types`: returns a minimal dict with raw OSM tags and geometry instead of `None`, which is the expected behavior for direct ID-based lookups where tag filtering is not the intent.
    -   Reuses `_make_request` for retry/backoff logic and `_process_node_relation`/`_process_way` for consistent element normalization, adding no new I/O or processing surface.

### Changed

-   **Optional Dependency: `unicefdata` (UNICEF SDG Handler)** - Refactored `UnicefDataFetcher` to make the `unicefdata` library optional.
    -   Implemented defensive loading in `gigaspatial/handlers/unicef/sdg.py`: `import gigaspatial` no longer fails if `unicefdata` is missing.
    -   Detailed `ImportError` messages now guide users to install the dependency via `pip install "giga-spatial[unicef]"` only when accessing UNICEF data functionality.
    -   Added `unicef` extra to `setup.py` and updated `requirements.txt` with the appropriate git installation URL.

-   **UNICEF Handler Reorganization** - Migrated the standalone `gigaspatial/handlers/unicef_georepo.py` to `gigaspatial/handlers/unicef/georepo.py`, centralizing all UNICEF-related providers under a single module. Updated imports in `AdminBoundaries` and top-level `__init__.py` for consistency.

### Fixed

-   **`TifProcessor._create_clipped_processor`: Fixed FileNotFoundError when using non-local DataStores**
    -   Resolved a `FileNotFoundError` during `clip_to_geometry(..., return_clipped_processor=True)` that occurred when the parent processor used a cloud-based `DataStore` (e.g., `ADLSDataStore`).
    -   Fixed the initialization of the new processor instance to use `LocalDataStore()` for its local temporary path validation, even when the parent uses a non-local store.
    -   Restored the placeholder initialization pattern to ensure the clipped file is saved in the new processor's `_temp_dir`, preventing accidental deletion when the original processor is cleaned up.

### Dependencies

-   `[unicef]`: `unicefdata` (install via git URL from `unicef-drp/unicefData` or `pip install unicefdata` from source).

## [v0.9.1] - 2026-03-26

### Added

-   **`read_dataset`: New geospatial format support**
    -   Added `.fgb` (FlatGeobuf) to `GEO_READERS` via `gpd.read_file`: FlatGeobuf is one of GigaSpatial's three primary export formats and was previously unhandled, raising an unsupported format error.
    -   Added `.kml` (standalone KML) to `GEO_READERS` via `gpd.read_file`: previously only `.kmz` archives were supported; standalone KML exports were rejected.
    -   Added `.geojsonl` and `.ndjson` (newline-delimited / GeoJSON lines) to `GEO_READERS` via `gpd.read_file`: supports NDJSON outputs common in data lake and streaming export pipelines.
    -   Added `validate_crs` parameter (default `True`) to `read_dataset`: emits a `UserWarning` when a GeoDataFrame is returned with no CRS defined, surfacing silent projection issues that previously caused hard-to-debug downstream failures. Implemented via a new `_maybe_warn_crs()` internal helper applied consistently across all geo read paths.

### Changed

-   **Refactored BaseHandlerDownloader and Handler Cleanup**
    -   Promoted `download_data_units` to a concrete non-abstract method in `BaseHandlerDownloader` (`gigaspatial/handlers/base.py`).
    -   Centralized download logic including optional parallel execution (`multiprocessing.Pool`), `tqdm` progress tracking, and support for `pandas.DataFrame` units into the base class.
    -   Added automatic result filtering (removing `None`) and flattening of list-based results (e.g., from extracted archives) to the base implementation.
    -   Systematically removed redundant `download_data_units` logic from several subclasses, significantly reducing boilerplate and improving maintainability:
        -   `microsoft_global_buildings.py`
        -   `google_open_buildings.py`
        -   `google_ms_combined_buildings.py`
        -   `nasa_srtm.py`
        -   `ookla_speedtest.py`
        -   `hdx.py`
        -   `ghsl.py`
        -   `opencellid.py`
        -   `worldpop.py`
    -   Ensured proper `kwargs` propagation to `download_data_unit` in the base implementation to preserve handler-specific parameters (e.g., `extract`, `file_pattern`, `data_type`).

-   **`read_dataset`: Format registry and routing improvements**
    -   Removed `.zip` from `COMPRESSION_FORMATS`: it is a container format, not a compression wrapper like `.gz`/`.bz2`. Previously, `.zip` matched the compression branch before the geo-container branch, causing ambiguous routing and silent failures for zipped geospatial data.
    -   Moved `.json` from `PANDAS_READERS` to `GEO_READERS` (using `gpd.read_file`): plain `.json` files in GigaSpatial pipelines are predominantly GeoJSON. The previous routing to `pd.read_json` returned geometry as raw strings with no spatial awareness. A pandas fallback remains for non-spatial JSON.
    -   Removed `**kwargs` forwarding from `read_json`: `json.load` only accepts `cls` and `object_hook`; forwarding arbitrary kwargs (e.g. `compression=...`) caused `TypeError` at runtime.
    -   Renamed inner `suffixes` variable in the `.shp` sidecar branch to `SHAPEFILE_SIDECAR_EXTENSIONS` (promoted to module-level constant): the previous local redeclaration silently shadowed `path_obj.suffixes` in the outer scope.
    -   Unified bare `.gz` fallback to delegate to `read_gzipped_json_or_csv` instead of blindly assuming CSV: a `.gz` file containing JSON (without a compound `.json.gz` extension) previously returned a malformed DataFrame without error.
    -   Replaced fragile `data_store.__class__.__name__.replace('DataStore', '').lower()` string manipulation with a `_storage_display_name()` helper using `in`-based class name detection, consistent with the DataStore cross-platform improvements in v0.9.0.
    -   Added `raise ... from e` throughout all `except` blocks to preserve original tracebacks and improve debuggability.

-   **OpenCellID Handler Refactor** - Complete architectural refactor to align with the `BaseHandler` design pattern.
    -   Decomposed the legacy monolithic class into specialized `OpenCellIDConfig`, `OpenCellIDDownloader`, and `OpenCellIDReader` components, orchestrating them via a new `OpenCellIDHandler`.
    -   Updated `extract_search_geometry` to return ISO 3166-1 alpha-2 country codes, matching the OpenCellID database's primary indexing method.
    -   Implemented `get_relevant_data_units_by_geometry` to dynamically resolve download links from the OpenCellID web portal based on the resolved country code.
    -   Standardized the local storage hierarchy to utilize country-specific subdirectories (`bronze/opencellid/{alpha2}/`) for better data organization and multi-country support.
    -   Decoupled processing parameters (`created_newer`, `created_before`, and `drop_duplicates`) from the static configuration, moving them to `load_from_paths` to enable granular, call-time filtering of cell tower data.
    -   Introduced `load_as_geodataframe` as a high-level convenience method for direct loading into spatial data structures.

### Fixed

-   **Robust Quadkey Handling in MercatorTiles and RWIHandler**
    -   Fixed a potential `TypeError` in `MercatorTiles.from_quadkeys` by ensuring all input quadkeys are mapped to strings before calculating zoom levels. This protects the initialization workflow when quadkeys are passed as integers.
    -   Updated RWIHandler to explicitly cast the quadkey column to strings after loading, preventing downstream failures during tiled aggregations when source data stores quadkeys in numeric format.

-   **`TifProcessor._create_clipped_processor`: Robust temp file initialization and data store handling**
    -   Fixed a `FileNotFoundError` raised during `clip_to_geometry(..., return_clipped_processor=True)` caused by the new `TifProcessor` being initialized with `data_store=self.data_store` while the placeholder file was written to a local temp path. When `data_store` is not a `LocalDataStore`, `__post_init__` routed to `data_store.file_exists()` which could not locate the locally created placeholder, raising the error.
    -   Eliminated the unnecessary two-step placeholder pattern (create dummy `.tif` in a separate `tempfile.mkdtemp()` dir, then overwrite with clipped data). Clipped data is now written directly to a single temp file within `self._temp_dir`, matching the pattern used by `_reproject_to_temp_file`.
    -   The new `TifProcessor` is now always initialized with `LocalDataStore()` for temp-file-backed instances, regardless of the parent processor's `data_store`, ensuring `__post_init__` correctly resolves the local absolute path.
    -   Explicitly sets `clipped_file_path`, `dataset_path`, and `dataset_paths` on the returned processor so `open_dataset` correctly routes reads to the clipped file, consistent with the behavior of `clip_to_bounds`.

## [v0.9.0] - 2026-03-17

### Added

-   **Giga Spatial Entity Schemas `gigaspatial/core/schemas/`** - New core module for managing structured infrastructure and administrative entities using a Medallion Architecture (Bronze-Silver-Gold) pattern.
    -   **Base Entity Framework (`entity.py`)**: 
        -   `BaseGigaEntity`, `GigaEntity`, `GigaGeoEntity`, `GigaEntityNoLocation`: Pydantic-validated base models for all GigaSpatial entities, supporting stable UUID-based ID generation, coordinate validation (including "Null Island" checks), and geometry parsing (WKT/WKB/Shapely).
        -   `EntityTable`: Generic container class for collections of entities, providing bulk loading from files/dataframes, serialization, spatial filtering (admin, polygon, bounds), and advanced spatial analysis (KDTree-based nearest neighbors, distance graph construction).
    -   **Silver-Layer Data Processing (`gigaspatial/processing/entity_processor.py`)**:
        -   `EntityProcessor`: A robust, multi-step cleaning pipeline designed to transform raw "Bronze" data into "Silver" validated structures. Features include:
            -   Automatic coordinate column detection and repair (merged lat/lon columns, trailing commas).
            -   NFKC string normalization and whitespace stripping.
            -   Sentinel-aware null coercion (handling "n/a", "none", etc.).
            -   Geometry-safe row operations (deduplication, empty row removal).
            -   Change tracking via `track_changes` decorator for detailed pipeline logging of row/column shifts.
    -   **Domain-Specific Entity Schemas**:
        -   **Connectivity**: `CellTower`, `Cell`, `TransmissionNode`, `MobileCoverage`. Includes logic for backhaul normalization, radio technology mapping, and cross-entity enrichment (e.g., populating tower performance from child cells).
        -   **Infrastructure**: `BuildingFootprint` for managing building geometries.
        -   **Administrative**: `AdminBoundary` for structured administrative region management and parent/child relationship mapping.
    -   **Shared Infrastructure**: 
        -   Centralized enums for `DataConfidence`, `PowerSource`, `RadioType`, and `BackhaulType` ensuring cross-module consistency.
        -   **Configurable Entity ID Namespace**: Relocated `ENTITY_UUID_NAMESPACE` to `shared.py` and linked it to a new `ENTITY_ID_NAMESPACE` global configuration field, allowing the UUID namespace used for stable ID generation to be overridden via environment variables (defaults to `uuid.NAMESPACE_DNS`).

-   **BigQuery Client `gigaspatial/core/io/bigquery_client.py`** - New reusable BigQuery client layer for interacting with Google BigQuery datasets.
    -   `BigQueryClientConfig` - Pydantic-validated configuration supporting service account key file authentication and Application Default Credentials (ADC) fallback, mirroring the credential pattern used by `GEEProfiler`. Defaults for `project`, `service_account`, and `service_account_key_path` are resolved from `global_config` (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_SERVICE_ACCOUNT`, `GOOGLE_SERVICE_ACCOUNT_KEY_PATH`).
    -   `BigQueryClient` - Generic, dataset-agnostic client exposing:
        -   `list_datasets(project_id)`: List available datasets within a GCP project.
        -   `list_tables(dataset_id)`: List all tables within a dataset.
        -   `get_table_schema(dataset_id, table_id)`: Retrieve field names, types, modes, and descriptions for a table.
        -   `query(sql)`: Execute SQL and return a raw `RowIterator`.
        -   `query_to_dataframe(sql)`: Execute SQL and return a `pd.DataFrame`, with optional BigQuery Storage API acceleration (`use_bq_storage`), configurable byte billing cap (`max_gb_allowed`, default 10 GB), and `tqdm` progress bar support.
        -   `get_query_cost_estimate(sql)`: Dry-run cost estimation in USD using on-demand pricing ($6.25/TiB).
    -   Designed for composition: dataset-specific handlers instantiate `BigQueryClient` internally rather than re-implementing auth or query logic.

-   **M-Lab NDT7 Handler (`gigaspatial/handlers/mlab.py`)** - New handler for querying M-Lab network measurement data from the `measurement-lab.ndt.ndt7` BigQuery public dataset.
    -   `MLabConfig` - Pydantic config holding dataset-level parameters (`project_id`, `dataset`, `default_start_date`). Auth and billing credentials are fully delegated to `BigQueryClientConfig` / `global_config`, keeping dataset config free of auth logic.
    -   `MLabHandler` - Composes `BigQueryClient` to query NDT7 upload and download measurements:
        -   `query_ndt7(country_code, start_date, end_date, measurement)`: Returns a `pd.DataFrame` of NDT7 measurements filtered by ISO alpha-2 country code and date range. Selects `id`, `date`, client geo fields (`country_code`, `city`, `lat`, `lon`, `postal_code`), `a.*` scalar metrics (`MeanThroughputMbps`, `MinRTT`, `LossRate`, `CongestionControl`), raw common fields (`ServerIP`, `ClientIP`, `StartTime`, `EndTime`), and direction-specific `ServerMeasurements` fields (AppInfo, BBRInfo, TCPInfo) based on the `measurement` parameter (`"download"`, `"upload"`, or `"both"`).
            -   **Automatic Schema Flattening**: Handles complex BigQuery ARRAY and STRUCT navigation (e.g., `ServerMeasurements[SAFE_OFFSET(0)]`) internally, exposing a flat `pd.DataFrame` to the user.
        -   `query_ndt7_gdf(country_code, start_date, end_date, measurement)`: Convenience wrapper returning a `gpd.GeoDataFrame` (EPSG:4326) from `client.Geo.Latitude` / `client.Geo.Longitude`; rows with null coordinates are dropped.
        -   `estimate_query_cost(country_code, start_date, end_date, measurement)`: Pre-execution cost estimate in USD via `BigQueryClient.get_query_cost_estimate()`, recommended before running wide date-range queries given the scale of the public dataset.
    -   Country codes validated via `pycountry` (raises `ValueError` for unknown codes).
    -   Partition filter on `date` and direct `client.Geo.CountryCode` field filter minimise bytes scanned for cost efficiency.

-   **WorldPop Handler: Degree of Urbanisation datasets (`degree_of_urbanization`)**
    -   Added `project="degree_of_urbanization"` as a new supported project type in `WPPopulationConfig` and `WPPopulationHandler`, providing access to WorldPop's Degree of Urbanisation (DUG) datasets built on the R2025A Global2 methodology.
    -   Covers years 2015–2030 via API category `dug_g2_v1` (`"Degree of Urbanisation 2015-2030 using WorldPop Global2 R2025A"`).
    -   Added `dug_level: Literal["L1", "L2"]` field (default `"L1"`) to select between:
        -   `L1`: 3-class grid classification (urban / suburban / rural) — `*_GRID_L1_R2025A_v1.tif`
        -   `L2`: 7-class detailed grid classification — `*_GRID_L2_R2025A_v1.tif`
    -   Only `.tif` grid files are fetched; accompanying `.zip` (entities, statistics) files are automatically excluded.
    -   `constrained`, `un_adjusted`, `school_age`, and `under_18` fields are not applicable for DUG and are silently normalised to `False` with a warning if set.
    -   `get_relevant_data_units_by_geometry` updated to route DUG requests to the correct API `dataset_type` (`"dug"`) and filter files by the selected `dug_level` pattern.
    -   `__repr__` updated to show `dug_level` instead of population-specific fields when `project="degree_of_urbanization"`.
    -   Fully compatible with existing download and reader pipeline — DUG `.tif` files follow the same `GIS/` path convention and require no changes to `get_data_unit_path`, `get_data_unit_paths`, or `WPPopulationReader`.

-   **HTTP Module `gigaspatial/core/http/`** - New submodule providing a reusable, composable HTTP client layer for interacting with REST APIs across all GigaSpatial handlers.
    -   `AuthConfig` and `AuthType` (`auth.py`) - Pydantic-validated authentication configuration supporting Bearer token, API key (header or query param), Basic auth, and no-auth modes. `AuthConfig.build()` resolves to `(headers, query_params, httpx_auth)` for direct use by `httpx.Client`.
    -   `BaseRestApiClient` and `RestApiClientConfig` (`client.py`) - Abstract base class for REST API clients with built-in retry logic (exponential backoff), `Retry-After`-aware rate limiting, session lifecycle management via context manager (`__enter__`/`__exit__`), and convenience `get`/`post` wrappers. Configuration is fully Pydantic-validated (`base_url`, `auth`, `timeout`, `max_retries`, `retry_backoff`, `default_headers`).
    -   `BasePaginationStrategy`, `OffsetPagination`, `CursorPagination`, `PageNumberPagination` (`pagination.py`) - Strategy-based pagination abstractions. Subclasses override `extract_records` and `next_request` to express any pagination pattern (offset/limit, cursor, page number) without modifying the client. `PageNumberPagination` added to support page/size APIs (used by Giga handlers).
    -   All classes exported from `gigaspatial/core/http/__init__.py`.

-   **Giga Handlers Submodule `gigaspatial/handlers/giga/`** - Reorganised Giga School API fetchers into a dedicated submodule with a shared internal client.
    -   `GigaApiClient` (`api_client.py`) - Internal `BaseRestApiClient` subclass shared across all Giga fetchers. Uses Bearer token auth and `PageNumberPagination` with `records_key="data"`.
    -   `GigaSchoolLocationFetcher` (`school_locations.py`) - Refactored to use `GigaApiClient`. Eliminates manual `requests` pagination loop; retry logic and rate limiting inherited from `BaseRestApiClient`.
    -   `GigaSchoolProfileFetcher` (`school_profile.py`) - Refactored to use `GigaApiClient`. Supports optional `giga_id_school` filter; single-school requests are short-circuited to one page.
    -   `GigaSchoolMeasurementsFetcher` (`school_measurements.py`) - Refactored to use `GigaApiClient`. Date range parameters (`start_date`, `end_date`) now overridable per-call in addition to instance level. `_format_date` promoted to `@staticmethod`.

### Changed

-   **`HealthSitesFetcher._convert_country`** - Added retry loop with exponential backoff and escalating timeouts (2000ms → 4000ms → 6000ms) for OSM country name resolution via `OSMLocationFetcher.get_osm_countries`, improving resilience against transient network timeouts.

-   **Optional Dependency Implementation (Core Framework)** - Refactored the library to make heavy dependencies optional, reducing the base installation size and improving import times.
    -   Implemented "Defensive Loading" pattern across all heavy modules: dependencies are only required when the specific functionality is accessed.
    -   Added clear `ImportError` messages with tailored installation instructions (e.g., `pip install "giga-spatial[gee]"`) for missing optional packages.
    -   Refactored the following modules to be optional:
        -   **Earth Engine**: `GEEProfiler` and related GEE handlers (requires `[gee]`).
        -   **BigQuery**: `BigQueryClient` and `MLabHandler` (requires `[bq]`).
        -   **Azure**: `ADLSDataStore` (requires `[azure]`).
        -   **Snowflake**: `SnowflakeDataStore` (requires `[snowflake]`).
        -   **Delta Sharing**: `DeltaSharingDataStore` (requires `[delta]`).
        -   **Database**: `DBConnection`, Trino support, and Dask integration (requires `[db]`).
        -   **DuckDB**: `OvertureAmenityFetcher` (requires `[duckdb]`).
    -   Updated `setup.py` with `extras_require` for all groups and an `[all]` extra for full installation.
    -   Reorganized `requirements.txt` to clearly distinguish between core and optional dependencies.

-   **Giga fetchers** - Removed `sleep_time` field and manual `time.sleep` calls; rate limiting is now handled transparently by `BaseRestApiClient` via `Retry-After` header respect and configurable backoff.

-   **`GigaSchoolMeasurementsFetcher.get_performance_summary`** - Consolidated repetitive boolean flag summarisation into a loop over `(flag, label)` pairs for maintainability.

-   **Healthsites Handler `gigaspatial/handlers/healthsites.py`** - Reorganised `HealthSitesFetcher` with its own internal client.
    -   `_HealthSitesPaginationStrategy`: Custom `BasePaginationStrategy` handling both GeoJSON (`features[]`) and JSON (direct list) response formats from the Healthsites API.
    -   `_HealthSitesApiClient`: Internal `BaseRestApiClient` subclass using `AuthType.API_KEY_QUERY` (`api-key` query parameter) and `_HealthSitesPaginationStrategy`.
    -   `HealthSitesFetcher`: Refactored to use `_HealthSitesApiClient`. All fetch parameters (`country`, `extent`, `output_format`, `flat_properties`, `from_date`, `to_date`) are now overridable per call. `fetch_statistics` and `fetch_facility_by_id` use the client directly for non-paginated requests.

-   **`WPPopulationConfig.validate_configuration` docstring**: Added DUG availability section:
    ```
    DUG (degree_of_urbanization):
    - Only available via GR2 release, years 2015-2030.
    - constrained, un_adjusted, school_age, under_18 are not applicable.
    - dug_level selects between L1 (3-class) and L2 (7-class) grid tifs.
    ```

-   **WorldPop Handler default configuration**: Updated the default dataset parameters in `WPPopulationHandler` to use `release="GR2"`, `year=2025`, and `constrained=True` (with `un_adjusted=False`), shifting the default data source to the more recent R2025A series.

### Fixed

-   **BaseHandler: Robust CRS handling in cropping and loading**
    -   Fixed a critical issue where `crop_to_geometry()` used a hardcoded `"EPSG:4326"` CRS when reprojecting geometries to match data projections. It now correctly uses the CRS provided in `kwargs` or falls back to the handler's default.
    -   Added a `crs` property to `BaseHandlerConfig` (defaulting to `"EPSG:4326"`) to allow handler-specific coordinate systems (e.g., GHSL's Mollweide `ESRI:54009`) to be correctly propagated during the loading pipeline.
    -   Updated `BaseHandlerReader.load()` to automatically retrieve and pass the configuration's CRS to `crop_to_geometry()`, ensuring that cached search geometries are correctly reprojected even when the handler uses a non-WGS84 coordinate system.
    -   Ensured `crs` is popped from `kwargs` in `crop_to_geometry()` to avoid downstream conflicts with processing algorithms that might receive the same keyword arguments.

### Dependencies

-   Added `httpx>=0.27` as a new dependency.
-   Refactored the library to use **Optional Dependencies** (Extras). See `setup.py` or `requirements.txt` for details.
    -   `[gee]`: `earthengine-api`, `geemap`
    -   `[bq]`: `google-cloud-bigquery`, `google-cloud-bigquery-storage`, `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2`, `db-dtypes`
    -   `[azure]`: `azure-storage-blob`
    -   `[snowflake]`: `snowflake-connector-python`
    -   `[delta]`: `delta-sharing`
    -   `[db]`: `SQLAlchemy`, `sqlalchemy-trino`, `dask`
    -   `[duckdb]`: `duckdb`
    -   `[all]`: Install all optional dependencies.

## [v0.8.2] - 2026-02-19

### Changed

-   **WorldPop Population Mapping Robustness**
    -   Refactored `GeometryBasedZonalViewGenerator.map_wp_pop` and `PoiViewGenerator.map_wp_pop` to safely handle `WPPopulationHandler.load_data` returning either a single `TifProcessor` or `List[TifProcessor]`.
    -   Introduced `_ensure_tif_list` helper to normalize handler outputs into flat lists, preventing crashes when filters (e.g., `under_18=True`, `min_age`/`max_age`) yield single rasters.
    -   Non-`age_structures` raster paths now build flat `List[TifProcessor]` across countries using `extend`, avoiding nested lists.
    -   Preserved per-raster summing semantics for `age_structures` + `centroid_within` (zonal or POI), now robust to single/list outputs.

-   **PoiViewGenerator: Removed unused kwargs from aggregation calls**
    -   Cleaned up `map_points()` and `map_polygons()` methods by removing `**kwargs` arguments passed to `aggregate_points_to_zones()` and `aggregate_polygons_to_zones()`, which do not accept additional keyword arguments.

### Fixed

-   Updated `requests` dependency from 2.32.3 to 2.32.4 to address CVE-2024-47081 (URL parsing issue leaking .netrc credentials).

## [v0.8.1] - 2026-02-19

### Added

-   **GeometryBasedZonalViewGenerator: Relative Wealth Index mapping**
    -   Added `map_rwi()` helper to aggregate Relative Wealth Index values to arbitrary zones using `RWIHandler` as the data source.
    -   Supports both point-based (`predicate="centroid_within"`, using quadkey centroids) and polygon-based (`"intersects"`, `"within"`) enrichment with configurable aggregations (`mean`, `median`, `max`, `min`) into a new output column (e.g., `rwi_mean`).

### Changed

-   **WorldPop Handler (`WPPopulationConfig`): Multi-release support with GR1/GR2**
    -   Introduced a `release` field (`Literal["GR1", "GR2"]`, default `"GR2"`) to `WPPopulationConfig` and `WPPopulationHandler` to explicitly select between the two WorldPop Global dataset releases, replacing the previous implicit year-based release inference.
    -   **GR1** (legacy release): covers years 2000–2020 for both `pop` and `age_structures` projects, with constrained and unconstrained variants and full UN adjustment support. All previously supported dataset categories (`wpgp`, `wpic1km`, `cic2020_100m`, `aswpgp`, `ascic_2020`, `sapya1km`, etc.) are preserved unchanged under this release.
    -   **GR2** (WorldPop R2025A v1): covers years 2015–2030, constrained only, no UN adjustment. Supports:
        -   `pop`: 100m (`G2_CN_POP_R25A_100m`) and 1km (`G2_CN_POP_R25A_1km`) resolution.
        -   `age_structures` (full age breakdown): 100m (`G2_CN_Age_R25A_100m`) and 1km (`G2_CN_Age_R25A_1km`) resolution.
        -   `age_structures` (under-18 population, `under_18=True`): 100m only (`G2_Age_U18_R25A_100m`).
    -   Added `under_18: bool` field (default `False`) to `WPPopulationConfig` and `WPPopulationHandler` for accessing WorldPop R2025A under-18 population datasets (`project="age_structures"`, `release="GR2"` only).
    -   Removed the R2024-specific special-case branches (`year == 2024`, `G2_CN_POP_2024_100m`, `G2_UC_POP_2024_100m`, `G2_CN_Age_2024_100m`, `G2_UC_Age_2024_100m`) from `validate_configuration`; users requiring those datasets should remain on GR1 with `year=2020` or use GR2 for the overlapping year range.
    -   Updated `AVAILABLE_YEARS_GR1` (formerly `AVAILABLE_YEARS`) to `range(2000, 2021)` and added `AVAILABLE_YEARS_GR2` as `range(2015, 2031)`.
    -   Extended `_filter_age_sex_paths` to correctly handle under-18 filename patterns (`ISO3_T/F/M_Under_18_YEAR_...tif`): detects `UNDER_18` keyword to avoid misclassification as non-school-age files, supports `T` (total) sex value, and defaults to returning only the `T` aggregate file when no sex filter is provided.
    -   Extended the zip-extraction branch in `get_data_unit_paths` to cover `under_18=True` datasets (previously only `school_age=True`).
    -   Extended `get_relevant_data_units_by_geometry` zip passthrough guard to include `G2_Age_U18_R25A_100m` alongside `sapya1km`.
    -   Updated `validate_configuration` docstring to reflect GR1/GR2 availability matrix for all project and resolution combinations.

-   **Relative Wealth Index Handler (`RWIHandler`): Quadkey-enriched loading**
    -   Updated `load_data()` to automatically compute and attach a `quadkey` column (zoom level 14) for all RWI records when only `latitude`/`longitude` are present, making tiled aggregation the default behavior.
    -   Added `load_as_geodataframe()` convenience method to return RWI data as a `GeoDataFrame` with Mercator quadkey geometries joined to the underlying `rwi` and `error` attributes for direct spatial analysis.

-   **`aggregate_points_to_zones` / `aggregate_polygons_to_zones`: Semantically correct fill values for empty zones**
    -   Changed the fill behavior for zones with no overlapping point or polygon data: non-`count` aggregations (e.g., `mean`, `sum`, `min`, `max`) now correctly fill missing values with `np.nan` instead of `0`, distinguishing "no data" from a true zero measurement.
    -   `count` aggregation continues to fill with `0`, preserving the expected behavior that a zone with no matched features has a count of zero.
    -   Updated docstrings for both functions to explicitly document the `np.nan` vs. `0` fill semantics per aggregation method.
    -   Fixed a missing column rename step in `aggregate_points_to_zones` for non-MultiIndex aggregation results, ensuring `output_suffix` is consistently applied to all output columns.
    -   Fixed geometry column drop in `aggregate_points_to_zones` being incorrectly gated on aggregation type; geometry is now always dropped before non-count aggregation to avoid pandas errors.

### Fixed

-   **BaseHandler.ensure_data_available: Correct handling of post-download paths**
    -   Fixed an issue where `ensure_data_available()` could still report missing data on the first download for handlers whose `get_data_unit_paths()` mapping changes after download (e.g., WorldPop `age_structures` with `school_age=True`, where ZIP resources are extracted to `.tif` files).
    -   After downloading, the method now refreshes the resolved data paths before performing the final existence check, ensuring that ZIP → `.tif` extraction workflows correctly pass availability checks on the first run.

## [v0.8.0] - 2026-02-10

### Added

-   **Google Earth Engine Handler (`GEEProfiler`)**
    -   **Complete GEE Integration**: New `gigaspatial/handlers/gee/` module with full Google Earth Engine support via `GEEProfiler` class.
    -   **Context & Impact: Multi-Sector GEE Intelligence**
        -   **Critical Data Access**: Built-in support for global infrastructure and environmental datasets (Nightlights, Population, Land Cover) provides instant, ready-to-use insights for emergency response, climate resilience, and urban development.  
        -   **Beyond Built-ins**: While the registry comes pre-configured for high-value datasets, `GEEProfiler` is fully flexible, unlocking the entire multi-petabyte Google Earth Engine public catalog (Landsat, Sentinel, MODIS, etc.) for custom analysis.
    -   **Intelligent Asset Handling**: Robust initialization automatically distinguishes `ee.Image` vs `ee.ImageCollection`, preventing backend loading errors during band validation.
    -   **Dataset Registry**: `GEEDatasetRegistry` and `GEEDatasetEntry` for managing built-in datasets (nightlights, population, etc.) with metadata (bands, scales, temporal coverage).
    -   **Inspection Utilities**: Comprehensive collection inspection including `display_collection_info()`, `get_band_names()`, `get_date_range()`, `display_band_names()`, `display_properties()`.
    -   **`map_to_points()`**: Extracts GEE raster values at point locations with optional circular buffers (metric CRS buffering, automatic UTM reprojection).
    -   **`map_to_zones()`**: Spatial aggregation of GEE rasters within polygon boundaries (admin zones, grids) using configurable reducers.
    -   **Automatic Chunking**: Handles GEE API limits with intelligent chunking (`chunk_size=1000` default), processing large feature sets without timeouts.
    -   **Unified Processing Pipeline**: Single `_reduce_regions_with_chunking()` core handles both points and zones with robust error recovery per chunk.
    -   **Temporal Filtering**: `start_date`/`end_date` filtering with automatic temporal reduction (`mean`, `median`, etc.) for ImageCollections.
    -   **NRT-Ready Logic**: Advanced date handling for Near Real-Time datasets, automatically defaulting to most recent data windows.
    -   **Config-Driven**: `GEEConfig` supports dataset lookup, authentication (service accounts), and defaults (band, reducer, scale, chunk_size).
    -   **Production-Ready**: Full validation (bands, dates, geometries), detailed logging, error handling, and config fallbacks.

-   **GEE Built-in Datasets**
    -   Pre-configured registry entries for popular datasets: `nightlights` (VIIRS), `population` (various sources), land cover, etc.
    -   Automatic collection loading via `dataset_id`: `profiler = GEEProfiler("nightlights")`.
    -   Standardized metadata: resolution, default bands, reducers, temporal cadence.

-   **DeltaSharingDataStore (formerly GigaDataAPI)**
    -   Refactored internal `GigaDataAPI` into public `DeltaSharingDataStore` as a reusable data-access layer for Delta Sharing tables.
    -   Replaced hardcoded global config with flexible initialization supporting global config (`API_PROFILE_FILE_PATH`, `API_SHARE_NAME`, `API_SCHEMA_NAME`) + selective per-instance overrides via keyword arguments.
    -   Added `DeltaSharingConfig` (Pydantic model) for validated configuration with profile file existence checks and `enable_cache` flag for in-memory table caching.
    -   Implemented lazy `SharingClient` initialization via `client` property and comprehensive cache management (`get_table_metadata()`, `clear_cache()`, `get_cached_tables()`, `cache_size_mb`).
    -   Maintained backward-compatible API: `list_tables()` → `get_country_list()`, `load_table()` → `load_country_data()`.

### Changed

-   **OvertureAmenityFetcher: Updated to latest Overture Places release and S3 layout**
    -   Updated default Overture release to `2026-01-21.0` for Places theme queries.
    -   Switched to the current S3 URL pattern for Places GeoParquet (`release/{release}/theme=places/...`) and now build the read path via a configurable `base_url` instead of hardcoding it in the SQL query.
    -   Fixed IO errors caused by outdated bucket patterns by reading directly from the official Overture S3 bucket with DuckDB’s `read_parquet()`.
    -   Enabled S3 access in DuckDB by installing and loading the `httpfs` extension and configuring the AWS region (`s3_region='us-west-2'`) during connection setup.

-   **DeltaSharingDataStore: Robust table discovery**
    -   Enhanced `list_tables()` to handle `SharingClient.list_all_tables()` returning empty lists (common in constrained shares) by falling back to explicit `list_shares()` → `list_schemas()` → `list_tables()` enumeration on configured share/schema.
    -   Added structured logging for debugging share/schema mismatches while preserving stable public API contract.

-   **DataStore: Cross-Platform Path Handling**
    -   **`ADLSDataStore` and `LocalDataStore`: Enhanced path type support**
        -   All path-accepting methods now support both `str` and `PathLike` objects (e.g., `pathlib.Path`) for improved type flexibility and cross-platform compatibility.
        -   Introduced `Pathish = Union[str, PathLike[str]]` type alias for consistent path parameter signatures across both data store implementations.
    
    -   **`ADLSDataStore`: Centralized blob key normalization**
        -   Added `_to_blob_key()` method as the single source of truth for converting input paths to Azure-compatible blob keys.
        -   Automatically handles Windows backslashes → forward slashes conversion via `PurePosixPath` for cross-platform compatibility.
        -   Strips leading slashes (Azure convention) and optionally ensures trailing slashes for directory operations.
        -   All methods now use `_to_blob_key()` internally, eliminating inconsistent path handling across different operations.
        -   `_normalize_path()` deprecated in favor of `_to_blob_key()` but kept for backward compatibility.
    
    -   **`LocalDataStore`: Improved path resolution**
        -   Enhanced `_resolve_path()` to accept `PathLike` objects alongside strings.
        -   Properly handles both absolute and relative paths with automatic resolution relative to `base_path`.
        -   All methods updated to use `Pathish` type signature for consistency with `ADLSDataStore`.
    
    -   **Benefits:**
        -   **Cross-platform safety**: Eliminates Windows vs. Linux path separator issues when handlers build paths using `pathlib` and pass to data stores.
        -   **Type flexibility**: Handlers can now pass `Path` objects, strings, or any `PathLike` without manual `str()` conversion.
        -   **Backward compatible**: Existing handler code with `str()` wrappers continues to work unchanged.
        -   **Maintainability**: Path normalization logic centralized in data store boundary, not scattered across handlers.


### Documentation

-   **OvertureAmenityFetcher: Amenity category documentation**
    -   Expanded the class docstring to explain that `amenity_types` should correspond to `categories.primary` values in the Overture Places schema.
    -   Linked to the authoritative Overture category list CSV on GitHub so users can discover all valid amenity categories without hardcoding them in giga-spatial:  
        `https://github.com/OvertureMaps/schema/blob/main/docs/schema/concepts/by-theme/places/overture_categories.csv`.

## [v0.7.6] - 2026-01-27

### Changed

-   **Maxar Imagery Handler: API Migration to GEGD Pro Platform**
    -   Updated `MaxarConfig` to support the new Maxar GEGD Pro API infrastructure:
        -   Replaced username/password/connection string authentication with API key-based authentication.
        -   Added support for OAuth bearer tokens as an alternative authentication method.
        -   Updated base URL from `evwhs.digitalglobe.com` to `pro.gegd.com/streaming/v1/ogc/wms`.
        -   Added `auth_method` parameter supporting three authentication modes: `api_key` (query parameter), `header` (custom header), and `bearer_token`.
        -   Updated layer names from `DigitalGlobe:Imagery`/`DigitalGlobe:ImageryFootprint` to `Maxar:Imagery`/`Maxar:FinishedFeature`.
        -   Replaced deprecated `featureProfile` parameter with optional `profile` parameter for stacking profiles.
        -   Renamed `coverage_cql_filter` to `cql_filter` for consistency with updated API specification.
        -   Added `styles` parameter supporting `raster` (imagery) and `footprints` (feature visualization) rendering modes.
        -   Updated to WMS version 1.3.0 as the default and recommended version.
    
    -   **Enhanced Initialization Flexibility:**
        -   `MaxarImageDownloader` now accepts configuration as `MaxarConfig` object, dictionary, or keyword arguments.
        -   Supports mixing configuration sources with kwargs taking precedence for convenient overrides.
        -   Added validation for API key or bearer token requirement via Pydantic field validators.
    
    -   **Updated Authentication Implementation:**
        -   Replaced basic authentication with header-based and query-parameter authentication methods.
        -   Implemented `_build_auth_headers()` for dynamic header construction based on auth method.
        -   Modified `_initialize_wms()` to handle API key injection and header configuration.
        -   Updated `_download_single_image()` to use `srs` parameter (OWSLib compatibility) while supporting WMS 1.3.0 `crs` in actual requests.

-   **Maxar Imagery Handler: WFS Metadata Integration**
    -   Added Web Feature Service (WFS) support for querying imagery metadata alongside image downloads:
        -   Implemented `_initialize_wfs()` for WFS 2.0.0 service initialization with authentication.
        -   Added `get_imagery_metadata()` method supporting bbox and CQL filter queries with configurable output formats.
        -   Implemented `get_metadata_for_bbox()` convenience method providing summary statistics (feature count, sensors, date ranges, cloud cover, resolution).
        -   Added `_download_single_image_with_metadata()` for atomic image download with associated feature metadata.
    
    -   **Metadata Features:**
        -   Retrieves comprehensive imagery attributes including: acquisition dates, sensor/source information, cloud cover percentage, ground sample distance, sun angles, off-nadir angle, product names, band descriptions, NIIRS quality ratings, and processing levels.
        -   Returns metadata as `GeoDataFrame` for seamless spatial analysis and filtering.
        -   Supports CQL filtering by date ranges, sensors, product types, and custom attribute queries.
        -   Includes BBOX-in-CQL support for combined spatial and attribute filtering (WFS limitation workaround).
    
    -   **Bulk Download Metadata Support:**
        -   Added `save_metadata` parameter to all bulk download methods (`download_images_by_tiles`, `download_images_by_bounds`, `download_images_by_coordinates`).
        -   When enabled, automatically saves JSON metadata files alongside downloaded images with matching filenames.
        -   Metadata JSON includes feature properties with proper datetime serialization (ISO 8601 format).
        -   Geometry column excluded from JSON output to minimize file size while preserving spatial reference in GeoDataFrame workflows.

-   **Maxar Imagery Handler: Date Filtering Convenience**
    -   Added `build_date_filter()` helper method for constructing CQL date range filters:
        -   Accepts string dates (`YYYY-MM-DD`), `datetime`, or `date` objects.
        -   Supports start-only, end-only, or date range filtering via `acquisitionDate` field.
        -   Uses `BETWEEN` syntax for cleaner queries when both dates provided.
        -   Configurable `date_field` parameter for filtering alternative temporal fields.
    
    -   **Integrated Date Filtering in Bulk Downloads:**
        -   Added `start_date` and `end_date` parameters to all bulk download methods.
        -   Date filters automatically combined with existing `cql_filter` configuration via logical AND.
        -   Original CQL filter state preserved and restored after download completion.
        -   Includes informative logging when date filters are applied.

-   **ADLSDataStore: Performance Optimizations for File and Directory Operations**
    -   **Optimized `list_files()` Method:**
        -   Replaced `list_blobs()` with `list_blob_names()` for performance improvement.
        -   Reduced memory usage by returning blob name strings instead of full `BlobProperties` objects.
        -   Added `_normalize_path()` helper method for consistent path handling (removes leading slashes, ensures trailing slashes for directories, converts backslashes to forward slashes).
        -   Maintains backward compatibility—still returns a list of file paths.
        -   Performance: ~2MB memory usage vs ~500MB for large directories.
    
    -   **New `list_files_iter()` Method:**
        -   Memory-efficient generator-based iteration over files in large directories.
        -   Enables early exit and lazy evaluation without loading entire file lists into memory.
        -   Returns iterator of blob name strings, supporting streaming workflows.
        -   Ideal for directories with 100K+ files where full materialization is unnecessary.
    
    -   **Optimized `walk()` Method:**
        -   Replaced `list_blobs()` with `list_files_iter()` for lazy evaluation.
        -   Eliminated full materialization of blob lists into memory.
        -   Performance: 2-3x faster and 100-500x less memory usage for large directory trees.
    
    -   **Optimized `list_directories()` Method:**
        -   Replaced `list_blobs()` iteration with Azure's `walk_blobs(delimiter='/')` for hierarchical listing.
        -   Azure now returns directory prefixes directly without scanning all files.
        -   Performance: 100-1000x faster for large directories (1-5 seconds vs 30-60 seconds for 1M files).
        -   No longer requires iterating through all files to identify subdirectories.
    
    -   **New Utility Methods:**
        -   `has_files_with_extension()`: Fast early-exit check for files with specific extensions without full directory scan.
        -   `count_files()`: Memory-efficient file counting using generator iteration.
        -   `count_files_with_extension()`: Memory-efficient counting of files by extension.

-   **SnowflakeDataStore: Multiprocessing Support**
    -   Implemented lazy connection initialization enabling safe pickling and multiprocessing usage.
    -   Added thread-safe connection creation via `_get_connection()` with double-check locking pattern.
    -   Implemented custom `__getstate__()` and `__setstate__()` for proper serialization, excluding non-picklable connection and lock objects.
    -   Each worker process creates its own database connection on first access.
    -   Maintained full backward compatibility via `connection` property accessor.
    -   Improved startup performance by deferring connection creation until first use.

### Performance

-   **Significant speedups for Azure Blob Storage operations:**
    -   File listing operations now faster due to `list_blob_names()` optimization.
    -   Directory listing operations now 100-1000x faster using Azure's hierarchical `walk_blobs()` with delimiter.
    -   Memory usage reduced by 100-500x for large directory operations (2-5MB vs 500MB-1GB).
    -   Directory tree walking now 2-3x faster with lazy evaluation via `list_files_iter()`.
    -   Benefits scale with directory size—most dramatic improvements for directories with 100K+ files.

-   **Improved startup time for SnowflakeDataStore:**
    -   Lazy connection initialization eliminates unnecessary connection overhead when data store is instantiated but not immediately used.

### Developer Notes

-   **Migration from Legacy Maxar API:**
    -   Existing code using `MAXAR_USERNAME`, `MAXAR_PASSWORD`, and `MAXAR_CONNECTION_STRING` environment variables must migrate to `MAXAR_API_KEY`.
    -   Old base URL and authentication methods are no longer supported by Maxar's API infrastructure.
    -   Layer names in existing configurations must be updated to new `Maxar:` namespace.
    -   CQL filter parameter names updated throughout codebase for consistency with new API specification.

-   **ADLSDataStore Improvements:**
    -   Existing `list_files()` calls remain fully compatible—no code changes required.
    -   For large directories (100K+ files), consider using `list_files_iter()` for memory efficiency.
    -   The `_normalize_path()` helper ensures consistent path handling across all methods.
    -   All optimizations leverage Azure SDK's native capabilities for maximum performance.

### Dependencies

-   Updated `owslib` usage to support WFS 2.0.0 specification for metadata retrieval.

## [v0.7.5] - 2026-01-20

### Added

-   **Google-Microsoft Combined Buildings Handler (VIDA)**
    -   Added `GoogleMSBuildingsHandler` (`gigaspatial/handlers/google_ms_combined_buildings.py`) to access the merged Google V3 Open Buildings (1.8B footprints) and Microsoft Global Building Footprints (1.24B footprints) dataset hosted by VIDA/source.coop.
    -   Supports multiple data formats: GeoParquet (default), FlatGeobuf, and PMTiles.
    -   Flexible partition strategies: country-level (single file per country) or S2 grid partitioning (tiled by S2 cells for large countries).
    -   Download strategies: S3 (default) or HTTPS for cloud-native access.
    -   Source filtering: filter buildings by data source (`google`, `microsoft`, or both).
    -   Integrated with `BaseHandler` architecture for consistent data lifecycle management (download, cache, read).
    -   Automatic partition discovery: resolves relevant S2 tiles or country files based on query geometry.
    -   Streaming support: can stream GeoParquet row groups directly from cloud storage without full download.

-   **Shared Building-Processing Engine**
    -   Added `GoogleMSBuildingsEngine` (`gigaspatial/processing/buildings_engine.py`) as a reusable, high-performance building workflow engine.
    -   Encapsulates common building-processing logic (S2 tile job creation, per-tile processing loops, and result accumulation) previously duplicated across view generators.
    -   Provides two main entrypoints:
        -   `count_buildings_in_zones()`: Efficiently counts buildings intersecting zones using `STRtree` spatial indexing.
        -   `nearest_buildings_to_pois()`: Computes nearest-building distances for POIs using KD-tree nearest-neighbor search with haversine distance calculation.
    -   Handles both single-file countries (processes entire dataset) and partitioned countries (loads only intersecting S2 tiles).
    -   Supports source filtering (`google`, `microsoft`, or both) at the engine level.
    -   Designed for reuse: both zonal and POI view generators now delegate to this engine, reducing code duplication and ensuring consistent performance optimizations.

-   **High-Performance Buildings Mapping for View Generators**
    -   Added `GeometryBasedZonalViewGenerator.map_buildings()` to efficiently count buildings per zone using the combined dataset.
        -   Uses S2 grid partitioning to load only intersecting building tiles for partitioned countries.
        -   Leverages `STRtree` spatial indexing for fast intersection queries between buildings and zones.
        -   Supports source filtering to count buildings from specific providers.
    -   Added `PoiViewGenerator.find_nearest_buildings()` to efficiently compute nearest-building distances for POIs.
        -   Uses S2 grid partitioning with configurable search radius to limit tile processing.
        -   Implements KD-tree nearest-neighbor search for fast candidate selection.
        -   Computes final distances using haversine (great-circle) distance in meters.
        -   Supports global nearest-building search with progressive radius expansion for partitioned countries.
        -   Returns distance metrics and boolean flags indicating buildings within specified search radius.

-   **Geo Processing Utilities**
    -   Added `estimate_utm_crs_with_fallback()` (`gigaspatial/processing/geo.py`) as a robust utility for UTM CRS estimation.
        -   Wraps `GeoDataFrame.estimate_utm_crs()` with comprehensive error handling and fallback logic.
        -   Automatically falls back to a configurable CRS (default: Web Mercator EPSG:3857) when UTM estimation fails or returns `None`.
        -   Handles edge cases: empty GeoDataFrames, estimation exceptions, and `None` return values.
        -   Centralizes the common UTM estimation pattern used across the codebase, reducing duplication.
        -   Provides optional logger parameter for warning messages when fallbacks occur.

### Fixed

-   **Spatial Matching Graph Construction (`build_distance_graph`)**
    -   Fixed critical bug where `exclude_same_index=True` returned fewer matches than requested when querying same dataframe against itself.
    -   Previously, when excluding self-matches, the function would query for `max_k` neighbors and then filter out self-matches, resulting in only `max_k - 1` actual matches returned.
    -   Most critically, `max_k=1` with `exclude_same_index=True` would always return zero matches (since the only neighbor found was the point itself).
    -   Now queries for `max_k + 1` neighbors when `exclude_same_index=True`, ensuring `max_k` valid matches are returned after self-match removal.
    -   Affects `build_distance_graph()` in spatial matching workflows where the same dataframe is matched against itself.

- **S2 cell polygon generation (`to_geoms`)**
    -   Now conversion of S2 cells to Shapely polygons enforces a consistent counter‑clockwise winding order using  shapely.geometry.polygon.orient() , avoiding accidental orientation flips or hole-like polygons when rendering or exporting.
    -   Added validation and automatic repair of invalid polygons via  buffer(0)  to handle rare projection-related self-intersections, logging the number of repaired cells and skipping any that remain invalid after repair.
    -   Improved logging for debugging by including the S2 token in warnings and errors when a cell fails conversion or cannot be repaired.

### Performance

-   **Significant speedups for buildings enrichment**
    -   Zonal building counts and POI nearest-building mapping are now substantially faster than mapping Google and Microsoft buildings separately, achieving performance gains through:
        -   **Single combined dataset**: Eliminates redundant I/O and repeated processing of overlapping building footprints from separate sources.
        -   **S2 tile filtering**: For partitioned countries, loads only intersecting S2 building tiles instead of scanning entire country files, dramatically reducing memory usage and processing time.
        -   **Spatial indexing**: Uses `STRtree` for O(log k) zone intersection queries and KD-tree for fast nearest-neighbor search, replacing slower sequential scans.
        -   **Shared engine architecture**: Centralized optimizations benefit both zonal and POI workflows, ensuring consistent performance improvements across use cases.

### Dependencies

-   Added `s3fs>=2024.12.0` as a new dependency

## [v0.7.4] - 2025-11-24

### Added

-   **TifProcessor: Raster Export Methods** 
    -   **`save_to_file()` method:** Comprehensive raster export functionality with flexible compression and optimization options.
        -   Supports multiple compression algorithms: LZW (default), DEFLATE, ZSTD, JPEG, WEBP, and NONE.
        -   Configurable compression parameters: `ZLEVEL` for DEFLATE (default: 6), `ZSTD_LEVEL` for ZSTD (default: 9), `JPEG_QUALITY` (default: 85), `WEBP_LEVEL` (default: 75).
        -   Predictor support for improved compression: predictor=2 for integer data (horizontal differencing), predictor=3 for floating-point data.
        -   Tiled output enabled by default (512×512 blocksize) for optimal random access performance.
        -   Cloud-Optimized GeoTIFF (COG) support via `cog=True` parameter with automatic overview generation.
        -   Customizable overview levels and resampling methods for COG creation.
        -   BigTIFF support for files >4GB via `bigtiff` parameter.
        -   Multi-threading support for compatible compression algorithms via `num_threads` parameter.
        -   Integrates with `self.open_dataset()` context manager, automatically handling merged, reprojected, and clipped rasters.
        -   Writes through `self.data_store` abstraction layer, supporting both local and remote storage (e.g., ADLS).
        -   Preserves all bands from source rasters without skipping.
    
    -   **`save_array_to_file()` method:** Export processed numpy arrays while preserving georeferencing metadata.
        -   Accepts 2D or 3D numpy arrays (with automatic dimension handling).
        -   Inherits CRS, transform, and nodata values from source raster or accepts custom values.
        -   Supports same compression options as `save_to_file()`.
        -   Enables saving modified/processed raster data while maintaining spatial reference.
        -   Writes through `self.data_store` for consistent storage abstraction.

-   **TifProcessor: Value-Based Filtering for DataFrame and GeoDataFrame Conversion**
    -   **`min_value` and `max_value` parameters:** Added optional filtering thresholds to `to_dataframe()` and `to_geodataframe()` methods.
        -   `min_value`: Filters out pixels with values ≤ threshold (exclusive).
        -   `max_value`: Filters out pixels with values ≥ threshold (exclusive).
        -   Filtering occurs **before** geometry creation in `to_geodataframe()`, significantly improving performance for sparse datasets.
        -   Supports both single-band and multi-band rasters with consistent behavior.
    
    -   **Enhanced `_build_data_mask()` method:** Extended to incorporate value threshold filtering alongside nodata filtering.
        -   Combines multiple mask conditions using logical AND for efficient filtering.
        -   Maintains backward compatibility when no thresholds are specified.
    
    -   **Enhanced `_build_multi_band_mask()` method:** Extended for multi-band value filtering.
        -   Drops pixels where ANY band has nodata or fails value thresholds.
        -   Ensures consistent filtering behavior across RGB, RGBA, and multi-band modes.

-   **TifProcessor: Raster Statistics in `get_raster_info()`**
    -   Added `include_statistics` and `approx_ok` flags to optionally return pixel statistics alongside metadata.
    -   New `_get_basic_statistics()` helper streams through raster blocks to compute per-band and overall min, max, mean, std, sum, and count with nodata-aware masking.
    -   Results are cached for reuse within the processor lifecycle to avoid repeated scans.

-   **BaseHandler: Tabular Load Progress**
    -   `_load_tabular_data()` now supports a `tqdm` progress bar, showing file-level load progress for large tabular batches.
    -   Added `show_progress` and `progress_desc` parameters so handlers can toggle or customize the indicator while keeping existing callers backward compatible.

-   **Improved developer usability by enabling easier access to primary components without deep module references:**
    - Exposed core handlers, view generators, and processing modules at the top-level `gigaspatial` package namespace for improved user experience and simplified imports.
    - Added convenient aliases for `gigaspatial.core.io` as `io` and `gigaspatial.processing.algorithms` as `algorithms` directly accessible from `gigaspatial`.
    - Declared explicit public API in `__init__.py` to clarify stable, supported components.

### Changed

-   **GHSLDataConfig: Improved SSL Certificate Handling for Tile Downloads**
    -   Replaced `ssl._create_unverified_context` approach with a robust two-tier fallback strategy for downloading GHSL tiles shapefile.
    
    -   **Primary method:** Attempts download via `gpd.read_file()` with unverified SSL context (fast, direct access).
    
    -   **Fallback method:** Uses `requests.get()` with `verify=False` for environments where `gpd.read_file()` fails (e.g., cloud compute instances with Anaconda certificate bundles).
    -   Downloads tiles to temporary local file before reading when fallback is triggered, ensuring compatibility across different Python environments.
    
    -   **Tile caching:** Implemented GeoJSON-based caching in `base_path/cache/` directory to minimize redundant downloads.
        -   Cache checked before any download attempts.
        -   Invalid cache automatically triggers re-download.
        -   Uses `write_dataset()` for consistent storage abstraction across local and remote data stores.
    
    -   **Enhanced error handling:** 
        -   Logs specific exception types (`type(e).__name__`) for better debugging.
        -   Graceful fallback with informative warning messages.
        -   Preserves exception chain for traceback analysis.
    -   Improved compatibility with Azure ML compute instances where multiple certificate stores (system, Anaconda, certifi) coexist.
    -   Temporary file cleanup guaranteed via `finally` block, preventing orphaned downloads.

### Fixed

-   **GHSLDataConfig: SSL certificate verification failures in cloud environments**
    -   Resolved `CERTIFICATE_VERIFY_FAILED` errors when downloading GHSL tiles shapefile on cloud compute instances.
  
### Performance

-   **Reduced network overhead for GHSL tile metadata**:
    -   Tiles shapefile downloaded only once per coordinate system (WGS84/Mollweide) and cached locally.
    -   Subsequent `GHSLDataConfig` instantiations load from cache, eliminating repeated ~MB shapefile downloads.
    -   Benefits scale with number of GHSL queries across application lifecycle.


### Documentation

- Improved `README` with clearer key workflows, core concepts, and updated overview text.

## [v0.7.3] - 2025-11-11

### Added

-   **SnowflakeDataStore Support**
    -   New `SnowflakeDataStore` class implementing the `DataStore` interface for Snowflake stages.
    -   Supports file operations (read, write, list, delete) on Snowflake internal stages.
    -   Integrated with `gigaspatial/config.py` for centralized configuration via environment variables.
    -   Provides directory-like operations (`mkdir`, `rmdir`, `walk`, `is_dir`, `is_file`) for conceptual directories in Snowflake stages.
    -   Includes context manager support and connection management.
    -   Full compatibility with existing `DataStore` abstraction.

-   **BaseHandler: Config-Level Data Unit Caching**
    -   `BaseHandlerConfig` now maintains internal `_unit_cache` for data unit and geometry caching.
    -   Cache stores tuples of `(units, search_geometry)` for efficient reuse across handler, downloader, and reader operations.
    -   New methods:
        -   `get_cached_search_geometry()`: Retrieve cached geometry for a source.
        -   `clear_unit_cache()`: Clear cached data for testing or manual refreshes.
        -   `_cache_key()`: Generate canonical cache keys from various source types.
    -   Benefits all components (handler, downloader, reader) regardless of entry point.

-   **Unified Geometry Extraction in BaseHandlerConfig**
    -   New `extract_search_geometry()` method providing standardized geometry extraction from various source types:
        -   Country codes (via `AdminBoundaries`)
        -   Shapely geometries (`BaseGeometry`)
        -   GeoDataFrames with automatic CRS handling
        -   Lists of points or coordinate tuples (converted to `MultiPoint`)
    -   Centralizes geometry conversion logic, eliminating duplication across handler methods.

-   **BaseHandler: Crop-to-Source Feature for Handlers**
    -   New `crop_to_source` parameter in `BaseHandlerReader.load()` and `BaseHandler.load_data()` methods.
    -   Allows users to load data clipped to exact source boundaries rather than full data units (e.g., tiles).
    -   Particularly useful for tile-based datasets (Google Open Buildings, GHSL) where tiles extend beyond requested regions.
    -   Implemented `crop_to_geometry()` method in `BaseHandlerReader` for spatial filtering:
        -   Supports `(Geo)DataFrame` clipping using geometry intersection.
        -   Supports raster clipping using `TifProcessor`'s `clip_to_geometry` method.
        -   Extensible for future cropping implementations.
    -   Search geometries are now cached alongside data units for efficient cropping operations.

-   **S2 Zonal View Generator (`S2ViewGenerator`)**
    -   New generator for producing zonal views using Google S2 cells (levels 0–30).
    -   Supports sources:
        -   Country name (`str`) via `CountryS2Cells.create(...)`
        -   Shapely geometry or `gpd.GeoDataFrame` via `S2Cells.from_spatial(...)`
        -   Points (`List[Point | (lon, lat)]`) via `S2Cells.from_points(...)`
        -   Explicit cells (`List[int | str]`, S2 IDs or tokens) via `S2Cells.from_cells(...)`
    -   Uses `cell_token` as the zone identifier.
    -   Includes `map_wp_pop()` convenience method (auto-uses stored country when available).

-   **H3 Zonal View Generator (`H3ViewGenerator`)**
    -   New generator for producing zonal views using H3 hexagons (resolutions 0–15).
    -   Supports sources:
        -   Country name (`str`) via `CountryH3Hexagons.create(...)`
        -   Shapely geometry or `gpd.GeoDataFrame` via `H3Hexagons.from_spatial(...)`
        -   Points (`List[Point | (lon, lat)]`) via `H3Hexagons.from_spatial(...)`
        -   Explicit H3 indexes (`List[str]`) via `H3Hexagons.from_hexagons(...)`
    -   Uses `h3` as the zone identifier.
    -   Includes `map_wp_pop()` convenience method (auto-uses stored country when available).

-   **TifProcessor: MultiPoint clipping support**
    -   `_prepare_geometry_for_clipping()` now accepts `MultiPoint` inputs and uses their bounding box for raster clipping.
    -   Enables passing collections of points as a `MultiPoint` to `clip_to_geometry()` without pre-converting to a polygon.

### Changed

-   **Configuration**
    -   Added Snowflake connection parameters to `gigaspatial/config.py`:
        -   `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`
        -   `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`
        -   `SNOWFLAKE_STAGE_NAME`
    -   Added Snowflake configuration variables to `.env_sample`

-   **BaseHandler**
    -   **Streamlined Data Unit Resolution**
        -   Consolidated `get_relevant_data_units_by_country()`, `get_relevant_data_units_by_points()`, and `get_relevant_data_units_by_geometry()` into a unified workflow.
        -   All source types now convert to geometry via `extract_search_geometry()` before unit resolution.
        -   Subclasses now only need to implement `get_relevant_data_units_by_geometry()` for custom logic.
        -   Significantly reduces code duplication in handler subclasses.
    
    -   **Optimized Handler Workflow**
        -   Eliminated redundant `get_relevant_data_units()` calls across handler, downloader, and reader operations.
        -   `ensure_data_available()` now uses cached units and paths, preventing multiple lookups per request.
        -   Data unit resolution occurs at most once per unique source query, improving performance for:
            -   Repeated `load_data()` calls with the same source.
            -   Operations involving both download and read steps.
            -   Direct usage of downloader or reader components.

-   **Enhanced BaseHandlerReader**
    -   `resolve_source_paths()` now primarily handles explicit file paths.
    -   Geometry/country/point conversion delegated to handler and config layers.
    -   `load()` method updated to support `crop_to_source` parameter with automatic geometry retrieval from cache.
    -   Fallback geometry computation if cache miss occurs (e.g., when reader used independently).


-   **BaseHandlerConfig Caching Logic**
    -   `get_relevant_data_units()` now checks cache before computing units.
    -   Added `force_recompute` parameter to bypass cache when needed (e.g., `force_download=True`).
    -   Cache operations include debug logging for transparency during development.


-   **TifProcessor temp-file handling**
    -   Simplified `_create_clipped_processor()` to mirror `_reproject_to_temp_file`: write clipped output to the new processor’s `_temp_dir`, set `_clipped_file_path`, update `dataset_path`, and reload metadata.
    -   `open_dataset()` now prioritizes `_merged_file_path`, `_reprojected_file_path`, then `_clipped_file_path`, and opens local files directly.
    -   Clipped processors consistently use `LocalDataStore()` for local temp files to avoid data-store path resolution issues.

## Fixed

-   **TifProcessor: clip_to_geometry() open failure after merge**
    -   Fixed a bug where `open_dataset()` failed for processors returned by `clip_to_geometry()` when the source was initialized with multiple paths and loaded via handlers with `merge_rasters=True`.
    -   The clipped raster is now saved directly into the new processor’s temp directory and tracked via `_clipped_file_path`, ensuring reliable access by `open_dataset()`.
    -   Absolute path checks in `__post_init__` now use `os.path.exists()` for absolute paths with `LocalDataStore`, preventing false negatives for temp files.

### Performance

-   **Significant reduction in redundant computations in handlers**:
    -   Single geometry extraction per source query (previously up to 3 times).
    -   Single data unit resolution per source query (previously 2-3 times).
    -   Cached geometry reuse for cropping operations.
    -   Benefits scale with:
        -   Number of repeated queries.
        -   Complexity of geometry extraction (especially country boundaries).
        -   Number of data units per query.

### Developer Notes

-   Subclass implementations should now:
    -   Only override `get_relevant_data_units_by_geometry()` for custom unit resolution.
    -   Use `extract_search_geometry()` for any geometry conversion needs.
    -   Optionally override `crop_to_geometry()` for dataset-specific cropping logic.


### Dependencies

-   Added `snowflake-connector-python>=3.0.0` as a new dependency

## [v0.7.2] - 2025-10-27

### Added

-   **Ookla Speedtest Handler Integration (`OoklaSpeedtestHandler`)**
    -   New classes `OoklaSpeedtestHandler`, `OoklaSpeedtestConfig`, `OoklaSpeedtestDownloader`, and `OoklaSpeedtestReader` for managing Ookla Speedtest data.
    -   `OoklaSpeedtestHandler.load_data` method supports Mercator tile filtering by country or spatial geometry and includes an optional `process_geospatial` parameter for WKT to GeoDataFrame conversion.
    -  In `OoklaSpeedtestConfig`, `year` and `quarter` fields are optional (defaulting to `None`) and `__post_init__` logs warnings if they are not explicitly provided, using the latest available data.
    -  In `OoklaSpeedtestReader`, `resolve_source_paths` method overridden to appropriately handle `None` or non-path sources by returning the `DATASET_URL`.
    -  `OoklaSpeedtestHandler`, the `__init__` method requires `type` as a mandatory argument, with `year` and `quarter` being optional.

-   **S2 Grid Generation Support (`S2Cells`)**
    -   Introduced `S2Cells` class for managing Google S2 cell grids using the `s2sphere` library.
    -   Supports S2 levels 0-30, providing finer granularity than H3 (30 levels vs 15).
    -   Provides multiple creation methods:
        -   `from_cells()`: Create from lists of S2 cell IDs (integers or tokens).
        -   `from_bounds()`: Create from geographic bounding box coordinates.
        -   `from_spatial()`: Create from various spatial sources (geometries, GeoDataFrames, points).
        -   `from_json()`: Load S2Cells from JSON files via DataStore.
    -   Includes methods for spatial operations:
        -   `get_neighbors()`: Get edge neighbors (4 per cell) with optional corner neighbors (8 total).
        -   `get_children()`: Navigate to higher resolution child cells.
        -   `get_parents()`: Navigate to lower resolution parent cells.
        -   `filter_cells()`: Filter cells by a given set of cell IDs.
    -   Provides conversion methods:
        -   `to_dataframe()`: Convert to pandas DataFrame with cell IDs, tokens, and centroid coordinates.
        -   `to_geoms()`: Convert cells to shapely Polygon geometries (square cells).
        -   `to_geodataframe()`: Convert to GeoPandas GeoDataFrame with geometry column.
    -   Supports saving to JSON, Parquet, or GeoJSON files via `save()` method.
    -   Includes `average_cell_area` property for approximate area calculation based on S2 level.
    
-   **Country-Specific S2 Cells (`CountryS2Cells`)**
    -   Extends `S2Cells` for generating S2 grids constrained by country boundaries.
    -   Integrates with `AdminBoundaries` to fetch country geometries for precise cell generation.
    -   Factory method `create()` enforces proper instantiation with country code validation via `pycountry`.

-   Expanded `write_dataset` to support generic JSON objects.
    -   The `write_dataset` function can now write any serializable Python object (like a dict or list) directly to a `.json` file by leveraging the dedicated write_json helper.

-   **NASA SRTM Elevation Data Handler (`NasaSRTMHandler`)**
    -   New handler classes for downloading and processing NASA SRTM elevation data (30m and 90m resolution).
    -   Supports Earthdata authentication via `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables.
    -   `NasaSRTMConfig` provides dynamic 1°x1° tile grid generation covering the global extent.
    -   `NasaSRTMDownloader` supports parallel downloads of SRTM .hgt.zip tiles using multiprocessing.
    -   `NasaSRTMReader` loads SRTM data with options to return as pandas DataFrame or list of `SRTMParser` objects.
    -   Integrated with `BaseHandler` architecture for consistent data lifecycle management.

-   **SRTM Parser (`SRTMParser`)**
    -   Efficient parser for NASA SRTM .hgt.zip files using memory mapping.
    -   Supports both SRTM-1 (3601x3601, 1 arc-second) and SRTM-3 (1201x1201, 3 arc-second) formats.
    -   Provides methods for:
        -   `get_elevation(latitude, longitude)`: Get interpolated elevation for specific coordinates.
        -   `get_elevation_batch(coordinates)`: Batch elevation queries with NumPy array support.
        -   `to_dataframe()`: Convert elevation data to pandas DataFrame with optional NaN filtering.
        -   Automatic tile coordinate extraction from filename (e.g., N37E023, S10W120).

-   **SRTM Manager (`SRTMManager`)**
    -   Manager class for accessing elevation data across multiple SRTM tiles with lazy loading.
    -   Implements LRU caching (default cache size: 10 tiles) for efficient memory usage.
    -   Methods include:
        -   `get_elevation(latitude, longitude)`: Get interpolated elevation for any coordinate.
        -   `get_elevation_batch(coordinates)`: Batch elevation queries across multiple tiles.
        -   `get_elevation_profile(latitudes, longitudes)`: Generate elevation profiles along paths.
        -   `check_coverage(latitude, longitude)`: Check if a coordinate has SRTM coverage.
        -   `get_available_tiles()`: List available SRTM tiles.
        -   `clear_cache()` and `get_cache_info()`: Cache management utilities.
    -   Automatically handles tile boundary crossings for elevation profiles.

-   **Earthdata Session (`EarthdataSession`)**
    -   Custom `requests.Session` subclass for NASA Earthdata authentication.
    -   Maintains Authorization headers through redirects to/from Earthdata hosts.
    -   Required for accessing NASA's SRTM data repository.

### Changed

-   **ADLSDataStore Enhancements**
    -   Modified `__init__` method to support initialization using either `ADLS_CONNECTION_STRING` or a combination of `ADLS_ACCOUNT_URL` and `ADLS_SAS_TOKEN`.
    -   Improved flexibility for authenticating with Azure Data Lake Storage.

-   **Configuration**
    -   Added `ADLS_ACCOUNT_URL` and `ADLS_SAS_TOKEN` to `gigaspatial/config.py` and `.env_sample` for alternative ADLS authentication.
    -   Added `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` to `gigaspatial/config.py` and `.env_sample` for NASA Earthdata authentication.

### Fixed

-   **WorldPop: `RuntimeError` during `school_age=True` data availability check:**
    -   Resolved a `RuntimeError: Could not ensure data availability for loading` that occurred when `school_age=True` and WorldPop data was not yet present in the data store.
    -   `WPPopulationConfig.get_data_unit_paths` now correctly returns the original `.zip` URLs to trigger the download/extraction process when filtered `.tif` files are missing.
    -   After successful download and extraction, it now accurately identifies and returns the paths to the local `.tif` files, allowing `BaseHandler` to confirm availability and proceed with loading.
-   **WorldPop: `list index out of range` when no datasets found**:
    -   Added a `RuntimeError` in `WPPopulationConfig.get_relevant_data_units_by_country` when `self.client.search_datasets` returns no results, providing a clearer error message with the search parameters.

-   **WorldPop: Incomplete downloads with `min_age`/`max_age` filters for non-school-age `age_structures`**:
    -   Fixed an issue where `load_data` with `min_age` or `max_age` filters (when `school_age=False`) resulted in incomplete downloads.
    -   `WPPopulationConfig.get_data_unit_paths` now returns all potential `.tif` URLs for non-school-age `age_structures` during the initial availability check, ensuring all necessary files are downloaded.
    -   Age/sex filtering is now deferred and applied by `WPPopulationReader.load_from_paths` using `WPPopulationConfig._filter_age_sex_paths` *after* download, guaranteeing data integrity.

-   **HealthSitesFetcher**
    -   Ensured correct Coordinate Reference System (CRS) assignment (`EPSG:4326`) when returning `GeoDataFrame` from fetched health facility data.

### Dependencies

-   Added `s2sphere` as a new dependency for S2 geometry operations

## [v0.7.1] - 2025-10-15

### Added

-   **Healthsites.io API Integration (`HealthSitesFetcher`):**
    -   New class `HealthSitesFetcher` to fetch and process health facility data from the Healthsites.io API.
    -   Supports filtering by country, bounding box extent, and date ranges (`from_date`, `to_date`).
    -   Provides methods for:
        -   `fetch_facilities()`: Retrieves health facility locations, returning a `pd.DataFrame` or `gpd.GeoDataFrame` based on output format.
        -   `fetch_statistics()`: Fetches aggregated statistics for health facilities based on provided filters.
        -   `fetch_facility_by_id()`: Retrieves details for a specific facility using its OSM type and ID.
    -   Includes robust handling for API pagination, different output formats (JSON, GeoJSON), and nested data structures.
    -   Integrates with `OSMLocationFetcher` and `pycountry` to standardize country names to OSM English names for consistent querying.
    -   Configurable parameters for API URL, API key, page size, flat properties, tag format, output format, and request sleep time.

-   **OSMLocationFetcher Enhancements:**
    -   **Historical Data Fetching (`fetch_locations_changed_between`):**
        -   New method `fetch_locations_changed_between()` to retrieve OSM objects that were created or modified within a specified date range. This enables historical analysis and change tracking.
        -   Defaults `include_metadata` to `True` for this method, as it's typically used for change tracking.
    -   **Comprehensive OSM Country Information (`get_osm_countries`):**
        -   New static method `get_osm_countries()` to fetch country-level administrative boundaries directly from the OSM database.
        -   Supports fetching all countries or a specific country by ISO 3166-1 alpha-3 code.
        -   Option to include various name variants (e.g., `name:en`, `official_name`) and ISO codes.
    -   **Metadata Inclusion in Fetched Locations:**
        -   Added `include_metadata` parameter to `fetch_locations()` to optionally retrieve change tracking metadata (timestamp, version, changeset, user, uid) for each fetched OSM element.
        -   This metadata is now extracted and included in the DataFrame for nodes, relations, and ways.
    -   **Flexible Date Filtering in Overpass Queries:**
        -   Introduced `date_filter_type` (`newer`, `changed`) and `start_date`/`end_date` parameters to `_build_queries()` for more granular control over time-based filtering in Overpass QL.
    -   **Date Normalization Utility:**
        -   Added `_normalize_date()` helper method to convert various date inputs (string, datetime object) into a standardized ISO 8601 format for Overpass API queries.

-   **TifProcessor**
    -   **Comprehensive Memory Management:**
        -   Introduced `_check_available_memory()`, `_estimate_memory_usage()`, and `_memory_guard()` methods for proactive memory assessment across various operations.
        -   Added warnings (`ResourceWarning`) for potentially high memory usage in batched operations, with suggestions for optimizing `n_workers`.
    -   **Chunked DataFrame Conversion:**
        -   Implemented `to_dataframe_chunked()` for memory-efficient processing of large rasters by converting them to DataFrames in manageable chunks.
        -   Automatic calculation of optimal `chunk_size` based on target memory usage via `_calculate_optimal_chunk_size()`.
        -   New helper methods: `_get_chunk_windows()`, `_get_chunk_coordinates()`.
    -   **Raster Clipping Functionality:**
        -   `clip_to_geometry()`: New method to clip rasters to arbitrary geometries (Shapely, GeoDataFrame, GeoSeries, GeoJSON-like dicts).
        -   `clip_to_bounds()`: New method to clip rasters to rectangular bounding boxes, supporting optional CRS transformation for the bounds.
        -   New helper methods for clipping: `_prepare_geometry_for_clipping()`, `_validate_geometry_crs()`, `_create_clipped_processor()`.

-   **WorldPopDownloader Zip Handling:**
    -   Modified `download_data_unit` in `WPPopulationDownloader` to correctly handle `.zip` files (e.g., school age datasets) by downloading them to a temporary location and extracting the contained `.tif` files.
    -   Updated `download_data_units` to correctly flatten the list of paths returned by `download_data_unit` when zip extraction results in multiple files.
    -   Adjusted `WPPopulationConfig.get_data_unit_paths` to correctly identify and return paths for extracted `.tif` files from zip resources. It is now intelligently resolves paths. For school-age datasets, it returns paths to extracted `.tif` files if available; otherwise, it returns the original `.zip` path(s) to trigger download and extraction.
    -   Added filter support to `WPPopulationConfig.get_data_unit_paths` hence to the `WPPopulationHandler` for:
        -   School-age datasets: supports `sex` (e.g., "F", "M", "F_M") and `education_level` (e.g., "PRIMARY", "SECONDARY") filters on extracted `.tif` filenames.
        -   Non-school-age age_structures: supports `sex`, `ages`, `min_age`, and `max_age` filters on `.tif` filenames.
-   **WorldPop: Filtered aggregation in `GeometryBasedZonalViewGenerator.map_wp_pop`**:
    -   `map_wp_pop` now enforces a single country input when `handler.config.project` is "age_structures".
    -   When `predicate` is "centroid_within" and the project is "age_structures", individual `TifProcessor` objects (representing age/sex combinations) are loaded, sampled with `map_rasters(stat="sum")`, and their results are summed per zone, preventing unintended merging.

-   **PoiViewGenerator: Filtered aggregation in `PoiViewGenerator.map_wp_pop`**:
    -   `map_wp_pop` now enforces a single country input when `handler.config.project` is "age_structures".
    -   When `predicate` is "centroid_within" and the project is "age_structures", individual `TifProcessor` objects (representing age/sex combinations) are loaded, sampled with `map_zonal_stats(stat="sum")`, and their results are summed per POI, preventing unintended merging.

-   **TifProcessor Multi-Raster Merging in Handlers and Generators:**
    -   Extended `_load_raster_data` in `BaseHandlerReader` to support an optional `merge_rasters` argument. When `True` and multiple raster paths are provided, `TifProcessor` now merges them into a single `TifProcessor` object during loading.
    -   Integrated `merge_rasters` argument into `GHSLDataReader` and `WPPopulationReader`'s `load_from_paths` and `load` methods, enabling control over raster merging at the reader level.
    -   Propagated `merge_rasters` to `GHSLDataHandler`'s `load_into_dataframe`, and `load_into_geodataframe` methods for consistent behavior across the handler interface.

### Changed

-   **TifProcessor**
    -   **Unified DataFrame Conversion:**
        -   Refactored `to_dataframe()` to act as a universal entry point, dynamically routing to internal, more efficient methods for single and multi-band processing.
        -   Deprecated the individual `_to_band_dataframe()`, `_to_rgb_dataframe()`, `_to_rgba_dataframe()`, and `_to_multi_band_dataframe()` methods in favor of the new unified `_to_dataframe()`.
        -   `to_dataframe()` now includes a `check_memory` parameter.
    -   **Optimized `open_dataset` Context Manager:**
        -   The `open_dataset` context manager now directly opens local files when `LocalDataStore` is used, avoiding unnecessary `rasterio.MemoryFile` creation for improved performance and reduced memory overhead.
    -   **Enhanced `to_geodataframe` and `to_graph`:**
        -   Added `check_memory` parameter to `to_geodataframe()` and `to_graph()` for memory pre-checks.
    -   **Refined `sample_by_polygons_batched`:**
        -   Included `check_memory` parameter for memory checks before batch processing.
        -   Implemented platform-specific warnings for potential multiprocessing issues on Windows/macOS.
    -   **Improved Multiprocessing Initialization:**
        -   The `_initializer_worker()` method now prioritizes merged, reprojected, or original local file paths for opening, ensuring workers access the most relevant data.
    -   **Modular Masking and Coordinate Extraction:**
        -   Introduced new private helper methods: `_extract_coordinates_with_mask()`, `_build_data_mask()`, `_build_multi_band_mask()`, and `_bands_to_dict()` to centralize and improve data masking and coordinate extraction logic.
    -   **Streamlined Band-Mode Validation:**
        -   Moved the logic for validating `mode` and band count compatibility into a dedicated `_validate_mode_band_compatibility()` method for better code organization.

-   **GigaSchoolLocationFetcher**
    -   `fetch_locations()` method:
        -   Added `process_geospatial` parameter (defaults to `False`) to optionally process geospatial data and return a `gpd.GeoDataFrame`.
    -   `_process_geospatial_data()` method:
        -   Modified to return a `gpd.GeoDataFrame` by converting the `pd.DataFrame` with a `geometry` column and `EPSG:4326` CRS.

-   **OSMLocationFetcher Refactoring:**
    -   **Unified Query Execution and Processing:** Refactored the core logic for executing Overpass queries and processing their results into a new private method `_execute_and_process_queries()`. This centralizes common steps and reduces code duplication between `fetch_locations()` and the new `fetch_locations_changed_between()`.
    -   **Enhanced `_build_queries`:** Modified `_build_queries` to accept `date_filter_type`, `start_date`, `end_date`, and `include_metadata` to construct more dynamic and feature-rich Overpass QL queries.
    -   **Updated `fetch_locations` Signature:**
        -   Replaced `since_year` parameter with `since_date` (which can be a `str` or `datetime` object) for more precise time-based filtering.
        -   Added `include_metadata` parameter.
    -   **Improved Logging of Category Distribution:**
        -   Modified the logging for category distribution to correctly handle cases where categories are combined into a list (when `handle_duplicates='combine'`).
    -   **`since_year` Parameter:** Removed `since_year` from `fetch_locations()` as its functionality is now superseded by the more flexible `since_date` parameter and the `_build_queries` enhancements.

-   **`PoiViewGenerator` Mapping Methods (`map_zonal_stats`, `map_nearest_points`, `map_google_buildings`, `map_ms_buildings`, `map_built_s`, `map_smod`):**
    -   Changed `map_zonal_stats` and `map_nearest_points` to return `pd.DataFrame` results (including `'poi_id'` and new mapped columns) instead of directly updating the internal view.
    -   Updated `map_google_buildings`, `map_ms_buildings`, `map_built_s`, and `map_smod` to capture the `pd.DataFrame` returned by their respective underlying mapping calls (`map_nearest_points` or `map_zonal_stats`) and then explicitly call `self._update_view()` with these results.
    -   This enhances modularity and allows for more flexible result handling and accumulation.

-   **`ZonalViewGenerator.map_rasters` Enhancements:**
    -   Modified `map_rasters` to accept `raster_data` as either a single `TifProcessor` or a `List[TifProcessor]`.
    -   Implemented internal merging of `List[TifProcessor]` into a single `TifProcessor` before performing zonal statistics.
    -   Replaced `sample_multiple_tifs_by_polygons` with the `TifProcessor.sample_by_polygons` method.

### Fixed

-   **TifProcessor**:
    -   **`to_graph()` Sparse Matrix Creation:**
        -   Corrected the sparse matrix creation logic in `to_graph()` to ensure proper symmetric graph representation when `graph_type="sparse"`.
    -   **Coordinate System Handling in `_initializer_worker`:**
        -   Ensured that `_initializer_worker` correctly handles different data storage scenarios to provide the correct dataset handle to worker processes, preventing `RuntimeError` due to uninitialized raster datasets.

### Removed

-   **OSMLocationFetcher**
    -   **Redundant Category Distribution Logging:** Removed the explicit category distribution logging for `handle_duplicates == "separate"` since the `value_counts()` method on the 'category' column already provides this.


## [v0.7.0] - 2025-09-17

### Added

- **TifProcessor Revamp**
    - **Explicit Reprojection Method:** Introduced `reproject_to()` method, allowing on-demand reprojection of rasters to a new CRS with customizable `resampling_method` and `resolution`.
    - **Reprojection Resolution Control:** Added `reprojection_resolution` parameter to `TifProcessor` for precise control over output pixel size during reprojection.
    - **Advanced Raster Information:** Added `get_raster_info()` method to retrieve a comprehensive dictionary of raster metadata.
    - **Graph Conversion Capabilities:** Implemented `to_graph()` method to convert raster data into a graph (NetworkX or sparse matrix) based on pixel adjacency (4- or 8-connectivity).
    - **Internal Refactoring: `_reproject_to_temp_file`:** Introduced `_reproject_to_temp_file` as a helper for reprojection into temporary files.

- **H3 Grid Generation**
    - **H3 Grid Generation Module (`gigaspatial/grid/h3.py`):**
        - Introduced `H3Hexagons` class for managing H3 cell IDs.
        - Supports creation from lists of hexagons, geographic bounds, spatial geometries, or points.
        - Provides methods to convert H3 hexagons to pandas DataFrames and GeoPandas GeoDataFrames.
        - Includes functionalities for filtering, getting k-ring neighbors, compacting hexagons, and getting children/parents at different resolutions.
        - Allows saving H3Hexagons to JSON, Parquet, or GeoJSON files.
    - **Country-Specific H3 Hexagons (`CountryH3Hexagons`):**
        - Extends `H3Hexagons` for generating H3 grids constrained by country boundaries.
        - Integrates with `AdminBoundaries` to fetch country geometries for precise H3 cell generation.

- **Documentation**
    - Improved `tif.md` example to showcase multi-raster initialization, explicit reprojection, and graph conversion.

### Changed

- **TifProcessor**
  - **Improved Temporary File Management:** Refactored temporary file handling for merging and reprojection using `tempfile.mkdtemp()` and `shutil.rmtree` for more robust and reliable cleanup. Integrated with context manager (`__enter__`, `__exit__`) and added a dedicated `cleanup()` method.
  - **Reprojection during Initialization:** Implemented automatic reprojection of single rasters to a specified `target_crs` during `TifProcessor` initialization.
  - **Enhanced `open_dataset` Context Manager:** The `open_dataset` context manager now intelligently opens the most up-to-date (merged or reprojected) version of the dataset.
  - **More Flexible Multi-Dataset Validation:** Modified `_validate_multiple_datasets` to issue a warning instead of raising an error for CRS mismatches when `target_crs` is not set.
  - **Optimized `_get_reprojection_profile`:** Dynamically calculates transform and dimensions based on `reprojection_resolution` and added LZW compression to reprojected TIFF files to reduce file size.

- **ADLSDataStore Enhancements**
    - **New `copy_file` method:** Implemented a new method for copying individual files within ADLS, with an option to overwrite existing files.
    - **New `rename` method:** Added a new method to rename (move) files in ADLS, which internally uses `copy_file` and then deletes the source, with options for overwrite, waiting for copy completion, and polling.
    - **Revamped `rmdir` method:** Modified `rmdir` to perform batch deletions of blobs, addressing the Azure Blob batch delete limit (256 sub-requests) and improving efficiency for large directories.

- **LocalDataStore Enhancements**
    - **New `copy_file` method:** Implemented a new method for copying individual files.

### Removed

- Removed deprecated `tabular` property and `get_zoned_geodataframe` method from `TifProcessor`. Users should now use `to_dataframe()` and `to_geodataframe()` respectively.

### Dependencies

- Added `networkx` and `h3` as new dependencies.

### Fixed

- Several small fixes and improvements to aggregation methods.

## [v0.6.9] - 2025-07-26

### Fixed

- Resolved a bug in the handler base class where non-hashable types (dicts) were incorrectly used as dictionary keys in `unit_to_path` mapping, preventing potential runtime errors during data availability checks.

## [v0.6.8] - 2025-07-26

### Added

- **OSMLocationFetcher Enhancements**
  - Support for querying OSM locations by arbitrary administrative levels (e.g., states, provinces, cities), in addition to country-level queries.
  - New optional parameters:
    - `admin_level`: Specify OSM administrative level (e.g., 4 for states, 6 for counties).
    - `admin_value`: Name of the administrative area to query (e.g., "California").
  - New static method `get_admin_names(admin_level, country=None)`:
    - Fetch all administrative area names for a given `admin_level`, optionally filtered by country.
    - Helps users discover valid admin area names for constructing precise queries.

- **Multi-Raster Merging Support in TifProcessor**
  - Added ability to initialize `TifProcessor` with **multiple raster datasets**.
  - Merges rasters on load with configurable strategies:
    - Supported `merge_method` options: `first`, `last`, `min`, `max`, `mean`.
  - Supports **on-the-fly reprojection** for rasters with differing coordinate reference systems via `target_crs`.
  - Handles **resampling** using `resampling_method` (default: `nearest`).
  - Comprehensive validation to ensure compatibility of input rasters (e.g., resolution, nodata, dtype).
  - Temporary file management for merged output with automatic cleanup.
  - Backward compatible with single-raster use cases.

  **New TifProcessor Parameters:**
  - `merge_method` (default: `first`) – How to combine pixel values across rasters.
  - `target_crs` (optional) – CRS to reproject rasters before merging.
  - `resampling_method` – Resampling method for reprojection.

  **New Properties:**
  - `is_merged`: Indicates whether the current instance represents merged rasters.
  - `source_count`: Number of raster datasets merged.

### Changed

- **OSMLocationFetcher Overpass Query Logic**
  - Refactored Overpass QL query builder to support **subnational queries** using `admin_level` and `admin_value`.
  - Improved flexibility and precision for spatial data collection across different administrative hierarchies.

### Breaking Changes

- None. All changes are fully backward compatible.

## [v0.6.7] - 2025-07-16

### Fixed

- Fixed a bug in WorldPopHandler/ADLSDataStore integration where a `Path` object was passed instead of a string, causing a `quote_from_bytes() expected bytes` error during download.

## [v0.6.6] - 2025-07-15

### Added

- **`AdminBoundaries.from_global_country_boundaries(scale="medium")`**
  - New class method to load global admin level 0 boundaries from Natural Earth.
  - Supports `"large"` (10m), `"medium"` (50m), and `"small"` (110m) scale options.

- **WorldPop Handler Refactor (API Integration)**
  - Introduced `WPPopulationHandler`, `WPPopulationConfig`, `WPPopulationDownloader`, and `WPPopulationReader`.
  - Uses new `WorldPopRestClient` to dynamically query the WorldPop REST API.
  - Replaces static metadata files and hardcoded logic with API-based discovery and download.
  - Country code lookup and dataset filtering now handled at runtime.
  - Improved validation, extensibility, logging, and error handling.

- **POI-Based WorldPop Mapping**
  - `PoiViewGenerator.map_wp_pop()` method:
    - Maps WorldPop population data around POIs using flexible spatial predicates:
      - `"centroid_within"`, `"intersects"`, `"fractional"` (1000m only), `"within"`
    - Supports configurable radius and resolution (100m or 1000m).
    - Aggregates population data and appends it to the view.

- **Geometry-Based Zonal WorldPop Mapping**
  - `GeometryBasedZonalViewGenerator.map_wp_pop()` method:
    - Maps WorldPop population data to polygons/zones using:
      - `"intersects"` or `"fractional"` predicate
    - Returns zonal population sums as a new view column.
    - Handles predicate-dependent data loading (raster vs. GeoDataFrame).

### Changed

- **Refactored `BaseHandler.ensure_data_available`**
  - More efficient data check and download logic.
  - Downloads only missing units unless `force_download=True`.
  - Cleaner structure and better reuse of `get_relevant_data_units()`.

- **Refactored WorldPop Module**
  - Complete handler redesign using API-based architecture.
  - Dataset paths and URLs are now dynamically constructed from API metadata.
  - Resolution/year validation is more robust and descriptive.
  - Removed static constants, gender/school_age toggles, and local CSV dependency.

### Fixed
- Several small fixes and improvements to zonal aggregation methods, especially around CRS consistency, missing values, and result alignment.

## [v0.6.5] - 2025-07-01

### Added

- **`MercatorTiles.get_quadkeys_from_points()`**  
  New static method for efficient 1:1 point-to-quadkey mapping using coordinate-based logic, improving performance over spatial joins.

- **`AdminBoundariesViewGenerator`**  
  New generator class for producing zonal views based on administrative boundaries (e.g., districts, provinces) with flexible source and admin level support.

- **Zonal View Generator Enhancements**  
  - `_view`: Internal attribute for accumulating mapped statistics.  
  - `view`: Exposes current state of zonal view.  
  - `add_variable_to_view()`: Adds mapped data from `map_points`, `map_polygons`, or `map_rasters` with robust validation and zone alignment.  
  - `to_dataframe()` and `to_geodataframe()` methods added for exporting current view in tabular or spatial formats.

- **`PoiViewGenerator` Enhancements**  
  - Consistent `_view` DataFrame for storing mapped results.  
  - `_update_view()`: Central method to update POI data.  
  - `save_view()`: Improved format handling (CSV, Parquet, GeoJSON, etc.) with geometry recovery.  
  - `to_dataframe()` and `to_geodataframe()` methods added for convenient export of enriched POI view.  
  - Robust duplicate ID detection and CRS validation in `map_zonal_stats`.

- **`TifProcessor` Enhancements**  
  - `sample_by_polygons_batched()`: Parallel polygon sampling.  
  - Enhanced `sample_by_polygons()` with nodata masking and multiple stats.  
  - `warn_on_error`: Flag to suppress sampling warnings.

- **GeoTIFF Multi-Band Support**  
  - `multi` mode added for multi-band raster support.  
  - Auto-detects band names via metadata.  
  - Strict validation of band count based on mode (`single`, `rgb`, `rgba`, `multi`).

- **Spatial Distance Graph Algorithm**  
  - `build_distance_graph()` added for fast KD-tree-based spatial matching.  
  - Supports both `DataFrame` and `GeoDataFrame` inputs.  
  - Outputs a `networkx.Graph` with optional DataFrame of matches.  
  - Handles projections, self-match exclusion, and includes verbose stats/logs.

- **Database Integration (Experimental)**  
  - Added `DBConnection` class in `core/io/database.py` for unified Trino and PostgreSQL access.  
  - Supports schema/table introspection, query execution, and reading into `pandas` or `dask`.  
  - Handles connection creation, credential management, and diagnostics.  
  - Utility methods for schema/view/table/column listings and parameterized queries.

- **GHSL Population Mapping**  
  - `map_ghsl_pop()` method added to `GeometryBasedZonalViewGenerator`.  
  - Aggregates GHSL population rasters to user-defined zones.  
  - Supports `intersects` and `fractional` predicates (latter for 1000m resolution only).  
  - Returns population statistics (e.g., `sum`) with customizable column prefix.

### Changed

- **`MercatorTiles.from_points()`** now internally uses `get_quadkeys_from_points()` for better performance.

- **`map_points()` and `map_rasters()`** now return `Dict[zone_id, value]` to support direct usage with `add_variable_to_view()`.

- **Refactored `aggregate_polygons_to_zones()`**  
  - `area_weighted` deprecated in favor of `predicate`.  
  - Supports flexible predicates like `"within"`, `"fractional"` for spatial aggregation.  
  - `map_polygons()` updated to reflect this change.

- **Optional Admin Boundaries Configuration**  
  - `ADMIN_BOUNDARIES_DATA_DIR` is now optional.  
  - `AdminBoundaries.create()` only attempts to load if explicitly configured or path is provided.  
  - Improved documentation and fallback behavior for missing configs.

### Fixed

- **GHSL Downloader**  
  - ZIP files are now downloaded into a temporary cache directory using `requests.get()`.  
  - Avoids unnecessary writes and ensures cleanup.

- **`TifProcessor`**  
  - Removed polygon sampling warnings unless explicitly enabled.

### Deprecated

- `TifProcessor.tabular` → use `to_dataframe()` instead.  
- `TifProcessor.get_zoned_geodataframe()` → use `to_geodataframe()` instead.  
- `area_weighted` → use `predicate` in aggregation methods instead.

## [v0.6.4] - 2025-06-19

### Added

- **GigaSchoolProfileFetcher**  
  - New class to fetch and process school profile data from the Giga School Profile API  
  - Supports paginated fetching, filtering by country and school ID  
  - Includes methods to generate connectivity summary statistics by region, connection type, and source

- **GigaSchoolMeasurementsFetcher**  
  - New class to fetch and process daily real-time connectivity measurements from the Giga API  
  - Supports filtering by date range and school  
  - Includes performance summary generation (download/upload speeds, latency, quality flags)

- **AdminBoundaries.from_geoboundaries**  
  - New class method to download and process geoBoundaries data by country and admin level  
  - Automatically handles HDX dataset discovery, downloading, and fallback logic

- **HDXConfig.search_datasets**  
  - Static method to search HDX datasets without full handler initialization  
  - Supports query string, sort order, result count, HDX site selection, and custom user agent

### Fixed

- Typo in `MaxarImageDownloader` causing runtime error

### Documentation

- **Improved Configuration Guide** (`docs/user-guide/configuration.md`)  
  - Added comprehensive table of environment variables with defaults and descriptions  
  - Synced `.env_sample` and `config.py` with docs  
  - Example `.env` file and guidance on path overrides using `config.set_path`  
  - New section on `config.ensure_directories_exist` and troubleshooting tips  
  - Clearer handling of credentials and security notes  
  - Improved formatting and structure for clarity

## [v0.6.3] - 2025-06-16

### Added
- Major refactor of `HDX` module to align with unified `BaseHandler` architecture:
  - `HDXConfig`: fully aligned with `BaseHandlerConfig` structure.
  - Added flexible pattern matching for resource filtering.
  - Improved data unit resolution by country, geometry, and points.
  - Enhanced resource filtering with exact and regex options.
- `HDXDownloader` fully aligned with `BaseHandlerDownloader`:
  - Simplified sequential download logic.
  - Improved error handling, validation, and logging.
- `HDXReader` fully aligned with `BaseHandlerReader`:
  - Added `resolve_source_paths` and `load_all_resources` methods.
  - Simplified source handling for single and multiple files.
  - Cleaned up redundant and dataset-specific logic.
- Introduced `HDXHandler` as unified orchestration layer using factory methods.

- Refactor of `RelativeWealthIndex (RWI)` module:
  - Added new `RWIHandler` class aligned with `HDXHandler` and `BaseHandler`.
  - Simplified class names: `RWIDownloader` and `RWIReader`.
  - Enhanced configuration with `latest_only` flag to select newest resources automatically.
  - Simplified resource filtering and country resolution logic.
  - Improved code maintainability, type hints, and error handling.


- **New raster multi-band support in TifProcessor:**
  - Added new `multi` mode for handling multi-band raster datasets.
  - Automatic band name detection from raster metadata.
  - Added strict mode validation (`single`, `rgb`, `rgba`, `multi`).
  - Enhanced error handling for invalid modes and band counts.

### Fixed
- Fixed GHSL tiles loading behavior for correct coordinate system handling:
  - Moved `TILES_URL` formatting and tile loading to `validate_configuration`.
  - Ensures proper tile loading after CRS validation.

### Documentation
- Updated and standardized API references across documentation.
- Standardized handler method names and usage examples.
- Added building enrichment examples for POI processing.
- Updated installation instructions.

### Deprecated
- Deprecated direct imports from individual handler modules.

## [v0.6.2] - 2025-06-11

### Added
- New `ROOT_DATA_DIR` configuration option to set a base directory for all data tiers
  - Can be configured via environment variable `ROOT_DATA_DIR` or `.env` file
  - Defaults to current directory (`.`) if not specified
  - All tier data paths (bronze, silver, gold, views) are now constructed relative to this root directory
  - Example: Setting `ROOT_DATA_DIR=/data/gigaspatial` will store all data under `/data/gigaspatial/bronze`, `/data/gigaspatial/silver`, etc.

### Fixed
- Fixed URL formatting in GHSL tiles by using Enum value instead of Enum member
  - Ensures consistent URL formatting with numeric values (4326) instead of Enum names (WGS84)
  - Fixes URL formatting issue across different Python environments

- Refactored GHSL downloader to follow DataStore abstraction
  - Directory creation is now handled by DataStore implementation
  - Removed redundant directory creation logic from download_data_unit method
  - Improves separation of concerns and makes the code more maintainable

## [v0.6.1] - 2025-06-09

### Fixed

- Gracefully handle missing or invalid GeoRepo API key in `AdminBoundaries.create()`:
  - Wrapped `GeoRepoClient` initialization in a `try-except` block
  - Added fallback to GADM if GeoRepo client fails
  - Improved logging for better debugging and transparency

## [v0.6.0] - 2025-06-09

### Added

#### POI View Generator
- **`map_zonal_stats`**: New method for enriched spatial mapping with support for:
  - Raster point sampling (value at POI location)
  - Raster zonal statistics (with buffer zone)
  - Polygon aggregation (with optional area-weighted averaging)
- **Auto-generated POI IDs** in `_init_points_gdf` for consistent point tracking.
- **Support for area-weighted aggregation** for polygon-based statistics.

#### BaseHandler Orchestration Layer
- New abstract `BaseHandler` class providing unified lifecycle orchestration for config, downloader, and reader.
- High-level interface methods:
  - `ensure_data_available()`
  - `load_data()`
  - `download_and_load()`
  - `get_available_data_info()`
- Integrated factory pattern for safe and standardized component creation.
- Built-in context manager support for resource cleanup.
- Fully backwards compatible with existing handler architecture.

#### Handlers Updated to Use BaseHandler
- `GoogleOpenBuildingsHandler`
- `MicrosoftBuildingsHandler`
- `GHSLDataHandler`
  - All now inherit from `BaseHandler`, supporting standardized behavior and cleaner APIs.

---

### Changed

#### POI View Generator
- `map_built_s` and `map_smod` now internally use the new `map_zonal_stats` method.
- `tif_processors` renamed to `data` to support both raster and polygon inputs.
- Removed parameters:
  - `id_column` (now handled internally)
  - `area_column` (now automatically calculated)

#### Internals and Usability
- Improved error handling with clearer validation messages.
- Enhanced logging for better visibility during enrichment.
- More consistent use of coordinate column naming.
- Refined type hints and parameter documentation across key methods.

---

### Notes

- Removed legacy POI generator classes and redundant `poi.py` file.
- Simplified imports and removed unused handler dependencies.
- All POI generator methods now include updated docstrings, parameter explanations, and usage examples.
- Added docs on the new `BaseHandler` interface and handler refactors.

## [v0.5.0] - 2025-06-02

### Changed

- **Refactored data loading architecture**:
  - Introduced **dedicated reader classes** for major datasets (Microsoft Global Buildings, Google Open Buildings, GHSL), each inheriting from a new `BaseHandlerReader`.
  - Centralized **file existence checks** and **raster/tabular loading** methods in `BaseHandlerReader`.
  - Improved maintainability by encapsulating dataset-specific logic inside each reader class.

- **Modularized source resolution**:
  - Each reader now supports resolving data **by country, geometry, or individual points**, improving code reuse and flexibility.

- **Unified POI enrichment**:
  - Merged all POI generators (Google Open Buildings, Microsoft Global Buildings, GHSL Built Surface, GHSL SMOD) into a single `PoiViewGenerator` class.
  - Supports flexible inputs: list of `(lat, lon)` tuples, list of dicts, DataFrame, or GeoDataFrame.
  - Maintains consistent internal state via `points_gdf`, updated after each mapping.
  - Enables **chained enrichment** of POI data using multiple datasets.

- **Modernized internal data access**:
  - All data loading now uses dedicated **handler/reader classes**, improving consistency and long-term maintainability.

### Fixed

- **Full DataStore integration**:
  - Fixed `OpenCellID` and `HDX` handlers to fully support the `DataStore` abstraction.
  - All file reads, writes, and checks now use the configured `DataStore` (local or cloud).
  - Temporary files are only used during downloads; final data is always stored and accessed via the DataStore interface.

### Removed

- Removed deprecated POI generator classes and the now-obsolete poi submodule. All enrichment is handled through the unified `PoiViewGenerator`.

### Notes

- This release finalizes the architectural refactors started in `v0.5.0`.
- While marked stable, please report any issues or regressions from the new modular structure.

## [v0.5.0b1] - 2025-05-27

### Added
- **New Handlers**:
  - `hdx.py`: Handler for downloading and managing Humanitarian Data Exchange datasets.
  - `rwi.py`: Handler for the Relative Wealth Index dataset.
  - `opencellid.py`: Handler for OpenCellID tower locations.
  - `unicef_georepo.py`: Integration with UNICEF’s GeoRepo asset repository.
- **Zonal Generators**:
  - Introduced the `generators/zonal/` module to support spatial aggregations of various data types (points, polygons, rasters)
    to zonal geometries such as grid tiles or catchment areas.
- **New Geo-Processing Methods**:
  - Added methods to compute centroids of (Multi)Polygon geometries.
  - Added methods to calculate area of (Multi)Polygon geometries in square meters.

### Changed
- **Refactored**:
  - `config.py`: Added support for new environment variables (OpenCellID and UNICEF GeoRepo keys).
  - `geo.py`: Enhanced spatial join functions for improved performance and clarity.
  - `handlers/`: 
    - Minor robustness improvements in `google_open_buildings` and `microsoft_global_buildings`.
    - Added a new class method in `boundaries` for initializing admin boundaries from UNICEF GeoRepo.
  - `core/io/`: 
    - Added `list_directories` method to both ADLS and local storage backends.
- **Documentation & Project Structure**:
  - Updated `.env_sample` and `.gitignore` to align with new environment variables and data handling practices.

### Dependencies
- Updated `requirements.txt` and `setup.py` to reflect new dependencies and ensure compatibility.

### Notes
- This is a **pre-release** (`v0.5.0b1`) and is intended for testing and feedback.
- Some new modules, especially in `handlers` and `generators`, are experimental and may be refined in upcoming releases.

## [v0.4.1] - 2025-04-17
### Added
- **Documentation**:
	- Added **API Reference** documentation for all modules, classes, and functions.
	- Added a **Configuration Guide** to explain how to set up paths, API keys, and other.
- **TifProcessor**: added new to_dataframe method.
- **config**: added set_path method for dynamic path management.

### Changed
- **Documentation**:
	- Restructured the `docs/` directory to improve organization and navigation.
	- Updated the `index.md` for the **User Guide** to provide a clear overview of available documentation.
	- Updated **Examples** for downloading, processing, and storing geospatial data - more to come.
- **README**:
	- Updated the README with a clear description of the package’s purpose and key features.
	- Added a section on **View Generators** to explain spatial context enrichment and mapping to grid or POI locations.
	- Included a **Supported Datasets** section with an image of dataset provider logos.

### Fixed
- Handled errors when processing nodes, relations, and ways in **OSMLocationFetcher**.
- Made `admin1` and `admin1_id_giga` optional in GigaEntity instances for countries with no admin level 1 divisions.

## [v0.4.0] - 2025-04-01
### Added
- **POI View Generators**: Introduced a new module, generators, containing a base class for POI view generation.
- **Expanded POI Support**: Added new classes for generating POI views from:
	- **Google Open Buildings**
	- **Microsoft Global Buildings**
	- **GHSL Settlement Model**
	- **GHSL Built Surface**
- **New Reader**: Added read_gzipped_json_or_csv to handle compressed JSON/CSV files.

### Changed
- **ADLSDataStore Enhancements**: Updated methods to match LocalDataStore for improved consistency.
- **Geo Processing Updates**:
	- Improved convert_to_dataframe for more efficient data conversion.
	- Enhanced annotate_with_admin_regions to improve spatial joins.
- **New TifProcessor Methods**:
	- sample_by_polygons for polygon-based raster sampling.
	- sample_multiple_tifs_by_coordinates & sample_multiple_tifs_by_polygons to manage multi-raster sampling.
- **Fixed Global Config Handling**: Resolved issues with handling configurations inside classes.

## [v0.3.2] - 2025-03-21
### Added
- Added a method to efficiently assign unique IDs to features.

### Changed
- Enhanced logging for better debugging and clarity.

### Fixed
- Minor bug fix in config.py

## [0.3.1] - 2025-03-20

### Added
- Enhanced AdminBoundaries handler with improved error handling for cases where administrative level data is unavailable for a country.
- Added pyproject.toml and setup.py, enabling pip install support for the package.
- Introduced a new method annotate_with_admin_regions in geo.py to perform spatial joins between input points and administrative boundaries (levels 1 and 2), handling conflicts where points intersect multiple admin regions.

### Removed
- Removed the utils module containing logger.py and integrated LOG_FORMAT and get_logger into config.py for a more streamlined logging approach.

## [0.3.0] - 2025-03-18  
### Added  
- Compression support in readers for improved efficiency  
- New GHSL data handler to manage GHSL dataset downloads  

### Fixed  
- Small fixes/improvements in Microsoft Buildings, Maxar, and Overture handlers  

## [v0.2.2] - 2025-03-12
- **Refactored Handlers**: Improved structure and performance of maxar_image.py, osm.py and overture.py to enhance geospatial data handling.

- **Documentation Improvements**:
	- Updated index.md, advanced.md, and use-cases.md for better clarity.
	- Added installation.md under docs/getting-started for setup guidance.
	- Refined API documentation in docs/api/index.md.

- **Configuration & Setup Enhancements**:
	•	Improved .gitignore to exclude unnecessary files.
	•	Updated mkdocs.yml for better documentation structuring.
- **Bug Fixes & Minor Optimizations**: Small fixes and improvements across the codebase for stability and maintainability.

## [v0.2.1] - 2025-02-28
### Added
- Introduced WorldPopDownloader feature to handlers
- Refactored TifProcessor class for better performance

### Fixed
- Minor bug fixes and performance improvements

## [v0.2.0] - MaxarImageDownloader & Bug Fixes - 2025-02-24
- **New Handler**: MaxarImageDownloader for downloading Maxar images.
- **Bug Fixes**: Various improvements and bug fixes.
- **Enhancements**: Minor optimizations in handlers.

## [v0.1.1] - 2025-02-24

### Added
- **Local Data Store**: Introduced a new local data store alongside ADLS to improve data storage and read/write functionality.
- **Boundaries Handler**: Added `boundaries.py`, a new handler that allows to read administrative boundaries from GADM.

### Changed
- **Handler Refactoring**: Refactored existing handlers to improve modularity and data handling.
- **Configuration Management**: Added `config.py` to manage paths, runtime settings, and environment variables.

### Removed
- **Administrative Schema**: Removed `administrative.py` since its functionality is now handled by the `boundaries` handler.
- **Globals Module**: Removed `globals.py` and replaced it with `config.py` for better configuration management.

### Updated Files
- `config.py`
- `boundaries.py`
- `google_open_buildings.py`
- `mapbox_image.py`
- `microsoft_global_buildings.py`
- `ookla_speedtest.py`
- `mercator_tiles.py`
- `adls_data_store.py`
- `data_store.py`
- `local_data_store.py`
- `readers.py`
- `writers.py`
- `entity.py`

## [v0.1.0] - 2025-02-07
### Added
- New data handlers: `google_open_buildings.py`, `microsoft_global_buildings.py`, `overture.py`, `mapbox_image.py`, `osm.py`
- Processing functions in `tif_processor.py`, `geo.py` and `transform.py`
- Grid generation modules: `h3_tiles.py`, `mercator_tiles.py`
- View managers: `grid_view.py` and `national_view.py`
- Schemas: `administrative.py`

### Changed
- Updated `requirements.txt` with new dependencies
- Improved logging and data storage mechanisms

### Removed
- Deprecated views: `h3_view.py`, `mercator_view.py`
