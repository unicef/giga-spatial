# Changelog

All notable changes to this project will be documented in this file.

## [v0.6.7] - 2024-07-16

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