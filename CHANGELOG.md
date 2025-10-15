# Changelog

All notable changes to this project will be documented in this file.

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