# Changelog

All notable changes to this project will be documented in this file.

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