# Changelog

All notable changes to this project will be documented in this file.

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