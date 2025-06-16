# Changelog

## [Unreleased]

### Changed
- Simplified import structure: all handlers now available directly from `gigaspatial.handlers`
- Updated `LocalDataStore` import path to `gigaspatial.core.io`
- Standardized handler method names across the codebase
- Updated documentation and examples to reflect current API

### Added
- New `PoiViewGenerator` for mapping nearest buildings to points
- Enhanced building data handling with improved point-to-building mapping

### Fixed
- Import path inconsistencies
- Documentation and installation instructions

### Deprecated
- Direct imports from individual handler modules
- Old downloader classes in favor of unified handlers