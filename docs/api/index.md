# API Reference

Welcome to the API reference for the `gigaspatial` package. This documentation provides detailed information about the modules, classes, and functions available in the package.

## Modules

The `gigaspatial` package is organized into several modules, each serving a specific purpose:

### 1. **Handlers**
The `handlers` module contains classes for downloading and processing geospatial data from various sources, such as OpenStreetMap (OSM) and the Global Human Settlement Layer (GHSL).

- **OSMLocationFetcher**: Fetches and processes location data from OpenStreetMap.
- **GHSLDataDownloader**: Downloads and processes data from the Global Human Settlement Layer.

[Learn more about the Handlers module](handlers.md)

### 2. **Processing**
The `processing` module provides tools for processing geospatial data, such as GeoTIFF files.

- **TifProcessor**: Processes GeoTIFF files and extracts relevant data.

[Learn more about the Processing module](processing.md)

### 3. **Core**
The `core` module contains essential utilities and base classes used throughout the package.

- **DataStore**: Handles the storage and retrieval of geospatial data in various formats.
- **Config**: Manages configuration settings, such as paths and API keys.

[Learn more about the Core module](core.md)

### 4. **Generators**
The `generators` module includes tools for generating geospatial data, such as grids and synthetic datasets.

[Learn more about the Generators module](generators.md)

### 5. **Grid**
The `grid` module provides utilities for working with geospatial grids, such as creating and manipulating grid-based data.

[Learn more about the Grid module](grid.md)

---

## Getting Started

To get started with the `gigaspatial` package, follow the [Quick Start Guide](../getting-started/quickstart.md).

---

## Additional Resources

- [Examples](../examples/): Real-world examples and use cases.
- [Changelog](../changelog.md): Information about the latest updates and changes.
- [Contributing](../contributing.md): Guidelines for contributing to the project.

---

## Support

If you encounter any issues or have questions, feel free to [open an issue](https://github.com/unicef/giga-spatial/issues) or [join our Discord community](https://discord.com/invite/NStBwE7kyv).
