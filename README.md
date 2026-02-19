<div style="padding-left: 20px; padding-right: 10px;">
<a href="https://giga.global/">
    <img src="https://s41713.pcdn.co/wp-content/uploads/2018/11/2020.05_GIGA-visual-identity-guidelines_v1-25.png" alt="Giga logo" title="Giga" align="right" height="60" style="padding-top: 10px;"/>
</a>

# GigaSpatial

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-brightgreen.svg)](https://opensource.org/license/agpl-v3)
[![PyPI version](https://badge.fury.io/py/giga-spatial.svg)](https://badge.fury.io/py/giga-spatial)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/giga-spatial.svg?color=dark-green)](https://pypi.org/project/giga-spatial/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI Downloads](https://static.pepy.tech/badge/giga-spatial)](https://pepy.tech/projects/giga-spatial)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/y/unicef/giga-spatial.svg?color=dark-green)](https://github.com/unicef/giga-spatial/graphs/contributors)

**Table of contents**

- [About Giga](#about-giga)
- [About GigaSpatial](#about-gigaspatial)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Key workflows](#key-workflows)
- [Core concepts](#core-concepts)
- [Supported datasets](#supported-datasets)
- [Why use GigaSpatial?](#why-use-gigaspatial)
- [Why open source?](#why-open-source)
- [How to contribute](#how-to-contribute)
- [Code of conduct](#code-of-conduct)
- [Stay connected](#stay-connected)


## About Giga

[Giga](https://giga.global/) is a UNICEF-ITU initiative to connect every school to the Internet and every young person to information, opportunity and choice. 
Giga maps schools' Internet access in real time, creates models for innovative financing, and supports governments contracting for connectivity. 

## About GigaSpatial

**GigaSpatial** is a Python toolkit for scalable geospatial data download, processing, and enrichment, designed for use across diverse domains such as infrastructure mapping, accessibility analysis, and environmental studies.

> Originally developed within UNICEF’s Giga initiative, GigaSpatial now provides a general‑purpose geospatial toolkit that can be applied to many contexts, including but not limited to school connectivity analysis.

### Who is this for?

- Data engineers building reproducible geospatial pipelines
- Data scientists analyzing school connectivity and infrastructure
- Researchers working with large, multi‑source spatial datasets
- GIS analysts requiring planetary-scale Earth observation data

## Installation

GigaSpatial requires Python 3.10 or above.

```console
pip install giga-spatial
```

The package depends on:

- geopandas
- pandas
- shapely
- rasterio
- earthengine-api (optional, for Google Earth Engine features)

For detailed setup instructions (including recommended environments and system dependencies), see the [installation docs](https://unicef.github.io/giga-spatial/getting-started/installation/).

We recommend using a virtual environment for installation.

## Quick start

```python
import geopandas as gpd
from gigaspatial.handlers import GoogleOpenBuildingsHandler, GHSLDataHandler
from gigaspatial.generators import POIViewGenerator

# 1. Load school locations
schools = gpd.read_file("schools.geojson")

# 2. Prepare data sources (downloads / caching handled by handlers)
buildings = GoogleOpenBuildingsHandler().load_data(source=schools, data_type="points")
ghsl = GHSLDataHandler(product="GHS_SMOD").load_data(source=schools, merge_rasters=True)

# 3. Generate school mappings with buildings + settlement model
view = POIViewGenerator(points=points)
ghsl_mapping = view.map_zonal_stats(data=ghsl, stat="median", output_column="smod_median")

print(ghsl_mapping.head())

buildings_mapping = view.map_zonal_stats(data=ghsl, stat="median", output_column="smod_median")
buildings_mapping = view.map_nearest_points(
    points_df=buildings,
    id_column="full_plus_code",
    output_prefix="nearest_google_building",
)

print(buildings_mapping.head())

```

## Key Features

- **Data Downloading**
  Download geospatial data from various sources including GHSL, Microsoft Global Buildings, Google Open Buildings, OpenCellID, and HDX datasets.

- **Data Processing** 
  Process and transform geospatial data, such as GeoTIFF files and vector data, with support for compression and efficient handling.

- **View Generators** 
  - Enrich spatial context with POI (Point of Interest) data
  - Support for raster point sampling and zonal statistics
  - Area-weighted aggregation for polygon-based statistics
  - Temporal aggregation for time-series Earth observation data

- **Grid System**
  Create and manipulate grid-based geospatial data for analysis and modeling using H3, S2, or Mercator tile systems.

- **Data Storage**
  Flexible storage options with local, cloud (ADLS), and Snowflake stage support.

- **Configuration Management**
  - Centralized configuration via environment variables or `.env` file
  - Easy setup of API keys and paths

## Key Workflows

- **Fetch POI data**
  Retrieve points of interest from OpenStreetMap, Healthsites.io, and Giga-maintained sources for any area of interest.

- **Enrich POI locations**
  Join POIs with Google/Microsoft building footprints, GHSL population and settlement layers, Earth Engine satellite data, and other contextual datasets.

- **Analyze Earth observation time series**
  Extract and analyze multi-temporal satellite data (vegetation indices, land surface temperature, precipitation, etc.) for any location using Google Earth Engine

- **Build and analyze grids**
  Generate national or sub‑national grids and aggregate multi‑source indicators (e.g. coverage, population, infrastructure) into each cell.

- **End‑to‑end pipelines**
  Use handlers, readers, and view generators together to go from raw data download to analysis‑ready tables in local storage, ADLS, or Snowflake.


## Core concepts

- **Handlers**: Orchestrate dataset lifecycle (download, cache, read) for sources like GHSL, Google/Microsoft buildings, OSM, and HDX.
- **Readers**: Low‑level utilities that parse and standardize raster and vector formats.
- **View generators**: High‑level components that enrich points or grids with contextual variables (POIs, buildings, population, etc.).
- **Grid system**: Utilities to build and manage grid cells for large‑scale analysis.
- **Storage backends**: Pluggable interfaces for local disk, Azure Data Lake Storage, and Snowflake stages.

## Supported Datasets

The `gigaspatial` package supports data from the following providers:

<div align="center">
    <img src="https://raw.githubusercontent.com/unicef/giga-spatial/main/docs/assets/datasets.png" alt="Dataset Providers" style="width: 75%; height: auto;"/>
</div>

**Google Earth Engine Catalog**

GigaSpatial now provides access to Google Earth Engine’s comprehensive data catalog, including:

- **Satellite imagery**: Landsat (30+ years), Sentinel-1/2, MODIS, Planet
- **Climate & weather**: ERA5, CHIRPS precipitation, NOAA temperature
- **Land cover**: Dynamic World, ESA WorldCover, MODIS land cover
- **Terrain**: SRTM, ASTER DEM, ALOS elevation data
- **Population & infrastructure**: GHSL, WorldPop, nighttime lights
- **Environmental**: Soil properties, vegetation indices, surface water

For a complete list of available datasets, visit the [Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets).

---

## View Generators

The **view generators** in GigaSpatial are designed to enrich the spatial context of school locations and map data into grid or POI locations. This enables users to analyze and visualize geospatial data in meaningful ways.

### Key Capabilities
1. **Spatial Context Enrichment**:
   - Automatic attribution of geospatial variables to school locations
   - Contextual layers for environmental, infrastructural, and socioeconomic factors
   - Multi-resolution data availability for different analytical needs
   - Support for both point and polygon-based enrichment

2. **Mapping to Grid or POI Locations**:
   - Map geospatial data to grid cells for scalable analysis
   - Map data to POI locations for detailed, location-specific insights
   - Support for chained enrichment using multiple datasets
   - Built-in support for administrative boundary annotations

---

## Why use GigaSpatial?

- **End-to-end geospatial pipelines**: Go from raw open datasets (OSM, GHSL, global buildings, HDX, etc.) to analysis-ready tables with a consistent set of handlers, readers, and view generators.

- **Planetary-scale analysis**: Leverage Google Earth Engine’s cloud infrastructure to process petabytes of satellite imagery without downloading data or managing compute resources.

- **Scalable analysis**: Work seamlessly with both point and grid representations, making it easy to aggregate indicators at national scale or zoom into local POIs.

- **Batteries included for enrichment**: Fetch POIs, buildings, and population layers and join them onto schools or other locations with a few lines of code.

- **Flexible storage**: Run the same workflows against local files, Azure Data Lake Storage (ADLS), or Snowflake stages without changing core logic.

- **Modern, extensible architecture**: Base handler orchestration, dataset-specific readers, modular source resolution, and structured logging make it straightforward to add new sources and maintain production pipelines.

- **Open and collaborative**: Developed in the open under an AGPL-3.0 license, with contributions and reviews from the wider geospatial and data-for-development community.

## Why Open Source?  

At Giga, we believe in the power of open-source technologies to accelerate progress and innovation. By keeping our tools and systems open, we:  
- Encourage collaboration and contributions from a global community.  
- Ensure transparency and trust in our methodologies.  
- Empower others to adopt, adapt, and extend our tools to meet their needs.  

## How to Contribute  

We welcome contributions to our repositories! Whether it's fixing a bug, adding a feature, or improving documentation, your input helps us move closer to our goal of universal school connectivity.  

### Steps to Contribute  
1. Fork the repository you'd like to contribute to.  
2. Create a new branch for your changes.  
3. Submit a pull request with a clear explanation of your contribution. 

To go through the “contribution” guidelines in detail you can visit the following link. 

[Click here for the detailed Contribution guidelines](https://github.com/unicef/giga-spatial/blob/main/CONTRIBUTING.md)

---

## Code of Conduct  

At Giga, we're committed to maintaining an environment that's respectful, inclusive, and harassment-free for everyone involved in our project and community. We welcome contributors and participants from diverse backgrounds and pledge to uphold the standards.

[Click here for the detailed Code of Conduct.](https://github.com/unicef/giga-spatial/blob/main/CODE_OF_CONDUCT.md)

---

## Stay Connected  

To learn more about Giga and our mission, visit our official website: [Giga.Global](https://giga.global)

## Join Us

Join us in creating an open-source future for education! 🌍  
