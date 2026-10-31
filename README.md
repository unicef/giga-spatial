<p align="center">
  <a href="https://unicef.github.io/giga-spatial/">
    <img src="https://raw.githubusercontent.com/unicef/giga-spatial/main/docs/assets/gigaspatial-logo.png" alt="GigaSpatial" width="420">
  </a>
</p>

<p align="center">
  <a href="https://pypi.org/project/giga-spatial/"><img src="https://img.shields.io/pypi/v/giga-spatial.svg?color=277aff" alt="PyPI version"></a>
  <a href="https://pypi.org/project/giga-spatial/"><img src="https://img.shields.io/pypi/pyversions/giga-spatial.svg?color=277aff" alt="Python versions"></a>
  <a href="https://github.com/unicef/giga-spatial/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-277aff.svg" alt="License: AGPL-3.0"></a>
  <a href="https://pepy.tech/projects/giga-spatial"><img src="https://static.pepy.tech/badge/giga-spatial/month" alt="Downloads"></a>
  <a href="https://unicef.github.io/giga-spatial/"><img src="https://img.shields.io/badge/docs-unicef.github.io-277aff.svg" alt="Documentation"></a>
</p>

GigaSpatial acquires published geospatial datasets into a data store you control, loads them for a source, and processes them. A generator maps a dataset onto points of interest or zones, and the result is a view: those places with attributes from the datasets attached. It started in UNICEF's [Giga](https://giga.global/) initiative.

## What it covers

- **[Handlers](https://unicef.github.io/giga-spatial/latest/api/handlers/)** acquire datasets published as files, such as population, built-up area, and buildings.
- **[Fetchers](https://unicef.github.io/giga-spatial/latest/api/fetchers/)** query datasets served by an API, such as OpenStreetMap and Giga school data.
- **[Generators](https://unicef.github.io/giga-spatial/latest/api/generators/)** map datasets onto points of interest or zones and return a view.
- **[Processing](https://unicef.github.io/giga-spatial/latest/user-guide/processing/)** provides raster and vector tools for the stored files.
- **[Road networks](https://unicef.github.io/giga-spatial/latest/user-guide/road-networks/)**: `OSMRoadsFetcher` fetches OpenStreetMap roads for a source, `OSMRoadsHandler` builds them for whole countries from Geofabrik's extracts, and `SpatialNetwork` joins schools or other points to them, finds shortest paths, and keeps the roads that connect a set of points.

Files live in a [data store](https://unicef.github.io/giga-spatial/latest/api/data-stores/): local disk, Azure Data Lake Storage, or Snowflake. Handlers only depend on the abstract `DataStore`, so other stores can be added by implementing it.

The full list of datasets is on the [Datasets](https://unicef.github.io/giga-spatial/latest/getting-started/datasets/) page.

## Installation

GigaSpatial requires Python 3.10 or above.

```console
pip install giga-spatial
```

To install every optional integration (Earth Engine, BigQuery, Azure, Snowflake, and others):

```console
pip install "giga-spatial[all]"
```

The individual extras, such as `roads` for OpenStreetMap road networks, are listed on the [Installation](https://unicef.github.io/giga-spatial/latest/getting-started/installation/) page.

## Example

Map WorldPop population onto H3 cells for Rwanda. The first run downloads the population raster into the data store, and later runs read the stored file.

```python
from gigaspatial import H3ViewGenerator

zones = H3ViewGenerator(source="RWA", resolution=6)
view = zones.map_wp_pop(country="RWA")
print(view.head())
```

`view` is the H3 cells with population attached. The same pattern works with `S2ViewGenerator` and `MercatorViewGenerator`.

<p align="center">
  <img src="https://raw.githubusercontent.com/unicef/giga-spatial/main/docs/assets/readme-rwanda-h3-population.png" alt="Rwanda H3 cells at resolution 6 colored by WorldPop population" width="720">
</p>

## Documentation

- [Installation](https://unicef.github.io/giga-spatial/latest/getting-started/installation/)
- [Tutorial](https://unicef.github.io/giga-spatial/latest/getting-started/quickstart/)
- [Datasets](https://unicef.github.io/giga-spatial/latest/getting-started/datasets/)
- [Configuration](https://unicef.github.io/giga-spatial/latest/getting-started/configuration/)
- [User guide](https://unicef.github.io/giga-spatial/latest/user-guide/)
- [API reference](https://unicef.github.io/giga-spatial/latest/api/)

## Why open source?

At Giga, we believe in the power of open-source technologies to accelerate progress and innovation. By keeping our tools and systems open, we:

- Encourage collaboration and contributions from a global community.
- Ensure transparency and trust in our methodologies.
- Empower others to adopt, adapt, and extend our tools to meet their needs.

## Contributing

Contributions are welcome, from bug reports to new datasets. See the [contribution guidelines](https://github.com/unicef/giga-spatial/blob/main/CONTRIBUTING.md) to get started.

Everyone taking part is expected to follow the [Code of Conduct](https://github.com/unicef/giga-spatial/blob/main/CODE_OF_CONDUCT.md).

## Get in touch

- Report bugs and ask questions in [GitHub Issues](https://github.com/unicef/giga-spatial/issues).
- Follow Giga on [X](https://x.com/Gigaglobal) and [LinkedIn](https://www.linkedin.com/company/gigaglobal/).
- Learn more about Giga at [giga.global](https://giga.global/).

## About Giga

[Giga](https://giga.global/) is a UNICEF-ITU initiative to connect every school to the Internet and every young person to information, opportunity and choice. Giga maps schools' Internet access in real time, creates models for innovative financing, and supports governments contracting for connectivity.

GigaSpatial is developed by Giga's applied science team at UNICEF.

Licensed under [AGPL-3.0-or-later](https://github.com/unicef/giga-spatial/blob/main/LICENSE).
