# About

GigaSpatial started inside UNICEF's [Giga](https://giga.global/) initiative. Development began in January 2025.

## How GigaSpatial compares to other tools

- **Hand-built `geopandas`, `rasterio`, `rasterstats`, `pyrosm`, and `osmnx` pipelines**: GigaSpatial uses `geopandas`, `rasterio`, and `h3` under the hood, and replaces custom per-dataset download scripts, tile stitching, CRS projection, and buffer or zonal loops with handlers, fetchers, and generators that run in a few lines of Python.
- **Grid libraries (`h3-py`, `h3ronpy`, `h3-toolkit`)**: `h3-py` indexes coordinates and geometries into H3 cell IDs. `H3ViewGenerator`, `S2ViewGenerator`, `MercatorViewGenerator`, `AdminBoundariesViewGenerator`, and `GeometryBasedZonalViewGenerator` in GigaSpatial acquire published country datasets (WorldPop, GHSL, Google and Microsoft buildings, Relative Wealth Index) into your data store and map them directly onto H3 hexagons, S2 cells, Mercator tiles, administrative boundaries, or custom polygons.
- **Cloud catalogs and spatial engines (Google Earth Engine, Microsoft Planetary Computer, PostGIS, DuckDB, Apache Sedona)**: Cloud platforms and spatial databases run queries inside their own infrastructure. GigaSpatial downloads and caches published datasets into a data store you control (local disk, Azure Data Lake Storage, or Snowflake) and also provides optional `GEEProfiler` (`giga-spatial[gee]`), `BigQueryClient` (`giga-spatial[bq]`), and DuckDB or Trino integrations when you want them.

## Timeline

- **January 2025.** Development began.
- **7 February 2025.** v0.1.0, the first public release, and the first handlers.

Releases after that are in the [changelog](../changelog.md).

<div class="grid cards" markdown>

-   <a class="gs-card-link" href="team/"><span class="gs-card-title">Team</span>Who maintains GigaSpatial.</a>

-   <a class="gs-card-link" href="roadmap/"><span class="gs-card-title">Roadmap</span>What the project is aiming at. A line here is not a promise to ship.</a>

</div>
