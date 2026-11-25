# Handlers

A **handler** persists a dataset's files in a **data store** you control and loads those files for a **source**. It is how the library keeps products such as WorldPop rasters, GHSL products, building footprint archives, and Geofabrik OpenStreetMap extracts.

This page explains why handlers exist, how they differ from **fetchers**, how data stores and lazy caching work, and how requests are scoped and cropped. It is written for data scientists who know Python, pandas, and geopandas. For the full catalog of built-in datasets, see [Datasets](../getting-started/datasets.md). For method signatures and dataset-specific options, see the [Handlers API](../api/handlers.md) and [Fetchers API](../api/fetchers.md).

## Why handlers

Geospatial datasets arrive in incompatible shapes. WorldPop publishes one GeoTIFF per country and demographic group. GHSL distributes global rasters tiled in Mollweide projection. Google Open Buildings and Microsoft Global Buildings partition footprints across S2 cells, quadkeys, or GeoParquet files. OpenStreetMap extracts come as regional `.osm.pbf` files from Geofabrik.

Without a common interface, every analysis project ends up maintaining its own download scripts, folder layouts, decompression logic, and readers. Switching from Rwanda to Kenya, or moving a pipeline from a laptop to cloud storage, means rewriting acquisition code before any analysis starts.

A handler puts one repeatable download-and-load interface over all of these datasets. You construct a handler with the product parameters you want (such as `year`, `resolution`, or `product`), pass a source, and get back a ready-to-use Python object: a pandas `DataFrame`, a geopandas `GeoDataFrame`, a `TifProcessor` for rasters, or a `SpatialNetwork` for roads.

## What it is like to use

A few principles run through every handler:

- **One import per dataset.** Every built-in handler is importable from `gigaspatial` and accepts its configuration fields as keyword arguments (for example, `GHSLDataHandler(product="GHS_POP", year=2020, resolution=1000)`). Passing an unknown keyword raises `TypeError` right away rather than failing silently.
- **Lazy caching.** The first load acquires only the data units needed for your source into your data store. Later loads read the stored files directly without re-downloading.
- **Source-driven loading.** You ask for a place (a country, a geometry, a GeoDataFrame, or points) or explicit stored file paths as the first argument to `load_data` (or a dataset-specific convenience method such as `load_into_dataframe` or `load_into_geodataframe`).
- **Storage-agnostic.** The same handler code runs unchanged against a local directory (`LocalDataStore`) or cloud storage (`ADLSDataStore`, `SnowflakeDataStore`, `DeltaSharingDataStore`).

```python
from gigaspatial import GHSLDataHandler

population = GHSLDataHandler(product="GHS_POP", year=2020, resolution=1000)
frame = population.load_into_dataframe("RWA", crop_to_source=True)
```

## Handlers vs. fetchers

A handler and a fetcher are two ways to acquire a dataset. They differ by what comes back, and by whether the library imposes a data-store layout.

- A **handler** persists files in a data store and loads them for a source. It acquires **data units** (tiles, country files, or extracts) and loads those **stored files**. The files stay in the data store, so a later run can reload them offline with `ensure_available=False`.
- A **fetcher** returns its result in memory and imposes no data-store layout. The result may be records (a `DataFrame` or `GeoDataFrame`), a `SpatialNetwork`, or another in-memory dataset. A **client** such as `BigQueryClient` is how a fetcher reaches a warehouse. Publication as files or as a query does not decide this: `OvertureAmenityFetcher` reads remote GeoParquet in place and does not mirror it.

| Aspect | Handler | Fetcher |
| --- | --- | --- |
| What comes back | Stored files, loaded for a source | The result in memory, with no data-store layout |
| Kept in a data store | Yes, as stored files | No |
| Offline reuse | Reload stored files with `ensure_available=False` | Depends on the service the call reaches |
| Typical return value | `TifProcessor`, `GeoDataFrame`, `DataFrame`, or `SpatialNetwork` | `GeoDataFrame`, `DataFrame`, `SpatialNetwork`, or another in-memory dataset |
| Examples | `WPPopulationHandler`, `GHSLDataHandler`, `GoogleMSBuildingsHandler`, `RWIHandler`, `OSMRoadsHandler` | `OSMLocationFetcher`, `OSMRoadsFetcher`, `GigaSchoolLocationFetcher`, `OvertureAmenityFetcher`, `MLabFetcher` |

OpenStreetMap roads is two acquisitions of related data. `OSMRoadsHandler` persists dated Geofabrik extracts and loads them for a source. `OSMRoadsFetcher` returns a `SpatialNetwork` from Overpass and saves nothing. A handler and a fetcher need not share a call.

A third, narrower category is the **image downloader** (such as `MapboxImageDownloader` or `MaxarImageDownloader`): it saves imagery from an API into a data store as image files, with no source-loading step.

## Data stores and the download-to-load lifecycle

A **data store** is where a handler keeps its stored files. `LocalDataStore` is the default and writes under the configured `ROOT_DATA_DIR` / `BRONZE_DIR` hierarchy (see [Configuration](../getting-started/configuration.md)). Pass `data_store=` when constructing a handler to target another directory or remote storage such as `ADLSDataStore` (Azure Data Lake Storage), `SnowflakeDataStore`, or `DeltaSharingDataStore`. A **client** (such as `BigQueryClient`) queries a database and is not a data store.

```python
from gigaspatial import ADLSDataStore, WPPopulationHandler

store = ADLSDataStore(container="geospatial")
wp = WPPopulationHandler(year=2020, resolution=1000, data_store=store)
```

Every handler follows the same lifecycle from availability check to in-memory load:

```text
source -> resolve data units -> check data store -> download missing units -> load stored files (optional crop)
```

Four public methods expose each stage of that lifecycle:

- **`get_available_data_info(source)`**: checks which data units and stored files for `source` already exist in the data store and which are missing, without downloading anything.
- **`ensure_data_available(source, force_download=False)`**: fetches any missing data units into the data store and returns `True` when all required stored files are present. Pass `force_download=True` to re-download even when stored files already exist.
- **`load_data(source, crop_to_source=False, ensure_available=True)`**: the standard entry point. With `ensure_available=True` (the default), it acquires missing units lazily and then reads the stored files into memory. With `ensure_available=False`, it skips downloading and raises `RuntimeError` if any required stored file is missing, which makes offline or read-only pipelines explicit.
- **`download_and_load(source, crop_to_source=False, force_download=False)`**: combines explicit download control (including `force_download=True`) and loading in a single call.

One data unit can unpack into several stored files (for example, an extracted archive). Once stored, those files are reused across every future call and across any [view generator](view-generators.md) pointed at the same data store.

## Scoping requests and cropping at load time

The first argument to a handler is a **source**, which tells the handler what to resolve and load. A source can be:

- **A country** name or ISO code (such as `"RWA"` or `"Rwanda"`), which resolves the data units covering that country.
- **A spatial extent**: a Shapely geometry (for example a polygon or a bounding box built with `shapely.geometry.box`), a `GeoDataFrame`, a `DataFrame` with coordinates, or a sequence of points (`Point` objects or coordinate pairs).
- **Explicit stored file paths**: a `Path`, a path string, or a list of paths pointing to files already in the data store.

Downloading happens at the granularity of whole **data units** (whole tiles, country files, or extracts) so that stored files remain intact and reusable for neighbouring requests. By default (`crop_to_source=False`), `load_data` returns the full contents of the resolved data units.

Pass `crop_to_source=True` to clip rasters or filter tables to your source geometry in memory at load time:

- The handler aligns your source geometry to the dataset's coordinate reference system automatically before clipping or filtering.
- Stored files in the data store are never modified; cropping applies only to the loaded result.
- Explicit path sources are never cropped because paths do not carry an area geometry.
- Country sources are not cropped when a dataset is already published as one file per country (for example, country-level WorldPop rasters).

## Where to go next

- **Using built-in datasets**: browse the full list in [Datasets](../getting-started/datasets.md), set up storage paths and credentials in [Configuration](../getting-started/configuration.md), and see step-by-step examples in [Combining datasets for a country](../user-guide/country-datasets.md) and [Processing a raster or a table](../user-guide/processing.md).
- **Mapping datasets onto places**: see [View generators](view-generators.md) to understand how generators orchestrate handlers to build analysis-ready tables.
- **API details**: see [Handlers](../api/handlers.md), [Fetchers](../api/fetchers.md), and [Data stores](../api/data-stores.md) for class signatures and options.
- **Contributing a new dataset**: internal handler-base classes and extension hooks are documented once for contributors in [Handler base](../api/handler-base.md) (see [ADR 0002](../adr/0002-handler-base-in-the-api-reference.md)).
