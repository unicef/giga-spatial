# Datasets

A dataset is a named product the library can acquire, such as WorldPop population or Google Open Buildings.

## Handler

A handler is a dataset published as files. It keeps those files in a data store and loads them for a source.

A source is the place you load for, or an explicit list of stored files. A place is a country, a geometry, points, or a frame.

```python
from gigaspatial import GHSLDataHandler

population = GHSLDataHandler(product="GHS_POP", year=2020, resolution=1000)
frame = population.load_into_dataframe("RWA")
```

The first call downloads the tiles that cover Rwanda. `load_data` and `load_into_dataframe` take the source as the first argument.

`crop_to_source=True` cuts the loaded data to that source. A path source is not cropped. A country source is not cropped when the dataset is already one file for that country. An unknown keyword raises `TypeError`.

```python
frame = population.load_into_dataframe("RWA", crop_to_source=True)
```

### Built-in handlers

Every handler is importable from `gigaspatial`. See [Handlers](../api/handlers.md) for method signatures and options.

| Dataset | Handler | What it acquires |
| --- | --- | --- |
| WorldPop | `WPPopulationHandler` | Country population rasters, age and sex structures, school-age and under-18 population, and degree of urbanisation |
| Global Human Settlement Layer (GHSL) | `GHSLDataHandler` | Population (`GHS_POP`), built-up surface (`GHS_BUILT_S`), settlement model (`GHS_SMOD`), building height (`GHS_BUILT_H`), and building volume (`GHS_BUILT_V`) |
| Google-Microsoft combined buildings | `GoogleMSBuildingsHandler` | Country and S2-partitioned building footprints from Google Open Buildings and Microsoft Global Buildings in GeoParquet |
| Google Open Buildings | `GoogleOpenBuildingsHandler` | Building footprint polygons and points by S2 tile |
| Microsoft Global Buildings | `MSBuildingsHandler` | Building footprint polygons by country and quadkey |
| OpenStreetMap roads | `OSMRoadsHandler` | Country-scale road networks built from Geofabrik OpenStreetMap extracts, returned as a `SpatialNetwork` |
| Meta Relative Wealth Index (RWI) | `RWIHandler` | Relative wealth index tiles per country from HDX |
| Humanitarian Data Exchange (HDX) | `HDXHandler` | Datasets and resources published on HDX |
| UNOSAT | `UnosatHandler` | Historical disaster products and assessments for a country, across every activation type |
| Ookla Speedtest | `OoklaSpeedtestHandler` | Fixed and mobile network performance tiles |
| OpenCellID | `OpenCellIDHandler` | Crowd-sourced cell tower locations by country |
| NASA SRTM | `NasaSRTMHandler` | 30 m digital elevation model tiles |
| Copernicus GloFAS | `GlofasHandler` | Global river flood hazard depth and water masks by return period |
| WRI Aqueduct | `AqueductFloodHandler` | Coastal and riverine flood hazard rasters |
| Administrative boundaries | `AdminBoundaries` | Country and subnational administrative boundary polygons |

## Fetcher

A fetcher is a dataset served as a query, from an API or a database. It returns records, or a road network. It does not store the dataset as files.

```python
from gigaspatial import OSMLocationFetcher

clinics = OSMLocationFetcher(country="RWA", location_types=["clinic"]).fetch_locations()
```

`MLabFetcher` queries M-Lab speed tests in BigQuery. `OSMRoadsFetcher` returns a road network instead of records; see [Road networks](../user-guide/road-networks.md).

### Built-in fetchers

Every fetcher is importable from `gigaspatial`. See [Fetchers](../api/fetchers.md) for method signatures.

| Dataset | Fetcher | What it queries |
| --- | --- | --- |
| OpenStreetMap places | `OSMLocationFetcher` | Points of interest (schools, clinics, hospitals, amenities) from the Overpass API |
| OpenStreetMap roads | `OSMRoadsFetcher` | Road networks from the Overpass API, returned as a `SpatialNetwork` |
| Giga schools | `GigaSchoolLocationFetcher`, `GigaSchoolProfileFetcher`, `GigaSchoolMeasurementsFetcher`, `GigaCountriesFetcher` | School locations, profiles, connectivity measurements, and country metadata from the Giga API |
| Healthsites.io | `HealthSitesFetcher` | Health facility locations and attributes |
| Overture Maps | `OvertureAmenityFetcher` | Amenity places from Overture Maps |
| UNICEF SDGs | `UnicefDataFetcher` | UNICEF indicator and SDG datasets |
| M-Lab | `MLabFetcher` | NDT7 internet speed test records in BigQuery |
| Google Earth Engine | `GEEProfiler` | Raster and vector collections in Google Earth Engine |

## Data store

A data store is where a handler keeps its files. `LocalDataStore` is the default. It may also be remote (`ADLSDataStore`, `SnowflakeDataStore`, `DeltaSharingDataStore`). A client, such as `BigQueryClient`, queries a database and is not a data store.

Pass `data_store=` when you construct a handler or a generator to choose the store. Paths and credentials are in [Configuration](configuration.md).
