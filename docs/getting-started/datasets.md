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

## Fetcher

A fetcher is a dataset served as a query, from an API or a database. It returns records. It does not store the dataset as files.

```python
from gigaspatial import OSMLocationFetcher

clinics = OSMLocationFetcher(country="RWA", location_types=["clinic"]).fetch_locations()
```

`MLabHandler` is a fetcher: it queries BigQuery. The class name still says handler.

## Data store

A data store is where a handler keeps its files. `LocalDataStore` is the default. It may also be remote. A client, such as `BigQueryClient`, queries a database and is not a data store.

Pass `data_store=` when you construct a handler or a generator to choose the store. Paths and credentials are in [Configuration](configuration.md).
