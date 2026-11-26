# Fetchers

A fetcher returns its result in memory and imposes no data-store layout. Publication as files or as a query does not decide this. The result is records, a road network, or another in-memory dataset.

```python
from gigaspatial import OSMLocationFetcher, OSMRoadsFetcher, GigaSchoolLocationFetcher
from gigaspatial.fetchers import MLabConfig, GEEConfig
```

`MLabFetcher` queries M-Lab speed tests in BigQuery. [`GEEProfiler`](earth-engine.md) is also a fetcher and has its own page.

## OpenStreetMap

::: gigaspatial.fetchers.osm.pois.OSMLocationFetcher
    options:
      show_source: false
      members:
        - fetch_locations
        - fetch_locations_changed_between
        - fetch_by_osmid

`OSMRoadsFetcher` returns a road network rather than records: a [`SpatialNetwork`](network.md). It needs `pip install "giga-spatial[roads]"`, and the [road networks guide](../user-guide/road-networks.md) says which area sizes suit Overpass.

::: gigaspatial.fetchers.osm.roads.OSMRoadsFetcher
    options:
      show_source: false
      members:
        - fetch
        - query_area

## Giga schools

::: gigaspatial.fetchers.giga.school_locations.GigaSchoolLocationFetcher
    options:
      show_source: false
      members:
        - fetch_locations

::: gigaspatial.fetchers.giga.school_profile.GigaSchoolProfileFetcher
    options:
      show_source: false
      members:
        - fetch_profiles

::: gigaspatial.fetchers.giga.school_measurements.GigaSchoolMeasurementsFetcher
    options:
      show_source: false
      members:
        - fetch_measurements

::: gigaspatial.fetchers.giga.countries.GigaCountriesFetcher
    options:
      show_source: false
      members:
        - fetch_countries

## Healthsites

::: gigaspatial.fetchers.healthsites.HealthSitesFetcher
    options:
      show_source: false
      members:
        - fetch_facilities
        - fetch_statistics
        - fetch_facility_by_id

## PeeringDB

`PeeringDBFetcher` fetches facilities from the [PeeringDB API](https://www.peeringdb.com/api). `fetch_facilities` returns a GeoDataFrame in EPSG:4326. Pass a country, a box `(min_lon, min_lat, max_lon, max_lat)`, or both. Omit both for the default listing. A facility with no coordinates stays in the frame with a missing geometry.

The call is anonymous when `PEERINGDB_API_KEY` is empty. A key is sent as `Authorization: Api-Key`. PeeringDB's [acceptable use policy](https://www.peeringdb.com/aup) limits reproduction, storage, and bulk redistribution.

::: gigaspatial.fetchers.peeringdb.PeeringDBFetcher
    options:
      show_source: false
      members:
        - fetch_facilities

## STORM tropical cyclone wind speeds

`STORMWindSpeedFetcher` fetches global tropical cyclone maximum wind speed (m/s) at 10 km resolution. `fetch` returns a `TifProcessor`. Pass a country or a geometry to clip the map to that place. Omit the source for the global raster. `model` is `constant` for the present-day baseline, or `CMCC-CM2-VHR4`, `CNRM-CM6-1-HR`, `EC-Earth3P-HR`, or `HadGEM3-GC31-HM` for 2050 under RCP8.5. `return_period` is the return period in years and defaults to 100. Keep one fetcher and ask it for each map. A `with` block downloads the Zenodo zip once (about 1.1 GB) into a temporary directory and deletes that directory when the block ends. The returned processors keep their own copies. Nothing is written to a data store. Licence: CC0. DOI: [10.5281/zenodo.10931452](https://doi.org/10.5281/zenodo.10931452).

```python
with STORMWindSpeedFetcher() as storm:
    present = storm.fetch()
    future = storm.fetch("KEN", model="HadGEM3-GC31-HM", return_period=500)
```

::: gigaspatial.fetchers.storm.STORMWindSpeedFetcher
    options:
      show_source: false
      members:
        - fetch
        - close

## Overture

::: gigaspatial.fetchers.overture.OvertureAmenityFetcher
    options:
      show_source: false
      members:
        - get_latest_release
        - get_available_releases
        - fetch_locations

## UNICEF

::: gigaspatial.fetchers.unicef.sdg.UnicefDataFetcher
    options:
      show_source: false
      members:
        - fetch_as_geodataframe

## M-Lab

::: gigaspatial.fetchers.mlab.MLabFetcher
    options:
      show_source: false
      members:
        - query_ndt7
        - query_ndt7_gdf
