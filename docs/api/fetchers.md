# Fetchers

A fetcher queries an API or a database and returns records. It does not store the dataset as files.

```python
from gigaspatial import OSMLocationFetcher, OSMRoadsFetcher, GigaSchoolLocationFetcher
from gigaspatial.fetchers import MLabConfig, GEEConfig
```

`MLabFetcher` queries M-Lab speed tests in BigQuery. [`GEEProfiler`](earth-engine.md) is also a fetcher and has its own page.

## OpenStreetMap

::: gigaspatial.fetchers.osm.OSMLocationFetcher
    options:
      show_source: false
      members:
        - fetch_locations
        - fetch_locations_changed_between
        - fetch_by_osmid

`OSMRoadsFetcher` returns a road network rather than records: a [`SpatialNetwork`](network.md). It needs `pip install "giga-spatial[roads]"`, and the [road networks guide](../user-guide/road-networks.md) says which area sizes suit Overpass.

::: gigaspatial.fetchers.osm_roads.OSMRoadsFetcher
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

## Overture

::: gigaspatial.fetchers.overture.OvertureAmenityFetcher
    options:
      show_source: false
      members:
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
