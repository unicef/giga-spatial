# Fetchers

A fetcher queries an API or a database and returns records. It does not store the dataset as files.

```python
from gigaspatial import OSMLocationFetcher, GigaSchoolLocationFetcher
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
