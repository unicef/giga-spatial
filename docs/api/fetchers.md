# Fetchers

A fetcher queries an API or a database and returns records. It does not store the dataset as files.

`MLabHandler` is a fetcher. It queries BigQuery. The class still has its current name.

## OpenStreetMap

::: gigaspatial.handlers.osm.OSMLocationFetcher
    options:
      show_source: false
      members:
        - fetch_locations
        - fetch_locations_changed_between
        - fetch_by_osmid

## Giga schools

::: gigaspatial.handlers.giga.school_locations.GigaSchoolLocationFetcher
    options:
      show_source: false
      members:
        - fetch_locations

::: gigaspatial.handlers.giga.school_profile.GigaSchoolProfileFetcher
    options:
      show_source: false
      members:
        - fetch_profiles

::: gigaspatial.handlers.giga.school_measurements.GigaSchoolMeasurementsFetcher
    options:
      show_source: false
      members:
        - fetch_measurements

::: gigaspatial.handlers.giga.countries.GigaCountriesFetcher
    options:
      show_source: false
      members:
        - fetch_countries

## Healthsites

::: gigaspatial.handlers.healthsites.HealthSitesFetcher
    options:
      show_source: false
      members:
        - fetch_facilities
        - fetch_statistics
        - fetch_facility_by_id

## Overture

::: gigaspatial.handlers.overture.OvertureAmenityFetcher
    options:
      show_source: false
      members:
        - fetch_locations

## UNICEF

::: gigaspatial.handlers.unicef.sdg.UnicefDataFetcher
    options:
      show_source: false
      members:
        - fetch_as_geodataframe

## M-Lab

::: gigaspatial.handlers.mlab.MLabHandler
    options:
      show_source: false
      members:
        - query_ndt7
        - query_ndt7_gdf
