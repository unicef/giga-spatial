# Fetchers

A fetcher returns its result in memory and imposes no data-store layout. Publication as files or as a query does not decide this. The result is records, a road network, or another in-memory dataset.

```python
from gigaspatial import OSMLocationFetcher, OSMRoadsFetcher, GigaSchoolLocationFetcher
from gigaspatial.fetchers import MLabConfig, GEEConfig
```

`MLabFetcher` queries M-Lab speed tests in BigQuery. [`GEEProfiler`](earth-engine.md) is also a fetcher and has its own page.

## GloFAS forecast

`GlofasForecastFetcher` queries the EWDS dataset `cems-glofas-forecast` for a country and returns the forecast in memory. It does not store the files. Set `EWDS_API_KEY` and accept the CEMS-FLOODS licence before a forecast query. Auxiliary maps are public NetCDF files and do not use that key.

The country boundary, including islands, selects native grid cells. Values are not resampled. `include_auxiliary=True` also retrieves the GloFAS v4.0 upstream-area and flood-threshold maps and copies them onto a v4 forecast. `exceeded_return_period_years` is the highest return period the discharge exceeds.

```python
from gigaspatial import GlofasForecastFetcher

GlofasForecastFetcher().fetch(
    "RWA",
    variable="river_discharge_in_the_last_24_hours",
    product_type="control_forecast",
    reference_start="2026-11-15",
    reference_end="2026-11-15",
)
```

::: gigaspatial.fetchers.cems.glofas_forecast.GlofasForecastFetcher
    options:
      show_source: false
      members:
        - fetch

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
