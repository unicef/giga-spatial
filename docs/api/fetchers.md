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

## ERA5 and ERA5-Land

`fetch_era5` reads [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) (`reanalysis-era5-single-levels`). `fetch_era5land` reads [ERA5-Land hourly data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview) (`reanalysis-era5-land`). Both return an in-memory `xarray.Dataset`. They do not write a data store and they do not keep a country cache. Install `pip install "giga-spatial[cds]"` and set `CDS_API_KEY` (or `CDSAPI_KEY`, or `key` in `~/.cdsapirc`). Accept each dataset's terms on the CDS website once. Both datasets are CC BY 4.0.

`area` is an ISO3 code or country name (`AdminBoundaries` resolves it), a country geometry, a Shapely geometry, or a GeoJSON mapping. The CDS request uses the geometry's bounding boxes. The returned grid is clipped to that geometry and keeps the native spacing: 0.25 degrees for ERA5, 0.1 degrees for ERA5-Land. A polygon that crosses a border is not forced into one country code. `countries_intersecting` lists every country the geometry touches.

`variables` is one CDS parameter name, a list of names, or `"all"`. The catalogues are separate (262 ERA5 variables, 60 ERA5-Land variables) and include each name, description, and units. A request for every variable is split into CDS-sized groups. A combination CDS rejects is split again; a variable that still cannot be retrieved raises, with that variable's name in the error.

`time` is a year (`"2024"`), a month (`"2025-01"`), a day, an ISO timestamp, or a `(start, end)` pair. `year`, `month`, `day`, and `hour` select the same hours. ERA5 starts in 1940 and ERA5-Land in 1950. Hours after the hour CDS currently publishes are rejected. The historical archive is downloaded only when `full_history=True`.

`data_format` is `"netcdf"` by default, or `"grib"`. Both formats are standardized to the same variable names, coordinates, units, and attributes. GRIB reading needs `cfgrib` and ecCodes in addition to the `cds` extra. Longitudes above 180 degrees are shifted into [-180, 180]. The native grid is not resampled.

The dataset attributes record the dataset id, name, source URL, requested variables, area, the coverage of the returned grid, start and end, temporal and spatial resolution, data format, retrieval time, and the latest hour CDS published. Variable attributes keep the description, units, and whether the field is an hourly accumulation.

A request larger than `max_bytes` (default 2 GiB) raises before the download. Repeated calls wait on the CDS queue again. Nothing is cached inside the fetcher. To update a dataset you already hold, compare it with `latest_available` or `period_available`, call the fetcher for the missing hours only, and merge with `merge_climate`. The first finite value at a repeated timestamp is kept, including accumulated precipitation.

```python
from gigaspatial import fetch_era5, fetch_era5land
from gigaspatial.fetchers.climate import latest_available, merge_climate, period_available

ds = fetch_era5(area="ITA", variables=["2m_temperature"], time="2025-01")
land = fetch_era5land(
    area=geom,
    variables=["2m_temperature", "total_precipitation"],
    time="2025-01",
    data_format="netcdf",
)

latest = latest_available("reanalysis-era5-single-levels")
status = period_available("reanalysis-era5-single-levels", "2024-11-16", "2024-11-17", latest=latest)
if status["missing_hours"]:
    fresh = fetch_era5(
        area="ITA",
        variables=["2m_temperature"],
        start=status["missing_hours"][0],
        end=status["missing_hours"][-1],
    )
    ds = merge_climate([ds, fresh])
```

::: gigaspatial.fetchers.climate.era5.fetch_era5
    options:
      show_source: false

::: gigaspatial.fetchers.climate.era5land.fetch_era5land
    options:
      show_source: false

## M-Lab

::: gigaspatial.fetchers.mlab.MLabFetcher
    options:
      show_source: false
      members:
        - query_ndt7
        - query_ndt7_gdf
