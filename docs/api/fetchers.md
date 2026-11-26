# Fetchers

A fetcher returns its result in memory and imposes no data-store layout. Publication as files or as a query does not decide this. The result is records, a road network, or another in-memory dataset.

```python
from gigaspatial import OSMLocationFetcher, OSMRoadsFetcher, GigaSchoolLocationFetcher
from gigaspatial.fetchers import MLabConfig, GEEConfig
```

`MLabFetcher` queries M-Lab speed tests in BigQuery. [`GEEProfiler`](earth-engine.md) is also a fetcher and has its own page.

## ECMWF open data

`EcmwfOpenDataFetcher` downloads the public IFS and AIFS forecast fields and returns them in memory. It does not store the files. The catalogue is the [ECMWF open data](https://www.ecmwf.int/en/forecasts/datasets/open-data) subset, licensed CC-BY-4.0.

IFS deterministic and IFS ENS are published at 0.25° and 0.1°. AIFS Single and AIFS ENS are published at 0.25°. At 0.1° each step is a surface file and a pressure-level file. Omitting `param` downloads every field in the step file. `fetch_wind` reads `10u` and `10v`, adds `wind_speed` in m/s, and, for the IFS systems, the maximum 10 m gust (`10fg` through 90 h, `10fg3` after that). A track table crops each storm, wind and gust together, to the box around that storm's positions.

```python
from gigaspatial import EcmwfOpenDataFetcher

EcmwfOpenDataFetcher().fetch_wind(
    system="ifs-ens",
    date="2026-11-26",
    time=0,
    step=24,
    resol="0p25",
    tracks=tracks,
)
```

Forecast steps: IFS deterministic runs to 360 h at 00 and 12 UTC and to 90 h at 06 and 18 UTC. IFS ENS runs to 360 h at 00 and 12 UTC and to 144 h at 06 and 18 UTC. AIFS Single and AIFS ENS run to 360 h at every cycle, every 6 h.

::: gigaspatial.fetchers.ecmwf.open_data.EcmwfOpenDataFetcher
    options:
      show_source: false
      members:
        - fetch
        - fetch_wind
        - download_fields

## ECMWF tropical cyclone tracks

`EcmwfTropicalCycloneFetcher` returns track positions for one cycle and does not store them. Per-storm files come from the [essential portal](https://essential.ecmwf.int/). IFS deterministic files use the ECMF heading and IFS ENS files use ECEP. AIFS Single, AIFS ENS, and an IFS system with no essential file use the open-data BUFR track. The same cycle can be kept on disk with `EcmwfTropicalCycloneHandler`.

```python
from gigaspatial import EcmwfTropicalCycloneFetcher

EcmwfTropicalCycloneFetcher().fetch(date="2026-10-15", time=0, system="ifs")
```

Pass a place as the first argument to keep positions inside it: `fetch(area, system="ifs", date="2026-10-15", time=0)`.

::: gigaspatial.fetchers.ecmwf.tropical_cyclone.EcmwfTropicalCycloneFetcher
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
