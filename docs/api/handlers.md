# Handlers

A handler acquires a dataset published as files, stores the files, and loads them for a source. Construct the handler, then call `load_data`, `load_into_dataframe`, or the method named below. The shared machinery is documented in the [handler base](handler-base.md), not on these pages.

```python
from gigaspatial import GHSLDataHandler, WPPopulationHandler
from gigaspatial.handlers import GHSLDataConfig, GHSLDataDownloader, GHSLDataReader
```

Every handler is importable from `gigaspatial`. Its config, downloader, and reader are in `gigaspatial.handlers`. A handler takes its config fields as keyword arguments, so most scripts only need the handler.

## GHSL

::: gigaspatial.handlers.ghsl.GHSLDataHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info
        - load_into_dataframe
        - load_into_geodataframe

## WorldPop

::: gigaspatial.handlers.worldpop.WPPopulationHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info
        - load_into_dataframe
        - load_into_geodataframe

## Google Open Buildings

::: gigaspatial.handlers.google_open_buildings.GoogleOpenBuildingsHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info
        - load_points
        - load_polygons

## Microsoft buildings

::: gigaspatial.handlers.microsoft_global_buildings.MSBuildingsHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info

## Google and Microsoft buildings

::: gigaspatial.handlers.google_ms_combined_buildings.GoogleMSBuildingsHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info
        - load_centroids

## HDX

::: gigaspatial.handlers.hdx.HDXHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info
        - load_all_resources

## Relative Wealth Index

::: gigaspatial.handlers.rwi.RWIHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info
        - load_as_geodataframe

## Ookla

::: gigaspatial.handlers.ookla_speedtest.OoklaSpeedtestHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info

## OpenCellID

::: gigaspatial.handlers.opencellid.OpenCellIDHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info
        - load_as_geodataframe

## NASA SRTM

::: gigaspatial.handlers.nasa.srtm.NasaSRTMHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info

## CEMS river flood hazard

::: gigaspatial.handlers.cems.flood_hazard.CEMSFloodHazardHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info

## OpenStreetMap roads

`OSMRoadsHandler` returns a [`SpatialNetwork`](network.md) built from Geofabrik's extracts, the same network [`OSMRoadsFetcher`](fetchers.md) returns for the same area. It needs `pip install "giga-spatial[roads]"`; the [road networks guide](../user-guide/road-networks.md#whole-countries) explains which extracts a source uses and how they are stored.

::: gigaspatial.handlers.osm.roads.OSMRoadsHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info

::: gigaspatial.handlers.osm.roads.OSMRoadsResolveOptions
    options:
      show_source: false
      members: false

::: gigaspatial.handlers.osm.roads.OSMRoadsReadOptions
    options:
      show_source: false
      members: false

## ERA5 and ERA5-Land

`ERA5Handler` (name `cds-era5`) reads [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) (`reanalysis-era5-single-levels`). `ERA5LandHandler` (name `cds-era5land`) reads [ERA5-Land hourly data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview) (`reanalysis-era5-land`). Both are CC BY 4.0. Install `pip install "giga-spatial[cds]"` and set `CDS_API_KEY` (or `CDSAPI_KEY`, or `key` in `~/.cdsapirc`). Accept each dataset's terms on the CDS website once.

A country name or code is resolved with `AdminBoundaries`. The CDS request uses the country's bounding boxes. The stored file is clipped to the country geometry and keeps the native grid: 0.25° for ERA5, 0.1° for ERA5-Land. Pass a geometry when you do not want a country.

`variables` is one CDS name, a sequence of names, or `"all"`. The two catalogues are not the same. Time is `start` and `end`, or `year`, `month`, `day`, and `hour`. ERA5 starts in 1940 and ERA5-Land in 1950. Omit the time arguments and the handler raises, so the historical archive is not downloaded unless you pass `full_history=True`.

```python
from gigaspatial import ERA5Handler, ERA5LandHandler

ERA5Handler().load_data(
    "ITA",
    variables="2m_temperature",
    start="2024-11-01",
    end="2024-11-02",
)
ERA5LandHandler().load_data(
    "ITA",
    variables=("2m_temperature", "total_precipitation"),
    year=2024,
    month=11,
    day=1,
    hour=0,
)
```

`update()` asks that dataset's CDS catalogue for its latest hour and downloads only hours that are not stored yet. ERA5 and ERA5-Land are checked on their own. With no country, every country that already has a status file is updated. A failed hour is recorded and the stored files are left in place. Running it again does not duplicate timestamps.

There is no scheduler inside the process. Call `update()`, or run one check from cron:

```bash
python -m gigaspatial.handlers.cds.era5 update --handler cds-era5 --country ITA
python -m gigaspatial.handlers.cds.era5 update --handler cds-era5land --force
```

`update_frequency_hours` (default 24) skips a country checked more recently than that. `--force` checks anyway.

::: gigaspatial.handlers.cds.era5.ERA5Handler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info
        - update

::: gigaspatial.handlers.cds.era5land.ERA5LandHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info
        - update

## WRI Aqueduct

::: gigaspatial.handlers.wri.aqueduct_flood.AqueductFloodHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info
        - load_into_dataframe
        - load_into_geodataframe
