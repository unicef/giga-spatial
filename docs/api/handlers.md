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
