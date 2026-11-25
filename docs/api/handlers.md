# Handlers

A handler persists a dataset's files in a data store and loads them for a source. Construct the handler, then call `load_data`, `load_into_dataframe`, or the method named below. The shared machinery is documented in the [handler base](handler-base.md), not on these pages.

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

## Global Power Plant Database

::: gigaspatial.handlers.wri.global_power_plant.GlobalPowerPlantHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info

## Global Seismic Hazard Map

`GSHMHandler` loads the GEM Global Seismic Hazard Map v2026.1. The map is one global point layer of peak ground acceleration (g) with a 10% probability of exceedance in 50 years, for reference rock (Vs30 800 m/s), including the oceans. Licence: CC BY-NC-SA 4.0. A source resolves to that single file. `crop_to_source=True` reads points inside the search geometry's bounds, then keeps the points that intersect the geometry. An uncropped load reads the full layer, about 1.76 GB.

```python
from shapely.geometry import box

from gigaspatial import GSHMHandler

kigali = box(29.9, -1.95, 30.2, -1.85)
frame = GSHMHandler().load_data(kigali, crop_to_source=True)
```

::: gigaspatial.handlers.gshm.GSHMHandler
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
