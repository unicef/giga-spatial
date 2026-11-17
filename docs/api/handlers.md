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

## GloFAS forecast

`GlofasForecastHandler` downloads the EWDS dataset `cems-glofas-forecast` for a country, and the GloFAS v4.0 auxiliary maps published beside it. Set `EWDS_API_KEY` and accept the CEMS-FLOODS licence on the dataset page before a forecast download. The auxiliary maps are public NetCDF files and do not use that key.

The country boundary, including islands and other disconnected parts, is read with `AdminBoundaries`. Nearby parts share one native-grid rectangle. Distant parts, including the two sides of the antimeridian, stay separate. EWDS subsets the forecast to that rectangle and snaps it to the GloFAS grid, so the cells keep their published coordinates. This handler does not resample, interpolate, or reproject them.

### Forecast variables

The operational system version is `operational`. `version_2_1` and `version_3_1` are the legacy 0.1° archives and publish river discharge only.

| Variable | EWDS name | Documented units | Operational coverage |
| --- | --- | --- | --- |
| River discharge in the last 24 hours | `river_discharge_in_the_last_24_hours` | m³ s⁻¹ | From 5 November 2019. HTESSEL-LISFLOOD through 25 May 2021, then LISFLOOD. |
| Runoff water equivalent (surface plus subsurface) | `runoff_water_equivalent` | kg m⁻² | From 29 February 2024 in the constraints read on 16 November 2026. |
| Snow depth water equivalent | `snow_depth_water_equivalent` | kg m⁻² | From 28 February 2024. |
| Soil wetness index (root zone) | `soil_wetness_index` | dimensionless | From 28 February 2024. |

The handler reads these dates from the catalogue constraints on each run. It does not assume the four variables share a calendar. Lead times in that catalogue are 24 to 720 hours at a 24-hour step. Control forecasts and perturbed ensemble members are separate products. The file keeps its reference time, valid time, lead time, ensemble dimension, variable names, and units. GRIB2 needs `pip install "giga-spatial[glofas]"`; the default download format is NetCDF-4.

GloFAS v4.0 (released 26 July 2023) and v4.1 (28 February 2024) are the 0.05° operational grid. Operational runs before 26 July 2023, and both legacy system versions, are 0.1°. A file on the wrong step is rejected rather than resampled.

Forecast files are stored under `bronze/cems/glofas/forecast/{ISO3}/{system_version}/{model}/{variable}/{product}/{area}/`. One NetCDF holds one reference date. The catalogue is read first. A stored run is downloaded again only when it is missing, when it does not yet contain the lead times the constraints list and the catalogue update time has changed, or when it is the latest reference date and that update time has changed.

Omitting the dates syncs every published run. Bound a request with `reference_start` and `reference_end` when the full archive is not needed.

### Auxiliary maps

The v4.0 maps apply to operational forecasts from 26 July 2023 onward, including v4.1, which reuses them. They are not paired with `version_2_1`, `version_3_1`, or earlier operational runs. They are stored under `bronze/cems/glofas/auxiliary/{ISO3}/{dataset}/`, separate from the forecast runs, and a global copy is kept so the same file is not downloaded for every country. A later request downloads a map again only when its size or modification time changes.

| Dataset | What it stores |
| --- | --- |
| `upstream_area` | Upstream catchment area, m² |
| `elevation` | Mean elevation above sea level, metres |
| `ldd` | Local drain direction, integer codes 1–9 (5 is a pit). The codes are not interpolated. |
| `flood_threshold_rl_1.5` through `flood_threshold_rl_500.0` | Discharge thresholds for the 1.5, 2, 5, 10, 20, 50, 100, 200, and 500 year return periods |
| `flood_threshold_mu`, `flood_threshold_sigma` | Parameters of the distribution used for those thresholds |

```python
from gigaspatial import GlofasForecastHandler

GlofasForecastHandler().load_data(
    "RWA",
    variable="river_discharge_in_the_last_24_hours",
    product_type="control_forecast",
    reference_start="2026-11-15",
    reference_end="2026-11-15",
)
GlofasForecastHandler().load_data("RWA", variable="ldd")
GlofasForecastHandler().load_data("RWA", include_auxiliary=True)
```

::: gigaspatial.handlers.cems.glofas_forecast.GlofasForecastHandler
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

## GloFAS historical

`GlofasHistoricalHandler` downloads the EWDS dataset `cems-glofas-historical` for a country. Set `EWDS_API_KEY` and accept the CEMS-FLOODS licence on the dataset page first. The operational system version is `version_4_0`. Omit `variable` to load all four daily variables, or pass one EWDS variable name. `include_auxiliary=True` also stores upstream area and elevation for version 4.0.

::: gigaspatial.handlers.cems.glofas_historical.GlofasHistoricalHandler
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
