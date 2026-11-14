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

## UNOSAT historical disaster products

`UnosatHandler` reads the catalogue behind [unosat.org/products](https://unosat.org/products/). `GET /our_products` supplies the country names and activation types. `POST /our_products/` with a date window and those country names returns the products, and the handler repeats that request year by year so older products are not left out. `GET /our_products/{id}` supplies the text, the GLIDE number and the file links.

The handler does not treat the publication timestamp as the day of the disaster. A GLIDE number such as `TC20240313MOZ` supplies the event date. Phrases such as "Imagery Date" and "as of" are stored as acquisition and observation dates, with the text they came from. Disaster types are the catalogue's own activation types, not a fixed flood list.

PDF, shapefile zip, geodatabase zip, KML, spreadsheet and preview images are saved in the published format. A link whose name is `None` is skipped. Web map and WMS addresses are kept on the product and are not saved as data files. UNOSAT does not publish one licence field for every product, so `data_license` is left null and the attribution stays "United Nations Satellite Centre (UNOSAT), UNITAR".

A country is a name or an ISO3 code, for example `Mozambique` or `MOZ`. Files are stored under `unosat/{ISO3}` in the bronze data folder: `catalogue.json`, `products/{id}/original/` with `metadata.json`, and `events/{GLIDE}/metadata.json` when the product has a GLIDE number whose date parses. Products without that number stay in the catalogue and are not given an invented event id. Running the handler again checks the catalogue and downloads only new or changed files.

On 14 November 2026 the catalogue listed 143 area names and these activation types: Cold Wave, Complex Emergency, Drought, Earthquake, Environmental Impact, Epidemic, Exercise / Simulation, Extratropical Cyclone, Extreme Temperature, Famine, Fire, Flash Flood, Floods, Heat Wave, Insect Infestation, Landslides, Mud Slide, Oil Spill, Other, Severe Local Storm, Snow Avalanche, Storm Surge, Tech. Disaster, Tornadoes, Tropical Cyclone, Tsunami, `Violent  Wind`, Volcano and Wild Fire. `None` and `No glide` also appear as catalogue labels. Conflict damage is published as Complex Emergency. Mozambique had 146 products, with publication years 2008 through 2026 and no products in 2009, 2010, 2018 or 2020. Those products were Floods (77), Tropical Cyclone (67), Severe Local Storm (1) and Complex Emergency (1). Every product carried a GLIDE number, and those numbers collapse to 21 events, from `FL20080109MOZ` to `FL20251230MOZ`. The files linked for Mozambique were PDF (133), shapefile zip, file geodatabase zip, XLSX and preview images. One spreadsheet field stores an external zip inside a site path; that absolute address is what gets downloaded. No GeoJSON, KML, CSV or GeoTIFF was linked for that country. The handler still keeps those formats when a product publishes them.

UNOSAT does not publish an event end date or a product update timestamp, so those fields stay null. A country filter with no dates returns an empty list, so the handler always sends a date window. History is requested year by year back to 2000, then in earlier decades until a decade comes back empty.

```python
UnosatHandler().load_data("Mozambique")
```

::: gigaspatial.handlers.unosat.UnosatHandler
    options:
      show_source: false
      inherited_members: true
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info

## GloFAS flood hazard

::: gigaspatial.handlers.glofas.GlofasHandler
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
