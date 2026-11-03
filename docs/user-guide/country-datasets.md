# Combining datasets for a country

This guide combines WorldPop population, Google and Microsoft building footprints, and OpenStreetMap places for a country in a few lines of Python. You can map all three onto an H3 hexagonal grid as a single view, or load each dataset directly as its own table.

## Map population, buildings, and OSM places onto H3 hexagons

`H3ViewGenerator` builds the H3 cells covering a country and attaches attributes from each dataset to the same view. The first run downloads the WorldPop raster and the Google-Microsoft building files into your data store.

```python
from gigaspatial import H3ViewGenerator, OSMLocationFetcher

zones = H3ViewGenerator(source="RWA", resolution=6)
zones.map_wp_pop(country="RWA")
zones.map_buildings(country="RWA")

schools = OSMLocationFetcher(country="RWA", location_types=["school"]).fetch_locations()
zones.add_variable_to_view(zones.map_points(schools), "school_count")

view = zones.view
print(view[["zone_id", "population", "building_count", "school_count"]].head())
```

`view` is a GeoDataFrame with one row per H3 cell, its polygon geometry, the summed WorldPop `population`, the `building_count` from Google and Microsoft building footprints, and the OpenStreetMap `school_count`.

You can attach additional datasets to the same view before reading `zones.view`:

- `zones.map_rwi(country="RWA")` attaches the mean Meta Relative Wealth Index (`rwi`)
- `zones.map_smod()` attaches the GHSL settlement model class (`smod_class`)
- `zones.map_ghsl_pop()` attaches GHSL population (`ghsl_pop`)

The same workflow applies to `S2ViewGenerator`, `MercatorViewGenerator`, and `AdminBoundariesViewGenerator`.

## Load each dataset directly for a country

When you want the datasets themselves as DataFrames or GeoDataFrames rather than aggregated onto zones, call each handler and fetcher directly with the country ISO3 code:

```python
from gigaspatial import (
    GoogleMSBuildingsHandler,
    OSMLocationFetcher,
    WPPopulationHandler,
)

population = WPPopulationHandler(year=2020, resolution=1000).load_into_geodataframe("RWA")
buildings = GoogleMSBuildingsHandler().load_data("RWA")
schools = OSMLocationFetcher(country="RWA", location_types=["school"]).fetch_locations()
```

- `WPPopulationHandler.load_into_geodataframe("RWA")` downloads the WorldPop raster for Rwanda into the data store and returns a GeoDataFrame of grid cells and population counts (`load_into_dataframe` returns a plain DataFrame with coordinates).
- `GoogleMSBuildingsHandler().load_data("RWA")` downloads the combined Google Open Buildings and Microsoft Global Buildings GeoParquet files for Rwanda and returns a GeoDataFrame of building footprints.
- `OSMLocationFetcher(country="RWA", location_types=["school"]).fetch_locations()` queries OpenStreetMap via the Overpass API and returns the matching places as a GeoDataFrame.
