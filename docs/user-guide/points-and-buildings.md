# Points and nearest buildings

This guide maps building footprints onto points you already have. The result is a table with the nearest building and the distance to it.

The points are `(latitude, longitude)` pairs. `find_nearest_buildings` downloads Microsoft and Google building files for the country on the first run.

```python
from gigaspatial import PoiViewGenerator

points = [(-1.9441, 30.0619), (-1.9500, 30.1000)]
places = PoiViewGenerator(points)
nearest = places.find_nearest_buildings(country="RWA", search_radius=1000)
print(nearest.head())
```

`search_radius` is in meters. A point with no building inside that radius is still in the table, with no nearest building.

`PoiViewGenerator` also accepts a DataFrame or a GeoDataFrame. Set `poi_id_column` when the table already has an id column.

## Giga school locations

The same mapping works for schools from the Giga API. That call needs `GIGA_SCHOOL_LOCATION_API_KEY`. See [Configuration](../getting-started/configuration.md).

```python
from gigaspatial import GigaSchoolLocationFetcher, PoiViewGenerator

schools = GigaSchoolLocationFetcher(country="RWA").fetch_locations(process_geospatial=True)
nearest = PoiViewGenerator(schools).find_nearest_buildings(country="RWA", search_radius=1000)
```

`fetch_locations` is a fetcher: it returns records from the API and does not store a schools dataset. The building files are still acquired by a handler.

## Download and load building footprints for a country

When you want the building footprints themselves as a GeoDataFrame for a country or a geometry, call one of the three building handlers directly:

- `GoogleMSBuildingsHandler` acquires the combined Google Open Buildings and Microsoft Global Buildings dataset in GeoParquet format.
- `GoogleOpenBuildingsHandler` acquires Google Open Buildings polygons or points by S2 tile.
- `MSBuildingsHandler` acquires Microsoft Global ML Building Footprints by country and quadkey.

```python
from gigaspatial import (
    GoogleMSBuildingsHandler,
    GoogleOpenBuildingsHandler,
    MSBuildingsHandler,
)

combined = GoogleMSBuildingsHandler().load_data("RWA")
google_buildings = GoogleOpenBuildingsHandler().load_polygons("RWA")
ms_buildings = MSBuildingsHandler().load_data("RWA")
```

For large countries where you only need building coordinates, `GoogleMSBuildingsHandler(partition_strategy="s2_grid").load_centroids("RWA")` caches and returns a NumPy array of building centroids.

To count building footprints inside H3 hexagons for a country, use `H3ViewGenerator.map_buildings`:

```python
from gigaspatial import H3ViewGenerator

zones = H3ViewGenerator(source="RWA", resolution=6)
view = zones.map_buildings(country="RWA")
print(view[["zone_id", "building_count"]].head())
```
