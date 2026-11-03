# Population around points

The [tutorial](../getting-started/quickstart.md) sums WorldPop onto H3 cells. This guide sums it inside a radius around each point, such as schools or clinics in a country. The result is a view: your points, with a `population` column.

```python
from gigaspatial import PoiViewGenerator

points = [(-1.9441, 30.0619), (-1.9500, 30.1000)]
places = PoiViewGenerator(points)
view = places.map_wp_pop(country="RWA", map_radius_meters=1000)
print(view.head())
```

`map_radius_meters` is the buffer around each point in meters. The first run downloads the WorldPop raster for the country into the data store.

## Estimate how many people live near each school in a country

Pass a GeoDataFrame of schools or clinics directly to `PoiViewGenerator` to estimate the population living within a catchment radius of each place.

With schools or clinics from OpenStreetMap (no API key required):

```python
from gigaspatial import OSMLocationFetcher, PoiViewGenerator

schools = OSMLocationFetcher(country="RWA", location_types=["school"]).fetch_locations()
view = PoiViewGenerator(schools).map_wp_pop(country="RWA", map_radius_meters=1000)
print(view[["population"]].head())
```

With school locations from the Giga API (requires `GIGA_SCHOOL_LOCATION_API_KEY`; see [Configuration](../getting-started/configuration.md)):

```python
from gigaspatial import GigaSchoolLocationFetcher, PoiViewGenerator

schools = GigaSchoolLocationFetcher(country="RWA").fetch_locations(process_geospatial=True)
view = PoiViewGenerator(schools, poi_id_column="giga_id_school").map_wp_pop(
    country="RWA",
    map_radius_meters=1000,
)
print(view[["giga_id_school", "population"]].head())
```

For school-age or under-18 population, pass WorldPop options such as `school_age=True` or `under_18=True` to `map_wp_pop`. See [Points and nearest buildings](points-and-buildings.md) to also attach the nearest building footprint to each school.
