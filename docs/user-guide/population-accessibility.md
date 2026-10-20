# Population around points

The [tutorial](../getting-started/quickstart.md) sums WorldPop onto H3 cells. This guide sums it inside a radius around each point. The result is a view: your points, with a `population` column.

```python
from gigaspatial import PoiViewGenerator

points = [(-1.9441, 30.0619), (-1.9500, 30.1000)]
places = PoiViewGenerator(points)
view = places.map_wp_pop(country="RWA", map_radius_meters=1000)
print(view.head())
```

`map_radius_meters` is the buffer around each point. The first run downloads the WorldPop raster for the country.

Pass a GeoDataFrame of schools, clinics, or any other points the same way. Giga school locations are loaded in [Points and nearest buildings](points-and-buildings.md#giga-school-locations).
