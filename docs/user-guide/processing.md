# Processing a raster or a table

The tutorial and the other guides stop at a view. This page uses the stored files and the vector helpers directly.

## A raster

`GHSLDataHandler` downloads the tiles that cover a country. `load_into_dataframe` turns those tiles into a table of pixel values. The first run downloads the files.

```python
from gigaspatial import GHSLDataHandler

population = GHSLDataHandler(product="GHS_POP", year=2020, resolution=1000)
frame = population.load_into_dataframe("RWA")
print(frame.head())
```

For a file you already have, `TifProcessor` reads it, clips it, and samples it.

```python
from gigaspatial import TifProcessor

raster = TifProcessor("bronze/ghsl/your-file.tif")
print(raster.get_raster_info())
pixels = raster.to_dataframe()
```

`clip_to_geometry` cuts the raster to a Shapely geometry. `sample_by_coordinates` reads values at points. `to_geodataframe` returns the pixels as polygons.

## A table

These helpers take a DataFrame or a GeoDataFrame:

- `convert_to_geodataframe` builds a GeoDataFrame from coordinate columns
- `buffer_geodataframe` buffers geometries in meters
- `aggregate_points_to_zones` and `aggregate_polygons_to_zones` summarize features onto zones
- `calculate_distance` returns the distance between two latitude/longitude pairs
- `map_points_within_polygons` finds which polygon contains each point

```python
import pandas as pd
from gigaspatial import convert_to_geodataframe, buffer_geodataframe

points = pd.DataFrame({"latitude": [-1.9441], "longitude": [30.0619]})
places = convert_to_geodataframe(points)
buffered = buffer_geodataframe(places, buffer_distance_meters=500)
```

`buffer_distance_meters` is the argument name on `buffer_geodataframe`. Check the [processing reference](../api/processing.md) if you call the others.
