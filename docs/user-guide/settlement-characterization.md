# Settlement

GHSL's settlement model (`GHS_SMOD`) classifies each cell as a kind of built-up area, from rural to urban centre. Mapping it onto points attaches that class to each point. The result is a view.

The first run downloads the settlement raster. It does not need an API key.

```python
from gigaspatial import PoiViewGenerator

points = [(-1.9441, 30.0619), (-1.9500, 30.1000)]
places = PoiViewGenerator(points)
view = places.map_smod()
print(view[["smod_class"]].head())
```

`map_smod` uses the 2020 release at 1000 m. `stat="median"` is the class at the point.

The same method exists on a zone generator. After the [tutorial](../getting-started/quickstart.md), `zones.map_smod()` adds `smod_class` to the H3 view.

To load the raster yourself, construct the handler:

```python
from gigaspatial import GHSLDataHandler

settlement = GHSLDataHandler(product="GHS_SMOD", year=2020, resolution=1000)
frame = settlement.load_into_dataframe("RWA")
```
