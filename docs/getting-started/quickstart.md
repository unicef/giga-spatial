# Tutorial

This tutorial builds H3 cells for Rwanda and attaches WorldPop population to each cell. The result is a view.

The first run downloads the country's population raster into the data store. That can take a few minutes. Later runs read the stored files. WorldPop does not need an API key.

## Build the view

```python
from gigaspatial import H3ViewGenerator

zones = H3ViewGenerator(source="RWA", resolution=6)
view = zones.map_wp_pop(country="RWA")
print(view.head())
```

`source` is the country the cells cover. `resolution=6` is the H3 size: about 36 km² per cell. `map_wp_pop` loads the WorldPop raster for that country and sums it onto the cells. `view` is a table of cells with a `population` column and a geometry.

`S2ViewGenerator` and `MercatorViewGenerator` follow the same pattern: construct them with a country, then call `map_wp_pop`.

## What happened

A handler acquired the WorldPop files and stored them. The generator mapped that dataset onto zones. The view is those zones with population attached. That attachment is enrichment.

The next pages name those pieces, then show other uses:

- [Datasets](datasets.md): handlers, fetchers, and cropping to a source
- [Points and nearest buildings](../user-guide/points-and-buildings.md)
- [Processing a raster or a table](../user-guide/processing.md), when you want the files themselves rather than a view
