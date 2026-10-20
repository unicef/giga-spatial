<div class="gs-hero" markdown>

# GigaSpatial

GigaSpatial acquires published geospatial datasets into a data store you control, loads them for a source, and processes them. A generator maps a dataset onto points of interest or zones. The result is a view: those places with attributes from the datasets attached.

[Tutorial](getting-started/quickstart.md){ .md-button .gs-button }
[API reference](api/index.md){ .md-button }

</div>

```python
from gigaspatial import H3ViewGenerator

zones = H3ViewGenerator(source="RWA", resolution=6)
view = zones.map_wp_pop(country="RWA")
print(view.head())
```

<div class="grid cards" markdown>

-   <a class="gs-card-link" href="getting-started/datasets/"><span class="gs-card-title">Acquire a dataset</span>A handler stores a published dataset and loads it for a source. A fetcher queries an API or a database and returns records.</a>

-   <a class="gs-card-link" href="getting-started/quickstart/"><span class="gs-card-title">Map it onto places</span>A generator attaches a dataset to points of interest or zones. The result is a view.</a>

-   <a class="gs-card-link" href="user-guide/processing/"><span class="gs-card-title">Process the files</span>Read a raster or a table yourself when you want the stored files rather than a view.</a>

-   <a class="gs-card-link" href="about/"><span class="gs-card-title">About GigaSpatial</span>Started inside UNICEF's Giga initiative. Who maintains it, and where it is going.</a>

</div>

## Guides

- [Points and nearest buildings](user-guide/points-and-buildings.md)
- [Population around points](user-guide/population-accessibility.md)
- [OSM places](user-guide/osm-places.md)
- [Processing a raster or a table](user-guide/processing.md)
- [Settlement](user-guide/settlement-characterization.md)
- [Infrastructure files](user-guide/infrastructure-normalization.md)

The [API reference](api/index.md) lists the classes a caller imports. The [handler base](api/handler-base.md) is for adding a dataset to this library.
