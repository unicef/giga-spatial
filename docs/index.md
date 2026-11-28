<div class="gs-hero" markdown>

# GigaSpatial

GigaSpatial acquires published geospatial datasets, loads them for a source, and processes them. A generator maps a dataset onto points of interest or zones. The result is a view: those places with attributes from the datasets attached.

[Tutorial](getting-started/quickstart.md){ .md-button .gs-button }
[API reference](api/index.md){ .md-button }

</div>

=== "Map onto H3 hexagons"

    ```python
    from gigaspatial import H3ViewGenerator

    zones = H3ViewGenerator(source="RWA", resolution=6)
    view = zones.map_wp_pop(country="RWA")
    print(view.head())
    ```

=== "Download building footprints"

    ```python
    from gigaspatial import GoogleMSBuildingsHandler

    buildings = GoogleMSBuildingsHandler()
    buildings.ensure_data_available("ATG")  # downloads country GeoParquet on first run; no API key needed
    gdf = buildings.load_data("ATG")
    print(len(gdf), gdf.columns.tolist())
    ```

=== "Aggregate WorldPop population"

    ```python
    from gigaspatial import AdminBoundaries, WPPopulationHandler

    pop = WPPopulationHandler(year=2025, resolution=1000)
    raster = pop.load_data("RWA", merge_rasters=True)  # downloads GeoTIFF on first run; no API key needed
    provinces = AdminBoundaries.create("RWA", admin_level=1).to_geodataframe()
    provinces["population"] = raster.sample_by_polygons(provinces.geometry, stat="sum")
    print(provinces[["name", "population"]])
    ```

<div class="grid cards" markdown>

-   <a class="gs-card-link" href="getting-started/datasets/"><span class="gs-card-title">Acquire a dataset</span>A handler persists files in a data store and loads them for a source. A fetcher returns its result in memory and imposes no data-store layout.</a>

-   <a class="gs-card-link" href="getting-started/quickstart/"><span class="gs-card-title">Map it onto places</span>A generator attaches a dataset to points of interest or zones. The result is a view.</a>

-   <a class="gs-card-link" href="user-guide/processing/"><span class="gs-card-title">Process the files</span>Read a raster or a table yourself when you want the stored files rather than a view.</a>

-   <a class="gs-card-link" href="about/"><span class="gs-card-title">About GigaSpatial</span>Started inside UNICEF's Giga initiative. Who maintains it, and where it is going.</a>

</div>

## Guides

- [Combining datasets for a country](user-guide/country-datasets.md)
- [Points and nearest buildings](user-guide/points-and-buildings.md)
- [Population around points](user-guide/population-accessibility.md)
- [OSM points of interest](user-guide/osm-places.md)
- [Road networks](user-guide/road-networks.md)
- [Processing a raster or a table](user-guide/processing.md)
- [Settlement](user-guide/settlement-characterization.md)
- [Infrastructure files](user-guide/infrastructure-normalization.md)

The [API reference](api/index.md) lists the classes a caller imports. The [handler base](api/handler-base.md) is for adding a dataset to this library.
