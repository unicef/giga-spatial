# View generators

A **generator** maps geospatial datasets onto **points of interest** or **zones**. That attachment is **enrichment**, and the result is a **view**: a geometry-free pandas `DataFrame` keyed by a canonical identifier column (`poi_id` or `zone_id`), with dataset attributes attached and geometries kept aside until you export geospatial data.

This page explains why view generators exist, the two shapes of views they produce, the uniform contract shared across every generator and `map_*` method, and how low-level spatial primitives relate to built-in dataset mappers. For a hands-on walkthrough, start with the [Tutorial](../getting-started/quickstart.md) or [Combining datasets for a country](../user-guide/country-datasets.md). For full method signatures, see the [Generators API](../api/generators.md).

## Why view generators

Most downstream modelling and reporting needs a flat table: one row per school, clinic, cell site, administrative district, or grid cell, and one column per feature (population within 1 km, nearest building distance, built-up surface, relative wealth index, flood depth).

Building that table by hand from raw geospatial files repeats the same fragile plumbing every time:

- Reprojecting geometries to match each raster or vector CRS (and converting degrees to metres for buffers and areas).
- Merging multi-tile rasters, sampling values at points or over polygons, and closing raster handles cleanly.
- Choosing spatial predicates (`within`, `intersects`, or area-weighted `fractional` aggregation) for points and polygons.
- Guarding against column collisions when chaining several enrichments onto the same table.
- Handling empty spatial intersections consistently so an empty dataset still yields predictable columns instead of breaking a pipeline.

A view generator encapsulates that table-building pipeline. You define your places once, attach as many datasets as your analysis requires, and get a clean `DataFrame` ready for pandas, scikit-learn, or export to Parquet, CSV, GeoJSON, or GeoPackage.

## What it is like to use

A few principles run through every view generator:

- **Geometry-free working table.** `generator.view` and `generator.to_dataframe()` return a defensive copy of a plain pandas `DataFrame` keyed by `poi_id` or `zone_id`. Call `generator.to_geodataframe()` whenever you need geometries back in the generator's CRS.
- **Stateful, chainable `map_*` methods.** Every `map_*` method updates the generator's internal view and returns `self.view`, so you can inspect the table after any step or run a declarative list of steps with `chain_operations`.
- **Collision protection by default.** Attempting to write to a column that already exists raises `ValueError` unless you pass `overwrite=True` or specify a new `output_column` (or `output_suffix` / `output_prefix`).
- **Predictable empty results, loud acquisition errors.** If a loaded dataset contains zero intersecting features, the expected columns are still added with semantic empty values (`0`, `False`, or `NaN`). If a handler cannot resolve or download required files, the error raises immediately.
- **Shared storage and offline controls.** Pass `data_store=` and `ensure_available=` once on the generator constructor; built-in dataset methods inherit them, and can also accept a pre-configured `handler=` or override `ensure_available` per call.

```python
from gigaspatial import H3ViewGenerator

zones = H3ViewGenerator(source="RWA", resolution=6)
zones.map_wp_pop(country="RWA")
zones.map_buildings(country="RWA")
zones.map_smod()

df = zones.view                # geometry-free DataFrame keyed by zone_id
gdf = zones.to_geodataframe()  # GeoDataFrame with cell geometries and CRS preserved
zones.save_view("views/rwanda_h3.parquet")
```

## Two shapes of views

Every view in GigaSpatial has one of two shapes, depending on whether your unit of analysis is a point or an area.

### Point-of-interest views (`PoiViewGenerator`)

Use `PoiViewGenerator` when each row is a **point of interest** analysed in its own right, such as a school, clinic, or cell site.

- **Key (`id_column`):** `poi_id`. If you pass a `DataFrame`, `GeoDataFrame`, or list of dicts with your own identifier column, pass `poi_id_column=` to standardize it to `poi_id` (duplicates, missing columns, or collisions with an existing `poi_id` column raise `ValueError`). If no identifier column is supplied (for example, a list of `(latitude, longitude)` tuples), `poi_0`, `poi_1`, ... are generated automatically.
- **Geometry handling:** input points are standardized to `EPSG:4326` with `latitude` and `longitude` columns preserved in the view.
- **Enrichment geometry:** operations either act at the point itself (raster sampling at coordinates, nearest point, nearest building) or inside a circular buffer around each point via `map_radius_meters` (for example, summing WorldPop population within 1,000 m of each school).
- **Building footprint methods:** nearest-building queries (`find_nearest_buildings`, `map_google_buildings`, `map_ms_buildings`).

### Zonal views (`ZonalViewGenerator` and built-in subclasses)

Use a zonal generator when each row is a **zone** (a polygon or multipolygon) that dataset values are summarised into via **zonal aggregation**.

- **Key (`id_column`):** `zone_id`.
- **Custom polygons:** instantiate `ZonalViewGenerator` directly with either a `GeoDataFrame` (setting `zone_id_column=` and `zone_data_crs=`) or a `dict` mapping zone identifiers to Shapely `Polygon` / `MultiPolygon` geometries.
- **Built-in zone generators:** four subclasses build standard zone geometries for you, standardize their native identifier to `zone_id`, and remember the country when initialized from a country string so country-scoped mappers like `map_wp_pop`, `map_buildings`, and `map_rwi` can use it automatically:
  - `AdminBoundariesViewGenerator`: political or statistical **administrative boundaries** for a `country` and `admin_level` (`boundary_id` -> `zone_id`).
  - `H3ViewGenerator`: Uber H3 hexagonal **grid system** cells for a `source` and `resolution` (`h3` -> `zone_id`).
  - `S2ViewGenerator`: Google S2 **grid system** cells for a `source` and `level` (`cell_token` -> `zone_id`).
  - `MercatorViewGenerator`: Web Mercator tile **grid system** cells for a `source` and `zoom_level` (`quadkey` -> `zone_id`).
- **Building footprint methods:** zonal counts and footprint areas (`map_buildings`, `map_google_buildings`, `map_ms_buildings`).

| Dimension | Point-of-interest view | Zonal view |
| --- | --- | --- |
| Primary classes | `PoiViewGenerator` | `ZonalViewGenerator`, `AdminBoundariesViewGenerator`, `H3ViewGenerator`, `S2ViewGenerator`, `MercatorViewGenerator` |
| Canonical `id_column` | `poi_id` | `zone_id` |
| Base geometry | Points (`EPSG:4326`) | Polygons or multipolygons (preserves `zone_data_crs`) |
| How spatial enrichment runs | At points or inside `map_radius_meters` buffers | Aggregated over each zone polygon |
| Building footprint operations | Nearest building distance and presence (`find_nearest_buildings`, `map_google_buildings`, `map_ms_buildings`) | Zonal building counts and areas (`map_buildings`, `map_google_buildings`, `map_ms_buildings`) |

## The view contract

All view generators follow one uniform contract (recorded in [ADR 0010](../adr/0010-view-generator-base-and-contract.md)):

- **Defensive copies and geometry re-attachment.** Both `generator.view` and `generator.to_dataframe()` return a fresh copy of the geometry-free `DataFrame`, so editing the returned frame never mutates the generator's internal state. `generator.to_geodataframe()` merges the current view onto the base geometries on `id_column` and preserves the generator's CRS.
- **Uniform `map_*` return value.** Every `map_*` method (both low-level spatial primitives and built-in dataset mappers) validates its output columns, merges the new attributes into the internal view by `id_column`, and returns `self.view`. You can also attach your own `dict` or `pd.Series` keyed by `id_column` with `add_variable_to_view`.
- **Overwrite guard (`overwrite=False`).** Output columns are checked before any computation runs. If a target column already exists in the view, the method raises `ValueError` unless `overwrite=True` is passed. This protects chained workflows such as calling `map_wp_pop` at two radii (pass distinct `output_column` names like `"pop_1km"` and `"pop_5km"`).
- **Empty data vs. errors.** When a loaded dataset contains zero features or zero intersecting rows, the generator still appends the expected output columns filled with semantic empty values:
  - `0` for counts (such as `point_count`, `polygon_count`, `building_count`, `google_buildings_count`, `ms_buildings_count`).
  - `False` for boolean presence flags (such as `building_within_{search_radius}m`).
  - `NaN` for continuous or categorical statistics, areas, and nearest-neighbour IDs or distances.
  By contrast, handler resolution errors, download failures, or missing stored files when `ensure_available=False` raise exceptions rather than producing silent empty columns.
- **Extension-dispatched persistence (`.save_view(path)`).** Pass a target file path directly to `.save_view(path)`. The generator writes through its `data_store` via `write_dataset`, selecting the format from the file extension (`.parquet`, `.csv`, `.xlsx`, `.json` for tabular views; `.geojson`, `.gpkg`, or GeoParquet for geospatial views). By default (`include_geometry=None`), geospatial extensions automatically attach geometries via `.to_geodataframe()`, while tabular extensions write the geometry-free table. Pass `include_geometry=True` to write GeoParquet (`.parquet`) with geometries included.

## Low-level primitives vs. built-in dataset methods

View generators expose two complementary layers of `map_*` methods, both following the exact same view contract:

### Low-level spatial primitives

Use these when you already have data in memory, whether loaded via a [handler](handlers.md), queried with a fetcher (such as `OSMLocationFetcher`), or read from your own local files:

- **`map_points`**: aggregates point data (counts or column statistics like `sum`, `mean`, `min`, `max`) into zones or POI buffers (`map_radius_meters` is required on `PoiViewGenerator`).
- **`map_polygons`**: aggregates polygon data onto zones or POIs/buffers with `intersects`, `within`, or area-weighted `fractional` predicates.
- **`map_rasters`**: samples one or more `TifProcessor` rasters, automatically reprojecting geometries to the raster CRS, merging multiple rasters when needed, and sampling at point coordinates or over polygons/buffers.
- **`map_nearest_points`** (`PoiViewGenerator`): computes the nearest target point and distance for each POI, adding `{output_prefix}_id` and `{output_prefix}_distance`.
- **`map_zonal_stats`** (`PoiViewGenerator`): convenience entry point that routes `TifProcessor` inputs to `map_rasters`, point `GeoDataFrame` inputs to `map_points`, and polygon `GeoDataFrame` inputs to `map_polygons`.

### Built-in dataset methods

Use these when you want a published dataset acquired and enriched in a single call. Each method wires up the corresponding handler, picks sensible defaults (CRS, resolution, statistic, and output column name), and delegates to the spatial primitives above:

- **Shared across POI and zonal generators:** `map_wp_pop` (WorldPop population and degree of urbanisation), `map_ghsl` (any GHSL raster product), `map_ghsl_pop` (GHSL population), `map_built_s` (GHSL built-up surface), `map_smod` (GHSL settlement model), `map_rwi` (Meta Relative Wealth Index), and `map_aqueduct_flood_hazard` (WRI Aqueduct flood hazard).
- **Building footprint methods:** `find_nearest_buildings`, `map_google_buildings`, and `map_ms_buildings` on `PoiViewGenerator`; `map_buildings`, `map_google_buildings`, and `map_ms_buildings` on `ZonalViewGenerator`.

Every built-in dataset method accepts an optional `handler=` argument. Passing a pre-configured handler lets you customize dataset parameters, reuse a handler across calls, or inject a handler backed by a local `LocalDataStore` fixture in offline tests while keeping the exact same enrichment contract.

## Where to go next

- **Step-by-step guides:** [Tutorial](../getting-started/quickstart.md), [Combining datasets for a country](../user-guide/country-datasets.md), [Points and nearest buildings](../user-guide/points-and-buildings.md), [Population around points](../user-guide/population-accessibility.md), and [Settlement](../user-guide/settlement-characterization.md).
- **How datasets are acquired underneath:** [Handlers](handlers.md).
- **API reference:** [Generators](../api/generators.md) for all classes, parameters, and return types.
- **Design rationale:** [ADR 0010](../adr/0010-view-generator-base-and-contract.md) records why the POI and zonal generators share a single contract.
