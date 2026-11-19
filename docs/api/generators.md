# Generators

A generator maps datasets onto points of interest or zones. The result is a view: a geometry-free table keyed by `poi_id` or `zone_id` with dataset attributes attached. Every `map_*` method on `PoiViewGenerator` and `ZonalViewGenerator` (including `map_google_buildings` and `map_ms_buildings`) updates and returns `self.view`, filling expected columns with `0`, `False`, or `NaN` when loaded spatial data is empty. Call `.to_geodataframe()` when you want geometries re-attached.

```python
from gigaspatial import (
    PoiViewGenerator,
    ZonalViewGenerator,
    H3ViewGenerator,
    S2ViewGenerator,
    MercatorViewGenerator,
    AdminBoundariesViewGenerator,
)
```

## Points

::: gigaspatial.generators.poi.PoiViewGenerator
    options:
      show_source: false
      inherited_members: true
      members:
        - find_nearest_buildings
        - map_wp_pop
        - map_ghsl_pop
        - map_smod
        - map_built_s
        - map_ghsl
        - map_rwi
        - map_google_buildings
        - map_ms_buildings
        - map_nearest_points
        - map_zonal_stats
        - map_points
        - map_polygons
        - map_rasters
        - map_aqueduct_flood_hazard
        - save_view
        - to_dataframe
        - to_geodataframe

## Zones

::: gigaspatial.generators.zonal.base.ZonalViewGenerator
    options:
      show_source: false
      inherited_members: true
      members:
        - map_wp_pop
        - map_ghsl_pop
        - map_smod
        - map_built_s
        - map_buildings
        - map_rwi
        - map_ghsl
        - map_google_buildings
        - map_ms_buildings
        - map_aqueduct_flood_hazard
        - map_points
        - map_polygons
        - map_rasters
        - save_view
        - to_dataframe
        - to_geodataframe

::: gigaspatial.generators.zonal.h3.H3ViewGenerator
    options:
      show_source: false
      inherited_members: true
      members:
        - map_wp_pop
        - map_buildings
        - map_rwi

::: gigaspatial.generators.zonal.s2.S2ViewGenerator
    options:
      show_source: false
      inherited_members: true
      members:
        - map_wp_pop
        - map_buildings
        - map_rwi

::: gigaspatial.generators.zonal.mercator.MercatorViewGenerator
    options:
      show_source: false
      inherited_members: true
      members:
        - map_wp_pop
        - map_buildings
        - map_rwi

::: gigaspatial.generators.zonal.admin.AdminBoundariesViewGenerator
    options:
      show_source: false
      inherited_members: true
      members:
        - map_wp_pop
        - map_buildings
        - map_rwi
