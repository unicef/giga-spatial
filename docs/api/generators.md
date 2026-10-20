# Generators

A generator maps datasets onto points of interest or zones. The result is a view: those places with dataset attributes attached.

```python
from gigaspatial import PoiViewGenerator, H3ViewGenerator, AdminBoundariesViewGenerator
from gigaspatial.generators import PoiViewGeneratorConfig, ZonalViewGeneratorConfig
```

## Points

::: gigaspatial.generators.poi.PoiViewGenerator
    options:
      show_source: false
      members:
        - find_nearest_buildings
        - map_wp_pop
        - map_smod
        - map_built_s
        - map_google_buildings
        - map_ms_buildings
        - map_nearest_points
        - map_zonal_stats
        - map_aqueduct_flood_hazard

## Zones

::: gigaspatial.generators.zonal.h3.H3ViewGenerator
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
