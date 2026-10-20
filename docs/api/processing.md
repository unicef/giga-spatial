# Processing

`TifProcessor` reads, clips, and samples a GeoTIFF. The functions below work on tables and geometries. The [processing guide](../user-guide/processing.md) shows both.

::: gigaspatial.processing.tif_processor.TifProcessor
    options:
      show_source: false
      members:
        - to_dataframe
        - to_geodataframe
        - clip_to_geometry
        - clip_to_bounds
        - sample_by_coordinates
        - sample_by_polygons
        - reproject_to
        - get_raster_info

::: gigaspatial.processing.geo.convert_to_geodataframe
    options:
      show_source: false

::: gigaspatial.processing.geo.buffer_geodataframe
    options:
      show_source: false

::: gigaspatial.processing.geo.aggregate_points_to_zones
    options:
      show_source: false

::: gigaspatial.processing.geo.aggregate_polygons_to_zones
    options:
      show_source: false

::: gigaspatial.processing.geo.calculate_distance
    options:
      show_source: false

::: gigaspatial.processing.geo.map_points_within_polygons
    options:
      show_source: false
