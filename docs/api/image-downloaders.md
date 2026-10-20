# Image downloaders

These classes download image files into a data store. They are not handlers.

::: gigaspatial.handlers.maxar_image.MaxarImageDownloader
    options:
      show_source: false
      members:
        - download_images_by_coordinates
        - download_images_by_bounds
        - download_images_by_tiles

::: gigaspatial.handlers.mapbox_image.MapboxImageDownloader
    options:
      show_source: false
      members:
        - download_images_by_coordinates
        - download_images_by_bounds
        - download_images_by_tiles
