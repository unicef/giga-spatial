# Image downloaders

These classes download image files into a data store. They are not handlers and not fetchers.

```python
from gigaspatial import MaxarImageDownloader, MapboxImageDownloader
from gigaspatial.image_downloaders import MaxarConfig
```

::: gigaspatial.image_downloaders.maxar_image.MaxarImageDownloader
    options:
      show_source: false
      members:
        - download_images_by_coordinates
        - download_images_by_bounds
        - download_images_by_tiles

::: gigaspatial.image_downloaders.mapbox_image.MapboxImageDownloader
    options:
      show_source: false
      members:
        - download_images_by_coordinates
        - download_images_by_bounds
        - download_images_by_tiles
