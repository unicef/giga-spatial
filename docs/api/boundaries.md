# Boundaries

`AdminBoundaries` loads administrative boundaries for a country. It is not a handler and not a fetcher. Handlers use it when a country source needs a geometry.

```python
from gigaspatial import AdminBoundaries
from gigaspatial.handlers import GeoRepoClient
```

::: gigaspatial.handlers.boundaries.AdminBoundaries
    options:
      show_source: false
