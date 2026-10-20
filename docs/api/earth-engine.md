# Earth Engine

`GEEProfiler` inspects and samples datasets in Google Earth Engine. It is a fetcher: it queries Earth Engine and returns records. It needs the Earth Engine credentials in [Configuration](../getting-started/configuration.md) and the `gee` extra.

```python
from gigaspatial import GEEProfiler
from gigaspatial.fetchers import GEEConfig, GEEDatasetRegistry
```

::: gigaspatial.fetchers.gee.profiler.GEEProfiler
    options:
      show_source: false
