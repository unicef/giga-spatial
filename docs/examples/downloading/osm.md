# Downloading OSM Data

This example demonstrates how to fetch and process OpenStreetMap (OSM) data using the `OSMLocationFetcher` class.

## Prerequisites

Ensure you have installed the `gigaspatial` package and set up the necessary configuration. Follow the [Installation Guide](../getting-started/installation.md) if you haven’t already.

## Example Code

```python
from gigaspatial.handlers.osm import OSMLocationFetcher

# Initialize the fetcher
fetcher = OSMLocationFetcher(
    country="Spain",
    location_types=["amenity", "building", "shop"]
)

# Fetch and process OSM locations
locations = fetcher.fetch_locations(since_year=2020, handle_duplicates="combine")
print(locations.head())
```

## Explanation

- **OSMLocationFetcher**: This class fetches and processes location data from OpenStreetMap.
- **fetch_locations**: This method fetches and processes OSM data based on the specified criteria.

## Next Steps

Once the data is fetched, you can process it using the [Processing Examples](../processing/vector.md).

---

[Back to Examples](../index.md)