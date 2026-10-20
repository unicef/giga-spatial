# OSM places

OpenStreetMap places are a fetcher. `OSMLocationFetcher` queries the Overpass API and returns a table of records. It does not store the dataset as files. The public Overpass instance needs no API key.

```python
from gigaspatial import OSMLocationFetcher

clinics = OSMLocationFetcher(
    country="RWA",
    location_types=["clinic", "hospital"],
).fetch_locations()
print(clinics.head())
```

`location_types` is a list of OSM values, or a dict of key to values such as `{"amenity": ["clinic", "hospital"]}`.

To limit the query to one admin area, pass `admin_level` and `admin_value` instead of relying on the whole country. A country-sized query can take several minutes.

`fetch_locations_changed_between` returns places edited in a date range. `fetch_by_osmid` loads particular OSM ids.
