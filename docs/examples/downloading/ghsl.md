# Downloading GHSL Data

This example demonstrates how to download data from the Global Human Settlement Layer (GHSL) using the `GHSLDataDownloader` class.

## Prerequisites

Ensure you have installed the `gigaspatial` package and set up the necessary configuration. Follow the [Installation Guide](../getting-started/installation.md) if you haven’t already.

## Example Code

```python
from gigaspatial.handlers.ghsl import GHSLDataDownloader

# Initialize the downloader
downloader = GHSLDataDownloader({
    "product": "GHS_BUILT_S",
    "year": 2020,
    "resolution": 100,
    "coord_system": "WGS84"
})

# Download data for a specific country
country_code = "TUR"
downloader.download_by_country(country_code)
```

## Explanation

- **GHSLDataDownloader**: This class handles the downloading of GHSL data.
- **download_by_country**: This method downloads data for a specific country.

## Next Steps

Once the data is downloaded, you can process it using the [Processing Examples](../processing/tif.md).

---

[Back to Examples](../index.md)