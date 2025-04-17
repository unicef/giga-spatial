# Processing GeoTIFF Files

This example demonstrates how to process GeoTIFF files using the `TifProcessor` class.

## Prerequisites

Ensure you have installed the `gigaspatial` package and set up the necessary configuration. Follow the [Installation Guide](../getting-started/installation.md) if you haven’t already.

## Example Code

```python
from gigaspatial.processing.tif_processor import TifProcessor

# Initialize the processor
processor = TifProcessor("/path/to/ghsl/data/ghsl_data.tif")

# Process the GeoTIFF file
processed_data = processor.to_dataframe()
print(processed_data.head())
```

## Explanation

- **TifProcessor**: This class processes GeoTIFF files and extracts relevant data.
- **process**: This method processes the GeoTIFF file and returns the data as a NumPy array.

## Next Steps

Once the data is processed, you can store it using the [Storage Examples](../storage/geojson.md).

---

[Back to Examples](../index.md)