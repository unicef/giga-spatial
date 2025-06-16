# Downloading and Processing GHSL Data

This example demonstrates how to download and process data from the Global Human Settlement Layer (GHSL) using the `GHSLDataHandler` class.

## Prerequisites

Ensure you have installed the `gigaspatial` package and set up the necessary configuration. Follow the [Installation Guide](../getting-started/installation.md) if you haven't already.

## Example Code

```python
from gigaspatial.handlers import GHSLDataHandler

# Initialize the handler with desired product and parameters
ghsl_handler = GHSLDataHandler(
    product="GHS_BUILT_S",  # Built-up surface
    year=2020,
    resolution=100,  # 100m resolution
)

# Download and load data for a specific country
country_code = "TUR"
downloaded_files = ghsl_handler.load_data(country_code, ensure_available=True)

# Load the data into a DataFrame
df = ghsl_handler.load_into_dataframe(country_code, ensure_available=True)
print(df.head())

# You can also load data for specific points
points = [(38.404581, 27.4816677), (39.8915702, 32.7809618)]  # Example coordinates
df_points = ghsl_handler.load_into_dataframe(points, ensure_available=True)
```

## Explanation

- **GHSLDataHandler**: This class provides a unified interface for downloading and processing GHSL data.
- **Available Products**:
  - `GHS_BUILT_S`: Built-up surface
  - `GHS_BUILT_H_AGBH`: Average building height
  - `GHS_BUILT_H_ANBH`: Average number of building heights
  - `GHS_BUILT_V`: Building volume
  - `GHS_POP`: Population
  - `GHS_SMOD`: Settlement model
- **Parameters**:
  - `product`: The GHSL product to use
  - `year`: The year of the data (default: 2020)
  - `resolution`: The resolution in meters (default: 100)
- **Methods**:
  - `load_data()`: Downloads and loads the data
  - `load_into_dataframe()`: Loads the data into a pandas DataFrame

## Next Steps

Once the data is downloaded and processed, you can:
1. Store the data using the `DataStore` class
2. Visualize the data using `geopandas` and `matplotlib`
3. Process the data further using the [Processing Examples](../processing/tif.md)

---

[Back to Examples](../index.md)