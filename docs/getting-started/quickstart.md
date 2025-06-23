# Quick Start Guide

This guide will walk you through the basic usage of the `gigaspatial` package. By the end of this guide, you will be able to download, process, and store geospatial data using the package.

## Prerequisites

Before you begin, ensure that you have installed the `gigaspatial` package. If you haven't installed it yet, follow the [Installation Guide](installation.md).

## Importing the Package

Start by importing the `gigaspatial` package:

```python
import gigaspatial as gs
```

## Setting Up Configuration

The `gigaspatial` package uses a unified configuration system to manage paths, API keys, and other settings.

- **Environment Variables:** Most configuration is handled via environment variables, which can be set in a `.env` file at the project root. For a full list of supported variables and their descriptions, see the [Configuration Guide](../user-guide/configuration.md).
- **Defaults:** If not set, sensible defaults are used for all paths and keys.
- **Manual Overrides:** You can override data directory paths in your code using `config.set_path`.

### Example `.env` File

```bash
BRONZE_DIR=/path/to/your/bronze_tier_data
SILVER_DIR=/path/to/your/silver_tier_data
GOLD_DIR=/path/to/your/gold_tier_data
VIEWS_DIR=/path/to/your/views_data
CACHE_DIR=/path/to/your/cache
ADMIN_BOUNDARIES_DIR=/path/to/your/admin_boundaries
MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
# ... other keys ...
```

### Setting Paths Programmatically

```python
from gigaspatial.config import config

config.set_path("bronze", "/path/to/your/bronze_tier_data")
config.set_path("gold", "/path/to/your/gold_tier_data")
config.set_path("views", "/path/to/your/views_data")
```

> For more details and troubleshooting, see the [full configuration guide](../user-guide/configuration.md).

## Downloading and Processing Geospatial Data

The `gigaspatial` package provides several handlers for different types of geospatial data. Here are examples for two commonly used handlers:

### GHSL (Global Human Settlement Layer) Data

The `GHSLDataHandler` provides access to various GHSL products including built-up surface, building height, population, and settlement model data:

```python
from gigaspatial.handlers import GHSLDataHandler

# Initialize the handler with desired product and parameters
ghsl_handler = GHSLDataHandler(
    product="GHS_BUILT_S",  # Built-up surface
    year=2020,
    resolution=100,  # 100m resolution
)

# Download data for a specific country
country_code = "TUR"
downloaded_files = ghsl_handler.load_data(country_code, ensure_available = True)

# Load the data into a DataFrame
df = ghsl_handler.load_into_dataframe(country_code, ensure_available = True)
print(df.head())

# You can also load data for specific points or geometries
points = [(38.404581,27.4816677), (39.8915702, 32.7809618)]
df_points = ghsl_handler.load_into_dataframe(points, ensure_available = True)
```

### Google Open Buildings Data

The `GoogleOpenBuildingsHandler` provides access to Google's Open Buildings dataset, which includes building footprints and points:

```python
from gigaspatial.handlers import GoogleOpenBuildingsHandler

# Initialize the handler
gob_handler = GoogleOpenBuildingsHandler()

# Download and load building polygons for a country
country_code = "TUR"
polygons_gdf = gob_handler.load_polygons(country_code, ensure_available = True)

# Download and load building points for a country
points_gdf = gob_handler.load_points(country_code, ensure_available = True)

# You can also load data for specific points or geometries
points = [(38.404581, 27.4816677), (39.8915702, 32.7809618)]
polygons_gdf = gob_handler.load_polygons(points, ensure_available = True)
```

## Storing Geospatial Data

You can store the processed data in various formats using the `DataStore` class from the `core.io` module. Here's an example of saving data to a parquet file:

```python
from gigaspatial.core.io import LocalDataStore

# Initialize the data store
data_store = LocalDataStore()

# Save the processed data to a parquet file
with data_store.open("/path/to/your/output/processed_data.parquet", "rb") as f:
    processed_data.to_parquet(f)
```

If your dataset is already a `pandas.DataFrame` or `geopandas.GeoDataFrame`, `write_dataset` method from the `core.io.writers` module can be used to write the dataset in various formats. 

```python
from gigaspatial.core.io.writers import write_dataset

# Save the processed data to a GeoJSON file
write_dataset(data=processed_data, data_store=data_store, path="/path/to/your/output/processed_data.geojson")
```

## Visualizing Geospatial Data

To visualize the geospatial data, you can use libraries like `geopandas` and `matplotlib`. Here's an example of plotting the processed data on a map:

```python
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the GeoJSON file
gdf = gpd.read_file("/path/to/your/output/processed_data.geojson")

# Plot the data
gdf.plot()
plt.show()
```

`geopandas.GeoDataFrame.explore` can also be used to visualise the data on interactive map based on `GeoPandas` and `folium/leaflet.js`:
```python
# Visualize the data
gdf.explore("population", cmap="Blues")
```

## Next Steps

Now that you have a basic understanding of how to use the `gigaspatial` package, you can explore more advanced features and configurations. Check out the [User Guide](../user-guide/index.md) for detailed documentation and examples.

---

### Additional Resources

- [API Documentation](../api/index.md): Detailed documentation of all classes and functions.
- [Examples](../examples/basic.md): Real-world examples and use cases.
- [Changelog](../changelog.md): Information about the latest updates and changes.