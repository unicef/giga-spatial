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

The `gigaspatial` package uses a configuration file (`config.py`) to manage paths, API keys, and other settings. You can customize the configuration as needed.

### Using Environment Variables

The package can read configuration settings from an environment file (e.g., `.env`). Here's an example of how to set up the `.env` file based on the `env_sample`:

```bash
# Paths for different data types
BRONZE_DIR=/path/to/your/bronze_tier_data
SILVER_DIR=/path/to/your/silver_tier_data
GOLD_DIR=/path/to/your/gold_tier_data
VIEWS_DIR=/path/to/your/views_data
ADMIN_BOUNDARIES_DIR=/path/to/your/admin_boundaries_data

# API keys and tokens
MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
MAXAR_USERNAME=your_maxar_username_here
MAXAR_PASSWORD=your_maxar_password_here
MAXAR_CONNECTION_STRING=your_maxar_key_here
```

The `config.py` file will automatically read these environment variables and set the paths and keys accordingly.

### Setting Paths Manually

You can also set paths manually in your code:

```python
from gigaspatial.config import config

# Example: Setting custom data storage paths
config.set_path("bronze", "/path/to/your/bronze_tier_data")
config.set_path("gold", "/path/to/your/gold_tier_data")
config.set_path("views", "/path/to/your/views_data")
```

API keys and tokens should be set through environment variables.

## Downloading Geospatial Data

To download geospatial data, you can use the `GHSLDataDownloader` class from the `handlers` module. Here's an example of downloading data for a specific country:

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

## Processing Geospatial Data

Once the data is downloaded, you can process it using the `TifProcessor` class from the `processing` module. Here's an example of processing GHSL data:

```python
from gigaspatial.processing.tif_processor import TifProcessor

# Initialize the processor
processor = TifProcessor(dataset_path="/path/to/ghsl/data/ghsl_data.tif", data_store=None, mode="single")

# Process the GHSL data
processed_data = processor.to_dataframe()
print(processed_data.head())
```

## Storing Geospatial Data

You can store the processed data in various formats using the `DataStore` class from the `core.io` module. Here's an example of saving data to a parquet file:

```python
from gigaspatial.core.io.local_data_store import LocalDataStore

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