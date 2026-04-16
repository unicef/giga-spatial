# Infrastructure Data Normalization

Raw data from external partners (like KMZs from telecom providers) often arrives in proprietary formats. GigaSpatial provides dedicated schemas and processors to normalize this data into a standardized infrastructure database.

## The "Why": Schemas and Processors

- **Schemas** (e.g., `TransmissionNodeTable`): Define the "Structure" of how infrastructure data should look, ensuring that fields like `node_type` and `transmission_medium` are consistent across different sources.
- **Processors** (e.g., `EntityProcessor`): Handle the "Action" of cleaning and deduplicating raw dataframes to match the schema requirements.

## The Workflow

### 1. Reading Raw Datasets

GigaSpatial provides `read_dataset` as a unified entry point for KMZ, GPKG, Parquet, and Shapefiles.

```python
from gigaspatial.core.io import read_dataset

# Load a raw KMZ or GPKG file from partner data
raw_df = read_dataset("/path/to/partner_data/kenya_fiber.kmz")
```

### 2. Normalizing to a Schema

Once loaded, we wrap the dataframe in a Schema class to enforce consistency and apply default values.

```python
from gigaspatial.core.schemas.transmission_node import TransmissionNodeTable

# Initialize the table schema with our raw data
# This allows the library to map non-standard column names to our core schema
node_table = TransmissionNodeTable(raw_df)
```

### 3. Cleaning and Deduplication

We use the `EntityProcessor` to perform high-level cleaning operations like geometry validation and row-level deduplication.

```python
from gigaspatial.core.schemas.entity import EntityProcessor

# Initialize the processor for our node table
processor = EntityProcessor(node_table.df)

# Execute standard cleaning pipeline
# This removes duplicates based on coordinates and name
cleaned_df = processor.process(drop_duplicates=True)
```

## Rationale for this Combination

By separating the **Reading** (DataStore/IO), **Normalization** (Schemas), and **Processing** (EntityProcessor), GigaSpatial allows for repeatable data cleaning pipelines. 

For example, when processing `ken-joints-manholes-handholes.gpkg` seen in our production notebooks, this workflow allows the library to:
1. Normalize partner-specific names (like `Joint 5 - KITUI`) into a standard `TransmissionNode`.
2. Automatically extract longitude/latitude from nested KMZ geometries.
3. Identify "logical" nodes (exchanges) that serves as primary connectivity hubs.

This "Giga-ready" standardized data can then be passed into the `PoiViewGenerator` for catchment analysis.
