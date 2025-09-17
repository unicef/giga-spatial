# Processing Raster Files

This example demonstrates how to process raster files using the `TifProcessor` class.

## Prerequisites

Ensure you have installed the `gigaspatial` package and set up the necessary configuration. Follow the [Installation Guide](../getting-started/installation.md) if you haven’t already.

## Example Code

```python
from gigaspatial.processing import TifProcessor
from gigaspatial.core.io import LocalDataStore
from rasterio.warp import Resampling # For reprojection methods

# NOTE: For these examples, replace "/path/to/your/file.tif" with actual paths to your GeoTIFF files.
# You might need to create dummy files or use existing ones for local testing.

# 1. Initialize with a single TIFF file
print("--- Single TIFF File Processing ---")
single_processor = TifProcessor(
    "/path/to/single_band.tif", 
    mode="single" # Can be "rgb", "rgba", "multi"
)
df_single = single_processor.to_dataframe()
print("Single-band DataFrame head:")
print(df_single.head())
print("Raster Info for single_processor:")
print(single_processor.get_raster_info())


# 2. Initialize with multiple TIFF files for merging and reprojection
print("\n--- Multi-raster Merging and Reprojection ---")
# Replace with actual paths to your tif files. Ensure they are compatible for merging.
# Example: two adjacent tiles from a dataset.
tif_paths = [
    "/path/to/raster1.tif",
    "/path/to/raster2.tif"
]
merged_reprojected_processor = TifProcessor(
    dataset_path=tif_paths,
    mode="single", # Or "multi", "rgb", "rgba" depending on your data
    merge_method="mean", # Options: "first", "last", "min", "max", "mean"
    target_crs="EPSG:4326", # Reproject all rasters to WGS84 during initialization
)
df_merged = merged_reprojected_processor.to_dataframe()
print("Merged and Reprojected DataFrame head:")
print(df_merged.head())
print("Raster Info for merged_reprojected_processor:")
print(merged_reprojected_processor.get_raster_info())

# 3. Explicit Reprojection after initialization
print("\n--- Explicit Reprojection ---")
# Reproject the current raster (e.g., the merged one) to a different CRS or resolution
# In a real scenario, you'd save this to a persistent location.
reprojected_output_path = "./temp_reprojected_raster.tif" 
reprojected_path = merged_reprojected_processor.reproject_to(
    target_crs="EPSG:3857", # Web Mercator
    output_path=reprojected_output_path,
    resampling_method=Resampling.bilinear # Different resampling method
)
print(f"Raster reprojected to: {reprojected_path}")

# 4. Convert raster to a graph (NetworkX example)
print("\n--- Raster to Graph Conversion ---")
# Assuming '/path/to/single_band.tif' is a suitable single-band raster
graph_processor = TifProcessor(
    "/path/to/single_band.tif", 
    mode="single" # Graph conversion typically for single-band data
)
graph = graph_processor.to_graph(
    connectivity=8, # 4-connectivity (von Neumann) or 8-connectivity (Moore)
    include_coordinates=True, # Include 'x' and 'y' coordinates as node attributes
    graph_type="networkx" # Or "sparse" for scipy.sparse.csr_matrix
)
print(f"Generated a NetworkX graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
# Example: Access node attributes (first 5 nodes)
# for node_id, data in list(graph.nodes(data=True))[:5]:
#    print(f"Node {node_id}: Value={data['value']:.2f}, X={data.get('x'):.2f}, Y={data.get('y'):.2f}")
```

## Explanation

The `TifProcessor` class provides robust functionalities for handling GeoTIFF files, from single-band to multi-band (RGB, RGBA) datasets, with advanced processing capabilities including:

-   **Initialization**:
    -   Can be initialized with a single GeoTIFF file path.
    -   Supports a list of GeoTIFF file paths for **automatic merging** during initialization, configured via `merge_method` (`first`, `last`, `min`, `max`, `mean`).
    -   The `mode` parameter (`single`, `rgb`, `rgba`, `multi`) dictates how bands are interpreted and validated.
    -   `target_crs` and `reprojection_resolution` can be set during initialization to reproject rasters immediately to a consistent CRS and pixel size.
-   **Data Extraction**:
    -   `to_dataframe()`: Converts raster data into a pandas DataFrame, with columns for longitude, latitude, and pixel values (or band-specific values for multi-band modes).
    -   `to_geodataframe()`: Extends `to_dataframe()` by adding a `geometry` column, converting each pixel into a GeoDataFrame representing its bounding box, with the correct CRS.
-   **Reprojection (`reproject_to`)**:
    -   Allows explicit reprojection of the current raster to a new Coordinate Reference System (CRS) and/or resolution, saving the output to a specified path or a temporary file.
    -   Supports different `resampling_method` options (e.g., `Resampling.nearest`, `Resampling.bilinear`).
-   **Raster Information (`get_raster_info`)**:
    -   Provides a dictionary containing comprehensive metadata about the raster, such as band count, dimensions, CRS, bounds, transform, data types, nodata values, processing mode, and merge status.
-   **Graph Conversion (`to_graph`)**:
    -   Converts raster data into a graph (NetworkX graph or SciPy sparse matrix) based on pixel adjacency.
    -   Supports `connectivity` of 4 (von Neumann neighborhood) or 8 (Moore neighborhood).
    -   Can include geographic coordinates and pixel values as node attributes.
-   **Sampling**:
    -   `sample_by_coordinates()`: Extracts pixel values at specific geographic coordinates.
    -   `sample_by_polygons()`: Computes aggregate statistics (e.g., mean, sum, min, max, count) of pixel values within given polygon boundaries, supporting single or multiple statistics.
    -   `sample_by_polygons_batched()`: Provides a parallelized version of polygon sampling for performance-intensive tasks.

---

# Multi-raster reprojection

The differences in the reprojected metadata are expected and are a direct result of the order of operations: **reproject then merge** versus **merge then reproject**. The two processes follow different steps, leading to variations in the final raster's dimensions, bounds, and resolution.

---

### **Reproject then Merge**

When you specify `target_crs` at initialization, the code first **reprojects each individual raster** to the target CRS (`EPSG:4326`) and then **merges the reprojected outputs**.

- **Step 1: Reprojection**: Each input raster is reprojected from `ESRI:54009` to `EPSG:4326`. During this step, `rasterio`'s `calculate_default_transform` function computes a new transform and pixel dimensions (`width`, `height`) for each raster. The reprojected rasters are now in the same CRS with a consistent resolution (e.g., `0.00918...` degrees).
- **Step 2: Merging**: The reprojected rasters, which are now in the same CRS and have similar resolutions, are merged. The `rasterio.merge` function can combine these aligned rasters seamlessly. The final output's dimensions are calculated by finding the union of all reprojected rasters' bounds and applying the shared resolution, resulting in a single, larger raster.

This process ensures a uniform resolution and grid alignment across all parts of the final merged raster.

---

### **Merge then Reproject**

When `target_crs` is not specified at initialization, the code first **merges the two rasters** in their original `ESRI:54009` CRS and then **reprojects the single, merged output** to `EPSG:4326`.

- **Step 1: Merging**: The two rasters are merged in `ESRI:54009`. Since they are in the same CRS and have the same resolution (`1000.0` meters), `rasterio.merge` can simply combine them side-by-side. The original raster was `1000x1000`, so merging a second one next to it likely creates a `2000x1000` raster, as seen in the metadata. The resolution remains `1000.0` meters.
- **Step 2: Reprojection**: The single `2000x1000` raster is then reprojected to `EPSG:4326`. A new transform and pixel dimensions are calculated for this single, larger raster. Since `calculate_default_transform` is working on a different-shaped input, it will calculate a different output resolution and grid shape. The resulting resolution (`0.00973...`) and dimensions (`2076x832`) will be different because the reprojection is performed on a single, larger input rather than two smaller ones.

---

### **Why the Metadata is Different**

- **Resolution**: The `reproject-then-merge` approach maintains a consistent resolution that is calculated for a single tile and then applied to all. The `merge-then-reproject` approach calculates a single resolution for the entire, larger combined area. The process of resampling to a new grid (a core part of reprojection) is inherently sensitive to the input's size and shape.
- **Dimensions (`width`, `height`)**: The final pixel dimensions are a function of the total bounds and the final resolution. Since the resolution is different in the two methods, the width and height must also be different to cover the same geographic area.
- **Bounds**: The final bounds are nearly identical in latitude and longitude, which makes sense because both methods represent the same geographic area. Any slight differences are due to rounding and the nuances of resampling.

**Conclusion**: The differences are normal and reflect the non-commutative nature of these two geospatial operations. The **reproject then merge** approach is generally preferable as it ensures greater consistency and can be more accurate when dealing with rasters that have slightly different resolutions or alignments, as it creates a single, clean grid before combining the data.

---

[Back to Examples](../index.md)