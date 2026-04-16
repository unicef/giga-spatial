# Zonal Statistics on Grids

While POI-based analysis is useful for specific sites, large-scale planning often requires analyzing entire regions. GigaSpatial provides tools to aggregate high-resolution raster data (like elevation or nightlights) into standardized grid systems for comparative analysis.

## The "Why": Grids vs. Administrative Boundaries

- **Administrative Boundaries** vary wildly in size and shape, making statistical comparisons (like population density) difficult.
- **Grids** (H3 or S2) provide equal-area cells with consistent topology. 

Aggregating rasters into grids allows you to identify "hotspots" of demand or infrastructure gaps across an entire country with mathematical consistency.

## The Workflow

### 1. Initialize the Raster Processor

The `TifProcessor` is an optimized wrapper for raster operations, allowing for fast clipping and windowed reading.

```python
from gigaspatial import TifProcessor

# Load a raster (e.g., NASA SRTM Elevation data)
tif = TifProcessor("/path/to/elevation_data.tif")
```

### 2. Use a specialized View Generator

Instead of managing raw grids, GigaSpatial uses **View Generators** to handle the mapping between grids and data sources. The `H3ViewGenerator` automatically handles the hexagonal tessellation of your study area.

```python
from gigaspatial.generators.zonal.h3 import H3ViewGenerator

# Initialize an H3 generator for a specific country at resolution 9
# (~0.1 square kilometers per cell)
generator = H3ViewGenerator(source="BEN", resolution=9)
```

### 3. Aggregate Raster Data to the Grid

We can now "summarize" the raster values within each hexagonal cell using the generator's mapping methods.

```python
# Aggregate elevation data to the grid
# This computes the MEAN elevation for every hexagon in the country
result_view = generator.map_rasters(tif, stat="mean", output_column="elevation")
```

## Rationale for this Combination

Combining the **TifProcessor** with **Grid Modules** (H3/S2) enables "Spatial Benchmarking." 

Because the hexagons are standardized:
1. You can compare the average elevation (terrain difficulty) between different provinces directly.
2. You can overlay multiple layers (e.g., population + nightlights + elevation) onto the **same hex grid** to create composite indices.
3. The resulting dataset is optimized for modern web-mapping tools (like Mapbox or Kepler.gl).

## When to use H3 vs S2?

- **H3**: Best for spatial statistics, neighborhood analysis, and heatmaps due to its hexagonal shape and uniform adjacency.
- **S2**: Best for large-scale data indexing and tiling, especially when integrating with Google Earth Engine or specialized buildings datasets.
