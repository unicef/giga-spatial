# Population Accessibility Mapping

Quantifying the number of people served by a specific location is the primary metric for connectivity impact. GigaSpatial integrates WorldPop datasets directly into the POI analysis workflow to automate population catchment estimation.

## The "Why": Catchment vs. Point Data

- **WorldPop** provides high-resolution (100m) global population estimates in raster format.
- **PoiViewGenerator** allows for circular catchment analysis (buffers) around your POIs to sum up the underlying population.

This combination answers a critical Giga question: *"If we connect this school, how many people in the surrounding community might also gain access to the network?"*

## The Workflow

### 1. Initialize the WorldPop Handler

The `WorldPopHandler` handles the complex directory structure and metadata associated with WorldPop's annual releases.

```python
from gigaspatial import WorldPopHandler

# Initialize the handler for population count data
wp_handler = WorldPopHandler()
```

### 2. Map Population to POIs

We use the `PoiViewGenerator` specifically to perform zonal statistics within a defined radius of each school.

```python
from gigaspatial import PoiViewGenerator

# Create a view from your schools
view = PoiViewGenerator(gdf_schools)

# Sum the population within a 1km radius of each school
# GigaSpatial automatically selects the correct aggregation method (sum)
enriched_view = view.map_wp_pop(country="BEN", map_radius_meters=1000)
```

## Key Columns Explained

| Column | Description |
| :--- | :--- |
| `population_count_1000m` | The estimated total population living within 1000m of the POI. |

## Rationale for this Combination

Traditional GIS software requires several steps (buffer, clip, sum raster) to perform this calculation for a single site. By using the **WorldPop Handler** + **PoiView Generator**, GigaSpatial:
1. Automatically downloads the relevant country-level population raster.
2. Manages spatial projection changes (CRS) to ensure distance calculations are accurate.
3. Performs vectorized zonal statistics across all sites in parallel.

This allows you to calculate "Potential Reach" for thousands of sites across an entire country in seconds.
