# School Proximity Analysis

Mapping the physical infrastructure around schools is critical for estimating connectivity costs and assessing local demand. This guide demonstrates how to use specialized handlers and generators to perform this analysis.

## The "Why": Handlers vs. Generators

- **Handlers** (e.g., `GigaSchoolLocationFetcher`, `GoogleMSBuildingsHandler`) are responsible for the "What": fetching data from APIs or storage and standardizing it into GeoDataFrames.
- **Generators** (e.g., `PoiViewGenerator`) provide the "So What": they integrate disparate datasets to create a composite "view" of specific locations (POIs).

Combining them allows you to see schools not just as coordinates, but as physical structures within a built environment.

## The Workflow

### 1. Fetching School Locations

We use the specialized fetching handler for Giga's school database.

```python
from gigaspatial import GigaSchoolLocationFetcher

# Initialize the fetcher for a specific country (e.g., Aruba)
fetcher = GigaSchoolLocationFetcher(country_code="ABW")

# Fetch locations as a GeoDataFrame
gdf_schools = fetcher.fetch_locations(process_geospatial=True)
```

### 2. Initializing the Analytic View

The `PoiViewGenerator` takes your base POIs (schools) and prepares them for enrichment.

```python
from gigaspatial import PoiViewGenerator

# Create a view from our school locations
view = PoiViewGenerator(gdf_schools)
```

### 3. Mapping Building Footprints

To understand the school's physical context, we want to find the nearest building footprints. The `PoiViewGenerator` manages the complex spatial join and distance calculations automatically using the building handler.

```python
# Enrich the view with nearest building data
# This automatically triggers the GoogleMSBuildingsHandler
enriched_view = view.find_nearest_buildings(country="ABW", search_radius=1000)
```

## Key Columns Explained

After enrichment, your GeoDataFrame will contain:

| Column | Description |
| :--- | :--- |
| `nearest_building_distance_m` | Distance to the closest building footprint in meters. |
| `building_within_1000m` | Boolean flag indicating if any buildings were found in the buffer. |

## Rationale for this Combination

Using `GoogleMSBuildingsHandler` specifically provides high-resolution, global coverage. By wrapping it in `PoiViewGenerator`, the library handles the **tiling and spatial indexing** (using S2 or H3 cells) behind the scenes, so you can execute proximity analysis at a national scale without manual geoprocessing.
