# Data Handler Examples

This guide provides examples of how to use various data handlers in GigaSpatial to access and process different types of spatial data.

## Population Data (WorldPop)

```python
from gigaspatial.handlers import WorldPopHandler

# Get population data for a specific country and year
config = {
    "country_code": "KEN",
    "year": 2020,
}

# Initialize the WorldPop handler
worldpop = WorldPopDownloader(config = config)
path_to_data = worldpop.download_dataset()
```

## Building Footprints

### Google Open Buildings

```python
from gigaspatial.handlers import GoogleOpenBuildingsHandler
from gigaspatial.generators import PoiViewGenerator
from gigaspatial.core.io import LocalDataStore
import geopandas as gpd

# Initialize data store and handlers
data_store = LocalDataStore()
gob_handler = GoogleOpenBuildingsHandler(data_store=data_store)

# Example 1: Load building data for a country
country_code = "KEN"  # Kenya
polygons_gdf = gob_handler.load_polygons(country_code, ensure_available=True)
print(f"Loaded {len(polygons_gdf)} building polygons")

# Example 2: Load building data for specific points
points = [(36.8219, -1.2921), (36.8172, -1.2867)]  # Nairobi coordinates
points_gdf = gob_handler.load_polygons(points, ensure_available=True)
print(f"Loaded {len(points_gdf)} building polygons for points")

# Example 3: Map nearest buildings to points using PoiViewGenerator
# Initialize POI view generator with points
poi_generator = PoiViewGenerator(points=points)

# Map nearest Google buildings to POIs
result_gdf = poi_generator.map_google_buildings(gob_handler)
print("\nPOIs with nearest building information:")
print(result_gdf[["poi_id", "nearest_google_building_id", "nearest_google_building_distance"]].head())

# Example 4: Save the enriched POI view
output_path = poi_generator.save_view("nairobi_buildings", output_format="geojson")
print(f"\nSaved enriched POI view to: {output_path}")
```

This example demonstrates:
1. Loading building data for a country or specific points
2. Using PoiViewGenerator to map nearest buildings to points of interest
3. Saving the enriched POI view with building information

The resulting GeoDataFrame includes:
- Original POI information
- Nearest building ID
- Distance to the nearest building

### Microsoft Global Buildings

```python
from gigaspatial.handlers import MSBuildingsDownloader

# Initialize the handler
mgb = MSBuildingsDownloader()

points = [(1.25214, 5.5124), (3.45234, 12.51232)]

# Get building footprints
list_of_paths = mgb.download(
    points=points
)
```

## Satellite Imagery

### Maxar Imagery

```python
from gigaspatial.handlers import MaxarImageHandler

# Initialize woith default config which reads credentials config from your environment
maxar = MaxarImageDownloader()

# Download imagery
maxar.download_images_by_coordinates(
    data=coordinates,
    res_meters_pixel=0.6,
    output_dir="bronze/maxar",
    bbox_size = 300.0,
    image_prefix = "maxar_"
)

```

### Mapbox Imagery

```python
from gigaspatial.handlers import MapboxImageDownloader

# Initialize with your access token or config will be read from your environment
mapbox = MapboxImageDownloader(access_token="your_access_token", style_id="mapbox/satellite-v9")

# Get satellite imagery
mapbox.download_images_by_coordinates(
    data=coordinates,
    res_meters_pixel=300.0,
    output_dir="bronze/mapbox",
    image_size=(256,256),
    image_prefix="mapbox_"
)
```

## Internet Speed Data (Ookla)

```python
from gigaspatial.core.io import LocalDataStore
from gigaspatial.handlers import (
    OoklaSpeedtestTileConfig, CountryOoklaTiles
)

# Initialize OoklaSpeedtestTileConfig for a specific quarter and year
ookla_config = OoklaSpeedtestTileConfig(
    service_type="fixed", year=2023, quarter=3, data_store=LocalDataStore())

# Download and read the Ookla tile data
df = ookla_config.read_tile()
print(df.head())  # Display the first few rows of the dataset

# Generate country-specific Ookla tiles
country_ookla_tiles = CountryOoklaTiles.from_country("KEN", ookla_config)

# Convert to DataFrame and display
country_df = country_ookla_tiles.to_dataframe()
print(country_df.head())

# Convert to GeoDataFrame and display
country_gdf = country_ookla_tiles.to_geodataframe()
print(country_gdf.head())

```

## Administrative Boundaries

```python
from gigaspatial.handlers import AdminBoundaries

# Load level-1 administrative boundaries for Kenya
admin_boundaries = AdminBoundaries.create(country_code="KEN", admin_level=1)

# Convert to a GeoDataFrame
gdf = admin_boundaries.to_geodataframe()
```


## OpenStreetMap Data

```python
from gigaspatial.handlers.osm import OSMAmenityFetcher

# Example 1: Fetching school amenities in Kenya
fetcher = OSMAmenityFetcher(country_iso2="KE", amenity_types=["school"])
schools_df = fetcher.get_locations()
print(schools_df.head())

# Example 2: Fetching hospital and clinic amenities in Tanzania
fetcher = OSMAmenityFetcher(country_iso2="TZ", amenity_types=["hospital", "clinic"])
healthcare_df = fetcher.get_locations()
print(healthcare_df.head())

# Example 3: Fetching restaurant amenities in Ghana since 2020
fetcher = OSMAmenityFetcher(country_iso2="GH", amenity_types=["restaurant"])
restaurants_df = fetcher.get_locations(since_year=2020)
print(restaurants_df.head())
```