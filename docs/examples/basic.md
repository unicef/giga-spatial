# Data Handler Examples

This guide provides examples of how to use various data handlers in GigaSpatial to access and process different types of spatial data.

## Population Data (WorldPop)

```python
from gigaspatial.handlers.worldpop import WorldPopHandler

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
from gigaspatial.handlers.google_open_buildings import GoogleOpenBuildingsDownloader, GoogleOpenBuildingsMapper
from gigaspatial.handlers.boundaries import AdminBoundaries
from gigaspatial.core.io.local_data_store import LocalDataStore
import geopandas as gpd

# Initialize downloader
data_store = LocalDataStore()
downloader = GoogleOpenBuildingsDownloader(data_store=data_store)

# Example 1: Estimate download size for a given country
country_code = "KEN"  # Kenya
gdf_admin0 = AdminBoundaries.create(country_code=country_code, admin_level=0).to_geodataframe()
estimated_size = downloader.get_download_size_estimate(gdf_admin0)
print(f"Estimated download size for {country_code}: {estimated_size:.2f} MB")

# Example 2: Download buildings data for a country
file_paths = downloader.download_by_country(country_code, data_type="polygons")
print(f"Downloaded files: {file_paths}")

# Example 3: Download buildings data for specific points
points_gdf = gpd.GeoDataFrame(
    {"geometry": [gpd.points_from_xy([36.8219], [-1.2921])]}, crs="EPSG:4326"
)  # Nairobi, Kenya
file_paths = downloader.download_by_points(points_gdf, data_type="points")
print(f"Downloaded files: {file_paths}")

# Example 4: Load downloaded data and map nearest buildings
mapper = GoogleOpenBuildingsMapper()

tiles_gdf = downloader._get_intersecting_tiles(gdf_admin0)
polygon_data = mapper.load_data(tiles_gdf, data_type="polygons")
print(f"Loaded {len(polygon_data)} building polygons")

# Example 5: Find the nearest building for a given point
nearest_buildings = mapper.map_nearest_building(points_gdf)
print(nearest_buildings)
```

### Microsoft Global Buildings

```python
from gigaspatial.handlers.microsoft_global_buildings import MSBuildingsDownloader

# Initialize the handler
mgb = MSBuildingsDownloader()

points = [(1.25214, 5.5124), (3.45234, 12.51232)]

# Get building footprints
list_of_paths = mgb.download_by_points(
    points=points
)
```

## Satellite Imagery

### Maxar Imagery

```python
from gigaspatial.handlers.maxar_image import MaxarImageHandler

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
from gigaspatial.handlers.mapbox_image import MapboxImageDownloader

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
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.ookla_speedtest import (
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
from gigaspatial.handlers.boundaries import AdminBoundaries

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

## Overture Places

```python
from gigaspatial.handlers.overture import OvertureAmenityFetcher
from shapely.geometry import Polygon
import geopandas as gpd

# Initialize the fetcher with country ISO3 code and amenity types
fetcher = OvertureAmenityFetcher(country_iso3='KEN', amenity_types=['school', 'hospital'])

# Example 1: Fetching all amenities for a given release
gdf_all = fetcher.get_locations(release='2024-01-01')
print(gdf_all.head())

# Example 2: Fetching amenities within a specific bounding polygon
polygon = Polygon([(36.8, -1.3), (36.9, -1.3), (36.9, -1.2), (36.8, -1.2), (36.8, -1.3)])
gdf_filtered = fetcher.get_locations(release='2024-01-01', geometry=polygon)
print(gdf_filtered.head())
```

## Combining Multiple Data Sources

Here's an example of how to combine multiple data sources for analysis:

```python
# Initialize handlers
buildings = GoogleOpenBuildingsHandler()
population = WorldPopHandler()
boundaries = BoundariesHandler()

# Get administrative boundaries
admin_areas = boundaries.get_admin_boundaries(
    country_code="KEN",
    admin_level=2
)

# Get building footprints and population data
for area in admin_areas.itertuples():
    # Get buildings
    area_buildings = buildings.get_buildings(
        geometry=area.geometry
    )
    
    # Get population
    area_population = population.get_population_stats(
        boundaries=area.geometry,
        year=2020
    )
    
    # Calculate metrics
    building_density = len(area_buildings) / area.geometry.area
    population_density = area_population['total_population'] / area.geometry.area
    
    # Your analysis here...
```

## Best Practices

1. Cache downloaded data when possible
2. Use appropriate spatial indices for large datasets
3. Handle errors and edge cases gracefully
4. Consider memory usage when working with large areas

## Additional Resources

- Check the [API Reference](../api/core.md) for detailed method documentation
- See [Advanced Features](../user-guide/advanced-features.md) for more complex usage
- Review the [Contributing Guide](../contributing.md) if you want to add new handlers