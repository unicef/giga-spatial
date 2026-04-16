"""
Utilities for satellite imagery processing.
Currently focuses on coordinate-to-pixel resolution calculations for
Mercator projections.
"""
import numpy as np
import pyproj

def calculate_pixels_at_location(gdf, resolution, bbox_size=300, crs="EPSG:3857"):
    """
    Calculate the number of pixels required to cover a bounding box.

    Calculates the dimensions in pixels for a given physical size (meters)
    around a coordinate, accounting for Mercator scale distortion.

    Args:
        gdf: GeoDataFrame with Point geometries (WGS84).
        resolution: Target resolution in meters per pixel.
        bbox_size: Bounding box side length in meters.
        crs: Target projection CRS.

    Returns:
        Number of pixels per side (width and height).
    """

    # Calculate avg lat and lon
    lon = gdf.geometry.x.mean()
    lat = gdf.geometry.y.mean()

    # Define projections
    wgs84 = pyproj.CRS("EPSG:4326")  # Geographic coordinate system
    mercator = pyproj.CRS(crs)  # Target CRS (EPSG:3857)

    # Transform the center coordinate to EPSG:3857
    transformer = pyproj.Transformer.from_crs(wgs84, mercator, always_xy=True)
    x, y = transformer.transform(lon, lat)

    # Calculate scale factor (distortion) at given latitude
    scale_factor = np.cos(np.radians(lat))  # Mercator scale correction

    # Adjust the effective resolution
    effective_resolution = resolution * scale_factor

    # Compute number of pixels per side
    pixels = bbox_size / effective_resolution
    return int(round(pixels))