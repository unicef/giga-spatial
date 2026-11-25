import geopandas as gpd
from shapely.geometry import Point, Polygon

from gigaspatial.grid.mercator_tiles import MercatorTiles


def test_from_spatial_with_lon_lat_tuples_matches_from_points():
    points = [(2.3522, 48.8566), (-0.1276, 51.5072)]
    zoom_level = 10

    expected = MercatorTiles.from_points(points=points, zoom_level=zoom_level)
    result = MercatorTiles.from_spatial(points, zoom_level=zoom_level)

    assert result.zoom_level == expected.zoom_level
    assert set(result.quadkeys) == set(expected.quadkeys)


def test_from_spatial_with_shapely_points_matches_from_points():
    points = [Point(2.3522, 48.8566), Point(-0.1276, 51.5072)]
    zoom_level = 10

    expected = MercatorTiles.from_points(points=points, zoom_level=zoom_level)
    result = MercatorTiles.from_spatial(points, zoom_level=zoom_level)

    assert result.zoom_level == expected.zoom_level
    assert set(result.quadkeys) == set(expected.quadkeys)


def test_from_spatial_with_point_geodataframe_matches_from_points():
    points = [Point(2.3522, 48.8566), Point(-0.1276, 51.5072)]
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    zoom_level = 10

    expected = MercatorTiles.from_points(points=points, zoom_level=zoom_level)
    result = MercatorTiles.from_spatial(gdf, zoom_level=zoom_level)

    assert result.zoom_level == expected.zoom_level
    assert set(result.quadkeys) == set(expected.quadkeys)


def test_from_spatial_with_polygon_and_polygon_geodataframe_matches_from_geometry():
    polygon = Polygon([(2.30, 48.82), (2.40, 48.82), (2.40, 48.88), (2.30, 48.88)])
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    zoom_level = 10

    expected = MercatorTiles.from_geometry(geometry=polygon, zoom_level=zoom_level)
    result_poly = MercatorTiles.from_spatial(polygon, zoom_level=zoom_level)
    result_gdf = MercatorTiles.from_spatial(gdf, zoom_level=zoom_level)

    assert result_poly.zoom_level == expected.zoom_level
    assert set(result_poly.quadkeys) == set(expected.quadkeys)
    assert result_gdf.zoom_level == expected.zoom_level
    assert set(result_gdf.quadkeys) == set(expected.quadkeys)
