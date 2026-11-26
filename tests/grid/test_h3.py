from shapely.geometry import Point

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.grid.h3 import H3Hexagons


def test_from_geometry_single_point_passes_resolution_and_kwargs(tmp_path):
    store = LocalDataStore(base_path=tmp_path)
    point = Point(2.3522, 48.8566)

    result = H3Hexagons.from_geometry(point, resolution=7, data_store=store)

    assert result.resolution == 7
    assert len(result.hexagons) == 1
    assert result.to_geoms()[0].contains(point)
    assert result.data_store is store
