from shapely.geometry import Point

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.grid.s2 import S2Cells


def test_from_geometry_single_point_forwards_kwargs(tmp_path):
    store = LocalDataStore(base_path=tmp_path)
    point = Point(2.3522, 48.8566)

    result = S2Cells.from_geometry(point, level=10, data_store=store)

    assert result.level == 10
    assert len(result.cells) == 1
    assert result.to_geoms()[0].contains(point)
    assert result.data_store is store
