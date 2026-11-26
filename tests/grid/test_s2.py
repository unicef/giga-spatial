import logging

from shapely.geometry import Point

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.grid import s2 as s2_module
from gigaspatial.grid.s2 import S2Cells


def test_from_geometry_single_point_forwards_kwargs(tmp_path):
    store = LocalDataStore(base_path=tmp_path)
    point = Point(2.3522, 48.8566)

    result = S2Cells.from_geometry(point, level=10, data_store=store)

    assert result.level == 10
    assert len(result.cells) == 1
    assert result.to_geoms()[0].contains(point)
    assert result.data_store is store


class _UnrepairablePolygon:
    is_valid = False

    def buffer(self, _distance):
        return self


def test_to_geodataframe_drops_skipped_invalid_cell_and_preserves_alignment(
    monkeypatch, caplog
):
    pts = [Point(2.3522, 48.8566), Point(10.7522, 59.9139), Point(-73.9857, 40.7484)]
    cell_ids = S2Cells.get_cells_from_points(pts, level=10)
    grid = S2Cells(level=10, cells=cell_ids)

    real_orient = s2_module.orient
    call_idx = {"n": 0}

    def _orient_with_invalid_middle(poly, sign=1.0):
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx == 1:
            return _UnrepairablePolygon()
        return real_orient(poly, sign=sign)

    monkeypatch.setattr(s2_module, "orient", _orient_with_invalid_middle)

    with caplog.at_level(logging.WARNING):
        gdf = grid.to_geodataframe()

    assert len(gdf) == 2
    assert list(gdf["cell_id"]) == [cell_ids[0], cell_ids[2]]
    assert gdf.iloc[0].geometry.contains(pts[0])
    assert gdf.iloc[1].geometry.contains(pts[2])
    assert str(gdf.crs) == "EPSG:4326"
    assert any("Skipped 1" in rec.message for rec in caplog.records)


def test_to_geodataframe_drops_failed_conversion_cell_and_preserves_alignment(
    monkeypatch, caplog
):
    pts = [Point(2.3522, 48.8566), Point(10.7522, 59.9139), Point(-73.9857, 40.7484)]
    cell_ids = S2Cells.get_cells_from_points(pts, level=10)
    grid = S2Cells(level=10, cells=cell_ids)

    real_orient = s2_module.orient
    call_idx = {"n": 0}

    def _orient_raising_on_middle(poly, sign=1.0):
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx == 1:
            raise ValueError("simulated conversion failure")
        return real_orient(poly, sign=sign)

    monkeypatch.setattr(s2_module, "orient", _orient_raising_on_middle)

    with caplog.at_level(logging.WARNING):
        gdf = grid.to_geodataframe()

    assert len(gdf) == 2
    assert list(gdf["cell_id"]) == [cell_ids[0], cell_ids[2]]
    assert gdf.iloc[0].geometry.contains(pts[0])
    assert gdf.iloc[1].geometry.contains(pts[2])
    assert any("Skipped 1" in rec.message for rec in caplog.records)


def test_to_geodataframe_empty_cells():
    grid = S2Cells(level=10, cells=[])
    gdf = grid.to_geodataframe()

    assert gdf.empty
    assert list(gdf.columns) == ["cell_id", "cell_token", "geometry"]
    assert str(gdf.crs) == "EPSG:4326"
