from shapely.geometry import MultiPolygon, box

from gigaspatial.core.schemas.admin_boundary import AdminBoundary, AdminLevel
from gigaspatial.grid.h3 import CountryH3Hexagons
from gigaspatial.grid.mercator_tiles import CountryMercatorTiles
from gigaspatial.grid.s2 import CountryS2Cells
from gigaspatial.handlers.boundaries import AdminBoundaries


ISLAND_A = box(0.0, 0.0, 0.2, 0.2)
ISLAND_B = box(1.0, 1.0, 1.2, 1.2)
TWO_ISLANDS = MultiPolygon([ISLAND_A, ISLAND_B])


def _multi_part_boundaries() -> AdminBoundaries:
    """AdminBoundaries with one polygon entity per island from TWO_ISLANDS."""
    entities = [
        AdminBoundary(
            boundary_id=f"ISL_{idx}",
            admin_level=AdminLevel.COUNTRY,
            name=f"Island {idx}",
            country_iso="RWA",
            geometry=part,
        )
        for idx, part in enumerate(TWO_ISLANDS.geoms)
    ]
    return AdminBoundaries(entities=entities)


def _single_multipolygon_boundaries() -> AdminBoundaries:
    """AdminBoundaries with a single MultiPolygon entity covering TWO_ISLANDS."""
    return AdminBoundaries(
        entities=[
            AdminBoundary(
                boundary_id="ISL_ALL",
                admin_level=AdminLevel.COUNTRY,
                name="All Islands",
                country_iso="RWA",
                geometry=TWO_ISLANDS,
            )
        ]
    )


def test_country_h3_create_covers_all_parts(monkeypatch):
    monkeypatch.setattr(
        AdminBoundaries, "create", classmethod(lambda cls, **kwargs: _multi_part_boundaries())
    )

    result = CountryH3Hexagons.create("RWA", resolution=6)
    cell_geoms = result.to_geoms()

    assert any(g.intersects(ISLAND_A) for g in cell_geoms)
    assert any(g.intersects(ISLAND_B) for g in cell_geoms)


def test_country_s2_create_covers_all_parts(monkeypatch):
    monkeypatch.setattr(
        AdminBoundaries, "create", classmethod(lambda cls, **kwargs: _multi_part_boundaries())
    )

    result = CountryS2Cells.create("RWA", level=8, max_cells=1000)
    cell_geoms = result.to_geoms()

    assert any(g.intersects(ISLAND_A) for g in cell_geoms)
    assert any(g.intersects(ISLAND_B) for g in cell_geoms)


def test_country_mercator_create_covers_all_parts(monkeypatch):
    monkeypatch.setattr(
        AdminBoundaries, "create", classmethod(lambda cls, **kwargs: _multi_part_boundaries())
    )

    result = CountryMercatorTiles.create("RWA", zoom_level=10)
    cell_geoms = result.to_geoms()

    assert any(g.intersects(ISLAND_A) for g in cell_geoms)
    assert any(g.intersects(ISLAND_B) for g in cell_geoms)


def test_country_grids_cover_single_multipolygon_boundary(monkeypatch):
    monkeypatch.setattr(
        AdminBoundaries,
        "create",
        classmethod(lambda cls, **kwargs: _single_multipolygon_boundaries()),
    )

    for grid in (
        CountryH3Hexagons.create("RWA", resolution=6),
        CountryS2Cells.create("RWA", level=8, max_cells=1000),
        CountryMercatorTiles.create("RWA", zoom_level=10),
    ):
        cell_geoms = grid.to_geoms()
        assert any(g.intersects(ISLAND_A) for g in cell_geoms)
        assert any(g.intersects(ISLAND_B) for g in cell_geoms)
