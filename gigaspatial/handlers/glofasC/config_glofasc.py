"""GloFAS config on ``BaseHandlerConfig``.

The base resolves a source in two steps. ``get_search_key`` turns a country,
or a level 1-3 unit, into the search geometry. ``get_relevant_data_units_by_key``
turns that geometry into data units. ``get_data_unit_path`` is the stored file.
``get_relevant_data_units`` and the unit cache stay on the base.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import requests
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from shapely.geometry.base import BaseGeometry

from gigaspatial.handlers.base.config import BaseHandlerConfig
from gigaspatial.handlers.base.fetch import tiles_intersecting
from gigaspatial.handlers.base.source import SourceKind, classify_source
from gigaspatial.handlers.glofasC.options_glofasc import (
    TILES_URL,
    GlofasCResolveOptions,
    depth_filename,
    depth_url,
)


def default_base_path() -> Path:
    """Store downloaded tiles in this ``glofasC`` folder, next to these modules."""
    return Path(__file__).resolve().parent


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class GlofasCConfig(BaseHandlerConfig):
    """
    Configuration for GloFAS flood hazard depth tiles.

    Attributes:
        base_path: Directory of the stored files. Defaults to this ``glofasC`` folder.
        n_workers: Parallel downloads. Kept small for the JRC file server.
    """

    options_type = GlofasCResolveOptions
    units_match_country = False

    base_path: Path = Field(default_factory=default_base_path)
    n_workers: int = Field(default=4, ge=1)
    tiles_gdf: Optional[gpd.GeoDataFrame] = Field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()
        Path(self.base_path).mkdir(parents=True, exist_ok=True)

    def get_search_key(self, source, options: GlofasCResolveOptions):
        """
        Return the search geometry the tile index understands.

        Level 0 uses the base country boundary. Levels 1-3 use the named unit
        inside that country. The returned geometry is the search geometry, so
        the base crops with it.
        """
        if options.admin_level == 0:
            return super().get_search_key(source, options)
        if classify_source(source) is not SourceKind.COUNTRY:
            raise ValueError(
                "admin_level 1-3 selects a unit inside a country. "
                "Pass a country code or name as the source."
            )
        return self._admin_unit_geometry(source, options)

    def _admin_unit_geometry(self, country: str, options: GlofasCResolveOptions) -> BaseGeometry:
        from gigaspatial.handlers.boundaries import AdminBoundaries

        boundary_kwargs = {
            key: value
            for key, value in (
                ("path", options.country_geom_path),
                ("data_store", options.country_data_store),
            )
            if value is not None
        }
        table = AdminBoundaries.create(
            country_code=country,
            admin_level=options.admin_level,
            **boundary_kwargs,
        )
        wanted = options.admin_name.strip().casefold()
        matches = [
            entity
            for entity in table.entities
            if any(_label_matches(label, wanted) for label in _entity_labels(entity))
        ]
        if len(matches) != 1:
            found = ", ".join(sorted({entity.name for entity in matches})) or "none"
            raise ValueError(
                f"Expected one admin level {options.admin_level} unit named "
                f"{options.admin_name!r} in {country}, found: {found}"
            )
        geometry = matches[0].geometry
        if geometry is None or geometry.is_empty:
            raise ValueError(
                f"Admin unit {options.admin_name!r} in {country} has no geometry"
            )
        return geometry

    def get_relevant_data_units_by_key(
        self, search_key: BaseGeometry, options: GlofasCResolveOptions
    ) -> List[dict]:
        """
        Return one unit per tile that intersects the search geometry, for each
        requested return period.

        Each unit carries the tile id, tile name, return period, and URL. Those
        are everything that picks the file.
        """
        hits = tiles_intersecting(self._tiles(), search_key, self.crs)
        units = []
        for record in hits[["id", "name"]].to_dict("records"):
            tile_id = int(record["id"])
            tile_name = str(record["name"])
            for period in options.return_periods:
                units.append(
                    {
                        "tile_id": tile_id,
                        "tile_name": tile_name,
                        "return_period": period,
                        "url": depth_url(tile_id, tile_name, period),
                    }
                )
        return units

    def get_data_unit_path(self, unit: dict, options=None) -> Path:
        """
        Return the stored depth GeoTIFF of one unit.

        The return period comes from the unit, not from a separate option.
        """
        filename = depth_filename(
            unit["tile_id"], unit["tile_name"], unit["return_period"]
        )
        return self.base_path / f"RP{unit['return_period']}" / filename

    def _tiles(self) -> gpd.GeoDataFrame:
        """Load the published tile grid, from the data store when it is cached."""
        if self.tiles_gdf is not None:
            return self.tiles_gdf

        from gigaspatial.core.io.readers import read_dataset
        from gigaspatial.core.io.writers import write_dataset

        cache = self.base_path / "cache" / "tile_extents.geojson"
        if self.data_store.file_exists(str(cache)):
            try:
                grid = read_dataset(cache, data_store=self.data_store)
                self.tiles_gdf = _as_wgs84(grid)
                self.logger.info("Loaded GloFAS tile grid from cache")
                return self.tiles_gdf
            except Exception:
                self.logger.warning(
                    "Cached GloFAS tile grid is unreadable, downloading it again"
                )

        response = requests.get(TILES_URL, timeout=60)
        response.raise_for_status()
        grid = _as_wgs84(gpd.read_file(io.BytesIO(response.content)))
        missing = {"id", "name"} - set(grid.columns)
        if missing:
            raise ValueError(f"GloFAS tile grid is missing columns: {sorted(missing)}")
        write_dataset(grid, self.data_store, str(cache))
        self.tiles_gdf = grid
        self.logger.info("GloFAS tile grid cached")
        return self.tiles_gdf


def _entity_labels(entity) -> Tuple[Optional[str], ...]:
    return (
        getattr(entity, "boundary_id", None),
        getattr(entity, "iso_code", None),
        getattr(entity, "name", None),
        getattr(entity, "name_en", None),
        getattr(entity, "name_int", None),
    )


def _label_matches(label, wanted: str) -> bool:
    return isinstance(label, str) and label.strip().casefold() == wanted


def _as_wgs84(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if grid.crs is None:
        return grid.set_crs("EPSG:4326")
    return grid
