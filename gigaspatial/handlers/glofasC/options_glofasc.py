"""Resolve options for the GloFAS flood-hazard handler.

Mirrors ``gigaspatial.handlers.base.options``: choices that change which file
is fetched live here, and the frozen dataclass is the unit-cache key.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

from gigaspatial.handlers.base.options import ResolveOptions


BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"
TILES_URL = f"{BASE_URL}/tile_extents.geojson"
RETURN_PERIODS: Tuple[int, ...] = (10, 20, 50, 75, 100, 200, 500)


def depth_filename(tile_id: int, tile_name: str, return_period: int) -> str:
    """Return the published file name of one raw depth map."""
    return f"ID{tile_id}_{tile_name}_RP{return_period}_depth.tif"


def depth_url(tile_id: int, tile_name: str, return_period: int) -> str:
    """Return the HTTPS URL of one raw depth map."""
    return f"{BASE_URL}/RP{return_period}/{depth_filename(tile_id, tile_name, return_period)}"


@dataclasses.dataclass(frozen=True)
class GlofasCResolveOptions(ResolveOptions):
    """
    Resolve options for GloFAS flood hazard tiles.

    Attributes:
        admin_level: 0 for the whole country, or 1, 2, or 3 for one unit
            inside it.
        admin_name: Identifier of that unit: its boundary id, code, or exact
            name. Required when ``admin_level`` is 1, 2, or 3.
        return_periods: Return periods to fetch, in years. Defaults to every
            published period.
    """

    admin_level: int = 0
    admin_name: Optional[str] = None
    return_periods: Tuple[int, ...] = RETURN_PERIODS

    def __post_init__(self):
        if isinstance(self.return_periods, int):
            periods = (self.return_periods,)
        else:
            periods = tuple(self.return_periods)
        if not periods or any(period not in RETURN_PERIODS for period in periods):
            raise ValueError(
                f"return_periods must be chosen from {RETURN_PERIODS}, "
                f"got {self.return_periods!r}"
            )
        if len(set(periods)) != len(periods):
            raise ValueError(f"return_periods contains duplicates: {periods}")
        object.__setattr__(self, "return_periods", periods)

        if self.admin_level not in (0, 1, 2, 3):
            raise ValueError("admin_level must be 0 (country) or 1, 2, or 3")
        if self.admin_level == 0 and self.admin_name is not None:
            raise ValueError(
                "admin_name selects a level 1-3 unit. Leave it empty for the whole country."
            )
        if self.admin_level > 0 and not self.admin_name:
            raise ValueError("admin_name is required when admin_level is 1, 2, or 3")
