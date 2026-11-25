"""In-memory ERA5 and ERA5-Land retrieval."""

from gigaspatial.fetchers.climate.era5 import fetch_era5
from gigaspatial.fetchers.climate.era5land import fetch_era5land
from gigaspatial.fetchers.climate.shared import (
    clip,
    countries_intersecting,
    latest_available,
    merge_climate,
    open_climate,
    period_available,
    standardize,
)

__all__ = [
    "fetch_era5",
    "fetch_era5land",
    "countries_intersecting",
    "open_climate",
    "standardize",
    "clip",
    "latest_available",
    "period_available",
    "merge_climate",
]
