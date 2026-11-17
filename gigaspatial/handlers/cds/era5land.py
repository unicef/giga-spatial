"""
ERA5-Land hourly data, from the Copernicus Climate Data Store.

Dataset id: ``reanalysis-era5-land``.
Documentation: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview
Licence: CC BY 4.0. DOI: 10.24381/cds.e2161bac.

The handler name is ``cds-era5land``. The variable list, the 0.1 degree grid,
and the 1950 start are this dataset's own. The download form has no
``product_type``. Availability is read from this dataset's catalogue, not from
ERA5.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Optional

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from gigaspatial.config import config as global_config
from gigaspatial.handlers.cds.era5 import (
    CDSConfig,
    CDSDownloader,
    CDSHandler,
    CDSReader,
    _CATALOGUES,
    variable_records,
)


_CATALOGUES["reanalysis-era5-land"] = variable_records((
    ('2m_dewpoint_temperature', '2m dewpoint temperature', ('Temperature',)),
    ('2m_temperature', '2m temperature', ('Temperature',)),
    ('skin_temperature', 'Skin temperature', ('Temperature',)),
    ('soil_temperature_level_1', 'Soil temperature level 1', ('Temperature',)),
    ('soil_temperature_level_2', 'Soil temperature level 2', ('Temperature',)),
    ('soil_temperature_level_3', 'Soil temperature level 3', ('Temperature',)),
    ('soil_temperature_level_4', 'Soil temperature level 4', ('Temperature',)),
    ('lake_bottom_temperature', 'Lake bottom temperature', ('Lakes',)),
    ('lake_ice_depth', 'Lake ice depth', ('Lakes',)),
    ('lake_ice_temperature', 'Lake ice temperature', ('Lakes',)),
    ('lake_mix_layer_depth', 'Lake mix-layer depth', ('Lakes',)),
    ('lake_mix_layer_temperature', 'Lake mix-layer temperature', ('Lakes',)),
    ('lake_shape_factor', 'Lake shape factor', ('Lakes',)),
    ('lake_total_layer_temperature', 'Lake total layer temperature', ('Lakes',)),
    ('snow_albedo', 'Snow albedo', ('Snow',)),
    ('snow_cover', 'Snow cover', ('Snow',)),
    ('snow_density', 'Snow density', ('Snow',)),
    ('snow_depth', 'Snow depth', ('Snow',)),
    ('snow_depth_water_equivalent', 'Snow depth water equivalent', ('Snow',)),
    ('snowfall', 'Snowfall', ('Snow',)),
    ('snowmelt', 'Snowmelt', ('Snow',)),
    ('temperature_of_snow_layer', 'Temperature of snow layer', ('Snow',)),
    ('skin_reservoir_content', 'Skin reservoir content', ('Soil Water',)),
    ('volumetric_soil_water_layer_1', 'Volumetric soil water layer 1', ('Soil Water',)),
    ('volumetric_soil_water_layer_2', 'Volumetric soil water layer 2', ('Soil Water',)),
    ('volumetric_soil_water_layer_3', 'Volumetric soil water layer 3', ('Soil Water',)),
    ('volumetric_soil_water_layer_4', 'Volumetric soil water layer 4', ('Soil Water',)),
    ('forecast_albedo', 'Forecast albedo', ('Radiation and Heat',)),
    ('surface_latent_heat_flux', 'Surface latent heat flux', ('Radiation and Heat',)),
    ('surface_net_solar_radiation', 'Surface net solar radiation', ('Radiation and Heat',)),
    ('surface_net_thermal_radiation', 'Surface net thermal radiation', ('Radiation and Heat',)),
    ('surface_sensible_heat_flux', 'Surface sensible heat flux', ('Radiation and Heat',)),
    ('surface_solar_radiation_downwards', 'Surface solar radiation downwards', ('Radiation and Heat',)),
    ('surface_thermal_radiation_downwards', 'Surface thermal radiation downwards', ('Radiation and Heat',)),
    ('evaporation_from_bare_soil', 'Evaporation from bare soil', ('Evaporation and Runoff',)),
    ('evaporation_from_open_water_surfaces_excluding_oceans', 'Evaporation from open water surfaces excluding oceans', ('Evaporation and Runoff',)),
    ('evaporation_from_the_top_of_canopy', 'Evaporation from the top of canopy', ('Evaporation and Runoff',)),
    ('evaporation_from_vegetation_transpiration', 'Evaporation from vegetation transpiration', ('Evaporation and Runoff',)),
    ('potential_evaporation', 'Potential evaporation', ('Evaporation and Runoff',)),
    ('runoff', 'Runoff', ('Evaporation and Runoff',)),
    ('snow_evaporation', 'Snow evaporation', ('Evaporation and Runoff',)),
    ('sub_surface_runoff', 'Sub-surface runoff', ('Evaporation and Runoff',)),
    ('surface_runoff', 'Surface runoff', ('Evaporation and Runoff',)),
    ('total_evaporation', 'Total evaporation', ('Evaporation and Runoff',)),
    ('10m_u_component_of_wind', '10m u-component of wind', ('Wind, Pressure and Precipitation',)),
    ('10m_v_component_of_wind', '10m v-component of wind', ('Wind, Pressure and Precipitation',)),
    ('surface_pressure', 'Surface pressure', ('Wind, Pressure and Precipitation',)),
    ('total_precipitation', 'Total precipitation', ('Wind, Pressure and Precipitation',)),
    ('leaf_area_index_high_vegetation', 'Leaf area index, high vegetation', ('Vegetation',)),
    ('leaf_area_index_low_vegetation', 'Leaf area index, low vegetation', ('Vegetation',)),
    ('high_vegetation_cover', 'High vegetation cover', ('Invariant',)),
    ('glacier_mask', 'Glacier mask', ('Invariant',)),
    ('lake_cover', 'Lake cover', ('Invariant',)),
    ('low_vegetation_cover', 'Low vegetation cover', ('Invariant',)),
    ('lake_total_depth', 'Lake total depth', ('Invariant',)),
    ('geopotential', 'Geopotential', ('Invariant',)),
    ('land_sea_mask', 'Land-sea mask', ('Invariant',)),
    ('soil_type', 'Soil type', ('Invariant',)),
    ('type_of_high_vegetation', 'Type of high vegetation', ('Invariant',)),
    ('type_of_low_vegetation', 'Type of low vegetation', ('Invariant',)),
))


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class ERA5LandConfig(CDSConfig):
    """ERA5-Land configuration. Files live under ``bronze/cds/era5-land``."""

    HANDLER_NAME: ClassVar[str] = "cds-era5land"
    DATASET_ID: ClassVar[str] = "reanalysis-era5-land"
    DOCUMENTATION_URL: ClassVar[str] = (
        "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview"
    )
    DOI: ClassVar[str] = "10.24381/cds.e2161bac"
    START_YEAR: ClassVar[int] = 1950
    RESOLUTION: ClassVar[float] = 0.1
    HAS_PRODUCT_TYPE: ClassVar[bool] = False
    PRODUCT_TYPES: ClassVar[tuple] = ()
    DEFAULT_PRODUCT_TYPE: ClassVar[Optional[str]] = None

    base_path: Path = Field(
        default_factory=lambda: global_config.get_path("cds", "bronze") / "era5-land"
    )


class ERA5LandDownloader(CDSDownloader):
    """Downloads ERA5-Land hourly fields."""

    config_type: ClassVar[type] = ERA5LandConfig


class ERA5LandReader(CDSReader):
    """Reads stored ERA5-Land country files."""

    config_type: ClassVar[type] = ERA5LandConfig


class ERA5LandHandler(CDSHandler):
    """
    ERA5-Land hourly fields for a country.

    The handler name is ``cds-era5land``.

        ERA5LandHandler().load_data(
            "ITA",
            variables="2m_temperature",
            start="2024-11-01",
            end="2024-11-02",
        )
        ERA5LandHandler().update()
    """

    name = "cds-era5land"
    config_class = ERA5LandConfig
    downloader_class = ERA5LandDownloader
    reader_class = ERA5LandReader


if __name__ == "__main__":
    from gigaspatial.handlers.cds.era5 import main

    raise SystemExit(main())
