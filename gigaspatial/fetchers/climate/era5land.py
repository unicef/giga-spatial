"""ERA5-Land hourly data.

Dataset id: ``reanalysis-era5-land``.
Documentation: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview
Licence: CC BY 4.0. DOI: 10.24381/cds.e2161bac.

``fetch_era5land`` returns an in-memory ``xarray.Dataset``. The variable list,
0.1 degree grid, and 1950 start belong to this dataset. Repeated calls ask CDS
again; a caller that wants incremental updates compares ``latest_available``
with data it already holds and requests only the missing hours.
"""

from gigaspatial.fetchers.climate.shared import DatasetSpec, fetch_dataset, variables_from_rows

ERA5_LAND_ROWS = (
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
)

ERA5_LAND = DatasetSpec(
    name="ERA5-Land",
    dataset_id="reanalysis-era5-land",
    documentation_url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview",
    doi="10.24381/cds.e2161bac",
    start_year=1950,
    resolution=0.1,
    variables=variables_from_rows(ERA5_LAND_ROWS),
    product_types=(),
    default_product=None,
)


def fetch_era5land(area, variables, time=None, *, start=None, end=None, year=None, month=None, day=None, hour=None, full_history=False, data_format="netcdf", retriever=None, max_bytes=None, api_url=None, api_key=None, check_availability=True, max_retries=3):
    """Fetch ERA5-Land for a country or any geometry.

    Arguments match ``fetch_era5``. This dataset has no ``product_type``.
    """

    from gigaspatial.fetchers.climate.shared import DEFAULT_MAX_BYTES

    return fetch_dataset(
        ERA5_LAND,
        area,
        variables,
        time,
        start=start,
        end=end,
        year=year,
        month=month,
        day=day,
        hour=hour,
        full_history=full_history,
        data_format=data_format,
        retriever=retriever,
        max_bytes=DEFAULT_MAX_BYTES if max_bytes is None else max_bytes,
        api_url=api_url,
        api_key=api_key,
        check_availability=check_availability,
        max_retries=max_retries,
    )
