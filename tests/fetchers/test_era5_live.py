"""Live CDS checks. Skipped unless pytest is run with ``-m live``."""

from datetime import timezone

import pytest

from gigaspatial.fetchers.climate.shared import latest_available, resolve_credentials


@pytest.mark.live
def test_catalogue_publishes_an_end_for_each_dataset():
    era5 = latest_available("reanalysis-era5-single-levels")
    land = latest_available("reanalysis-era5-land")

    assert era5.tzinfo is not None
    assert land.tzinfo is not None
    assert era5.year >= 1940
    assert land.year >= 1950
    assert era5.astimezone(timezone.utc)
    assert land.astimezone(timezone.utc)


@pytest.mark.live
def test_one_hour_retrieve_when_credentials_are_configured():
    url, key = resolve_credentials()
    if not key:
        pytest.skip("CDS_API_KEY is not configured")

    from shapely.geometry import box

    from gigaspatial.fetchers.climate.era5 import fetch_era5

    data = fetch_era5(
        box(12.45, 41.90, 12.47, 41.91),
        ["2m_temperature"],
        year=2024,
        month=1,
        day=1,
        hour=0,
        api_url=url,
        api_key=key,
    )
    assert "2m_temperature" in data
    assert data.attrs["dataset_id"] == "reanalysis-era5-single-levels"
