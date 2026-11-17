"""Live CDS checks. Skipped unless pytest is run with ``-m live``."""

from datetime import timezone

import pytest

from gigaspatial.handlers.cds.era5 import (
    DEFAULT_API_URL,
    fetch_latest_available,
    resolve_credentials,
)


@pytest.mark.live
def test_catalogue_publishes_an_end_for_each_dataset():
    era5 = fetch_latest_available("reanalysis-era5-single-levels", DEFAULT_API_URL)
    land = fetch_latest_available("reanalysis-era5-land", DEFAULT_API_URL)

    assert era5.tzinfo is not None
    assert land.tzinfo is not None
    assert era5.year >= 1940
    assert land.year >= 1950
    assert era5.astimezone(timezone.utc)
    assert land.astimezone(timezone.utc)


@pytest.mark.live
def test_one_hour_retrieve_when_credentials_are_configured(tmp_path):
    url, key = resolve_credentials()
    if not key:
        pytest.skip("CDS_API_KEY is not configured")

    from gigaspatial.handlers.cds.era5 import ERA5Handler

    handler = ERA5Handler(base_path=tmp_path / "era5", api_url=url, api_key=key, n_workers=1)
    data = handler.load_data(
        "VAT",
        variables="2m_temperature",
        year=2024,
        month=1,
        day=1,
        hour=0,
        show_progress=False,
    )
    assert data is not None
