import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.base import DownloadOptions
from gigaspatial.handlers.nasa.srtm import NasaSRTMConfig, NasaSRTMDownloader
from gigaspatial.processing.elevation.los_analyzer import LOSAnalyzer, LOSResult
from gigaspatial.processing.elevation.srtm_manager import SRTMManager


def _create_synthetic_srtm3_zip(tile_stem: str = "N36E030") -> bytes:
    """Create in-memory bytes for a synthetic 1201x1201 SRTM-3 tile."""
    elevation_matrix = np.full((1201, 1201), 100, dtype=">i2")
    hgt_bytes = elevation_matrix.tobytes()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{tile_stem}.hgt", hgt_bytes)
    return buf.getvalue()


def test_nasa_srtm_downloader_accepts_optional_options():
    downloader = NasaSRTMDownloader()
    tile_info = {
        "tile_id": "N36E030",
        "tile_url": "https://example.test/N36E030.zip",
    }

    with patch("gigaspatial.handlers.nasa.srtm.stream_to_store") as mock_stream:
        # Call with 1 argument (options omitted)
        downloader.download_data_unit(tile_info)
        assert mock_stream.call_count == 1
        assert mock_stream.call_args.kwargs["show_progress"] is True

        # Call with 2 arguments (explicit options)
        downloader.download_data_unit(
            tile_info, DownloadOptions(show_progress=False)
        )
        assert mock_stream.call_count == 2
        assert mock_stream.call_args.kwargs["show_progress"] is False


def test_srtm_manager_passes_options_to_downloader(tmp_path):
    store = LocalDataStore(base_path=tmp_path)
    mock_downloader = MagicMock()
    mock_downloader.config._tile_name.return_value = "N36E030"
    mock_downloader.config.get_tile_url.return_value = "https://example.test/N36E030.zip"
    mock_downloader.options_type = DownloadOptions
    mock_downloader.data_store = store

    # Mock download_data_unit to write the synthetic zip into tmp_path
    def _fake_download(tile_info, options=None):
        out_path = tmp_path / "N36E030.SRTMGL1.hgt.zip"
        out_path.write_bytes(_create_synthetic_srtm3_zip("N36E030"))
        return out_path

    mock_downloader.download_data_unit.side_effect = _fake_download

    manager = SRTMManager(
        srtm_directory=tmp_path, downloader=mock_downloader, data_store=store
    )
    elev = manager.get_elevation(36.5, 30.5)

    assert elev == pytest.approx(100.0)
    assert mock_downloader.download_data_unit.call_count == 1
    args, kwargs = mock_downloader.download_data_unit.call_args
    assert args[0]["tile_id"] == "N36E030"
    # Verify options is passed with show_progress=False
    options = kwargs.get("options") or (args[1] if len(args) > 1 else None)
    assert isinstance(options, DownloadOptions)
    assert options.show_progress is False


def test_los_analyzer_auto_download_e2e(tmp_path):
    store = LocalDataStore(base_path=tmp_path)
    config = NasaSRTMConfig(base_path=tmp_path)
    downloader = NasaSRTMDownloader(config=config, data_store=store)

    def _fake_stream(tile_url, target_path, data_store, **kwargs):
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(_create_synthetic_srtm3_zip("N36E030"))
        return target_path

    with patch("gigaspatial.handlers.nasa.srtm.stream_to_store", side_effect=_fake_stream):
        analyzer = LOSAnalyzer(
            srtm_directory=tmp_path, downloader=downloader, data_store=store
        )
        result = analyzer.analyze(
            start_lat=36.0,
            start_lon=30.0,
            end_lat=36.01,
            end_lon=30.01,
            tx_height_m=20.0,
            rx_height_m=10.0,
            frequency_mhz=5800.0,
            num_points=50,
        )

        assert isinstance(result, LOSResult)
        assert result.is_visual_los is True
        summary = result.summary()
        assert "is_highly_available" in summary
        assert "margin_median_m" in summary
