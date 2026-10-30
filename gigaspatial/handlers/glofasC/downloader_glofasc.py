"""GloFAS downloader on ``BaseHandlerDownloader``.

``download_data_unit`` fetches one unit. Parallelism, progress, and skipping
files that are already stored stay on the base (``download`` and
``download_data_units``).
"""

from __future__ import annotations

from pathlib import Path

from gigaspatial.handlers.base.downloader import BaseHandlerDownloader
from gigaspatial.handlers.base.fetch import stream_to_store
from gigaspatial.handlers.base.options import DownloadOptions
from gigaspatial.handlers.glofasC.config_glofasc import GlofasCConfig


class GlofasCDownloader(BaseHandlerDownloader):
    """Downloads GloFAS depth tiles into the data store."""

    options_type = DownloadOptions
    config: GlofasCConfig

    def download_data_unit(self, unit: dict, options: DownloadOptions) -> Path:
        """
        Download one depth tile.

        Args:
            unit: A tile and return period from the config. The URL is on the unit.
            options: ``show_progress`` controls the byte progress bar.

        Returns:
            The stored file path.

        Raises:
            requests.RequestException: If the download fails. A partial file is
                removed by ``stream_to_store``.
        """
        return stream_to_store(
            unit["url"],
            self.config.get_data_unit_path(unit),
            self.data_store,
            show_progress=options.show_progress,
        )
