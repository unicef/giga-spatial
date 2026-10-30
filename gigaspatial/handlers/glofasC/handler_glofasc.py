"""GloFAS handler on ``BaseHandler``.

The base owns ``load_data``, ``download_and_load``, and ``ensure_data_available``.
This class only builds the config, downloader, and reader. A caller passes a
country or a level 1-3 unit. The base turns that into units, downloads the
missing depth tiles, then asks the reader to load them.
"""

from __future__ import annotations

from gigaspatial.handlers.base.handler import BaseHandler
from gigaspatial.handlers.glofasC.config_glofasc import GlofasCConfig
from gigaspatial.handlers.glofasC.downloader_glofasc import GlofasCDownloader
from gigaspatial.handlers.glofasC.reader_glofasc import GlofasCReader


class GlofasCHandler(BaseHandler):
    """
    Handler for GloFAS flood hazard depth maps.

    A country downloads the tiles that intersect it, for every return period:

        GlofasCHandler().load_data("KEN", crop_to_source=True)

    A level 1, 2, or 3 unit is the same call with ``admin_level`` and
    ``admin_name``. ``return_periods`` narrows the set; the default is
    ``(10, 20, 50, 75, 100, 200, 500)``. Files are stored in the ``glofasC`` folder.
    """

    def create_config(self, data_store, logger, **kwargs) -> GlofasCConfig:
        return GlofasCConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(self, config, data_store, logger, **kwargs) -> GlofasCDownloader:
        return GlofasCDownloader(config=config, data_store=data_store, logger=logger)

    def create_reader(self, config, data_store, logger, **kwargs) -> GlofasCReader:
        return GlofasCReader(config=config, data_store=data_store, logger=logger)
