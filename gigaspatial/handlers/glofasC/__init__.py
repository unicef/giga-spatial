"""GloFAS flood-hazard handler, split the same way as ``handlers/base``."""

from gigaspatial.handlers.glofasC.config_glofasc import GlofasCConfig
from gigaspatial.handlers.glofasC.downloader_glofasc import GlofasCDownloader
from gigaspatial.handlers.glofasC.handler_glofasc import GlofasCHandler
from gigaspatial.handlers.glofasC.options_glofasc import (
    RETURN_PERIODS,
    GlofasCResolveOptions,
)
from gigaspatial.handlers.glofasC.reader_glofasc import GlofasCReader

__all__ = [
    "RETURN_PERIODS",
    "GlofasCResolveOptions",
    "GlofasCConfig",
    "GlofasCDownloader",
    "GlofasCReader",
    "GlofasCHandler",
]
