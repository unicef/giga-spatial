"""
Base classes and abstractions for dataset handlers.

This module defines the foundational architecture for all data handlers in
GigaSpatial. It provides abstract base classes for:
- Configuration (`BaseHandlerConfig`)
- Data downloaders (`BaseHandlerDownloader`)
- Data readers (`BaseHandlerReader`)
- Orchestrating handlers (`BaseHandler`)

Standardizing these interfaces ensures consistency across various data sources
(e.g., GHSL, WorldPop, OSM, Google Buildings).
"""

from gigaspatial.handlers.base.config import BaseHandlerConfig
from gigaspatial.handlers.base.downloader import BaseHandlerDownloader
from gigaspatial.handlers.base.reader import BaseHandlerReader
from gigaspatial.handlers.base.handler import BaseHandler
from gigaspatial.handlers.base.options import (
    DownloadOptions,
    RasterReadOptions,
    ReadOptions,
    ResolveOptions,
)

__all__ = [
    "BaseHandlerConfig",
    "BaseHandlerDownloader",
    "BaseHandlerReader",
    "BaseHandler",
    "DownloadOptions",
    "RasterReadOptions",
    "ReadOptions",
    "ResolveOptions",
]
