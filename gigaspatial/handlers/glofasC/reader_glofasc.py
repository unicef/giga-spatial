"""GloFAS reader on ``BaseHandlerReader``.

``load_from_paths`` loads stored GeoTIFFs. Path resolution and
``crop_to_source`` stay on the base ``load``. Several return periods come back
as a list, and each map is cropped with the base raster clip.
"""

from __future__ import annotations

from typing import List

from gigaspatial.handlers.base.options import RasterReadOptions
from gigaspatial.handlers.base.reader import BaseHandlerReader
from gigaspatial.handlers.glofasC.config_glofasc import GlofasCConfig
from gigaspatial.processing.tif_processor import TifProcessor


class GlofasCReader(BaseHandlerReader):
    """Loads GloFAS depth tiles as ``TifProcessor`` objects."""

    options_type = RasterReadOptions
    config: GlofasCConfig

    def load_from_paths(
        self, source_data_path: List[str], options: RasterReadOptions
    ) -> TifProcessor | List[TifProcessor]:
        """
        Load stored depth maps.

        Args:
            source_data_path: Stored GeoTIFF paths from the config.
            options: ``merge_rasters`` merges every path into one processor.
                Leave it false across return periods: those maps are different
                datasets.

        Returns:
            One ``TifProcessor``, or a list of them when several files are
            loaded separately.
        """
        return self._load_raster_data(
            source_data_path, merge_rasters=options.merge_rasters
        )

    def crop_to_geometry(self, data, geometry, predicate="intersects", **kwargs):
        """Crop one depth map, or each map when several return periods were loaded."""
        if isinstance(data, list):
            return [
                super().crop_to_geometry(item, geometry, predicate=predicate, **kwargs)
                for item in data
            ]
        return super().crop_to_geometry(data, geometry, predicate=predicate, **kwargs)
