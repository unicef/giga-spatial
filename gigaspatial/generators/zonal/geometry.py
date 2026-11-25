import warnings

from gigaspatial.generators.zonal.base import (
    ZonalViewGenerator,
    ZonalViewGeneratorConfig,
    T,
)


class GeometryBasedZonalViewGenerator(ZonalViewGenerator[T]):
    """Deprecated alias for :class:`ZonalViewGenerator`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GeometryBasedZonalViewGenerator is deprecated and will be removed in a future minor release; use ZonalViewGenerator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
