from typing import TYPE_CHECKING

from gigaspatial._lazy import attach

__getattr__, __dir__, __all__ = attach(
    __name__,
    {
        "gigaspatial.processing.elevation.los_analyzer": [
            "LOSAnalyzerConfig",
            "LOSAnalyzer",
            "LOSResult",
        ],
        "gigaspatial.processing.elevation.srtm_manager": ["SRTMManager"],
        "gigaspatial.processing.elevation.refractivity": [
            "get_worst_case_k_factor",
            "get_median_k_factor",
            "get_clearance_k_factors",
        ],
    },
)

if TYPE_CHECKING:
    from gigaspatial.processing.elevation.los_analyzer import (
        LOSAnalyzerConfig,
        LOSAnalyzer,
        LOSResult,
    )
    from gigaspatial.processing.elevation.srtm_manager import SRTMManager
    from gigaspatial.processing.elevation.refractivity import (
        get_worst_case_k_factor,
        get_median_k_factor,
        get_clearance_k_factors,
    )
