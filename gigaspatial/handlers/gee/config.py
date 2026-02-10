from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Literal
from datetime import datetime, timedelta
import ee

from .registry import GEEDatasetRegistry

# Module-level default registry (loaded once)
_DEFAULT_REGISTRY = None


def get_default_registry() -> GEEDatasetRegistry:
    """Get or create the default registry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = GEEDatasetRegistry()
    return _DEFAULT_REGISTRY


@dataclass
class GEEConfig:
    """
    Configuration class for Google Earth Engine Handler operations.

    This config manages dataset metadata, authentication, and processing parameters
    for GEE operations within GigaSpatial.
    """

    # ===== Authentication & Initialization =====
    service_account: Optional[str] = None
    key_path: Optional[str] = None
    project_id: Optional[str] = None
    use_highvolume: bool = False  # For high-volume GEE endpoint

    # ===== Dataset Configuration =====
    dataset_id: Optional[str] = None  # e.g., "nightlights", "global_human_modification"
    collection: Optional[str] = (
        None  # GEE collection ID, e.g., "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"
    )

    # ===== Band & Variable Selection =====
    band: Optional[str] = None  # Selected band to process
    supported_bands: Optional[List[str]] = None  # Available bands for this dataset
    default_band: Optional[str] = None  # Default band if none specified

    # ===== Temporal Parameters =====
    start_date: Optional[Union[str, datetime]] = None  # "YYYY-MM-DD" or datetime
    end_date: Optional[Union[str, datetime]] = None
    temporal_cadence: Optional[Literal["daily", "monthly", "yearly", "single"]] = None
    temporal_reducer: Optional[str] = (
        "mean"  # How to aggregate over time: mean, median, max, etc.
    )

    # ===== Spatial Parameters =====
    scale: Optional[float] = None  # Spatial resolution in meters
    crs: str = "EPSG:4326"  # Coordinate reference system
    buffer_radius_m: Optional[float] = None  # Buffer radius for point-based queries

    # ===== Reduction & Aggregation =====
    reducer: str = "mean"  # Spatial reducer: mean, median, min, max, sum, etc.
    max_pixels: int = 100_000_000  # Maximum pixels for GEE operations
    best_effort: bool = True  # Use best effort for large operations
    tile_scale: int = 1  # Tile scale for large computations (1-16)

    # ===== Chunking & Performance =====
    chunk_size: int = 1000  # Number of features to process per chunk
    enable_chunking: bool = (
        True  # Enable/disable chunking for large feature collections
    )
    max_retries: int = 3  # Retry attempts for failed GEE operations
    timeout_seconds: int = 300  # Timeout for GEE operations

    # ===== Edge Case Handling =====
    fill_value: Optional[float] = (
        None  # Value to use when data unavailable (None uses NaN)
    )
    skip_water_bodies: bool = False  # Skip processing over oceans/water
    skip_polar_regions: bool = False  # Skip Arctic/Antarctica (>60° lat)

    # ===== Output Configuration =====
    output_column_prefix: Optional[str] = (
        None  # Prefix for output columns, e.g., "nightlights_"
    )
    keep_geometry: bool = True  # Keep original geometry in output
    return_format: Literal["geodataframe", "dict", "featurecollection"] = "geodataframe"

    # ===== Caching & Storage =====
    enable_cache: bool = False  # Enable local caching of GEE results
    cache_dir: Optional[str] = None  # Directory for cached data
    export_format: Optional[Literal["GeoTIFF", "GeoJSON", "shapefile", "netcdf"]] = None

    # ===== Metadata & Registry =====
    dataset_metadata: Dict = field(default_factory=dict)  # Full metadata from registry

    # ===== Validation & Logging =====
    validate_inputs: bool = True  # Validate dates, bands, geometries before processing
    verbose: bool = False  # Enable verbose logging

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""

        # 1. Handle Near Real-Time (NRT) Defaults
        # We only apply defaults if the dataset is NRT AND the user hasn't provided dates.
        if self.dataset_metadata.get("is_nrt") and not self.end_date:
            # If no end_date is provided, default to 'now'
            if self.end_date is None:
                self.end_date = datetime.now()

            # If no start_date is provided, default to 7 days before the end_date
            if self.start_date is None:
                # Ensure we are calculating relative to the end_date (whether it was
                # just set to 'now' or was manually provided by the user)
                reference_date = (
                    self.end_date
                    if isinstance(self.end_date, datetime)
                    else datetime.strptime(str(self.end_date), "%Y-%m-%d")
                )
                self.start_date = reference_date - timedelta(days=7)

        # 2. Normalization (Convert everything to "YYYY-MM-DD" strings)
        # This allows the rest of the profiler to treat dates consistently.
        if isinstance(self.start_date, datetime):
            self.start_date = self.start_date.strftime("%Y-%m-%d")
        if isinstance(self.end_date, datetime):
            self.end_date = self.end_date.strftime("%Y-%m-%d")

        # Set default band if none specified but supported_bands available
        if self.band is None and self.default_band:
            self.band = self.default_band

        # Set output column prefix if not specified
        if self.output_column_prefix is None and self.dataset_id:
            self.output_column_prefix = f"{self.dataset_id}_"

        # Validate configuration if enabled
        if self.validate_inputs:
            self._validate()

    def _validate(self):
        """Validate configuration parameters."""

        # Check band is in supported bands
        if self.band and self.supported_bands:
            if self.band not in self.supported_bands:
                raise ValueError(
                    f"Band '{self.band}' not in supported bands: {self.supported_bands}"
                )

        # Check date order
        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                raise ValueError(
                    f"start_date ({self.start_date}) must be before end_date ({self.end_date})"
                )

        # Check scale is positive
        if self.scale is not None and self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

        # Check chunk_size is reasonable
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        # Check tile_scale range
        if not (1 <= self.tile_scale <= 16):
            raise ValueError(
                f"tile_scale must be between 1 and 16, got {self.tile_scale}"
            )

    @classmethod
    def from_dataset_id(
        cls, dataset_id: str, registry: Optional[GEEDatasetRegistry] = None, **overrides
    ) -> "GEEConfig":
        """
        Create config from dataset ID using registry.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        registry : GEEDatasetRegistry, optional
            Custom registry. If None, uses built-in registry.
        **overrides
            Override any config parameters
        """
        # Use default registry if none provided
        if registry is None:
            registry = get_default_registry()

        # Get dataset entry
        dataset_entry = registry.get(dataset_id)

        # Ensure scale is never 0 or None if the dataset has a resolution
        scale = (
            dataset_entry.resolution
            if dataset_entry.resolution and dataset_entry.resolution > 0
            else overrides.get("scale")
        )

        # Build config from entry
        config_params = {
            "dataset_id": dataset_id,
            "collection": dataset_entry.collection,
            "supported_bands": dataset_entry.supported_bands,
            "default_band": dataset_entry.default_band,
            "scale": scale,
            "temporal_cadence": dataset_entry.temporal_cadence,
            "reducer": dataset_entry.recommended_reducer,
            "dataset_metadata": dataset_entry.to_dict(),
            **overrides,
        }

        return cls(**config_params)

    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def get_ee_reducer(self) -> ee.Reducer:
        """
        Get the appropriate Earth Engine Reducer based on config.

        Returns
        -------
        ee.Reducer
            Earth Engine reducer object
        """
        reducer_map = {
            "mean": ee.Reducer.mean(),
            "median": ee.Reducer.median(),
            "min": ee.Reducer.min(),
            "max": ee.Reducer.max(),
            "sum": ee.Reducer.sum(),
            "stdDev": ee.Reducer.stdDev(),
            "variance": ee.Reducer.variance(),
            "mode": ee.Reducer.mode(),
            "minMax": ee.Reducer.minMax(),
            "count": ee.Reducer.count(),
        }

        if self.reducer not in reducer_map:
            raise ValueError(
                f"Unsupported reducer: {self.reducer}. "
                f"Supported: {list(reducer_map.keys())}"
            )

        return reducer_map[self.reducer]

    def update(self, **kwargs):
        """Update configuration parameters in place."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"GEEConfig has no attribute '{key}'")

        # Re-validate after updates
        if self.validate_inputs:
            self._validate()
