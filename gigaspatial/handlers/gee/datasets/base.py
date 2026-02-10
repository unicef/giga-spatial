from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
from datetime import datetime


@dataclass
class GEEDatasetEntry:
    """
    Base class for GEE dataset registry entries.

    Defines the schema and validation for dataset metadata.
    """

    # Required fields
    name: str
    collection: str
    supported_bands: List[str]
    default_band: str
    resolution: float  # meters

    # Temporal metadata
    temporal_cadence: Literal["hourly", "daily", "monthly", "yearly", "single"]
    min_date: str  # "YYYY-MM-DD"
    max_date: str  # "YYYY-MM-DD"

    # Optional metadata
    description: Optional[str] = None
    source: Optional[str] = None  # e.g., "NOAA", "ESA", "NASA"
    units: Optional[Dict[str, str]] = field(
        default_factory=dict
    )  # band -> unit mapping
    scale_factor: Optional[Dict[str, float]] = field(
        default_factory=dict
    )  # band -> scale
    no_data_value: Optional[float] = None
    is_nrt: bool = False  # For Near Real-Time datasets

    # Processing hints
    recommended_reducer: str = "mean"
    max_cloud_cover: Optional[float] = None  # For optical datasets
    requires_masking: bool = False

    # Access metadata
    requires_authentication: bool = True
    is_public: bool = True
    license: Optional[str] = None
    citation: Optional[str] = None

    def __post_init__(self):
        """Validate entry after initialization."""
        self._validate()

    def _validate(self):
        """Validate dataset entry parameters."""
        # Check default band is in supported bands
        if self.default_band not in self.supported_bands:
            raise ValueError(
                f"default_band '{self.default_band}' must be in supported_bands: {self.supported_bands}"
            )

        # Validate date format
        try:
            datetime.strptime(self.min_date, "%Y-%m-%d")
            datetime.strptime(self.max_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Dates must be in YYYY-MM-DD format: {e}")

        # Check date order
        if self.min_date > self.max_date:
            raise ValueError(
                f"min_date ({self.min_date}) must be before max_date ({self.max_date})"
            )

        # Check resolution is positive
        if self.resolution <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")

    def to_dict(self) -> Dict:
        """Convert to dictionary for registry."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def is_date_valid(self, date: str) -> bool:
        """Check if a date is within the dataset's valid range."""
        return self.min_date <= date <= self.max_date

    def is_band_valid(self, band: str) -> bool:
        """Check if a band is supported."""
        return band in self.supported_bands
