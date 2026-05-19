"""
WRI Aqueduct Flood Hazard handler.

Integrates World Resources Institute AQUEDUCT global flood hazard layers
with the GigaSpatial handler architecture. Provides tile-level access to
riverine and coastal flood inundation depth rasters across historical and
future climate scenarios.

Data source: World Resources Institute AQUEDUCT Global Flood Hazard Layers (v2)
URL: https://www.wri.org/aqueduct
License: CC BY 4.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional, Union, ClassVar

import requests
from tqdm import tqdm
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from shapely.geometry.base import BaseGeometry

import geopandas as gpd
import pandas as pd

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.handlers.base import (
    BaseHandler,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
)
from gigaspatial.processing.tif_processor import TifProcessor

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# ─────────────────────────────────────────────
# Valid option sets (derived from WRI S3 inventory)
# ─────────────────────────────────────────────

FloodType = Literal["inuncoast", "inunriver"]
ClimateScenario = Literal["historical", "rcp4p5", "rcp8p5"]

# Coastal years: "hist" (historical only) + future projection years
CoastalYear = Literal["hist", "2030", "2050", "2080"]

# Riverine years: "1980" for historical, future projection years for rcp4p5
RiverineYear = Literal["1980", "2030", "2050", "2080"]

# Unified Year alias used on the public handler interface
Year = Union[CoastalYear, RiverineYear]

# Coastal return periods (4-digit zero-padded)
CoastalReturnPeriod = Literal[
    "rp0001",  # 1-in-1.5-year; requires projection "5", "5_perc_05", or "5_perc_50"
    "rp0002",
    "rp0005",
    "rp0010",
    "rp0025",
    "rp0050",
    "rp0100",
    "rp0250",
    "rp0500",
    "rp1000",
]

# Riverine return periods (5-digit zero-padded — different from coastal)
RiverineReturnPeriod = Literal[
    "rp00002",
    "rp00005",
    "rp00010",
    "rp00025",
    "rp00050",
    "rp00100",
    "rp00250",
    "rp00500",
    "rp01000",
]

# Unified ReturnPeriod alias used on the public handler interface
ReturnPeriod = Union[CoastalReturnPeriod, RiverineReturnPeriod]

# Coastal-only
Subsidence = Literal["nosub", "wtsub"]

# Coastal projection suffix:
#   "0", "0_perc_05", "0_perc_50" — for rp0002–rp1000
#   "5", "5_perc_05", "5_perc_50" — for rp0001 only
# Percentile variants (_perc_05 / _perc_50) exist only for rcp4p5 and rcp8p5,
# not for "historical".
Projection = Literal[
    "0",
    "0_perc_05",
    "0_perc_50",
    "5",
    "5_perc_05",
    "5_perc_50",
]

# Riverine-only GCM model tokens (exactly as they appear in S3 filenames)
RiverineModel = Literal[
    "000000000WATCH",  # historical only
    "00000NorESM1-M",  # rcp4p5 only
    "0000GFDL-ESM2M",  # rcp4p5 only  ← hyphen, not underscore
    "0000HadGEM2-ES",  # rcp4p5 only; rp00500 and rp01000 absent for year=2080
]

# ---------------------------------------------------------------------------
# Validation look-up tables
# ---------------------------------------------------------------------------

# Valid years per (flood_type, scenario) combination
_VALID_YEARS: dict[tuple[str, str], frozenset[str]] = {
    ("inuncoast", "historical"): frozenset({"hist", "2030", "2050", "2080"}),
    ("inuncoast", "rcp4p5"): frozenset({"2030", "2050", "2080"}),
    ("inuncoast", "rcp8p5"): frozenset({"2030", "2050", "2080"}),
    ("inunriver", "historical"): frozenset({"1980"}),
    ("inunriver", "rcp4p5"): frozenset({"2030", "2050", "2080"}),
    # rcp8p5 + inunriver does not exist on S3
}

# For coastal historical, nosub is further restricted to year="hist" only
_COASTAL_HISTORICAL_NOSUB_YEARS: frozenset[str] = frozenset({"hist"})

# rp0001 requires a "5*" projection; all other rps require a "0*" projection
_RP0001_PROJECTIONS: frozenset[str] = frozenset({"5", "5_perc_05", "5_perc_50"})
_OTHER_PROJECTIONS: frozenset[str] = frozenset({"0", "0_perc_05", "0_perc_50"})

# Percentile projections only exist for RCP scenarios (not historical)
_PERCENTILE_PROJECTIONS: frozenset[str] = frozenset(
    {"0_perc_05", "0_perc_50", "5_perc_05", "5_perc_50"}
)

# Models available per scenario
_RIVERINE_SCENARIO_MODELS: dict[str, frozenset[str]] = {
    "historical": frozenset({"000000000WATCH"}),
    "rcp4p5": frozenset({"00000NorESM1-M", "0000GFDL-ESM2M", "0000HadGEM2-ES"}),
}

# HadGEM2-ES 2080 is missing rp00500 and rp01000 on S3
_HADGEM2_2080_MISSING_RPS: frozenset[str] = frozenset({"rp00500", "rp01000"})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AqueductFloodConfig(BaseHandlerConfig):
    """Configuration for WRI Aqueduct flood hazard raster access.

    Each unique combination of parameters maps 1-to-1 to a single global
    GeoTIFF on the WRI S3 bucket, which is used as the **data unit**
    throughout the base-class workflow — identical to the Ookla speedtest
    pattern.

    Attributes:
        flood_type: ``"inunriver"`` (riverine) or ``"inuncoast"`` (coastal).
        climate_scenario: ``"historical"``, ``"rcp4p5"``, or ``"rcp8p5"``.
            ``"rcp8p5"`` is coastal-only; riverine data only has
            ``"historical"`` and ``"rcp4p5"``.
        year: Reference year for the hazard layer.
            Coastal: ``"hist"`` | ``"2030"`` | ``"2050"`` | ``"2080"``.
            Riverine: ``"1980"`` | ``"2030"`` | ``"2050"`` | ``"2080"``.
            The combination of ``climate_scenario`` and ``year`` must be
            consistent (see validation).
        return_period: Return-period key.
            Coastal (4-digit): ``"rp0001"``–``"rp1000"``.
            Riverine (5-digit): ``"rp00002"``–``"rp01000"``.
        subsidence: *Coastal only.* Sea-level-rise subsidence assumption —
            ``"nosub"`` or ``"wtsub"``. Required when
            ``flood_type="inuncoast"``.
        projection: *Coastal only.* Sea-level projection percentile suffix.
            Use ``"0"`` / ``"0_perc_05"`` / ``"0_perc_50"`` for
            ``return_period`` ≥ ``"rp0002"``, and ``"5"`` / ``"5_perc_05"``
            / ``"5_perc_50"`` exclusively for ``return_period="rp0001"``.
            Percentile variants (``_perc_05`` / ``_perc_50``) are only
            available for RCP scenarios, not ``"historical"``.
            Required when ``flood_type="inuncoast"``.
        model: *Riverine only.* GCM model token.
            ``"000000000WATCH"`` is the only model for the historical
            scenario; ``"00000NorESM1-M"``, ``"0000GFDL-ESM2M"``, and
            ``"0000HadGEM2-ES"`` are available for ``"rcp4p5"``.
            Note: ``"0000HadGEM2-ES"`` does not have ``rp00500`` or
            ``rp01000`` for ``year="2080"``.
            Required when ``flood_type="inunriver"``.
        base_path: Root directory for downloaded raster files.
    """

    # ── Required ────────────────────────────────────────
    flood_type: FloodType = "inunriver"
    climate_scenario: ClimateScenario = "historical"
    year: str = "1980"
    return_period: str = "rp00100"

    # ── Coastal-only ────────────────────────────────────
    subsidence: Optional[Subsidence] = None
    projection: Optional[Projection] = None

    # ── Riverine-only ───────────────────────────────────
    model: Optional[RiverineModel] = "000000000WATCH"

    # ── Storage ─────────────────────────────────────────
    file_extension: str = "tif"

    # ── Class-level constants ────────────────────────────
    BASE_URL: ClassVar[str] = (
        "https://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2"
    )
    base_path: Path = Field(
        default=global_config.get_path("wri", "bronze") / "AqueductFloodHazard"
    )
    DATASET_URL: str = Field(default="", init=False)

    def __post_init__(self):
        super().__post_init__()
        self._validate()
        self.DATASET_URL = self._get_dataset_url()
        self._log_config()

    def _log_config(self) -> None:
        """Log a human-readable summary of the active configuration."""
        if self.flood_type == "inunriver":
            params = (
                f"flood_type=inunriver | scenario={self.climate_scenario} | "
                f"model={self.model} | year={self.year} | "
                f"return_period={self.return_period}"
            )
        else:
            params = (
                f"flood_type=inuncoast | scenario={self.climate_scenario} | "
                f"subsidence={self.subsidence} | projection={self.projection} | "
                f"year={self.year} | return_period={self.return_period}"
            )
        self.logger.info(
            "AqueductFloodConfig initialised — %s",
            params,
        )

    # ────────────────────────────────────────────────────
    # Validation
    # ────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Validate parameter combinations against the WRI S3 inventory.

        Raises:
            ValueError: On any invalid parameter combination.
        """
        # ── 1. rcp8p5 is coastal-only ────────────────────
        if self.flood_type == "inunriver" and self.climate_scenario == "rcp8p5":
            raise ValueError(
                "rcp8p5 scenario is not available for riverine flood data. "
                "Use 'historical' or 'rcp4p5'."
            )

        # ── 2. Year must match flood_type × scenario ─────
        valid_years = _VALID_YEARS.get((self.flood_type, self.climate_scenario))
        if valid_years is None:
            raise ValueError(
                f"Unsupported combination: flood_type='{self.flood_type}', "
                f"climate_scenario='{self.climate_scenario}'."
            )
        if self.year not in valid_years:
            raise ValueError(
                f"year='{self.year}' is not valid for "
                f"flood_type='{self.flood_type}' + "
                f"climate_scenario='{self.climate_scenario}'. "
                f"Valid years: {sorted(valid_years)}."
            )

        # ── 3. Coastal-specific checks ───────────────────
        if self.flood_type == "inuncoast":
            missing = [
                name
                for name, val in [
                    ("subsidence", self.subsidence),
                    ("projection", self.projection),
                ]
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"Coastal flood hazard requires: {', '.join(missing)}."
                )

            # nosub + historical is only available for year="hist"
            if (
                self.climate_scenario == "historical"
                and self.subsidence == "nosub"
                and self.year not in _COASTAL_HISTORICAL_NOSUB_YEARS
            ):
                raise ValueError(
                    f"Coastal historical + nosub only has year='hist', "
                    f"got '{self.year}'."
                )

            # Validate return_period format (coastal: 4-digit rp####)
            if not self.return_period.startswith("rp") or len(self.return_period) != 6:
                raise ValueError(
                    f"Coastal return_period must be 4-digit zero-padded "
                    f"(e.g. 'rp0100'), got '{self.return_period}'."
                )

            # rp0001 requires a "5*" projection; all others require "0*"
            if self.return_period == "rp0001":
                if self.projection not in _RP0001_PROJECTIONS:
                    raise ValueError(
                        f"return_period='rp0001' requires projection in "
                        f"{sorted(_RP0001_PROJECTIONS)}, got '{self.projection}'."
                    )
            else:
                if self.projection not in _OTHER_PROJECTIONS:
                    raise ValueError(
                        f"return_period='{self.return_period}' requires projection in "
                        f"{sorted(_OTHER_PROJECTIONS)}, got '{self.projection}'."
                    )

            # Percentile projections only exist for RCP scenarios
            if (
                self.climate_scenario == "historical"
                and self.projection in _PERCENTILE_PROJECTIONS
            ):
                raise ValueError(
                    f"Percentile projections ({sorted(_PERCENTILE_PROJECTIONS)}) "
                    f"are only available for rcp4p5 and rcp8p5, not 'historical'."
                )

        # ── 4. Riverine-specific checks ──────────────────
        elif self.flood_type == "inunriver":
            if self.model is None:
                raise ValueError(
                    "Riverine flood hazard requires `model` " "(e.g. '000000000WATCH')."
                )

            # Model must be valid for the selected scenario
            valid_models = _RIVERINE_SCENARIO_MODELS.get(
                self.climate_scenario, frozenset()
            )
            if self.model not in valid_models:
                raise ValueError(
                    f"model='{self.model}' is not available for "
                    f"climate_scenario='{self.climate_scenario}'. "
                    f"Valid models: {sorted(valid_models)}."
                )

            # Validate return_period format (riverine: 5-digit rp#####)
            if not self.return_period.startswith("rp") or len(self.return_period) != 7:
                raise ValueError(
                    f"Riverine return_period must be 5-digit zero-padded "
                    f"(e.g. 'rp00100'), got '{self.return_period}'."
                )

            # HadGEM2-ES 2080 is missing the two highest return periods on S3
            if (
                self.model == "0000HadGEM2-ES"
                and self.year == "2080"
                and self.return_period in _HADGEM2_2080_MISSING_RPS
            ):
                raise ValueError(
                    f"return_period='{self.return_period}' is not available "
                    f"for model='0000HadGEM2-ES' + year='2080'. "
                    f"Missing RPs: {sorted(_HADGEM2_2080_MISSING_RPS)}."
                )

    # ────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────

    def _get_filename(self) -> str:
        """Build the canonical WRI Aqueduct hazard filename.

        Coastal pattern::

            inuncoast_{scenario}_{subsidence}_{year}_{rp}_{projection}.tif

        Riverine pattern::

            inunriver_{scenario}_{model}_{year}_{rp}.tif
        """
        ext = self.file_extension

        if self.flood_type == "inuncoast":
            return (
                f"inuncoast"
                f"_{self.climate_scenario}"
                f"_{self.subsidence}"
                f"_{self.year}"
                f"_{self.return_period}"
                f"_{self.projection}"
                f".{ext}"
            )

        return (
            f"inunriver"
            f"_{self.climate_scenario}"
            f"_{self.model}"
            f"_{self.year}"
            f"_{self.return_period}"
            f".{ext}"
        )

    def _get_dataset_url(self) -> str:
        """Remote S3 URL for this specific hazard file."""
        return f"{self.BASE_URL}/{self._get_filename()}"

    # ────────────────────────────────────────────────────
    # BaseHandlerConfig interface
    # ────────────────────────────────────────────────────

    def get_relevant_data_units(self, source=None, **kwargs) -> List[str]:
        """Return the S3 download URL for the configured dataset.

        Aqueduct rasters are global; geometry does not filter the units.

        Args:
            source: Ignored; present for interface compatibility.

        Returns:
            Single-element list containing the S3 URL.
        """
        return [self.DATASET_URL]

    def get_relevant_data_units_by_geometry(
        self,
        geometry: Union[BaseGeometry, gpd.GeoDataFrame],
        **kwargs,
    ) -> List[str]:
        """Return the S3 download URL for the configured dataset.

        Aqueduct rasters are global; geometry does not filter the units.

        Args:
            geometry: Ignored; present for interface compatibility.

        Returns:
            Single-element list containing the S3 URL.
        """
        return [self.DATASET_URL]

    def get_data_unit_path(self, unit: str, **kwargs) -> Path:
        """Resolve an S3 URL to its local storage path.

        Args:
            unit: S3 URL (the data-unit identifier).

        Returns:
            Absolute path under ``base_path``.
        """
        return self.base_path / unit.split("/")[-1]

    def get_dataset_info(self) -> dict:
        """Return provenance metadata for the configured Aqueduct dataset."""
        return {
            "name": "WRI AQUEDUCT Global Flood Hazard Layers (v2)",
            "source": "World Resources Institute",
            "url": "https://www.wri.org/aqueduct",
            "hazard_type": self.flood_type,
            "description": (
                "Riverine flood inundation depth rasters"
                if self.flood_type == "inunriver"
                else "Coastal flood inundation depth rasters"
            ),
            "scenario": self.climate_scenario,
            "return_period": self.return_period,
            "year": self.year,
            "model": self.model,
            "projection": self.projection,
            "resolution": "~1 km (30 arc-seconds)",
            "coverage": "Global",
            "units": "Flood inundation depth (metres)",
            "license": "CC BY 4.0",
            "citation": (
                "Ward et al. (2020). Aqueduct Floods Methodology. "
                "World Resources Institute, Washington, DC."
            ),
        }


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


class AqueductFloodDownloader(BaseHandlerDownloader):
    """Downloader for WRI Aqueduct Global Flood Hazard GeoTIFFs.

    Streams individual return-period rasters from the WRI S3 bucket to the
    configured ``DataStore``.
    """

    def __init__(
        self,
        config: Optional[AqueductFloodConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config or AqueductFloodConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(self, url: str, **kwargs) -> Optional[Path]:
        """Download a single Aqueduct raster file.

        Args:
            url: S3 URL of the raster to download.
            **kwargs: Ignored.

        Returns:
            Local path to the downloaded file, or ``None`` on failure.
        """
        output_path = self.config.get_data_unit_path(url)
        self.logger.info(f"Downloading {output_path.name} from {url}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            import math

            total_size = int(response.headers.get("content-length", 0))
            total_chunks = math.ceil(total_size / 8192) if total_size else 0

            with self.data_store.open(str(output_path), "wb") as fh:
                for chunk in tqdm(
                    response.iter_content(chunk_size=8192),
                    total=total_chunks,
                    unit="chunk",
                    desc=output_path.name,
                ):
                    fh.write(chunk)

            self.logger.info(f"Saved: {output_path}")
            return output_path

        except requests.exceptions.RequestException as exc:
            self.logger.error(f"Failed to download {url}: {exc}")
            return None
        except Exception as exc:
            self.logger.error(f"Unexpected error downloading {url}: {exc}")
            return None


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class AqueductFloodReader(BaseHandlerReader):
    """Reader for WRI Aqueduct Global Flood Hazard GeoTIFFs."""

    def __init__(
        self,
        config: Optional[AqueductFloodConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config or AqueductFloodConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self,
        source_data_path: List[Union[str, Path]],
        **kwargs,
    ) -> List[TifProcessor]:
        """Initialise a ``TifProcessor`` for each raster path.

        Args:
            source_data_path: List of ``.tif`` paths.
            **kwargs: Forwarded to ``_load_raster_data``.

        Returns:
            A list of ``TifProcessor`` instances.
        """
        return self._load_raster_data(source_data_path, merge_rasters=False, **kwargs)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class AqueductFloodHandler(BaseHandler):
    """Unified handler for WRI Aqueduct Global Flood Hazard data.

    Orchestrates configuration, downloading, and reading of global flood
    inundation depth rasters.

    Examples
    --------
    Riverine — historical 1-in-100-year flood:

    >>> handler = AqueductFloodHandler(
    ...     flood_type="inunriver",
    ...     climate_scenario="historical",
    ...     year="1980",
    ...     return_period="rp00100",
    ...     model="000000000WATCH",
    ... )
    >>> tif = handler.load_data(schools_gdf)

    Coastal — RCP 8.5 2050 with subsidence:

    >>> handler = AqueductFloodHandler(
    ...     flood_type="inuncoast",
    ...     climate_scenario="rcp8p5",
    ...     year="2050",
    ...     return_period="rp0100",
    ...     subsidence="wtsub",
    ...     projection="0_perc_50",
    ... )
    >>> tif = handler.load_data("RWA")
    """

    def __init__(
        self,
        flood_type: FloodType = "inunriver",
        climate_scenario: ClimateScenario = "historical",
        year: Year = "1980",
        return_period: ReturnPeriod = "rp00100",
        # Coastal-only
        subsidence: Optional[Subsidence] = None,
        projection: Optional[Projection] = None,
        # Riverine-only
        model: Optional[RiverineModel] = "000000000WATCH",
        # Infrastructure
        config: Optional[AqueductFloodConfig] = None,
        downloader: Optional[AqueductFloodDownloader] = None,
        reader: Optional[AqueductFloodReader] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        """Initialise the AqueductFloodHandler.

        Args:
            flood_type: Flood category — ``"inunriver"`` (riverine) or
                ``"inuncoast"`` (coastal). Defaults to ``"inunriver"``.
            climate_scenario: Climate scenario — ``"historical"``,
                ``"rcp4p5"``, or ``"rcp8p5"``. Defaults to ``"historical"``.
            year: Reference year — ``"hist"``, ``"2030"``, ``"2050"``, or
                ``"2080"``. Defaults to ``"hist"``.
            return_period: Return-period key (e.g. ``"rp0100"``).
                Defaults to ``"rp0100"``.
            subsidence: *Coastal only.* Sea-level-rise subsidence assumption —
                ``"nosub"`` or ``"wtsub"``. Required when
                ``flood_type="inuncoast"``.
            projection: *Coastal only.* Projection percentile —
                ``"0"``, ``"0_perc_05"``, or ``"0_perc_50"``. Required when
                ``flood_type="inuncoast"``.
            model: *Riverine only.* GCM model token (e.g.
                ``"000000000WATCH"``). Required when
                ``flood_type="inunriver"``.
            config: Optional pre-built :class:`AqueductFloodConfig`. When
                supplied, all dataset parameters above are ignored.
            downloader: Optional pre-built downloader.
            reader: Optional pre-built reader.
            data_store: Storage interface. Defaults to ``LocalDataStore``.
            logger: Custom logger instance.
            **kwargs: Additional parameters forwarded to ``create_config``.
        """
        self._flood_type = flood_type
        self._climate_scenario = climate_scenario
        self._year = year
        self._return_period = return_period
        self._subsidence = subsidence
        self._projection = projection
        self._model = model

        super().__init__(
            config=config,
            downloader=downloader,
            reader=reader,
            data_store=data_store,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # BaseHandler factory methods
    # ------------------------------------------------------------------

    def create_config(
        self,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> AqueductFloodConfig:
        """Create and return an ``AqueductFloodConfig`` instance."""
        return AqueductFloodConfig(
            flood_type=self._flood_type,
            climate_scenario=self._climate_scenario,
            year=self._year,
            return_period=self._return_period,
            subsidence=self._subsidence,
            projection=self._projection,
            model=self._model,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_downloader(
        self,
        config: AqueductFloodConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> AqueductFloodDownloader:
        """Create and return an ``AqueductFloodDownloader`` instance."""
        return AqueductFloodDownloader(
            config=config,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_reader(
        self,
        config: AqueductFloodConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> AqueductFloodReader:
        """Create and return an ``AqueductFloodReader`` instance."""
        return AqueductFloodReader(
            config=config,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def load_data(
        self,
        source: Union[
            str,
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,
            List[Union[str, Path]],
        ],
        crop_to_source: bool = True,
        ensure_available: bool = True,
        **kwargs,
    ):
        return super().load_data(
            source=source,
            crop_to_source=crop_to_source,
            ensure_available=ensure_available,
            **kwargs,
        )

    def load_into_dataframe(
        self,
        source: Union[
            str,
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,
            List[Union[str, Path]],
        ],
        crop_to_source: bool = True,
        ensure_available: bool = True,
        min_value: float = 0,
        **kwargs,
    ):
        tif_processors = self.load_data(
            source=source,
            crop_to_source=crop_to_source,
            ensure_available=ensure_available,
            **kwargs,
        )

        if isinstance(tif_processors, TifProcessor):
            return tif_processors.to_dataframe(min_value=min_value, **kwargs)
        return pd.concat(
            [tp.to_dataframe(min_value=min_value, **kwargs) for tp in tif_processors],
            ignore_index=True,
        )

    def load_into_geodataframe(
        self,
        source: Union[
            str,
            BaseGeometry,
            gpd.GeoDataFrame,
            Path,
            List[Union[str, Path]],
        ],
        crop_to_source: bool = True,
        ensure_available: bool = True,
        min_value: float = 0,
        **kwargs,
    ):
        tif_processors = self.load_data(
            source=source,
            crop_to_source=crop_to_source,
            ensure_available=ensure_available,
            **kwargs,
        )

        if isinstance(tif_processors, TifProcessor):
            return tif_processors.to_geodataframe(min_value=min_value, **kwargs)
        return pd.concat(
            [
                tp.to_geodataframe(min_value=min_value, **kwargs)
                for tp in tif_processors
            ],
            ignore_index=True,
        )
