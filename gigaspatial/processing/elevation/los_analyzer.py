from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import plotly.graph_objects as go

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.handlers.nasa.srtm import NasaSRTMDownloader
from gigaspatial.processing.elevation.srtm_manager import SRTMManager

logger = global_config.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LOSAnalyzerConfig(BaseModel):
    """
    Configuration for rigorous dual-state Line-of-Sight and Fresnel zone analysis.

    Validates infrastructure-grade links against both standard (median) atmospheric
    conditions and sub-refractive (worst-case) conditions to ensure high availability.

    Parameters
    ----------
    k_factor_median : float
        Effective Earth radius multiplier for median/standard atmospheric
        refraction (ITU-R standard atmosphere = 4/3 or ~1.333).
    fresnel_clearance_median : float
        Minimum required clearance expressed as a fraction of the Fresnel
        zone radius under median k-factor conditions. Typically 1.0 (100%).
    k_factor_min : float
        Effective Earth radius multiplier for worst-case sub-refractive
        conditions (Earth bulging). Typically 0.6 to 0.8 depending on path length.
    fresnel_clearance_min : float
        Minimum required clearance expressed as a fraction of the Fresnel
        zone radius under worst-case k-factor conditions. ITU-R P.530 recommends 0.6 (60%).
    apply_earth_curvature : bool
        Whether to apply Earth curvature + refraction correction to terrain heights.
    fresnel_zone_number : int
        Fresnel zone number to compute the radius for. Default is 1 (first Fresnel zone).
    """

    k_factor_median: float = Field(default=4.0 / 3.0, gt=0)
    fresnel_clearance_median: float = Field(default=1.0, ge=0.0, le=1.0)
    k_factor_min: float = Field(default=0.6, gt=0)
    fresnel_clearance_min: float = Field(default=0.6, ge=0.0, le=1.0)
    apply_earth_curvature: bool = True
    fresnel_zone_number: int = Field(default=1, ge=1)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LOSResult:
    """
    Result of a dual-state Line-of-Sight analysis between two points.

    Attributes
    ----------
    profile : pd.DataFrame
        Full per-sample profile. Contains base geometry, plus dual-state
        curvature corrections and clearance boundaries if frequency is provided.
    is_visual_los : bool
        True if the path has an unobstructed optical line of sight (ignoring
        Fresnel zones but accounting for curvature at the median k-factor).
    passes_median_clearance : bool or None
        True if the terrain clears the requested Fresnel fraction under
        median atmospheric conditions (k_factor_median). None if no frequency.
    passes_worst_case_clearance : bool or None
        True if the terrain clears the requested Fresnel fraction under
        worst-case sub-refraction (k_factor_min). None if no frequency.
    margin_median_m : float
        Minimum vertical margin (meters) between the terrain and the required
        median clearance boundary. Negative values indicate obstruction.
    margin_worst_case_m : float
        Minimum vertical margin (meters) between the terrain and the required
        worst-case clearance boundary. Negative values indicate obstruction.
    bottleneck_distance_median_km : float
        Distance from the transmitter where the median margin is minimized.
    bottleneck_distance_worst_case_km : float
        Distance from the transmitter where the worst-case margin is minimized.
    obstruction_count_worst_case : int
        Number of sample points where terrain breaches the worst-case clearance.
    knife_edge_loss_worst_case_db : float or None
        Estimated knife-edge diffraction loss (dB) at the worst-case bottleneck
        (ITU-R P.526 approximation). None if path clears or frequency is missing.
    tx_height_m : float
        Transmitter antenna height above ground used in the analysis.
    rx_height_m : float
        Receiver antenna height above ground used in the analysis.
    frequency_mhz : float or None
        Carrier frequency used in the analysis, if provided.
    config : LOSAnalyzerConfig
        The configuration used to produce this result.
    """

    profile: pd.DataFrame
    is_visual_los: bool
    passes_median_clearance: Optional[bool]
    passes_worst_case_clearance: Optional[bool]
    margin_median_m: float
    margin_worst_case_m: float
    bottleneck_distance_median_km: float
    bottleneck_distance_worst_case_km: float
    obstruction_count_worst_case: int
    knife_edge_loss_worst_case_db: Optional[float]
    tx_height_m: float
    rx_height_m: float
    frequency_mhz: Optional[float]
    config: LOSAnalyzerConfig

    @property
    def is_highly_available(self) -> bool:
        """
        Final verdict for infrastructure-grade deployments.
        Requires passing both standard median and worst-case sub-refractive checks.
        """
        if (
            self.passes_median_clearance is None
            or self.passes_worst_case_clearance is None
        ):
            return None

        return self.passes_median_clearance and self.passes_worst_case_clearance

    def summary(self) -> dict:
        """
        Return a concise summary dictionary of the dual-state analysis result.

        Returns
        -------
        dict
            Key metrics indicating path viability and critical bottlenecks.
        """
        return {
            "is_highly_available": self.is_highly_available,
            "is_visual_los": self.is_visual_los,
            "passes_median_clearance": self.passes_median_clearance,
            "passes_worst_case_clearance": self.passes_worst_case_clearance,
            "margin_median_m": round(self.margin_median_m, 2),
            "margin_worst_case_m": round(self.margin_worst_case_m, 2),
            "bottleneck_distance_worst_case_km": round(
                self.bottleneck_distance_worst_case_km, 3
            ),
            "obstruction_count_worst_case": self.obstruction_count_worst_case,
            "knife_edge_loss_worst_case_db": (
                round(self.knife_edge_loss_worst_case_db, 2)
                if self.knife_edge_loss_worst_case_db is not None
                else None
            ),
            "tx_height_m": self.tx_height_m,
            "rx_height_m": self.rx_height_m,
            "frequency_mhz": self.frequency_mhz,
        }

    def plot(self) -> go.Figure:
        """
        Generate an interactive 2D visualization of the path profile.

        Plots the effective terrain (including Earth curvature), the visual
        line of sight, the 1st Fresnel zone (if frequency was provided),
        and highlights any obstruction points.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive plot configured with a dark theme.
        """
        df = self.profile
        dist = df["distance_km"]
        terrain = df["effective_terrain_median_m"]
        los = df["los_height_m"]

        # 1. Initialize Figure with Dark Theme
        fig = go.Figure(
            layout=go.Layout(
                template="plotly_dark",
                title="Line of Sight & Fresnel Zone Analysis",
                xaxis_title="Distance from Transmitter (km)",
                yaxis_title="Elevation + Earth Bulge (m AMSL)",
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
        )

        # 2. Add Effective Terrain (Filled Area)
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=terrain,
                mode="lines",
                fill="tozeroy",
                name="Effective Terrain (Median k)",
                line=dict(color="#535c68", width=2),  # Slate grey
                fillcolor="rgba(83, 92, 104, 0.5)",
            )
        )

        # 3. Add Line of Sight (Dashed Line)
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=los,
                mode="lines",
                name="Visual Line of Sight",
                line=dict(color="#00d2d3", width=2, dash="dash"),  # Cyan
            )
        )

        # 4. Add Fresnel Zone and identify obstructions
        if self.frequency_mhz is not None:
            fresnel_radius = df["fresnel_radius_m"]

            # Upper Fresnel Boundary
            fig.add_trace(
                go.Scatter(
                    x=dist,
                    y=los + fresnel_radius,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Lower Fresnel Boundary (Filled to upper)
            fig.add_trace(
                go.Scatter(
                    x=dist,
                    y=los - fresnel_radius,
                    mode="lines",
                    fill="tonexty",
                    name="1st Fresnel Zone",
                    line=dict(width=0),
                    fillcolor="rgba(0, 210, 211, 0.15)",  # Transparent Cyan
                )
            )

            # Required Clearance Boundary (e.g., 60% of Fresnel)
            req_clearance_radius = fresnel_radius * self.config.fresnel_clearance_median
            fig.add_trace(
                go.Scatter(
                    x=dist,
                    y=los - req_clearance_radius,
                    mode="lines",
                    name=f"Req. Clearance ({self.config.fresnel_clearance_median*100}%)",
                    line=dict(color="#feca57", width=1, dash="dot"),  # Warning Yellow
                )
            )

            # Obstructions are where terrain breaks the required Fresnel margin
            obstruction_mask = df["fresnel_margin_median_m"] < 0
        else:
            # Obstructions are purely visual LOS blockages
            obstruction_mask = df["clearance_median_m"] < 0

        # 5. Mark Obstructions (Red Scatter Points)
        if obstruction_mask.any():
            obs_points = df[obstruction_mask]
            fig.add_trace(
                go.Scatter(
                    x=obs_points["distance_km"],
                    y=obs_points["effective_terrain_median_m"],
                    mode="markers",
                    name="Obstruction Points",
                    marker=dict(color="#ff6b6b", size=6, symbol="cross"),  # Soft Red
                    hoverinfo="skip",
                )
            )

        # 6. Adjust Axes for visual scaling
        min_y = max(0, terrain.min() - 50)
        max_y = max(los.max(), terrain.max()) + 50
        fig.update_yaxes(range=[min_y, max_y])

        return fig


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------


class LOSAnalyzer:
    """
    Line-of-Sight and Fresnel zone analyzer for radio link paths.

    Combines SRTM elevation data access with geometric LOS analysis,
    Fresnel zone radii, Earth curvature correction, and knife-edge
    diffraction estimation.

    The analyzer is stateless with respect to per-link parameters: antenna
    heights and frequency are passed at call time to ``analyze()`` or
    ``analyze_profile()``, so a single instance can be reused efficiently
    across many school-to-tower pairs.

    Parameters
    ----------
    srtm_directory : str or Path
        Directory containing SRTM ``.hgt.zip`` files.
    downloader : NasaSRTMDownloader, optional
        Downloader instance for auto-downloading missing tiles.
    cache_size : int, default=10
        Maximum number of SRTM tiles to keep in memory (LRU cache).
    data_store : DataStore, optional
        Data store for reading files. See ``SRTMManager`` for priority rules.
    config : LOSAnalyzerConfig, optional
        Analysis configuration. If omitted, a default config is created.
    **config_kwargs
        Keyword arguments forwarded to ``LOSAnalyzerConfig`` when no
        ``config`` object is supplied (e.g. ``effective_earth_factor=1.0``).

    Examples
    --------
    Standard usage - analyzer owns its SRTM manager:

    >>> from gigaspatial.processing.elevation.los_analyzer import (
    ...     LOSAnalyzer, LOSAnalyzerConfig
    ... )
    >>>
    >>> analyzer = LOSAnalyzer(
    ...     srtm_directory="/data/srtm",
    ...     apply_earth_curvature=True,
    ... )
    >>>
    >>> # Coordinate-based: profile is fetched internally
    >>> result = analyzer.analyze(
    ...     start_lat=36.8, start_lon=30.7,
    ...     end_lat=36.9,   end_lon=30.9,
    ...     tx_height_m=30.0,
    ...     rx_height_m=15.0,
    ...     frequency_mhz=5800.0,
    ... )
    >>> print(result.summary())
    >>>
    >>> # Profile-based: pass a pre-fetched profile directly
    >>> profile = analyzer.manager.get_elevation_profile(
    ...     36.8, 30.7, 36.9, 30.9
    ... )
    >>> result = analyzer.analyze_profile(
    ...     profile,
    ...     tx_height_m=30.0,
    ...     rx_height_m=15.0,
    ...     frequency_mhz=5800.0,
    ... )
    >>>
    >>> # Reuse across many links
    >>> for school, tower in school_tower_pairs:
    ...     result = analyzer.analyze(
    ...         school.lat, school.lon,
    ...         tower.lat, tower.lon,
    ...         tx_height_m=tower.antenna_height_m,
    ...         rx_height_m=school.antenna_height_m,
    ...         frequency_mhz=tower.frequency_mhz,
    ...     )
    ...     print(result.summary())
    >>>
    >>> # Inject a pre-configured SRTMManager (e.g. shared across analyzers)
    >>> from gigaspatial.processing.elevation.srtm_manager import SRTMManager
    >>> manager = SRTMManager("/data/srtm", cache_size=50)
    >>> analyzer = LOSAnalyzer.from_manager(manager)
    """

    EARTH_RADIUS_KM = 6371.0

    def __init__(
        self,
        srtm_directory: Union[str, Path] = global_config.get_path(
            "nasa_srtm", "bronze"
        ),
        downloader: Optional[NasaSRTMDownloader] = None,
        cache_size: int = 10,
        data_store: Optional[DataStore] = None,
        config: Optional[LOSAnalyzerConfig] = None,
        **config_kwargs,
    ):
        self.manager = SRTMManager(
            srtm_directory=srtm_directory,
            downloader=downloader,
            cache_size=cache_size,
            data_store=data_store,
        )
        self.config = config or LOSAnalyzerConfig(**config_kwargs)
        logger.debug("LOSAnalyzer initialized with config: %s", self.config)

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_manager(
        cls,
        manager: SRTMManager,
        config: Optional[LOSAnalyzerConfig] = None,
        **config_kwargs,
    ) -> "LOSAnalyzer":
        """
        Construct a ``LOSAnalyzer`` from an existing ``SRTMManager`` instance.

        Use this when the manager is already configured and potentially shared
        across multiple analyzers or other components.

        Parameters
        ----------
        manager : SRTMManager
            Initialized SRTM manager instance.
        config : LOSAnalyzerConfig, optional
            Analysis configuration. If omitted, built from ``config_kwargs``.
        **config_kwargs
            Forwarded to ``LOSAnalyzerConfig`` when no ``config`` is supplied.

        Returns
        -------
        LOSAnalyzer
        """
        instance = cls.__new__(cls)
        instance.manager = manager
        instance.config = config or LOSAnalyzerConfig(**config_kwargs)
        logger.debug(
            "LOSAnalyzer created from existing manager with config: %s",
            instance.config,
        )
        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        tx_height_m: float = 10.0,
        rx_height_m: float = 10.0,
        frequency_mhz: Optional[float] = None,
        num_points: int = 200,
    ) -> LOSResult:
        """
        Fetch an elevation profile and perform LOS analysis in one call.

        Convenience wrapper around ``analyze_profile()`` for the common case
        where coordinates are known but no profile has been pre-fetched.

        Parameters
        ----------
        start_lat : float
            Transmitter latitude in decimal degrees.
        start_lon : float
            Transmitter longitude in decimal degrees.
        end_lat : float
            Receiver latitude in decimal degrees.
        end_lon : float
            Receiver longitude in decimal degrees.
        tx_height_m : float, default=10.0
            Transmitter antenna height above ground in meters.
        rx_height_m : float, default=10.0
            Receiver antenna height above ground in meters.
        frequency_mhz : float, optional
            Carrier frequency in MHz. Enables Fresnel zone and diffraction
            output when provided.
        num_points : int, default=200
            Number of sample points along the path.

        Returns
        -------
        LOSResult
        """
        profile = self.manager.get_elevation_profile(
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            num_points=num_points,
        )
        return self.analyze_profile(
            profile=profile,
            tx_height_m=tx_height_m,
            rx_height_m=rx_height_m,
            frequency_mhz=frequency_mhz,
        )

    def analyze_profile(
        self,
        profile: pd.DataFrame,
        tx_height_m: float = 10.0,
        rx_height_m: float = 10.0,
        frequency_mhz: Optional[float] = None,
    ) -> LOSResult:
        """
        Perform dual-state LOS and Fresnel zone analysis on an elevation profile.

        Evaluates the path against both median atmospheric conditions and
        worst-case sub-refractive conditions simultaneously to ensure
        high-availability infrastructure planning.

        Parameters
        ----------
        profile : pd.DataFrame
            DataFrame with columns: ``distance_km``, ``latitude``,
            ``longitude``, ``elevation``.
        tx_height_m : float, default=10.0
            Transmitter antenna height above ground in meters.
        rx_height_m : float, default=10.0
            Receiver antenna height above ground in meters.
        frequency_mhz : float, optional
            Carrier frequency in MHz. Required to calculate Fresnel zones and margins.

        Returns
        -------
        LOSResult
            Comprehensive dual-state analysis result.

        Notes
        -----
        Output profile columns
        ~~~~~~~~~~~~~~~~~~~~~~
        ==============================  =============================================================
        Column                          Description
        ==============================  =============================================================
        distance_km                     Cumulative distance from TX (km)
        elevation                       Raw SRTM terrain elevation (m)
        los_height_m                    LOS visual straight-line height at this point (m AMSL)
        earth_correction_median_m       Earth bulge added to terrain at median k (m)
        earth_correction_worst_case_m   Earth bulge added to terrain at worst-case k (m)
        effective_terrain_median_m      elevation + earth_correction_median_m
        effective_terrain_worst_case_m  elevation + earth_correction_worst_case_m
        clearance_median_m              los_height_m − effective_terrain_median_m
        clearance_worst_case_m          los_height_m − effective_terrain_worst_case_m
        fresnel_radius_m                First Fresnel zone radius (m) [if freq given]
        fresnel_margin_median_m         Vertical gap between terrain and median req. boundary (m)
        fresnel_margin_worst_case_m     Vertical gap between terrain and worst-case req. boundary (m)
        ==============================  =============================================================
        """
        self._validate_profile(profile)

        profile = profile.copy()
        distances_km = profile["distance_km"].values
        elevations = profile["elevation"].values
        total_distance_km = float(distances_km[-1])

        # Distances from TX (d1) and RX (d2)
        d1 = distances_km
        d2 = total_distance_km - distances_km

        logger.info(
            "Analyzing LOS for %.3f km path (%d points), "
            "tx_height=%.1f m, rx_height=%.1f m, freq=%s MHz",
            total_distance_km,
            len(profile),
            tx_height_m,
            rx_height_m,
            frequency_mhz,
        )

        # 1. Absolute antenna heights
        tx_abs = elevations[0] + tx_height_m
        rx_abs = elevations[-1] + rx_height_m

        # 2. LOS visual line height at each point (Linear interpolation)
        t = distances_km / total_distance_km
        los_height = tx_abs + t * (rx_abs - tx_abs)
        profile["los_height_m"] = los_height

        # 3. Dual Earth curvature corrections
        earth_radius_km = 6371.0
        if self.config.apply_earth_curvature:
            bulge_median = self._earth_curvature_correction(
                distances_km, total_distance_km, self.config.k_factor_median
            )
            bulge_worst = self._earth_curvature_correction(
                distances_km, total_distance_km, self.config.k_factor_min
            )
        else:
            bulge_median = bulge_worst = np.zeros_like(distances_km)

        profile["earth_correction_median_m"] = bulge_median
        profile["earth_correction_worst_case_m"] = bulge_worst

        # 4. Effective terrain
        eff_terrain_median = elevations + bulge_median
        eff_terrain_worst = elevations + bulge_worst
        profile["effective_terrain_median_m"] = eff_terrain_median
        profile["effective_terrain_worst_case_m"] = eff_terrain_worst

        # 5. Base Clearances (Distance from LOS straight line to effective terrain)
        clearance_median = los_height - eff_terrain_median
        clearance_worst = los_height - eff_terrain_worst
        profile["clearance_median_m"] = clearance_median
        profile["clearance_worst_case_m"] = clearance_worst

        # Visual LOS relies purely on the median clearance > 0
        # We slice [1:-1] to ignore the exact TX and RX mast locations
        is_visual_los = bool((clearance_median[1:-1] >= 0.0).all())

        # 6. Optional RF Fresnel Analysis
        if frequency_mhz is not None:
            # Fresnel Radius Calculation
            freq_ghz = frequency_mhz / 1000.0
            # Note: np.sqrt safely handles d1*d2 == 0 at endpoints
            fresnel_radii = 17.32 * np.sqrt((d1 * d2) / (freq_ghz * total_distance_km))

            # Apply zone multiplier if looking at a zone other than 1
            if self.config.fresnel_zone_number != 1:
                fresnel_radii *= np.sqrt(self.config.fresnel_zone_number)

            profile["fresnel_radius_m"] = fresnel_radii

            # Required clearance boundaries (Fraction of Fresnel Zone)
            req_clearance_median = fresnel_radii * self.config.fresnel_clearance_median
            req_clearance_worst = fresnel_radii * self.config.fresnel_clearance_min

            # Vertical margin: > 0 means clearing the boundary. < 0 means obstruction.
            margin_median = clearance_median - req_clearance_median
            margin_worst = clearance_worst - req_clearance_worst
            profile["fresnel_margin_median_m"] = margin_median
            profile["fresnel_margin_worst_case_m"] = margin_worst

            # Calculate pass/fail using the internal span [1:-1]
            passes_median = bool((margin_median[1:-1] >= 0.0).all())
            passes_worst = bool((margin_worst[1:-1] >= 0.0).all())

            # Bottleneck identification
            min_margin_median = float(np.min(margin_median[1:-1]))
            min_margin_worst = float(np.min(margin_worst[1:-1]))

            # np.argmin on [1:-1] shifts index by -1, add 1 to map back to original array
            bottleneck_idx_median = int(np.argmin(margin_median[1:-1])) + 1
            bottleneck_idx_worst = int(np.argmin(margin_worst[1:-1])) + 1

            obstruction_count_worst = int((margin_worst[1:-1] < 0.0).sum())

        else:
            passes_median = None
            passes_worst = None
            min_margin_median = float(np.min(clearance_median[1:-1]))
            min_margin_worst = float(np.min(clearance_worst[1:-1]))

            bottleneck_idx_median = int(np.argmin(clearance_median[1:-1])) + 1
            bottleneck_idx_worst = int(np.argmin(clearance_worst[1:-1])) + 1

            obstruction_count_worst = int((clearance_worst[1:-1] < 0.0).sum())

        # Extract bottleneck distances
        bottleneck_dist_median_km = float(distances_km[bottleneck_idx_median])
        bottleneck_dist_worst_km = float(distances_km[bottleneck_idx_worst])

        # 7. Knife-edge diffraction loss (calculated at worst-case physical bottleneck)
        knife_edge_loss_db = None
        if passes_worst is False and frequency_mhz is not None:
            # ITU-R P.526 expects the height of the obstacle *above the LOS straight line*
            h_obstruction_m = -clearance_worst[bottleneck_idx_worst]
            wavelength_m = 300.0 / frequency_mhz
            knife_edge_loss_db = self._knife_edge_loss(
                h_obstruction_m=h_obstruction_m,
                d1_km=bottleneck_dist_worst_km,
                d2_km=total_distance_km - bottleneck_dist_worst_km,
                wavelength_m=wavelength_m,
            )

        logger.info(
            "LOS analysis complete: passes_median=%s, passes_worst=%s, "
            "worst_margin=%.1f m, worst_bottleneck_dist=%.2f km",
            passes_median,
            passes_worst,
            min_margin_worst,
            bottleneck_dist_worst_km,
        )

        return LOSResult(
            profile=profile,
            is_visual_los=is_visual_los,
            passes_median_clearance=passes_median,
            passes_worst_case_clearance=passes_worst,
            margin_median_m=min_margin_median,
            margin_worst_case_m=min_margin_worst,
            bottleneck_distance_median_km=bottleneck_dist_median_km,
            bottleneck_distance_worst_case_km=bottleneck_dist_worst_km,
            obstruction_count_worst_case=obstruction_count_worst,
            knife_edge_loss_worst_case_db=knife_edge_loss_db,
            tx_height_m=tx_height_m,
            rx_height_m=rx_height_m,
            frequency_mhz=frequency_mhz,
            config=self.config,
        )

    # ------------------------------------------------------------------
    # Geometric helpers
    # ------------------------------------------------------------------

    def _earth_curvature_correction(
        self,
        distances_km: np.ndarray,
        total_distance_km: float,
        k_factor: float,
    ) -> np.ndarray:
        """
        Compute per-point Earth bulge correction in meters for a specific k-factor.

        The terrain effectively rises relative to a flat visual LOS line due to
        Earth's curvature and atmospheric refraction.

        Formula:
            h_curve(d) = (d * (D - d)) / (2 * k * Re) * 1000   [meters]

        where:
        - d  : distance from TX (km)
        - D  : total path length (km)
        - k  : effective Earth radius factor (dimensionless)
        - Re : mean Earth radius (km) -> self.EARTH_RADIUS_KM

        Parameters
        ----------
        distances_km : np.ndarray
            Cumulative distances from TX in km.
        total_distance_km : float
            Total path length in km.
        k_factor : float
            The specific atmospheric k-factor to evaluate (e.g., 1.333 or 0.6).

        Returns
        -------
        np.ndarray
            Earth curvature correction in meters per sample point.
            All zeros when ``apply_earth_curvature`` is False.
        """
        if not self.config.apply_earth_curvature:
            return np.zeros_like(distances_km)

        d = distances_km
        D = total_distance_km
        Re = self.EARTH_RADIUS_KM

        # Division safely handled as k_factor and Re are constrained to be > 0
        return (d * (D - d)) / (2.0 * k_factor * Re) * 1000.0

    @staticmethod
    def _fresnel_radius(
        distances_km: np.ndarray,
        total_distance_km: float,
        wavelength_m: float,
        zone_number: int = 1,
    ) -> np.ndarray:
        """
        Compute the nth Fresnel zone radius at each point along the path.

            r_n(d) = sqrt(n * lambda * d1 * d2 / (d1 + d2))

        where d1 = distance from TX (m), d2 = distance from RX (m).

        Parameters
        ----------
        distances_km : np.ndarray
            Cumulative distances from TX in km.
        total_distance_km : float
            Total path length in km.
        wavelength_m : float
            Carrier wavelength in meters.
        zone_number : int
            Fresnel zone number (1 = first zone).

        Returns
        -------
        np.ndarray
            Fresnel zone radius in meters. Zero at TX and RX endpoints.
        """
        d1 = distances_km * 1000.0
        d2 = (total_distance_km - distances_km) * 1000.0

        denominator = d1 + d2
        safe_denom = np.where(denominator > 0, denominator, np.inf)

        return np.sqrt(zone_number * wavelength_m * d1 * d2 / safe_denom)

    @staticmethod
    def _knife_edge_loss(
        h_obstruction_m: float,
        d1_km: float,
        d2_km: float,
        wavelength_m: float,
    ) -> float:
        """
        Estimate knife-edge diffraction loss using the Fresnel-Kirchhoff
        diffraction parameter ν (nu).

            nu = h * sqrt(2 * (d1 + d2) / (lambda * d1 * d2))

        Loss J(nu) approximated per ITU-R P.526 piecewise formula:

        ============  ======================================================
        ν range       J(ν) formula
        ============  ======================================================
        ν ≤ −0.7      0 dB
        −0.7 < ν ≤ 0  20·log10(0.5 − 0.62·ν)
        0 < ν ≤ 1     20·log10(0.5·exp(−0.95·ν))
        1 < ν ≤ 2.4   20·log10(0.4 − √(0.1184 − (0.38 − 0.1·ν)²))
        ν > 2.4       20·log10(0.225 / ν)
        ============  ======================================================

        Parameters
        ----------
        h_obstruction_m : float
            Height of the obstruction above the LOS line in meters.
            Positive = above LOS (blocking), negative = below (clear).
        d1_km : float
            Distance from TX to the obstruction point in km.
        d2_km : float
            Distance from the obstruction point to RX in km.
        wavelength_m : float
            Carrier wavelength in meters.

        Returns
        -------
        float
            Estimated diffraction loss in dB (positive = attenuation).
            Returns 0.0 if geometry is degenerate.
        """
        d1 = d1_km * 1000.0
        d2 = d2_km * 1000.0

        if d1 <= 0 or d2 <= 0 or wavelength_m <= 0:
            return 0.0

        nu = h_obstruction_m * np.sqrt(2.0 * (d1 + d2) / (wavelength_m * d1 * d2))

        if nu <= -0.7:
            return 0.0
        elif nu <= 0.0:
            return float(20.0 * np.log10(0.5 - 0.62 * nu))
        elif nu <= 1.0:
            return float(20.0 * np.log10(0.5 * np.exp(-0.95 * nu)))
        elif nu <= 2.4:
            inner = 0.1184 - (0.38 - 0.1 * nu) ** 2
            return float(20.0 * np.log10(0.4 - np.sqrt(max(inner, 0.0))))
        else:
            return float(20.0 * np.log10(0.225 / nu))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_profile(profile: pd.DataFrame) -> None:
        """
        Validate that the input profile DataFrame has the required columns
        and sufficient rows.

        Parameters
        ----------
        profile : pd.DataFrame
            Input elevation profile to validate.

        Raises
        ------
        ValueError
            If required columns are missing or the profile has fewer than
            2 rows.
        """
        required = {"distance_km", "latitude", "longitude", "elevation"}
        missing = required - set(profile.columns)
        if missing:
            raise ValueError(
                f"Profile is missing required columns: {missing}. "
                "Expected output from SRTMManager.get_elevation_profile()."
            )
        if len(profile) < 2:
            raise ValueError("Profile must contain at least 2 sample points.")

    def __repr__(self) -> str:
        return (
            f"LOSAnalyzer("
            f"earth_curve={self.config.apply_earth_curvature}, "
            f"k_median={self.config.k_factor_median:.3f}, "
            f"k_min={self.config.k_factor_min:.3f}, "
            f"fresnel_zone={self.config.fresnel_zone_number}, "
            f"manager={self.manager!r})"
        )
