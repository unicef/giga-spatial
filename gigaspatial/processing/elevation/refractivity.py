"""
ITU-R P.453-14 refractivity model for deriving the effective Earth radius
k-factor from geographic location and climate zone.

This module provides both standard (median) k-factors for general propagation
loss calculations, and worst-case (sub-refractive) k-factors required for
rigorous Line-of-Sight (LOS) Fresnel clearance validation.

References
----------
ITU-R P.453-14 (08/2019), "The radio refractive index: its formula and
refractivity data", Annex 1, §§ 2-3.
ITU-R P.530-18, "Propagation data and prediction methods required for the
design of terrestrial line-of-sight systems", § 2.2.
"""

from __future__ import annotations

from enum import Enum

from gigaspatial.config import config

logger = config.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Climate zones
# ---------------------------------------------------------------------------


class ClimateZone(str, Enum):
    """
    Broad ITU climate zones used for regional k-factor estimation.

    Attributes
    ----------
    TROPICAL : str
        Equatorial and tropical humid regions (roughly |lat| < 23°).
        High humidity drives strong negative refractivity gradients.
    SUBTROPICAL_DRY : str
        Subtropical arid/semi-arid regions (roughly 23° <= |lat| < 35°).
        Low humidity, weaker gradients.
    TEMPERATE : str
        Mid-latitude temperate regions (roughly 35° <= |lat| < 60°).
        Standard atmosphere conditions.
    POLAR : str
        High-latitude and polar regions (|lat| >= 60°).
        Very weak refractivity gradients, k close to 1.
    """

    TROPICAL = "tropical"
    SUBTROPICAL_DRY = "subtropical_dry"
    TEMPERATE = "temperate"
    POLAR = "polar"


# ---------------------------------------------------------------------------
# Regional ΔN₁ median values (N-units/km)
# ---------------------------------------------------------------------------
# Derived from ITU-R P.453-14 Figures 4-7.
# Positive values represent a decrease in N over 1 km altitude (lapse).

_ZONE_DN1_MEDIAN: dict[ClimateZone, float] = {
    ClimateZone.TROPICAL: 55.0,  # N-units/km -> k ≈ 1.24
    ClimateZone.SUBTROPICAL_DRY: 32.0,  # N-units/km -> k ≈ 1.25
    ClimateZone.TEMPERATE: 40.0,  # N-units/km -> k ≈ 4/3 (~1.33)
    ClimateZone.POLAR: 27.0,  # N-units/km -> k ≈ 1.20
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def classify_climate_zone(lat: float, lon: float) -> ClimateZone:
    """
    Classify a geographic coordinate into a broad ITU climate zone.

    Uses latitude thresholds as the primary discriminator, with a
    longitude-based refinement to distinguish subtropical arid regions
    from tropical humid ones.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees (-90 to +90).
    lon : float
        Longitude in decimal degrees (-180 to +180).

    Returns
    -------
    ClimateZone
        Estimated climate zone for the given location.
    """
    abs_lat = abs(lat)

    if abs_lat >= 60.0:
        return ClimateZone.POLAR

    if abs_lat >= 35.0:
        return ClimateZone.TEMPERATE

    if abs_lat >= 23.0:
        # Subtropical band: distinguish arid from humid
        # Arid corridors: Sahara/Middle East (lon 20°W-60°E),
        # Australian interior (lon 115°E-155°E),
        # Atacama/W. South America (lon 68°W-78°W, S hemi.)
        lon_norm = lon % 360  # normalise to 0-360
        sahara_middle_east = 340 <= lon_norm or lon_norm <= 60
        australian_interior = 115 <= lon_norm <= 155 and lat < 0
        atacama = 282 <= lon_norm <= 292 and lat < 0

        if sahara_middle_east or australian_interior or atacama:
            return ClimateZone.SUBTROPICAL_DRY
        return ClimateZone.TEMPERATE  # subtropical humid -> closer to temperate

    return ClimateZone.TROPICAL


def k_factor_from_dn1(dn1: float) -> float:
    """
    Compute the effective Earth radius k-factor from ΔN₁.

    Uses the ITU-R P.453-14 relationship:
        dN/dh = -ΔN₁ / 1 km
        k = 1 / (1 + a * dN/dh)

    Parameters
    ----------
    dn1 : float
        Refractivity decrease over first 1 km (positive, N-units/km).

    Returns
    -------
    float
        Effective Earth radius factor k (dimensionless).
        Returns 4/3 as fallback if result is non-physical.
    """
    dn_dh = -dn1 * 1e-6  # negative: N decreases with height
    denominator = 1.0 + _EARTH_RADIUS_KM * dn_dh

    # Protect against zero division or extreme super-refraction
    if denominator <= 0.001:
        logger.warning(
            "Extreme sub-refraction or division by zero prevented for ΔN₁=%.1f. "
            "Falling back to standard atmosphere k=4/3.",
            dn1,
        )
        return 4.0 / 3.0

    k = 1.0 / denominator
    return k if k > 0 else 4.0 / 3.0


def get_median_k_factor(lat: float, lon: float) -> float:
    """
    Estimate the median effective Earth radius k-factor for a location.

    For higher accuracy, install the ``itur`` package. This function will
    automatically delegate to it when available to use full digital maps.

    Parameters
    ----------
    lat, lon : float
        Coordinates in decimal degrees.

    Returns
    -------
    float
        Estimated median k-factor (p=50%) for the location.
    """
    try:
        return _get_k_factor_itur(lat, lon)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("itur k-factor lookup failed (%s); using fallback LUT.", exc)

    zone = classify_climate_zone(lat, lon)
    dn1 = _ZONE_DN1_MEDIAN[zone]
    return k_factor_from_dn1(dn1)


def get_worst_case_k_factor(path_length_km: float) -> float:
    """
    Estimate the worst-case (sub-refractive) k-factor for a link.

    Derived from ITU-R P.530 clearance criteria. Sub-refractive fading
    (Earth bulging) becomes significantly more severe and difficult to
    clear as path length increases.

    Parameters
    ----------
    path_length_km : float
        Total distance of the RF link in kilometers.

    Returns
    -------
    float
        Worst-case k-factor (k_e) for high-availability planning.
    """
    if path_length_km <= 15.0:
        return 0.8  # Short links are highly resilient to Earth bulge
    elif path_length_km <= 50.0:
        return 0.7  # Medium links require moderate bulge accommodation
    else:
        return 0.6  # Long PtP links must survive severe sub-refraction


def get_clearance_k_factors(
    lat: float, lon: float, path_length_km: float
) -> dict[str, float]:
    """
    Returns both the median and worst-case k-factors required for
    rigorous dual-state LOS clearance analysis.

    Parameters
    ----------
    lat, lon : float
        Coordinates of the path midpoint in decimal degrees.
    path_length_km : float
        Total distance of the RF link in kilometers.

    Returns
    -------
    dict
        Dictionary containing 'k_factor_median' and 'k_factor_min'.
    """
    return {
        "k_factor_median": get_median_k_factor(lat, lon),
        "k_factor_min": get_worst_case_k_factor(path_length_km),
    }


# ---------------------------------------------------------------------------
# Optional itur backend
# ---------------------------------------------------------------------------


def _get_k_factor_itur(lat: float, lon: float) -> float:
    """
    Compute k-factor using the ``itur`` package (ITU-R P.453 digital maps).
    """
    import itur  # noqa: PLC0415

    # p=50 -> median value
    dn1_quantity = itur.models.itu453.DN1(lat, lon, p=50)
    dn1_value = float(dn1_quantity.value)

    return k_factor_from_dn1(dn1_value)
