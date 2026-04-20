"""
AQUEDUCT Flood Hazard Handler for Giga Spatial
Integrates WRI AQUEDUCT 3.0 flood hazard layers with school locations.
Data source: World Resources Institute AQUEDUCT 3.0
URL: https://www.wri.org/aqueduct
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import rasterio
from rasterio.mask import mask
from pathlib import Path
from typing import Optional, Union, List, Literal
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from shapely.geometry import Point, mapping
import tempfile
import logging

logger = logging.getLogger(__name__)

RETURN_PERIODS = {
    "rp10":   "1-in-10 year flood (10% annual probability)",
    "rp25":   "1-in-25 year flood (4% annual probability)",
    "rp50":   "1-in-50 year flood (2% annual probability)",
    "rp100":  "1-in-100 year flood (1% annual probability)",
    "rp250":  "1-in-250 year flood (0.4% annual probability)",
    "rp500":  "1-in-500 year flood (0.2% annual probability)",
    "rp1000": "1-in-1000 year flood (0.1% annual probability)",
}

HAZARD_TYPES = {
    "inunriver": "Riverine flooding",
    "inuncoast": "Coastal flooding",
}


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AQUEDUCTFloodHandler:
    """
    Handler for WRI AQUEDUCT 3.0 flood hazard data.
    Extracts flood depth values at school locations for multiple
    return periods, enabling flood risk assessment.

    Parameters
    ----------
    return_periods : list of str
        Return periods. Options: rp10, rp25, rp50, rp100, rp250, rp500, rp1000
    hazard_type : str
        Flood type. Options: inunriver, inuncoast
    buffer_meters : int
        Buffer radius around each school in metres
    climate_scenario : str
        Climate scenario. Options: historical, rcp4p5, rcp8p5
    data_dir : str or Path
        Directory to store downloaded raster files

    Examples
    --------
    >>> handler = AQUEDUCTFloodHandler(
    ...     return_periods=["rp100", "rp1000"],
    ...     hazard_type="inunriver",
    ...     buffer_meters=500
    ... )
    >>> risk_df = handler.get_flood_risk(schools_df, id_col="school_id")
    """
    return_periods: List[str] = None
    hazard_type: Literal["inunriver", "inuncoast"] = "inunriver"
    buffer_meters: int = 500
    climate_scenario: Literal[
        "historical", "rcp4p5", "rcp8p5"
    ] = "historical"
    data_dir: Optional[Union[str, Path]] = None

    def __post_init__(self):
        if self.return_periods is None:
            self.return_periods = ["rp100"]
        for rp in self.return_periods:
            if rp not in RETURN_PERIODS:
                raise ValueError(
                    f"Invalid return period: {rp}. "
                    f"Valid options: {list(RETURN_PERIODS.keys())}"
                )
        if self.data_dir is None:
            self.data_dir = Path(tempfile.mkdtemp())
        else:
            self.data_dir = Path(self.data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"AQUEDUCTFloodHandler initialised: "
            f"{self.hazard_type} | {self.return_periods}"
        )

    def _get_filename(self, return_period: str) -> str:
        scenario_code = {
            "historical": "0", "rcp4p5": "4", "rcp8p5": "8"
        }[self.climate_scenario]
        return (f"{self.hazard_type}_{return_period}_"
                f"{scenario_code}_perc_50.tif")

    def _get_download_url(self, return_period: str) -> str:
        return (
            "https://wri-public-data.s3.amazonaws.com/"
            "resourcewatch/foo_024_aqueduct_global_flood_risk/"
            + self._get_filename(return_period)
        )

    def _download_raster(self, return_period: str) -> Path:
        filepath = self.data_dir / self._get_filename(return_period)
        if filepath.exists():
            return filepath
        r = requests.get(
            self._get_download_url(return_period),
            stream=True, timeout=60
        )
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return filepath

    def _extract_values(self, raster_path, schools_gdf, rp):
        values = []
        with rasterio.open(raster_path) as src:
            schools_proj = (
                schools_gdf.to_crs(src.crs)
                if schools_gdf.crs != src.crs
                else schools_gdf.copy()
            )
            schools_proj["geometry"] = schools_proj.geometry.buffer(
                self.buffer_meters / 111000
            )
            for _, school in schools_proj.iterrows():
                try:
                    out_image, _ = mask(
                        src, [mapping(school.geometry)], crop=True
                    )
                    data = out_image[0].astype(float)
                    if src.nodata:
                        data[data == src.nodata] = np.nan
                    val = np.nanmean(data)
                    values.append(
                        0.0 if np.isnan(val) else round(val, 3)
                    )
                except Exception:
                    values.append(0.0)
        return pd.Series(values, name=f"flood_depth_{rp}_m")

    def get_flood_risk(
        self,
        schools: Union[gpd.GeoDataFrame, pd.DataFrame],
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        id_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract flood hazard values for school locations.

        Parameters
        ----------
        schools : GeoDataFrame or DataFrame
            School locations
        lat_col : str
            Latitude column name
        lon_col : str
            Longitude column name
        id_col : str, optional
            School ID column to include in output

        Returns
        -------
        DataFrame
            School-level flood depth and risk scores
        """
        if isinstance(schools, pd.DataFrame):
            schools_gdf = gpd.GeoDataFrame(
                schools,
                geometry=[
                    Point(lon, lat) for lat, lon in
                    zip(schools[lat_col], schools[lon_col])
                ],
                crs="EPSG:4326"
            )
        else:
            schools_gdf = schools.copy()
            if schools_gdf.crs is None:
                schools_gdf = schools_gdf.set_crs("EPSG:4326")

        result = pd.DataFrame()
        if id_col and id_col in schools_gdf.columns:
            result[id_col] = schools_gdf[id_col].values
        result["longitude"] = schools_gdf.geometry.x.values
        result["latitude"]  = schools_gdf.geometry.y.values

        for rp in self.return_periods:
            try:
                path = self._download_raster(rp)
                result[f"flood_depth_{rp}_m"] =                     self._extract_values(path, schools_gdf, rp).values
            except Exception as e:
                logger.warning(f"Failed {rp}: {e}")
                result[f"flood_depth_{rp}_m"] = 0.0

        depth_cols = [
            c for c in result.columns
            if c.startswith("flood_depth_")
        ]
        if depth_cols:
            result["max_flood_depth_m"] =                 result[depth_cols].max(axis=1)
            result["flood_risk_score"] = (
                result["max_flood_depth_m"] /
                max(result["max_flood_depth_m"].max(), 0.01) * 100
            ).round(1)
            result["flood_risk_tier"] =                 result["flood_risk_score"].apply(self._assign_risk_tier)

        result["hazard_type"]      = self.hazard_type
        result["climate_scenario"] = self.climate_scenario
        result["return_periods"]   = str(self.return_periods)
        return result

    @staticmethod
    def _assign_risk_tier(score: float) -> str:
        if score >= 75:   return "CRITICAL"
        elif score >= 50: return "HIGH"
        elif score >= 25: return "MEDIUM"
        else:             return "LOW"

    def get_dataset_info(self) -> dict:
        """Return metadata about the AQUEDUCT dataset."""
        return {
            "name":          "WRI AQUEDUCT 3.0 Flood Hazard",
            "source":        "World Resources Institute",
            "url":           "https://www.wri.org/aqueduct",
            "hazard_type":   self.hazard_type,
            "description":   HAZARD_TYPES[self.hazard_type],
            "scenario":      self.climate_scenario,
            "return_periods": {
                rp: RETURN_PERIODS[rp]
                for rp in self.return_periods
            },
            "resolution":    "~1km (30 arc-seconds)",
            "coverage":      "Global",
            "units":         "Flood inundation depth (metres)",
            "license":       "CC BY 4.0",
            "citation": (
                "Ward et al. (2020). Aqueduct Floods Methodology. "
                "World Resources Institute, Washington, DC."
            )
        }

    def __repr__(self):
        return (
            f"AQUEDUCTFloodHandler("
            f"hazard_type={self.hazard_type!r}, "
            f"return_periods={self.return_periods!r}, "
            f"scenario={self.climate_scenario!r}, "
            f"buffer_meters={self.buffer_meters})"
        )
