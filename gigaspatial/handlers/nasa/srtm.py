import pandas as pd
from pathlib import Path
from typing import List, Union, Iterable, Literal, Optional
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
import logging
import itertools
import requests

from pydantic.dataclasses import dataclass
from pydantic import Field, ConfigDict

from gigaspatial.handlers.base import (
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    BaseHandler,
)
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.config import config as global_config
from gigaspatial.processing.elevation.srtm_parser import SRTMParser
from gigaspatial.handlers.nasa.utils import EarthdataSession, LPDAACS3CredentialProvider


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class NasaSRTMConfig(BaseHandlerConfig):
    """
    Configuration for NASA SRTM .hgt tiles (30m or 90m).
    Creates tile geometries dynamically for 1°x1° grid cells.

    Each tile file covers 1 degree latitude x 1 degree longitude.
    """

    earthdata_username: str = Field(
        default=global_config.EARTHDATA_USERNAME, description="Earthdata Login username"
    )
    earthdata_password: str = Field(
        default=global_config.EARTHDATA_PASSWORD, description="Earthdata Login password"
    )

    https_base_url: str = Field(
        default="https://data.lpdaac.earthdatacloud.nasa.gov",
        description="LP DAAC Earthdata Cloud HTTPS base URL",
    )
    s3_region: str = Field(
        default="us-west-2",
        description="AWS region for LP DAAC SRTM (reserved for future direct-S3 access)",
    )
    s3_bucket: Literal["lp-prod-protected", "lp-prod-public"] = Field(
        default="lp-prod-protected",
        description="LP DAAC collection bucket path segment in tile URLs",
    )
    s3_credentials_endpoint: str = Field(
        default=LPDAACS3CredentialProvider.DEFAULT_ENDPOINT,
        description="LP DAAC endpoint for temporary S3 credentials (future direct-S3 use)",
    )

    # user config
    base_path: Path = global_config.get_path("nasa_srtm", "bronze")
    resolution: Literal["30m", "90m"] = "30m"

    def __post_init__(self):
        super().__post_init__()
        self._res_arc = 3 if self.resolution == "90m" else 1
        self._credential_provider: Optional[LPDAACS3CredentialProvider] = None
        self.session = EarthdataSession(
            username=self.earthdata_username,
            password=self.earthdata_password,
        )
        self._generate_tile_grid()

    def get_credential_provider(self) -> LPDAACS3CredentialProvider:
        """Return LP DAAC STS credential provider (reserved for future direct-S3 access)."""
        if self._credential_provider is None:
            self._credential_provider = LPDAACS3CredentialProvider(
                username=self.earthdata_username,
                password=self.earthdata_password,
                endpoint=self.s3_credentials_endpoint,
                region=self.s3_region,
            )
        return self._credential_provider

    def get_tile_url(self, tile_id: str) -> str:
        """Build the HTTPS URL for a single SRTM tile."""
        product = f"SRTMGL{self._res_arc}"
        folder = f"{tile_id}.{product}.hgt"
        filename = f"{folder}.zip"
        return (
            f"{self.https_base_url}/{self.s3_bucket}/{product}.003/{folder}/{filename}"
        )

    def _generate_tile_grid(self):
        """
        Generate 1°x1° grid polygons covering global extent.
        """

        lats = range(-90, 90)
        lons = range(-180, 180)

        grid_records = []
        for lat, lon in itertools.product(lats, lons):
            tile_name = self._tile_name(lat, lon)
            tile_url = self.get_tile_url(tile_name)
            grid_records.append(
                {
                    "tile_id": tile_name,
                    "geometry": box(lon, lat, lon + 1, lat + 1),
                    "tile_url": tile_url,
                    "tile_uri": tile_url,
                }
            )

        self.grid_records = grid_records
        self.tile_tree = STRtree([r.get("geometry") for r in grid_records])

    def _tile_name(self, lat: int, lon: int) -> str:
        """Return the SRTM tile name like N37E023 or S10W120."""
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"

    def get_relevant_data_units(self, source, force_recompute: bool = False, **kwargs):
        return super().get_relevant_data_units(
            source, force_recompute, crs="EPSG:4326", **kwargs
        )

    def get_relevant_data_units_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> List[dict]:
        mask = self.tile_tree.query(geometry, predicate="intersects")
        filtered_grid = [self.grid_records[i] for i in mask]

        return gpd.GeoDataFrame(filtered_grid, crs="EPSG:4326").to_dict("records")

    def get_data_unit_path(self, unit: Union[pd.Series, dict, str], **kwargs) -> Path:
        """
        Given a tile unit or tile_id, return expected storage path.
        """
        tile_id = unit["tile_id"] if isinstance(unit, (pd.Series, dict)) else unit
        return self.base_path / f"{tile_id}.SRTMGL{self._res_arc}.hgt.zip"

    def get_data_unit_paths(
        self, units: Union[pd.DataFrame, Iterable[Union[dict, str]]], **kwargs
    ) -> list:
        """
        Given tile identifiers, return list of file paths.
        """
        if isinstance(units, pd.DataFrame):
            return [
                self.get_data_unit_path(row, **kwargs) for _, row in units.iterrows()
            ]
        return super().get_data_unit_paths(units, **kwargs)


class NasaSRTMDownloader(BaseHandlerDownloader):
    """A class to handle downloads of NASA SRTM elevation data."""

    def __init__(
        self,
        config: Optional[NasaSRTMConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the downloader.

        Args:
            config: Optional configuration for customizing download behavior and file paths.
                    If None, a default `NasaSRTMConfig` is used.
            data_store: Optional instance of a `DataStore` for managing data storage.
                        If provided, it overrides the `data_store` in the `config`.
                        If None, the `data_store` from the `config` is used.
            logger: Optional custom logger instance. If None, a default logger
                    named after the module is created and used.
        """
        config = config or NasaSRTMConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

    def _http_request(self, method: str, url: str, **kwargs):
        """Issue an HTTP request using Earthdata auth or anonymous access."""
        if self.config.s3_bucket == "lp-prod-public":
            return requests.request(method, url, **kwargs)
        return self.config.session.request(method, url, **kwargs)

    def validate_unit_exists(self, tile_url: str) -> bool:
        """Check whether a tile exists at the remote HTTPS URL."""
        try:
            response = self._http_request("HEAD", tile_url, timeout=30, allow_redirects=True)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Error validating unit {tile_url}: {e}")
            return False

    def download_data_unit(
        self,
        tile_info: Union[pd.Series, dict],
        **kwargs,
    ) -> Optional[str]:
        """Download data file for a single SRTM tile."""

        tile_url = tile_info.get("tile_url") or tile_info.get("tile_uri")
        file_path = str(self.config.get_data_unit_path(tile_info))

        try:
            response = self._http_request(
                "GET", tile_url, stream=True, timeout=30, allow_redirects=True
            )
            response.raise_for_status()

            with self.data_store.open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192 * 1024):
                    if chunk:
                        file.write(chunk)

            self.logger.debug(
                f"Successfully downloaded tile: {tile_info['tile_id']} from {tile_url}"
            )
            return file_path

        except requests.RequestException as e:
            self.logger.error(
                f"Failed to download tile {tile_info['tile_id']} from {tile_url}: {e}"
            )
            if self.data_store.file_exists(file_path):
                self.data_store.remove(file_path)
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error downloading tile {tile_info['tile_id']}: {e}"
            )
            if self.data_store.file_exists(file_path):
                self.data_store.remove(file_path)
            return None


class NasaSRTMReader(BaseHandlerReader):
    """A class to handle reading of NASA SRTM elevation data."""

    def __init__(
        self,
        config: Optional[NasaSRTMConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the reader.

        Args:
            config: Optional configuration for customizing reading behavior and file paths.
                    If None, a default `NasaSRTMConfig` is used.
            data_store: Optional instance of a `DataStore` for managing data storage.
                        If provided, it overrides the `data_store` in the `config`.
                        If None, the `data_store` from the `config` is used.
            logger: Optional custom logger instance. If None, a default logger
                    named after the module is created and used.
        """
        config = config or NasaSRTMConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> Union[pd.DataFrame, List[SRTMParser]]:
        """
        Load SRTM elevation data from file paths.

        Args:
            source_data_path: List of SRTM .hgt.zip file paths
            **kwargs: Additional parameters for data loading
                - as_dataframe: bool, default=True. If True, return concatenated DataFrame.
                               If False, return list of SRTMParser objects.
                - dropna: bool, default=True. If True, drop rows with NaN elevation values.

        Returns:
            Union[pd.DataFrame, List[SRTMParser]]: Loaded elevation data
        """
        as_dataframe = kwargs.get("as_dataframe", True)
        dropna = kwargs.get("dropna", True)

        parsers = []
        for file_path in source_data_path:
            try:
                parser = SRTMParser(file_path, data_store=self.data_store)
                parsers.append(parser)
                self.logger.debug(f"Successfully loaded SRTM tile: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to load SRTM tile {file_path}: {str(e)}")
                continue

        if not parsers:
            self.logger.warning("No SRTM tiles could be loaded")
            return pd.DataFrame() if as_dataframe else []

        if as_dataframe:
            # Concatenate all tile dataframes
            dataframes = [parser.to_dataframe(dropna=dropna) for parser in parsers]
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                self.logger.info(
                    f"Loaded {len(combined_df)} elevation points from {len(parsers)} tiles"
                )
                return combined_df
            else:
                return pd.DataFrame()
        else:
            self.logger.info(f"Loaded {len(parsers)} SRTM tiles")
            return parsers


class NasaSRTMHandler(BaseHandler):
    """Main handler class for NASA SRTM elevation data."""

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> NasaSRTMConfig:
        """Create and return a NasaSRTMConfig instance."""
        return NasaSRTMConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(
        self,
        config: NasaSRTMConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> NasaSRTMDownloader:
        """Create and return a NasaSRTMDownloader instance."""
        return NasaSRTMDownloader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def create_reader(
        self,
        config: NasaSRTMConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> NasaSRTMReader:
        """Create and return a NasaSRTMReader instance."""
        return NasaSRTMReader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )
