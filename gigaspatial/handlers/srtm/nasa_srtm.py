import numpy as np
import pandas as pd
from pathlib import Path
import functools
import multiprocessing
from typing import List, Union, Iterable, Literal, Optional, Tuple
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
import logging
import itertools
import requests
from tqdm import tqdm

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
from gigaspatial.handlers.srtm.srtm_parser import SRTMParser
from gigaspatial.handlers.srtm.utils import EarthdataSession


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

    BASE_URL: str = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL{}.003/2000.02.11/"

    # user config
    base_path: Path = global_config.get_path("nasa_srtm", "bronze")
    resolution: Literal["30m", "90m"] = "30m"

    def __post_init__(self):
        super().__post_init__()
        self._res_arc = 3 if self.resolution == "90m" else 1
        self.BASE_URL = self.BASE_URL.format(self._res_arc)
        # self.session = self._setup_earthdata_session()
        self.session = self._create_authenticated_session()
        self._generate_tile_grid()

    def _create_authenticated_session(self) -> requests.Session:
        """
        Create a persistent Earthdata-authenticated requests session
        that keeps Authorization headers through redirects.
        """
        logging.info("Setting up Earthdata session with header redirection...")

        session = EarthdataSession(
            username=self.earthdata_username,
            password=self.earthdata_password,
        )

        # Optionally verify credentials once (to pre-authenticate cookies)
        auth_test = "https://urs.earthdata.nasa.gov"
        try:
            r = session.get(auth_test, timeout=10)
            logging.debug(f"Earthdata auth test status: {r.status_code}")
        except requests.RequestException as e:
            logging.warning(f"Earthdata auth test failed: {e}")

        return session

    def _generate_tile_grid(self):
        """
        Generate 1°x1° grid polygons covering global extent.
        """

        lats = range(-90, 90)
        lons = range(-180, 180)

        grid_records = []
        for lat, lon in itertools.product(lats, lons):
            tile_name = self._tile_name(lat, lon)
            grid_records.append(
                {
                    "tile_id": tile_name,
                    "geometry": box(lon, lat, lon + 1, lat + 1),
                    "tile_url": f"{self.BASE_URL}/{tile_name}.SRTMGL{self._res_arc}.hgt.zip",
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

    def download_data_unit(
        self,
        tile_info: Union[pd.Series, dict],
        **kwargs,
    ) -> Optional[str]:
        """Download data file for a single SRTM tile."""

        tile_url = tile_info["tile_url"]

        try:
            response = self.config.session.get(tile_url, stream=True)
            response.raise_for_status()

            file_path = str(self.config.get_data_unit_path(tile_info))

            with self.data_store.open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

                self.logger.debug(
                    f"Successfully downloaded tile: {tile_info['tile_id']}"
                )
                return file_path

        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Failed to download tile {tile_info['tile_id']}: {str(e)}"
            )
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading dataset: {str(e)}")
            return None

    def download_data_units(
        self,
        tiles: Union[pd.DataFrame, List[dict]],
        **kwargs,
    ) -> List[str]:
        """Download data files for multiple SRTM tiles."""

        if len(tiles) == 0:
            self.logger.warning(f"There is no matching data")
            return []

        with multiprocessing.Pool(self.config.n_workers) as pool:
            download_func = functools.partial(self.download_data_unit)
            file_paths = list(
                tqdm(
                    pool.imap(
                        download_func,
                        (
                            [row for _, row in tiles.iterrows()]
                            if isinstance(tiles, pd.DataFrame)
                            else tiles
                        ),
                    ),
                    total=len(tiles),
                    desc=f"Downloading SRTM elevation data",
                )
            )

        return [path for path in file_paths if path is not None]

    # def download(
    #     self,
    #     source: Union[
    #         str,  # country
    #         List[Union[Tuple[float, float], Point]],  # points
    #         BaseGeometry,  # shapely geoms
    #         gpd.GeoDataFrame,
    #     ],
    #     **kwargs,
    # ) -> List[str]:
    #     """
    #     Download NASA SRTM elevation data for a specified geographic region.

    #     The region can be defined by a country, a list of points,
    #     a Shapely geometry, or a GeoDataFrame. This method identifies the
    #     relevant data tiles intersecting the region and downloads them in parallel.

    #     Args:
    #         source: Defines the geographic area for which to download data.
    #                 Can be:
    #                   - A string representing a country code or name.
    #                   - A list of (latitude, longitude) tuples or Shapely Point objects.
    #                   - A Shapely BaseGeometry object (e.g., Polygon, MultiPolygon).
    #                   - A GeoDataFrame with a geometry column in EPSG:4326.
    #         **kwargs: Additional parameters passed to data unit resolution methods

    #     Returns:
    #         A list of local file paths for the successfully downloaded tiles.
    #         Returns an empty list if no data is found for the region or if
    #         all downloads fail.
    #     """

    #     tiles = self.config.get_relevant_data_units(source, **kwargs)
    #     return self.download_data_units(tiles, **kwargs)


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
