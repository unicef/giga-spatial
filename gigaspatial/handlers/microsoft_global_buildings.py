from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from pathlib import Path
import functools
import multiprocessing
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, MultiPolygon
from difflib import SequenceMatcher
import pycountry
import requests
from tqdm import tqdm
import logging

from gigaspatial.core.io.readers import *
from gigaspatial.core.io.writers import *
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.grid.mercator_tiles import (
    MercatorTiles,
    CountryMercatorTiles,
)
from gigaspatial.config import config


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MSBuildingsConfig:
    """Configuration for Microsoft Global Buildings dataset files."""

    data_store: DataStore = field(
        default_factory=LocalDataStore
    )  # instance of DataStore for accessing data storage
    BASE_PATH: Path = config.get_path("microsoft_global_buildings", "bronze")

    TILE_URLS: str = (
        "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
    )
    MERCATOR_ZOOM_LEVEL: int = 9

    LOCATION_MAPPING_FILE: Path = BASE_PATH / "location_mapping.json"
    SIMILARITY_SCORE: float = 0.8
    DEFAULT_MAPPING: Dict[str, str] = field(
        default_factory=lambda: {
            "Bonaire": "BES",
            "Brunei": "BRN",
            "IvoryCoast": "CIV",
            "CongoDRC": "COD",
            "DemocraticRepublicoftheCongo": "COD",
            "RepublicoftheCongo": "COG",
            "TheGambia": "GMB",
            "FYROMakedonija": "MKD",
            "SultanateofOman": "OMN",
            "StateofQatar": "QAT",
            "Russia": "RUS",
            "KingdomofSaudiArabia": "SAU",
            "Svalbard": "SJM",
            "Swaziland": "SWZ",
            "StMartin": "SXM",
            "leSaint-Martin": "MAF",
            "Turkey": "TUR",
            "VaticanCity": "VAT",
            "BritishVirginIslands": "VGB",
            "USVirginIslands": "VIR",
            "RepublicofYemen": "YEM",
            "CzechRepublic": "CZE",
            "French-Martinique": "MTQ",
            "French-Guadeloupe": "GLP",
            "UnitedStates": "USA",
        }
    )
    CUSTOM_MAPPING: Optional[Dict[str, str]] = None

    n_workers: int = 4  # number of workers for parallel processing
    logger: Optional[logging.Logger] = config.get_logger(__name__)

    def __post_init__(self):
        """Validate inputs and set location mapping"""

        self._load_tile_urls()

        self.upload_date = self.df_tiles.upload_date[0]

        if self.data_store.file_exists(str(self.LOCATION_MAPPING_FILE)):
            self.location_mapping = read_json(
                self.data_store, str(self.LOCATION_MAPPING_FILE)
            )
        else:
            self.location_mapping = self.create_location_mapping(
                similarity_score_threshold=self.SIMILARITY_SCORE
            )
            self.location_mapping.update(self.DEFAULT_MAPPING)
            write_json(
                self.location_mapping, self.data_store, str(self.LOCATION_MAPPING_FILE)
            )

        self.location_mapping.update(self.CUSTOM_MAPPING or {})

        self._map_locations()

    def _load_tile_urls(self):
        """Load dataset links from csv file."""
        self.df_tiles = pd.read_csv(
            self.TILE_URLS,
            names=["location", "quadkey", "url", "size", "upload_date"],
            dtype={"quadkey": str},
            header=0,
        )

    def _map_locations(self):
        self.df_tiles["country"] = self.df_tiles.location.map(self.location_mapping)

    def create_location_mapping(self, similarity_score_threshold: float = 0.8):

        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        location_mapping = dict()

        for country in pycountry.countries:
            if country.name not in self.df_tiles.location.unique():
                try:
                    country_quadkey = CountryMercatorTiles.create(
                        country.alpha_3, self.MERCATOR_ZOOM_LEVEL
                    )
                except:
                    self.logger.warning(f"{country.name} is not mapped.")
                    continue
                country_datasets = country_quadkey.filter_quadkeys(
                    self.df_tiles.quadkey
                )
                matching_locations = self.df_tiles[
                    self.df_tiles.quadkey.isin(country_datasets.quadkeys)
                ].location.unique()
                scores = np.array(
                    [
                        (
                            similar(c, country.common_name)
                            if hasattr(country, "common_name")
                            else similar(c, country.name)
                        )
                        for c in matching_locations
                    ]
                )
                if any(scores > similarity_score_threshold):
                    matched = matching_locations[scores > similarity_score_threshold]
                    if len(matched) > 2:
                        self.logger.warning(
                            f"Multiple matches exist for {country.name}. {country.name} is not mapped."
                        )
                    location_mapping[matched[0]] = country.alpha_3
                    self.logger.debug(f"{country.name} matched with {matched[0]}!")
                else:
                    self.logger.warning(
                        f"No direct matches for {country.name}. {country.name} is not mapped."
                    )
                    self.logger.debug("Possible matches are: ")
                    for c, score in zip(matching_locations, scores):
                        self.logger.debug(c, score)
            else:
                location_mapping[country.name] = country.alpha_3

        return location_mapping

    def get_tiles_for_country(self, country_code: str) -> pd.DataFrame:
        country_tiles = self.df_tiles[self.df_tiles["country"] == country_code]

        if not country_tiles.empty:
            return country_tiles

        self.logger.warning(
            f"The country is not in location mapping. Manually checking if there are overlapping locations with country boundary."
        )

        country_mercator = CountryMercatorTiles.create(
            country_code, self.MERCATOR_ZOOM_LEVEL
        )
        country_tiles = country_mercator.filter_quadkeys(self.df_tiles.quadkey)

        if country_tiles:
            filtered_tiles = self.df_tiles[
                self.df_tiles.country.isnull()
                & self.df_tiles.quadkey.isin(country_tiles.quadkeys)
            ]
            # filtered_tiles["country"] = country_code
            return filtered_tiles

        return pd.DataFrame(columns=self.df_tiles.columns)

    def get_tiles_for_geometry(
        self, geometry: Union[Polygon, MultiPolygon]
    ) -> pd.DataFrame:
        if isinstance(geometry, MultiPolygon):
            geom_mercator = MercatorTiles.from_multipolygon(
                geometry, self.MERCATOR_ZOOM_LEVEL
            )
        elif isinstance(geometry, Polygon):
            geom_mercator = MercatorTiles.from_polygon(
                geometry, self.MERCATOR_ZOOM_LEVEL
            )

        geom_tiles = geom_mercator.filter_quadkeys(self.df_tiles.quadkey)

        if geom_tiles:
            return self.df_tiles[self.df_tiles.quadkey.isin(geom_tiles.quadkeys)]

        return pd.DataFrame(columns=self.df_tiles.columns)

    def get_tiles_for_points(
        self, points: List[Union[Point, Tuple[float, float]]]
    ) -> pd.DataFrame:

        points_mercator = MercatorTiles.from_points(points, self.MERCATOR_ZOOM_LEVEL)

        points_tiles = points_mercator.filter_quadkeys(self.df_tiles.quadkey)

        if points_tiles:
            return self.df_tiles[self.df_tiles.quadkey.isin(points_tiles.quadkeys)]

        return pd.DataFrame(columns=self.df_tiles.columns)

    def get_tile_path(self, quadkey: str, location: str) -> Path:
        return self.BASE_PATH / location / self.upload_date / f"{quadkey}.csv.gz"


class MSBuildingsDownloader:
    """A class to handle downloads of Microsoft's Global ML Building Footprints dataset."""

    def __init__(
        self,
        config: Optional[MSBuildingsConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the downloader.

        Args:
            config: Optional configuration for customizing file paths.
            logger: Optional custom logger. If not provided, uses default logger.
        """
        self.logger = logger or config.get_logger(__name__)
        self.config = config or MSBuildingsConfig(logger=self.logger)

    def _download_tile(
        self,
        tile_info: Union[pd.Series, dict],
    ) -> Optional[str]:
        """Download data file for a single tile."""

        # Modify URL based on data type if needed
        tile_url = tile_info["url"]

        try:
            response = requests.get(tile_url, stream=True)
            response.raise_for_status()

            file_path = str(
                self.config.get_tile_path(
                    quadkey=tile_info["quadkey"],
                    location=(
                        tile_info["country"]
                        if tile_info["country"]
                        else tile_info["location"]
                    ),
                )
            )

            with self.config.data_store.open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

                self.logger.debug(
                    f"Successfully downloaded tile: {tile_info['quadkey']}"
                )
                return file_path

        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Failed to download tile {tile_info['quadkey']}: {str(e)}"
            )
            return None

    def download_by_country(self, country_code: str) -> List[str]:
        """
        Download Microsoft Global ML Building Footprints data for a specific country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code

        Returns:
            List of paths to downloaded files
        """

        country_tiles = self.config.get_tiles_for_country(country_code)

        if not country_tiles.empty:

            with multiprocessing.Pool(self.config.n_workers) as pool:
                download_func = functools.partial(self._download_tile)
                file_paths = list(
                    tqdm(
                        pool.imap(
                            download_func, [row for _, row in country_tiles.iterrows()]
                        ),
                        total=len(country_tiles),
                        desc=f"Downloading tiles for {country_code}",
                    )
                )

            return [path for path in file_paths if path is not None]

        return []

    def download_by_geometry(self, geometry: Union[Polygon, MultiPolygon]) -> List[str]:
        """
        Download Microsoft Global ML Building Footprints data for a specific geometry.

        Args:
            geometry: Polygon or MultiPolygon geometry

        Returns:
            List of paths to downloaded files
        """

        geom_tiles = self.config.get_tiles_for_geometry(geometry)

        if not geom_tiles.empty:

            with multiprocessing.Pool(self.config.n_workers) as pool:
                download_func = functools.partial(self._download_tile)
                file_paths = list(
                    tqdm(
                        pool.imap(
                            download_func, [row for _, row in geom_tiles.iterrows()]
                        ),
                        total=len(geom_tiles),
                        desc="Downloading tiles for the geometry",
                    )
                )

            return [path for path in file_paths if path is not None]

        return []

    def download_by_points(
        self,
        points: List[Union[Point, Tuple[float, float]]],
    ) -> List[str]:
        """
        Download  Microsoft Global ML Building Footprints data for areas containing specific points.

        Args:
            points_gdf: GeoDataFrame containing points of interest

        Returns:
            List of paths to downloaded files
        """

        points_tiles = self.config.get_tiles_for_points(points)

        if not points_tiles.empty:

            with multiprocessing.Pool(self.config.n_workers) as pool:
                download_func = functools.partial(self._download_tile)
                file_paths = list(
                    tqdm(
                        pool.imap(
                            download_func, [row for _, row in points_tiles.iterrows()]
                        ),
                        total=len(points_tiles),
                        desc=f"Downloading tiles for the points",
                    )
                )

            return [path for path in file_paths if path is not None]

        return []
