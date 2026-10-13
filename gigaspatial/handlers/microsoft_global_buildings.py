"""
Microsoft Global ML Building Footprints dataset handler.

This module provides specialized handlers for the Microsoft Global Buildings dataset.
It manages location mapping between dataset identifiers and ISO country codes,
resolves Bing Maps Quadkeys for spatial querying, and handles multi-threaded
downloads of building footprints in compressed CSV format.
"""

from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from pathlib import Path
from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry
from difflib import SequenceMatcher
import pycountry
import requests
import logging
import geopandas as gpd
import re

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.grid.mercator_tiles import (
    MercatorTiles,
    CountryMercatorTiles,
)
from gigaspatial.handlers.base import (
    BaseHandlerReader,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandler,
)
from gigaspatial.handlers.base.source import SourceKind, classify_source
from gigaspatial.config import config as global_config


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MSBuildingsConfig(BaseHandlerConfig):
    """
    Configuration for Microsoft Global Buildings dataset.

    Manages dataset URLs, Bing Maps Quadkey zoom levels, and country-to-location
    mapping logic.

    Attributes:
        TILE_URLS: URL to the dataset links CSV.
        MERCATOR_ZOOM_LEVEL: Bing Maps zoom level for quadkey resolution.
        base_path: Root directory for storing building data.
        LOCATION_MAPPING_FILE: Path to the cached location mapping JSON.
    """

    README_URL = (
        "https://raw.githubusercontent.com/"
        "microsoft/GlobalMLBuildingFootprints/main/README.md"
    )
    MANIFEST_URL: Optional[str] = None

    MERCATOR_ZOOM_LEVEL: int = 9
    units_match_country = True

    base_path: Path = global_config.get_path("microsoft_global_buildings", "bronze")

    LOCATION_MAPPING_FILE: Path = base_path / "location_mapping.json"
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

    def __post_init__(self):
        """Initialize the configuration, load tile URLs, and set up location mapping."""
        super().__post_init__()
        manifest_url = (
            self.MANIFEST_URL if self.MANIFEST_URL else self.get_latest_manifest_url()
        )
        self.MANIFEST_URL = manifest_url
        self.manifest_date = self._get_manifest_date(manifest_url)

        self._load_tile_urls(manifest_url)
        self._setup_location_mapping()

    @classmethod
    def get_latest_manifest_url(cls) -> str:
        """
        Discover the latest Microsoft Global Buildings dataset manifest.

        Returns:
            URL of the latest ``dataset-links.csv`` manifest.

        Raises:
            RuntimeError: If no manifest URL can be found.
            requests.HTTPError: If the README cannot be retrieved.
        """
        response = requests.get(cls.README_URL, timeout=30)
        response.raise_for_status()

        pattern = re.compile(
            r"https://bfppub\.blob\.core\.windows\.net/"
            r"\%24web/(\d{4}-\d{2}-\d{2})/dataset-links\.csv"
        )

        matches = pattern.findall(response.text)

        if not matches:
            raise RuntimeError("No Microsoft Buildings dataset manifest found.")

        latest_date = max(matches)

        return (
            "https://bfppub.blob.core.windows.net/"
            f"$web/{latest_date}/dataset-links.csv"
        )

    @staticmethod
    def _get_manifest_date(manifest_url: str) -> str:
        """Extract the dataset manifest date from its URL.

        Args:
            manifest_url: URL of the Microsoft Buildings dataset manifest.

        Returns:
            Manifest date in ``YYYY-MM-DD`` format.

        Raises:
            ValueError: If the URL does not contain a valid manifest date.
        """
        match = re.search(
            r"/(\d{4}-\d{2}-\d{2})/dataset-links\.csv(?:\?.*)?$",
            manifest_url,
        )

        if not match:
            raise ValueError(
                f"Could not extract manifest date from URL: {manifest_url}"
            )

        return match.group(1)

    def _load_tile_urls(self, url: str):
        """Load dataset links from the remote manifest."""
        self.df_tiles = pd.read_csv(
            url,
            names=[
                "location",
                "quadkey",
                "url",
                "size",
                "upload_date",
            ],
            dtype={"quadkey": str},
            header=0,
        )

    def _setup_location_mapping(self):
        """Load or create the mapping between dataset locations and ISO country codes."""
        from gigaspatial.core.io.readers import read_json
        from gigaspatial.core.io.writers import write_json

        if self.data_store.file_exists(str(self.LOCATION_MAPPING_FILE)):
            self.location_mapping = read_json(
                str(self.LOCATION_MAPPING_FILE), data_store=self.data_store
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
        self.df_tiles.loc[self.df_tiles.country.isnull(), "country"] = None

    def _map_locations(self):
        """Map the 'location' column in the tiles DataFrame to ISO country codes."""
        self.df_tiles["country"] = self.df_tiles.location.map(self.location_mapping)

    def create_location_mapping(self, similarity_score_threshold: float = 0.8):
        """
        Map dataset location names to ISO 3166-1 alpha-3 country codes.

        Uses string similarity and spatial overlaps with Bing Maps tiles to
        automatically resolve which countries the dataset locations belong to.

        Args:
            similarity_score_threshold: Minimum string similarity ratio for matches.

        Returns:
            Dictionary mapping location names to ISO-3 country codes.
        """

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

    def get_relevant_data_units_by_key(
        self, search_key: Union[str, BaseGeometry], **kwargs
    ) -> List[dict]:
        """
        Find Microsoft Building tiles for a country or a geographic area.

        A country is looked up in the location mapping first. Otherwise, and
        for geometries, tiles are found by Mercator quadkey overlap.

        Args:
            search_key: ISO 3166-1 alpha-3 country code, or a search geometry.
            **kwargs: Additional filtering parameters.

        Returns:
            List of tile records containing quadkey, URL, and country metadata.
        """
        if isinstance(search_key, str):
            country_code = search_key
            mask = self.df_tiles["country"] == country_code

            if any(mask):
                return self.df_tiles.loc[
                    mask, ["quadkey", "url", "country", "location"]
                ].to_dict("records")

            self.logger.warning(
                f"The country code '{country_code}' is not directly in the location mapping. "
                "Manually checking for overlapping locations with the country boundary."
            )

            source_tiles = CountryMercatorTiles.create(
                country_code, self.MERCATOR_ZOOM_LEVEL
            )
        else:
            source_tiles = MercatorTiles.from_spatial(
                source=search_key, zoom_level=self.MERCATOR_ZOOM_LEVEL
            )

        filtered_tiles = source_tiles.filter_quadkeys(self.df_tiles.quadkey)

        mask = self.df_tiles.quadkey.isin(filtered_tiles.quadkeys)

        return self.df_tiles.loc[
            mask, ["quadkey", "url", "country", "location"]
        ].to_dict("records")

    def get_data_unit_path(self, unit: Union[pd.Series, dict], **kwargs) -> Path:

        tile_location = unit["country"] if unit["country"] else unit["location"]

        return (
            self.base_path
            / tile_location
            / self.manifest_date
            / f'{unit["quadkey"]}.csv.gz'
        )

    def get_search_key(self, source, **kwargs) -> Union[str, BaseGeometry]:
        """
        Return the ISO 3166-1 alpha-3 code of a country source, so its tiles
        can be looked up by country; other sources use the search geometry.
        """
        if classify_source(source) is SourceKind.COUNTRY:
            try:
                return pycountry.countries.lookup(source).alpha_3
            except LookupError:
                raise ValueError(f"Invalid country: {source}")
        return self.extract_search_geometry(source, **kwargs)


class MSBuildingsDownloader(BaseHandlerDownloader):
    """
    Downloader for Microsoft's Global ML Building Footprints dataset.

    Handles the parallel acquisition of gzipped building footprint CSVs
    from Azure Open Datasets.
    """

    def __init__(
        self,
        config: Optional[MSBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the downloader.

        Args:
            config: Optional configuration for customizing download behavior and file paths.
                    If None, a default `MSBuildingsConfig` is used.
            data_store: Optional instance of a `DataStore` for managing data storage.
                        If provided, it overrides the `data_store` in the `config`.
                        If None, the `data_store` from the `config` is used.
            logger: Optional custom logger instance. If None, a default logger
                    named after the module is created and used.
        """
        config = config or MSBuildingsConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(
        self,
        tile_info: Union[pd.Series, dict],
        **kwargs,
    ) -> Optional[str]:
        """
        Download a single building footprint tile.

        Args:
            tile_info: Metropolitan or quadkey-specific metadata including 'url' and 'quadkey'.
            **kwargs: Additional parameters.

        Returns:
            The path to the local gzipped CSV file on success, or None on failure.
        """

        tile_url = tile_info["url"]

        try:
            response = requests.get(tile_url, stream=True)
            response.raise_for_status()

            file_path = str(self.config.get_data_unit_path(tile_info))

            with self.data_store.open(file_path, "wb") as file:
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
        except Exception as e:
            self.logger.error(f"Unexpected error downloading dataset: {str(e)}")
            return None

    def download_by_country(
        self,
        country: str,
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """
        Download Microsoft buildings data for an entire country.

        Args:
            country: ISO country code or name.
            data_store: Optional DataStore for boundary lookups.
            country_geom_path: Path to a specific country boundary file.

        Returns:
            A list of local paths to the successfully downloaded tiles.
        """
        return self.download(
            source=country, data_store=data_store, path=country_geom_path
        )


class MSBuildingsReader(BaseHandlerReader):
    """
    Reader for Microsoft Global Buildings data.

    Manages the ingestion of gzipped footprint CSVs into GeoDataFrames,
    handling the conversion of string-based geometry into Shapely objects.
    """

    def __init__(
        self,
        config: Optional[MSBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config or MSBuildingsConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> gpd.GeoDataFrame:
        """
        Load building data from specific file paths.

        Args:
            source_data_path: List of local paths to building data files.
            **kwargs: Additional parameters for `read_ms_dataset`.

        Returns:
            A GeoDataFrame containing the aggregated building data.
        """
        from gigaspatial.core.io.readers import read_gzipped_json_or_csv
        from shapely.geometry import shape

        def read_ms_dataset(file_path: str, data_store: DataStore = None):
            df = read_gzipped_json_or_csv(file_path=file_path, data_store=data_store)
            df["geometry"] = df["geometry"].apply(shape)
            return gpd.GeoDataFrame(df, crs=4326)

        result = self._load_tabular_data(
            file_paths=source_data_path, read_function=read_ms_dataset
        )
        return result


class MSBuildingsHandler(BaseHandler):
    """
    Unified handler for the Microsoft Global Buildings dataset.

    Provides a top-level API for acquiring and accessing Microsoft building
    footprints for any geographic region.
    """

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> MSBuildingsConfig:
        """
        Create a configuration instance for Microsoft Buildings.

        Args:
            data_store: DataStore for location mapping logic.
            logger: Component logger.
            **kwargs: Configuration overrides.

        Returns:
            A configured MSBuildingsConfig.
        """
        return MSBuildingsConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(
        self,
        config: MSBuildingsConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> MSBuildingsDownloader:
        """
        Create a downloader instance for Microsoft Buildings.

        Args:
            config: Handler configuration.
            data_store: DataStore for saving footprints.
            logger: Component logger.
            **kwargs: Downloader parameters.

        Returns:
            A configured MSBuildingsDownloader.
        """
        return MSBuildingsDownloader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def create_reader(
        self,
        config: MSBuildingsConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> MSBuildingsReader:
        """
        Create a reader instance for Microsoft Buildings.

        Args:
            config: Handler configuration.
            data_store: DataStore for reading footprints.
            logger: Component logger.
            **kwargs: Reader parameters.

        Returns:
            A configured MSBuildingsReader.
        """
        return MSBuildingsReader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )
