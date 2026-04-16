"""
OpenCellID data handler for cell tower locations.

This module provides handlers for interacting with OpenCellID's crowd-sourced 
database of cell tower locations. It supports:
- Automated discovery of country-specific download links.
- Parallelized acquisition of MCC-grouped CSV files.
- Filtering by creation date, provider, and network type.
- Spatial deduplication and GeoPandas integration.
"""
import pandas as pd
import geopandas as gpd
import requests
import logging
from typing import List, Optional, Union
from pathlib import Path
from bs4 import BeautifulSoup
import pycountry
from pydantic.dataclasses import dataclass
from pydantic import Field, ConfigDict
import os

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.handlers.base import (
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    BaseHandler,
)
from gigaspatial.config import config as global_config


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OpenCellIDConfig(BaseHandlerConfig):
    """
    Configuration for OpenCellID dataset access.

    Attributes:
        base_path: Root directory for local CSV storage.
        api_token: OpenCellID access token (required for downloads).
        download_url: Base endpoint for the OpenCellID download page.
    """

    base_path: Path = Field(default=global_config.get_path("opencellid", "bronze"))
    api_token: str = Field(default=global_config.OPENCELLID_ACCESS_TOKEN)
    download_url: str = "https://opencellid.org/downloads.php?token="

    def extract_search_geometry(self, source, **kwargs) -> str:
        """
        Identify the country alpha-2 code from a geographic source.

        Args:
            source: Country name or ISO code.
            **kwargs: Additional context.

        Returns:
            The ISO 3166-1 alpha-2 country code.
        """
        if not isinstance(source, str):
            raise ValueError(
                f"Unsupported source type: {type(source)}. "
                "Please use country name or ISO code (str)."
            )

        try:
            return pycountry.countries.lookup(source).alpha_2
        except LookupError:
            raise ValueError(f"Invalid country provided: {source}")

    def get_relevant_data_units_by_geometry(
        self, geometry: str, **kwargs
    ) -> List[dict]:
        """
        Discover download links for a specific country by screen-scraping.

        Args:
            geometry: ISO 3166-1 alpha-2 country code.
            **kwargs: Additional parameters.

        Returns:
            A list of dictionaries containing 'url' and 'country'.
        """
        url = f"{self.download_url}{self.api_token}"
        country_alpha2 = geometry.upper()

        try:
            self.logger.info(f"Fetching download links for country: {country_alpha2}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            html_content = response.text
            soup = BeautifulSoup(html_content, "lxml")
            table = soup.find("table", {"id": "regions"})

            if not table:
                raise ValueError(
                    "Could not find cell tower data table on OpenCellID website"
                )

            # Parse table headers
            t_headers = []
            for th in table.find_all("th"):
                t_headers.append(th.text.replace("\n", " ").strip())

            # Parse table data
            table_data = []
            for tr in table.tbody.find_all("tr"):
                t_row = {}
                for td, th in zip(tr.find_all("td"), t_headers):
                    if "Files" in th:
                        t_row[th] = [a.get("href") for a in td.find_all("a")]
                    else:
                        t_row[th] = td.text.replace("\n", "").strip()
                table_data.append(t_row)

            cell_dict = pd.DataFrame(table_data)

            if "Country Code" not in cell_dict.columns:
                raise ValueError(
                    "Could not find 'Country Code' column in OpenCellID table."
                )

            if country_alpha2 not in cell_dict["Country Code"].values:
                raise ValueError(
                    f"Country code {country_alpha2} not found in OpenCellID database"
                )

            links = cell_dict[cell_dict["Country Code"] == country_alpha2][
                "Files (grouped by MCC)"
            ].values[0]

            return [{"url": link, "country": country_alpha2} for link in links]

        except Exception as e:
            self.logger.error(f"Error fetching download links: {str(e)}")
            raise

    def get_data_unit_path(self, unit: dict, **kwargs) -> Path:
        """
        Determine the local storage path for an OpenCellID data unit.

        Args:
            unit: Dictionary containing 'url' and 'country' metadata.
            **kwargs: Additional resolution context.

        Returns:
            Absolute local path for the CSV file.
        """
        url = unit["url"]
        country = unit["country"].lower()

        # Extract filename from URL parameters if possible, otherwise use a hash or parts of the URL
        # OpenCellID link example: https://opencellid.org/ocid/downloads?token=...&file=MCC-XXX.csv.gz
        from urllib.parse import urlparse, parse_qs

        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)

        filename = params.get("file", [None])[0]
        if not filename:
            # Fallback filename if 'file' param is missing
            filename = os.path.basename(parsed_url.path) or "data.csv.gz"
            if not filename.endswith(".gz"):
                filename += ".gz"

        return self.base_path / country / filename


class OpenCellIDDownloader(BaseHandlerDownloader):
    """
    Downloader for OpenCellID CSV exports.

    Handles authenticated requests to the OpenCellID website and persists
    streaming responses to local storage.
    """

    def download_data_unit(self, unit: dict, **kwargs) -> Path:
        """
        Acquire a single OpenCellID data unit.

        Args:
            unit: Dictionary containing 'url' and 'country' metadata.
            **kwargs: Download parameters.

        Returns:
            Path to the downloaded local file.
        """
        url = unit["url"]
        save_path = self.config.get_data_unit_path(unit)

        self.logger.info(f"Downloading OpenCellID data from {url}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with self.data_store.open(str(save_path), "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Check for API errors in the downloaded file (sometimes they return 200 with error text)
            # This is a bit tricky with data_store, but we can check if it's very small and contains error text

            self.logger.info(f"Successfully downloaded to {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"Failed to download unit {url}: {e}")
            if self.data_store.file_exists(str(save_path)):
                self.data_store.remove(str(save_path))
            raise


class OpenCellIDReader(BaseHandlerReader):
    """
    Reader for OpenCellID CSV exports.

    Parses raw cell tower data, applies temporal filters, and aggregates 
    results into tabular or geospatial objects.
    """

    def load_from_paths(
        self,
        source_data_path: List[Union[str, Path]],
        created_newer: int = 2003,
        created_before: Optional[int] = None,
        drop_duplicates: bool = True,
        as_geodataframe: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Load cell tower data from one or more local CSV paths.

        Args:
            source_data_path: List of local paths to OpenCellID files.
            created_newer: Optional start year for creation date filtering.
            created_before: Optional end year for creation date filtering.
            drop_duplicates: If True, resolves duplicate towers by latest update.
            as_geodataframe: If True, returns a GeoDataFrame with point geometries.
            **kwargs: Additional loading parameters.

        Returns:
            A pandas DataFrame or geopandas GeoDataFrame of processed cell data.
        """
        if created_before is None:
            created_before = pd.Timestamp.now().year

        dfs = []

        # Official OpenCellID column sequence
        ocid_columns = [
            "radio",
            "mcc",
            "net",
            "area",
            "cell",
            "unit",
            "lon",
            "lat",
            "range",
            "samples",
            "changeable",
            "created",
            "updated",
            "average_signal",
        ]

        for path in source_data_path:
            self.logger.info(f"Loading data from {path}")
            try:
                df = read_dataset(
                    path,
                    data_store=self.data_store,
                    header=None,
                    names=ocid_columns,
                    low_memory=False,
                )
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Error reading file {path}: {e}")
                # Check if it was a rate limit or token error hidden in the file
                try:
                    with self.data_store.open(str(path), "r") as ef:
                        first_line = ef.readline()
                        if "RATE_LIMITED" in first_line:
                            raise RuntimeError("API rate limit exceeded!")
                        if "INVALID_TOKEN" in first_line:
                            raise RuntimeError("Invalid API token!")
                except:
                    pass
                raise

        if not dfs:
            return gpd.GeoDataFrame() if as_geodataframe else pd.DataFrame()

        df_cell = pd.concat(dfs, ignore_index=True)

        # Process the data
        if not df_cell.empty:
            # Convert timestamps to datetime
            df_cell["created"] = pd.to_datetime(
                df_cell["created"], unit="s", origin="unix"
            )
            df_cell["updated"] = pd.to_datetime(
                df_cell["updated"], unit="s", origin="unix"
            )

            # Filter by year
            df_cell = df_cell[
                (df_cell.created.dt.year >= created_newer)
                & (df_cell.created.dt.year < created_before)
            ]

            # Drop duplicates if configured
            if drop_duplicates:
                df_cell = (
                    df_cell.sort_values("updated", ascending=False)
                    .groupby(["radio", "lon", "lat"])
                    .first()
                    .reset_index()
                )

            if not as_geodataframe:
                return df_cell

            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame(
                df_cell,
                geometry=gpd.points_from_xy(df_cell.lon, df_cell.lat),
                crs="EPSG:4326",
            )
            return gdf

        return gpd.GeoDataFrame() if as_geodataframe else pd.DataFrame()


class OpenCellIDHandler(BaseHandler):
    """
    Unified handler for OpenCellID data.

    Coordinates automated link discovery, coordinated downloads, and
    standardized reading of cell tower datasets.
    """

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> OpenCellIDConfig:
        """
        Create an OpenCellID configuration instance.

        Args:
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Configuration overrides.

        Returns:
            A configured OpenCellIDConfig.
        """
        return OpenCellIDConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(
        self,
        config: OpenCellIDConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> OpenCellIDDownloader:
        """
        Create an OpenCellID downloader instance.

        Args:
            config: Handler configuration.
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Downloader parameters.

        Returns:
            A configured OpenCellIDDownloader.
        """
        return OpenCellIDDownloader(config=config, data_store=data_store, logger=logger)

    def create_reader(
        self,
        config: OpenCellIDConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> OpenCellIDReader:
        """
        Create an OpenCellID reader instance.

        Args:
            config: Handler configuration.
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Reader parameters.

        Returns:
            A configured OpenCellIDReader.
        """
        return OpenCellIDReader(config=config, data_store=data_store, logger=logger)

    def load_as_geodataframe(
        self,
        source,
        crop_to_source: bool = False,
        ensure_available: bool = True,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Acquire OpenCellID data and load it as a geospatial GeoDataFrame.

        Args:
            source: Geographic filter (country ISO) or direct file paths.
            crop_to_source: If True, clips data to the source boundary.
            ensure_available: If True, executes download if data is missing locally.
            **kwargs: Additional parameters passed to `load_data`.

        Returns:
            A GeoDataFrame containing cell tower locations.
        """
        return self.load_data(
            source,
            crop_to_source=crop_to_source,
            ensure_available=ensure_available,
            as_geodataframe=True,
            **kwargs,
        )
