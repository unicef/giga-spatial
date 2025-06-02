import pandas as pd
import numpy as np
import geopandas as gpd
import requests
import logging
import gzip
import os
import tempfile
from datetime import datetime
from typing import List, Optional, Union
from pathlib import Path
from bs4 import BeautifulSoup
import pycountry
from pydantic import BaseModel, Field, HttpUrl, field_validator

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.config import config as global_config


class OpenCellIDConfig(BaseModel):
    """Configuration for OpenCellID data access"""

    # Base URLs
    BASE_URL: HttpUrl = Field(default="https://opencellid.org/")
    DOWNLOAD_URL: HttpUrl = Field(default="https://opencellid.org/downloads.php?token=")

    # User configuration
    country: str = Field(...)
    api_token: str = Field(
        default=global_config.OPENCELLID_ACCESS_TOKEN,
        description="OpenCellID API Access Token",
    )
    base_path: Path = Field(default=global_config.get_path("opencellid", "bronze"))
    created_newer: int = Field(
        default=2003, description="Filter out cell towers added before this year"
    )
    created_before: int = Field(
        default=datetime.now().year,
        description="Filter out cell towers added after this year",
    )
    drop_duplicates: bool = Field(
        default=True,
        description="Drop cells that are in the exact same location and radio technology",
    )

    @field_validator("country")
    def validate_country(cls, value: str) -> str:
        try:
            return pycountry.countries.lookup(value).alpha_3
        except LookupError:
            raise ValueError(f"Invalid country code provided: {value}")

    @property
    def output_file_path(self) -> Path:
        """Path to save the downloaded OpenCellID data"""
        return self.base_path / f"opencellid_{self.country.lower()}.csv.gz"

    def __repr__(self) -> str:
        return (
            f"OpenCellIDConfig(\n"
            f"  country='{self.country}'\n"
            f"  created_newer={self.created_newer}\n"
            f"  created_before={self.created_before}\n"
            f"  drop_duplicates={self.drop_duplicates}\n"
            f")"
        )


class OpenCellIDDownloader:
    """Downloader for OpenCellID data"""

    def __init__(
        self,
        config: Union[OpenCellIDConfig, dict],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(config, dict):
            self.config = OpenCellIDConfig(**config)
        else:
            self.config = config

        self.data_store = data_store or LocalDataStore()
        self.logger = logger or global_config.get_logger(self.__class__.__name__)

    @classmethod
    def from_country(
        cls,
        country: str,
        api_token: str = global_config.OPENCELLID_ACCESS_TOKEN,
        **kwargs,
    ):
        """Create a downloader for a specific country"""
        config = OpenCellIDConfig(country=country, api_token=api_token, **kwargs)
        return cls(config=config)

    def get_download_links(self) -> List[str]:
        """Get download links for the country from OpenCellID website"""
        url = f"{self.config.DOWNLOAD_URL}{self.config.api_token}"
        country_alpha2 = pycountry.countries.get(
            alpha_3=self.config.country.upper()
        ).alpha_2

        try:
            # Find table with cell tower data links
            self.logger.info(f"Fetching download links for {self.config.country}")
            html_content = requests.get(url).text
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
                        t_row[th] = []
                        for a in td.find_all("a"):
                            t_row[th].append(a.get("href"))
                    else:
                        t_row[th] = td.text.replace("\n", "").strip()

                table_data.append(t_row)

            cell_dict = pd.DataFrame(table_data)

            # Get links for the country code
            if country_alpha2 not in cell_dict["Country Code"].values:
                raise ValueError(
                    f"Country code {country_alpha2} not found in OpenCellID database"
                )
            else:
                links = cell_dict[cell_dict["Country Code"] == country_alpha2][
                    "Files (grouped by MCC)"
                ].values[0]

            return links

        except Exception as e:
            self.logger.error(f"Error fetching download links: {str(e)}")
            raise

    def download_and_process(self) -> str:
        """Download and process OpenCellID data for the configured country"""

        try:
            links = self.get_download_links()
            self.logger.info(f"Found {len(links)} data files for {self.config.country}")

            dfs = []

            for link in links:
                self.logger.info(f"Downloading data from {link}")
                response = requests.get(link, stream=True)
                response.raise_for_status()

                # Use a temporary file for download
                with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as tmpfile:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            tmpfile.write(chunk)
                    temp_file = tmpfile.name

                try:
                    # Read the downloaded gzipped CSV data
                    with gzip.open(temp_file, "rt") as feed_data:
                        dfs.append(
                            pd.read_csv(
                                feed_data,
                                names=[
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
                                ],
                            )
                        )
                except IOError as e:
                    with open(temp_file, "r") as error_file:
                        contents = error_file.readline()

                    if "RATE_LIMITED" in contents:
                        raise RuntimeError(
                            "API rate limit exceeded. You're rate-limited!"
                        )
                    elif "INVALID_TOKEN" in contents:
                        raise RuntimeError("API token rejected by OpenCellID!")
                    else:
                        raise RuntimeError(
                            f"Error processing downloaded data: {str(e)}"
                        )
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

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
                    (df_cell.created.dt.year >= self.config.created_newer)
                    & (df_cell.created.dt.year < self.config.created_before)
                ]

                # Drop duplicates if configured
                if self.config.drop_duplicates:
                    df_cell = (
                        df_cell.groupby(["radio", "lon", "lat"]).first().reset_index()
                    )

                # Save processed data using data_store
                output_path = str(self.config.output_file_path)
                self.logger.info(f"Saving processed data to {output_path}")
                with self.data_store.open(output_path, "wb") as f:
                    df_cell.to_csv(f, compression="gzip", index=False)

                return output_path
            else:
                raise ValueError(f"No data found for {self.config.country}")

        except Exception as e:
            self.logger.error(f"Error downloading and processing data: {str(e)}")
            raise


class OpenCellIDReader:
    """Reader for OpenCellID data"""

    def __init__(
        self,
        country: str,
        data_store: Optional[DataStore] = None,
        base_path: Optional[Path] = None,
    ):
        self.country = pycountry.countries.lookup(country).alpha_3
        self.data_store = data_store or LocalDataStore()
        self.base_path = base_path or global_config.get_path("opencellid", "bronze")

    def read_data(self) -> pd.DataFrame:
        """Read OpenCellID data for the specified country"""
        file_path = str(self.base_path / f"opencellid_{self.country.lower()}.csv.gz")

        if not self.data_store.file_exists(file_path):
            raise FileNotFoundError(
                f"OpenCellID data for {self.country} not found at {file_path}. "
                "Download the data first using OpenCellIDDownloader."
            )

        return read_dataset(self.data_store, file_path)

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert OpenCellID data to a GeoDataFrame"""
        df = self.read_data()
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
        )
        return gdf
