import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Literal
import pycountry
import tempfile

from pydantic import Field, field_validator

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.handlers.hdx import HDXConfig, HDXDownloader


class RWIConfig(HDXConfig):
    """Configuration for Relative Wealth Index data access"""

    # Override dataset_name to be fixed for RWI
    dataset_name: Literal["relative-wealth-index"] = Field(
        default="relative-wealth-index"
    )

    # Additional RWI-specific configurations
    country: Optional[str] = Field(
        default=None, description="Country ISO code to filter data for"
    )

    @field_validator("country")
    def validate_country(cls, value: str) -> str:
        try:
            return pycountry.countries.lookup(value).alpha_3
        except LookupError:
            raise ValueError(f"Invalid country code provided: {value}")


class RelativeWealthIndexDownloader(HDXDownloader):
    """Specialized downloader for the Relative Wealth Index dataset from HDX"""

    def __init__(
        self,
        config: Union[RWIConfig, dict] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if config is None:
            config = RWIConfig()
        elif isinstance(config, dict):
            config = RWIConfig(**config)

        super().__init__(config=config, data_store=data_store, logger=logger)

    @classmethod
    def from_config(
        cls,
        country: Optional[str] = None,
        **kwargs,
    ):
        """Create a downloader with RWI-specific configurations"""
        config = RWIConfig(country=country, **kwargs)
        return cls(config=config)

    def download_dataset(self) -> List[str]:
        """Download RWI dataset, optionally filtering for a specific country.

        If country is specified, attempts to find and download only the resources
        relevant to that country. Otherwise, downloads all RWI resources.

        Returns:
            List of paths to the downloaded files
        """
        # If no country specified, download all resources
        if self.config.country is None:
            return super().download_dataset()

        # Get all resources from the dataset
        try:
            resources = self.get_dataset_resources()
            if not resources:
                self.logger.warning(f"No resources found for RWI dataset")
                return []

            # Prepare country identifiers for matching
            country_code = self.config.country.lower()
            country_name = pycountry.countries.lookup(self.config.country).name.lower()
            country_alpha2 = pycountry.countries.lookup(
                self.config.country
            ).alpha_2.lower()

            # Try different matching patterns
            country_patterns = [
                f"/{country_code}_",  # URL path with ISO3 prefix
                f"/{country_code}.",  # URL path with ISO3 followed by extension
                f"_{country_code}_",  # Filename with ISO3 in middle
                f"_{country_code}.",  # Filename with ISO3 at end
                f"/{country_name.replace(' ', '')}_",  # URL with no spaces
                f"/{country_name.replace(' ', '-')}_",  # URL with hyphens
                f"/{country_alpha2}_",  # URL with ISO2 code
                country_name,  # Country name anywhere in URL
            ]

            # Find matching resources
            matching_resources = []
            for resource in resources:
                # Get the URL safely
                resource_url = resource.get("url", "")
                if not resource_url:
                    continue

                resource_url = resource_url.lower()

                # Check for matches with our patterns
                if any(pattern in resource_url for pattern in country_patterns):
                    matching_resources.append(resource)

            if not matching_resources:
                self.logger.warning(
                    f"No resources matching country '{self.config.country}' were found. "
                    f"Consider downloading the full dataset with country=None and filtering afterwards."
                )
                return []

            # Download the matching resources
            downloaded_paths = []
            for res in matching_resources:
                try:
                    resource_name = res.get("name", "Unknown")
                    self.logger.info(f"Downloading resource: {resource_name}")

                    # Download to a temporary directory
                    with tempfile.TemporaryDirectory() as tmpdir:
                        url, local_path = res.download(folder=tmpdir)
                        # Read the file and write to the DataStore
                        with open(local_path, "rb") as f:
                            data = f.read()
                        # Compose the target path in the DataStore
                        target_path = str(
                            self.config.output_dir_path / Path(local_path).name
                        )
                        self.data_store.write_file(target_path, data)
                        downloaded_paths.append(target_path)

                    self.logger.info(
                        f"Downloaded resource: {resource_name} to {target_path}"
                    )

                except Exception as e:
                    resource_name = res.get("name", "Unknown")
                    self.logger.error(
                        f"Error downloading resource {resource_name}: {str(e)}"
                    )

            return downloaded_paths

        except Exception as e:
            self.logger.error(f"Error during country-filtered download: {str(e)}")

            # Fall back to downloading all resources
            self.logger.info("Falling back to downloading all RWI resources")
            return super().download_dataset()
