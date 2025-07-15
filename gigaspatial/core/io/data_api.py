import pandas as pd
import delta_sharing
from typing import Union
from pathlib import Path

from gigaspatial.config import config


class GigaDataAPI:

    def __init__(
        self,
        profile_file: Union[str, Path] = config.API_PROFILE_FILE_PATH,
        share_name: str = config.API_SHARE_NAME,
        schema_name: str = config.API_SCHEMA_NAME,
    ):
        """
        Initialize the GigaDataAPI class with the profile file, share name, and schema name.

        profile_file: Path to the delta-sharing profile file.
        share_name: Name of the share (e.g., "gold").
        schema_name: Name of the schema (e.g., "school-master").
        """
        self.profile_file = profile_file
        self.share_name = share_name
        self.schema_name = schema_name
        self.client = delta_sharing.SharingClient(profile_file)

        self._cache = {}

    def get_country_list(self, sort=True):
        """
        Retrieve a list of available countries in the dataset.

        :param sort: Whether to sort the country list alphabetically (default is True).
        """
        country_list = [
            t.name
            for t in self.client.list_all_tables()
            if t.schema == self.schema_name
        ]
        if sort:
            country_list.sort()
        return country_list

    def load_country_data(self, country, filters=None, use_cache=True):
        """
        Load the dataset for the specified country with optional filtering and caching.

        country: The country code (e.g., "MWI").
        filters: A dictionary with column names as keys and filter values as values.
        use_cache: Whether to use cached data if available (default is True).
        """
        # Check if data is cached
        if use_cache and country in self._cache:
            df_country = self._cache[country]
        else:
            # Load data from the API
            table_url = (
                f"{self.profile_file}#{self.share_name}.{self.schema_name}.{country}"
            )
            df_country = delta_sharing.load_as_pandas(table_url)
            self._cache[country] = df_country  # Cache the data

        # Apply filters if provided
        if filters:
            for column, value in filters.items():
                df_country = df_country[df_country[column] == value]

        return df_country

    def load_multiple_countries(self, countries):
        """
        Load data for multiple countries and combine them into a single DataFrame.

        countries: A list of country codes.
        """
        df_list = []
        for country in countries:
            df_list.append(self.load_country_data(country))
        return pd.concat(df_list, ignore_index=True)

    def get_country_metadata(self, country):
        """
        Retrieve metadata (e.g., column names and data types) for a country's dataset.

        country: The country code (e.g., "MWI").
        """
        df_country = self.load_country_data(country)
        metadata = {
            "columns": df_country.columns.tolist(),
            "data_types": df_country.dtypes.to_dict(),
            "num_records": len(df_country),
        }
        return metadata

    def get_all_cached_data_as_dict(self):
        """
        Retrieve all cached data in a dictionary format, where each key is a country code,
        and the value is the DataFrame of that country.
        """
        return self._cache if self._cache else {}

    def get_all_cached_data_as_json(self):
        """
        Retrieve all cached data in a JSON-like format. Each country is represented as a key,
        and the value is a list of records (i.e., the DataFrame's `to_dict(orient='records')` format).
        """
        if not self._cache:
            return {}

        # Convert each DataFrame in the cache to a JSON-like format (list of records)
        return {
            country: df.to_dict(orient="records") for country, df in self._cache.items()
        }
