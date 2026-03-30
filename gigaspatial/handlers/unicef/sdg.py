import logging
from typing import Optional, List, Union

import geopandas as gpd
import pandas as pd
import pycountry
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from gigaspatial.config import config as global_config

import importlib.util

_HAS_UNICEF = importlib.util.find_spec("unicefdata") is not None


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class UnicefDataFetcher:
    """
    Fetch UNICEF child welfare indicators from the UNICEF SDMX Data Warehouse.

    Wraps the ``unicefdata`` Python package, providing access to 700+
    indicators across categories such as child mortality, education,
    nutrition, immunisation, WASH, and SDGs.

    Parameters
    ----------
    indicator : str
        SDMX indicator code, e.g. ``"CME_MRY0T4"`` (Under-5 mortality rate).
        Use :func:`search_indicators` or :func:`list_categories` from
        ``unicefdata`` to discover available codes.
    countries : list of str, optional
        ISO 3166 alpha-3 country codes (e.g. ``["KEN", "NGA", "BRA"]``).
        If None, fetches data for all available countries.
    year : str or int, optional
        Single year (``2020``), range (``"2015:2023"``), or list.
        If None, fetches all available years.
    sex : str, optional
        Sex disaggregation: ``"_T"`` (total), ``"F"``, ``"M"``, or ``"ALL"``.
        Defaults to ``"_T"``.
    format : str, optional
        Output format: ``"long"`` (default), ``"wide"``, or
        ``"wide_indicators"``.
    latest : bool, optional
        If True, keep only the most recent value per country.
    mrv : int, optional
        Keep the N most recent values per country.
    circa : bool, optional
        If True, return the closest available year when an exact match
        is not found.
    logger : logging.Logger, optional
        Logger instance. Defaults to the global GigaSpatial logger.

    Examples
    --------
    >>> fetcher = UnicefDataFetcher(
    ...     indicator="ED_CR_L1",
    ...     countries=["KEN", "NGA", "ETH"],
    ...     year="2015:2023",
    ... )
    >>> df = fetcher.fetch()
    >>> df = fetcher.fetch(latest=True)

    Spatial enrichment with admin boundaries:

    >>> gdf = fetcher.fetch_as_geodataframe(boundaries_gdf=admin_gdf, boundaries_iso3_col="iso3")
    """

    indicator: str = Field(..., description="SDMX indicator code, e.g. 'CME_MRY0T4'")
    countries: Optional[List[str]] = Field(
        default=None, description="ISO3 country codes; None fetches all countries"
    )
    year: Optional[Union[str, int]] = Field(
        default=None, description="Single year, range ('2015:2023'), or list"
    )
    sex: str = Field(default="_T", description="Sex filter: '_T', 'F', 'M', or 'ALL'")
    format: str = Field(
        default="long",
        description="Output format: 'long', 'wide', or 'wide_indicators'",
    )
    latest: bool = Field(
        default=False, description="Keep only the most recent value per country"
    )
    mrv: Optional[int] = Field(
        default=None, description="Keep N most recent values per country"
    )
    circa: bool = Field(default=False, description="Match closest available year")
    logger: Optional[logging.Logger] = Field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not _HAS_UNICEF:
            raise ImportError(
                "UnicefDataFetcher requires 'unicefdata'. "
                "Install it with: pip install 'giga-spatial[unicef]'"
            )
        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)

        # Validate and normalise country codes
        if self.countries is not None:
            validated = []
            for code in self.countries:
                try:
                    validated.append(pycountry.countries.lookup(code).alpha_3)
                except LookupError:
                    raise ValueError(f"Invalid country code: '{code}'")
            self.countries = validated

        valid_formats = {"long", "wide", "wide_indicators"}
        if self.format not in valid_formats:
            raise ValueError(
                f"format must be one of {valid_formats}, got '{self.format}'"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self) -> pd.DataFrame:
        """
        Fetch indicator data from the UNICEF SDMX Data Warehouse.

        Returns
        -------
        pd.DataFrame
            Indicator records. Returns an empty DataFrame if no data
            is available for the given parameters.

        Raises
        ------
        SDMXTimeoutError
            If the UNICEF SDMX API request times out.
        """
        self.logger.info(
            "Fetching UNICEF indicator '%s' | countries=%s | year=%s",
            self.indicator,
            self.countries or "ALL",
            self.year or "ALL",
        )

        kwargs = dict(
            indicator=self.indicator,
            sex=self.sex,
            format=self.format,
            latest=self.latest,
            circa=self.circa,
        )
        if self.countries is not None:
            kwargs["countries"] = self.countries
        if self.year is not None:
            kwargs["year"] = self.year
        if self.mrv is not None:
            kwargs["mrv"] = self.mrv

        from unicefdata import unicefData
        from unicefdata.sdmx_client import SDMXTimeoutError

        try:
            df: pd.DataFrame = unicefData(**kwargs)
        except SDMXTimeoutError:
            self.logger.error(
                "UNICEF SDMX API timed out for indicator '%s'", self.indicator
            )
            raise

        if df is None or df.empty:
            self.logger.warning(
                "No data returned for indicator '%s' — returning empty DataFrame",
                self.indicator,
            )
            return pd.DataFrame()

        self.logger.info(
            "Fetched %d records for indicator '%s'", len(df), self.indicator
        )
        return df

    def fetch_as_geodataframe(
        self,
        boundaries_gdf: gpd.GeoDataFrame,
        boundaries_iso3_col: str = "iso3",
        indicator_iso3_col: str = "REF_AREA",
    ) -> gpd.GeoDataFrame:
        """
        Fetch indicator data and spatially enrich it by joining to admin boundaries.

        Parameters
        ----------
        boundaries_gdf : gpd.GeoDataFrame
            GeoDataFrame with administrative boundary geometries (e.g. country polygons).
        boundaries_iso3_col : str
            Column in ``boundaries_gdf`` containing ISO3 country codes.
            Defaults to ``"iso3"``.
        indicator_iso3_col : str
            Column in the fetched indicator DataFrame containing ISO3 codes.
            UNICEF SDMX uses ``"REF_AREA"`` by default.

        Returns
        -------
        gpd.GeoDataFrame
            Boundaries GeoDataFrame enriched with UNICEF indicator values,
            preserving the original CRS.

        Raises
        ------
        ValueError
            If ``boundaries_iso3_col`` is missing from ``boundaries_gdf`` or
            ``indicator_iso3_col`` is missing from the fetched data.
        """
        df = self.fetch()

        if df.empty:
            self.logger.warning("No data to join — returning boundaries_gdf unchanged")
            return boundaries_gdf

        if boundaries_iso3_col not in boundaries_gdf.columns:
            raise ValueError(
                f"Column '{boundaries_iso3_col}' not found in boundaries_gdf. "
                f"Available columns: {list(boundaries_gdf.columns)}"
            )
        if indicator_iso3_col not in df.columns:
            raise ValueError(
                f"Column '{indicator_iso3_col}' not found in fetched indicator data. "
                f"Available columns: {list(df.columns)}"
            )

        gdf = boundaries_gdf.merge(
            df,
            left_on=boundaries_iso3_col,
            right_on=indicator_iso3_col,
            how="left",
        )
        self.logger.info(
            "Joined indicator '%s' to %d boundary features", self.indicator, len(gdf)
        )
        return gdf

    # ------------------------------------------------------------------
    # Discovery helpers (thin pass-throughs for convenience)
    # ------------------------------------------------------------------

    @staticmethod
    def search_indicators(query: str) -> pd.DataFrame:
        """Search available UNICEF indicators by keyword."""
        if not _HAS_UNICEF:
            raise ImportError(
                "search_indicators requires 'unicefdata'. "
                "Install it with: pip install 'giga-spatial[unicef]'"
            )
        from unicefdata import search_indicators
        return search_indicators(query)

    @staticmethod
    def list_categories() -> pd.DataFrame:
        """List all available indicator categories."""
        if not _HAS_UNICEF:
            raise ImportError(
                "list_categories requires 'unicefdata'. "
                "Install it with: pip install 'giga-spatial[unicef]'"
            )
        from unicefdata import list_categories
        return list_categories()
