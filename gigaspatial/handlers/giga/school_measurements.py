# gigaspatial/handlers/giga_school_measurements.py

import logging
from datetime import date, datetime
from typing import Optional, Union

import pandas as pd
import pycountry
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from gigaspatial.config import config as global_config
from gigaspatial.core.http import AuthConfig, AuthType, RestApiClientConfig
from gigaspatial.handlers.giga.api_client import GigaApiClient

_API_URL = "https://uni-ooi-giga-maps-service.azurewebsites.net/api/v1/all_measurements"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GigaSchoolMeasurementsFetcher:
    """
    Fetch and process school daily realtime connectivity measurements from the Giga API.

    Includes download/upload speeds, latency, and connectivity performance data.

    Parameters
    ----------
    country : str
        Country name or ISO 3166 alpha-2/alpha-3 code.
    start_date : str, date, or datetime
        Start of the measurement date range (inclusive).
    end_date : str, date, or datetime
        End of the measurement date range (inclusive).
    api_key : str
        Bearer token for the Giga API. Defaults to GIGA_SCHOOL_MEASUREMENTS_API_KEY.
    page_size : int
        Number of records per API page. Defaults to 1000.
    max_retries : int
        Maximum retry attempts on transient errors. Defaults to 3.
    giga_id_school : str, optional
        Filter results to a specific school by Giga ID.
    logger : logging.Logger, optional
        Logger instance. Defaults to the global config logger.

    Examples
    --------
    >>> fetcher = GigaSchoolMeasurementsFetcher(
    ...     country="Kenya",
    ...     start_date="2024-01-01",
    ...     end_date="2024-01-31",
    ... )
    >>> df = fetcher.fetch_measurements()
    >>> df = fetcher.fetch_measurements(max_pages=3)
    """

    country: str = Field(...)
    start_date: Union[str, date, datetime] = Field(...)
    end_date: Union[str, date, datetime] = Field(...)
    api_key: str = Field(default=global_config.GIGA_SCHOOL_MEASUREMENTS_API_KEY)
    page_size: int = Field(default=1000, description="Number of records per API page")
    max_retries: int = Field(default=3, description="Max retry attempts on errors")
    giga_id_school: Optional[str] = Field(
        default=None, description="Optional specific Giga school ID to fetch"
    )
    logger: Optional[logging.Logger] = Field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            self.country = pycountry.countries.lookup(self.country).alpha_3
        except LookupError:
            raise ValueError(f"Invalid country code provided: {self.country}")

        self.start_date = self._format_date(self.start_date)
        self.end_date = self._format_date(self.end_date)

        if self.start_date > self.end_date:
            raise ValueError("start_date must be before or equal to end_date")

        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_date(value: Union[str, date, datetime]) -> str:
        """
        Normalise any date-like input to YYYY-MM-DD string.

        Parameters
        ----------
        value : str, date, or datetime

        Returns
        -------
        str
            Date in YYYY-MM-DD format.

        Raises
        ------
        ValueError
            If the value cannot be parsed as a date.
        """
        if isinstance(value, (date, datetime)):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, str):
            try:
                datetime.strptime(value, "%Y-%m-%d")
                return value
            except ValueError:
                try:
                    return pd.to_datetime(value).strftime("%Y-%m-%d")
                except Exception:
                    raise ValueError(
                        f"Invalid date format: {value!r}. Expected YYYY-MM-DD."
                    )
        raise ValueError(f"Invalid date type: {type(value)}")

    def _build_client(self) -> GigaApiClient:
        config = RestApiClientConfig(
            base_url=_API_URL,
            auth=AuthConfig(auth_type=AuthType.BEARER, api_key=self.api_key),
            max_retries=self.max_retries,
            default_headers={"Accept": "application/json"},
        )
        return GigaApiClient(config, page_size=self.page_size)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def fetch_measurements(
        self,
        max_pages: Optional[int] = None,
        giga_id_school: Optional[str] = None,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
    ) -> pd.DataFrame:
        """
        Fetch school connectivity measurements.

        Parameters
        ----------
        max_pages : int, optional
            Limit the number of pages fetched. Useful for testing.
        giga_id_school : str, optional
            Override the instance-level school ID filter.
        start_date : str, date, or datetime, optional
            Override the instance-level start date.
        end_date : str, date, or datetime, optional
            Override the instance-level end date.

        Returns
        -------
        pd.DataFrame
            Measurements with processed performance columns.
            Empty DataFrame if no data is returned.
        """
        school_id = giga_id_school or self.giga_id_school
        _start = self._format_date(start_date) if start_date else self.start_date
        _end = self._format_date(end_date) if end_date else self.end_date

        base_params: dict = {
            "country_iso3_code": self.country,
            "start_date": _start,
            "end_date": _end,
            "size": self.page_size,
            "page": 1,
        }
        if school_id:
            base_params["giga_id_school"] = school_id
            self.logger.info("Filtering for specific school ID: %s", school_id)

        self.logger.info(
            "Fetching measurements for country: %s from %s to %s",
            self.country, _start, _end,
        )
        all_records: list[dict] = []

        with self._build_client() as client:
            for page_num, page in enumerate(
                client.paginate("/", params=base_params), start=1
            ):
                all_records.extend(page)
                self.logger.info("Fetched page %d — %d records", page_num, len(page))

                if max_pages and page_num >= max_pages:
                    self.logger.info("Reached max_pages limit (%d)", max_pages)
                    break

        self.logger.info("Total records fetched: %d", len(all_records))

        if not all_records:
            self.logger.warning("No data fetched — returning empty DataFrame")
            return pd.DataFrame()

        return self._process_measurements_data(pd.DataFrame(all_records))

    def _process_measurements_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich the raw measurements DataFrame with derived performance columns.

        Parameters
        ----------
        df : pd.DataFrame
            Raw records from the API.

        Returns
        -------
        pd.DataFrame
            Enhanced DataFrame with speed categories, quality flags, and
            temporal breakdown columns.
        """
        if df.empty:
            return df

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["date_only"] = df["date"].dt.date
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day_of_week"] = df["date"].dt.day_name()

        for col in ["download_speed", "upload_speed", "latency"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "download_speed" in df.columns:
            df["download_speed_category"] = pd.cut(
                df["download_speed"],
                bins=[0, 5, 25, 100, float("inf")],
                labels=["Very Slow (<5 Mbps)", "Slow (5-25 Mbps)", "Moderate (25-100 Mbps)", "Fast (>100 Mbps)"],
                include_lowest=True,
            )

        if "upload_speed" in df.columns:
            df["upload_speed_category"] = pd.cut(
                df["upload_speed"],
                bins=[0, 1, 10, 50, float("inf")],
                labels=["Very Slow (<1 Mbps)", "Slow (1-10 Mbps)", "Moderate (10-50 Mbps)", "Fast (>50 Mbps)"],
                include_lowest=True,
            )

        if "latency" in df.columns:
            df["latency_category"] = pd.cut(
                df["latency"],
                bins=[0, 50, 150, 300, float("inf")],
                labels=["Excellent (<50ms)", "Good (50-150ms)", "Fair (150-300ms)", "Poor (>300ms)"],
                include_lowest=True,
            )

        if "download_speed" in df.columns and "upload_speed" in df.columns:
            df["has_broadband"] = (df["download_speed"] >= 25) & (df["upload_speed"] >= 3)
            df["has_basic_connectivity"] = (df["download_speed"] >= 1) & (df["upload_speed"] >= 0.5)

        df["has_complete_measurement"] = (
            df["download_speed"].notna()
            & df["upload_speed"].notna()
            & df["latency"].notna()
        )

        self.logger.info("Processed measurement data for %d records", len(df))
        return df

    # ------------------------------------------------------------------
    # Analytics helpers (unchanged logic, cleaned style)
    # ------------------------------------------------------------------

    def get_performance_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a comprehensive summary of connectivity performance metrics.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame returned by fetch_measurements().

        Returns
        -------
        dict
            Summary statistics covering speeds, latency, quality, and temporality.
        """
        if df.empty:
            return {"error": "No data available"}

        summary: dict = {
            "total_measurements": len(df),
            "country": df["country_iso3_code"].iloc[0] if "country_iso3_code" in df.columns else "Unknown",
            "date_range": {
                "start": df["date"].min().strftime("%Y-%m-%d") if "date" in df.columns else None,
                "end": df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns else None,
            },
        }

        if "giga_id_school" in df.columns:
            n = df["giga_id_school"].nunique()
            summary["unique_schools_measured"] = n
            summary["avg_measurements_per_school"] = len(df) / n if n > 0 else 0

        for col in ["download_speed", "upload_speed"]:
            if col in df.columns:
                s = df[col].dropna()
                if len(s):
                    summary[f"{col}_stats"] = {
                        "mean": float(s.mean()), "median": float(s.median()),
                        "min": float(s.min()), "max": float(s.max()), "std": float(s.std()),
                    }

        if "latency" in df.columns:
            s = df["latency"].dropna()
            if len(s):
                summary["latency_stats"] = {
                    "mean": float(s.mean()), "median": float(s.median()),
                    "min": float(s.min()), "max": float(s.max()), "std": float(s.std()),
                }

        for cat in ["download_speed_category", "upload_speed_category", "latency_category"]:
            if cat in df.columns:
                summary[cat.replace("_category", "_breakdown")] = df[cat].value_counts().to_dict()

        for flag, key in [
            ("has_broadband", "broadband"),
            ("has_basic_connectivity", "basic_connectivity"),
            ("has_complete_measurement", "complete_measurements"),
        ]:
            if flag in df.columns:
                summary[f"{key}_count"] = int(df[flag].sum())
                summary[f"{key}_percentage"] = float(df[flag].mean() * 100)

        if "data_source" in df.columns:
            summary["data_sources"] = df["data_source"].value_counts().to_dict()

        if "day_of_week" in df.columns:
            summary["measurements_by_day_of_week"] = df["day_of_week"].value_counts().to_dict()

        self.logger.info("Generated performance summary")
        return summary

    def get_school_performance_comparison(
        self, df: pd.DataFrame, top_n: int = 10
    ) -> dict:
        """
        Compare download speed performance across schools.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame returned by fetch_measurements().
        top_n : int
            Number of top and bottom schools to include. Defaults to 10.

        Returns
        -------
        dict
            Top and bottom performing schools by mean download speed.
        """
        if df.empty or "giga_id_school" not in df.columns:
            return {"error": "No school data available"}

        agg: dict = {
            "download_speed": ["mean", "median", "count"],
            "upload_speed": ["mean", "median"],
            "latency": ["mean", "median"],
        }
        if "has_broadband" in df.columns:
            agg["has_broadband"] = "mean"

        school_stats = df.groupby("giga_id_school").agg(agg).round(2)
        school_stats.columns = ["_".join(col).strip() for col in school_stats.columns]

        if "download_speed_mean" not in school_stats.columns:
            return {"error": "Insufficient data for school comparison"}

        return {
            "top_performing_schools": school_stats.nlargest(top_n, "download_speed_mean").to_dict("index"),
            "bottom_performing_schools": school_stats.nsmallest(top_n, "download_speed_mean").to_dict("index"),
            "total_schools_analyzed": len(school_stats),
        }
