from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, Field, field_validator
import pycountry

from gigaspatial.core.io.bigquery_client import BigQueryClient, BigQueryClientConfig
from gigaspatial.config import config as global_config

logger = global_config.get_logger("MLabHandler")

_MLAB_PROJECT = "measurement-lab"
_MLAB_NDT7 = f"`{_MLAB_PROJECT}.ndt.ndt7`"

# Scalar fields from `a` record - all are directly selectable
_NDT7_A_FIELDS: dict[str, str] = {
    "a.UUID": "a_uuid",
    "a.TestTime": "test_time",
    "a.CongestionControl": "congestion_control",
    "a.MeanThroughputMbps": "mean_throughput_mbps",
    "a.MinRTT": "min_rtt_ms",
    "a.LossRate": "loss_rate",
}

# Scalar fields from raw.Download and raw.Upload sub-records
_NDT7_RAW_DOWNLOAD_FIELDS: dict[str, str] = {
    "raw.Download.UUID": "download_uuid",
    "raw.Download.StartTime": "download_start_time",
    "raw.Download.EndTime": "download_end_time",
    "raw.Download.ServerMeasurements[SAFE_OFFSET(0)].AppInfo.NumBytes": "download_server_num_bytes",
    "raw.Download.ServerMeasurements[SAFE_OFFSET(0)].AppInfo.ElapsedTime": "download_server_elapsed_us",
    "raw.Download.ServerMeasurements[SAFE_OFFSET(0)].BBRInfo.BW": "download_bbr_bw",
    "raw.Download.ServerMeasurements[SAFE_OFFSET(0)].BBRInfo.MinRTT": "download_bbr_min_rtt_us",
    "raw.Download.ServerMeasurements[SAFE_OFFSET(0)].TCPInfo.RTT": "download_tcp_rtt_us",
    "raw.Download.ServerMeasurements[SAFE_OFFSET(0)].TCPInfo.TotalRetrans": "download_tcp_total_retrans",
}

_NDT7_RAW_UPLOAD_FIELDS: dict[str, str] = {
    "raw.Upload.UUID": "upload_uuid",
    "raw.Upload.StartTime": "upload_start_time",
    "raw.Upload.EndTime": "upload_end_time",
    "raw.Upload.ServerMeasurements[SAFE_OFFSET(0)].AppInfo.NumBytes": "upload_server_num_bytes",
    "raw.Upload.ServerMeasurements[SAFE_OFFSET(0)].AppInfo.ElapsedTime": "upload_server_elapsed_us",
    "raw.Upload.ServerMeasurements[SAFE_OFFSET(0)].BBRInfo.BW": "upload_bbr_bw",
    "raw.Upload.ServerMeasurements[SAFE_OFFSET(0)].BBRInfo.MinRTT": "upload_bbr_min_rtt_us",
    "raw.Upload.ServerMeasurements[SAFE_OFFSET(0)].TCPInfo.RTT": "upload_tcp_rtt_us",
    "raw.Upload.ServerMeasurements[SAFE_OFFSET(0)].TCPInfo.TotalRetrans": "upload_tcp_total_retrans",
}

_NDT7_CLIENT_GEO_FIELDS: dict[str, str] = {
    "client.Geo.Latitude": "lat",
    "client.Geo.Longitude": "lon",
}

_NDT7_RAW_COMMON_FIELDS: dict[str, str] = {
    "raw.ServerIP": "server_ip",
    "raw.ClientIP": "client_ip",
    "raw.StartTime": "start_time",
    "raw.EndTime": "end_time",
}

MeasurementType = Literal["download", "upload", "both"]


class MLabConfig(BaseModel):
    """
    Dataset-specific configuration for M-Lab NDT7 BigQuery data.

    Auth and GCP credentials are handled by ``BigQueryClientConfig`` /
    ``BigQueryClient`` via global_config. This config only holds MLab-specific
    query parameters.

    Parameters
    ----------
    project_id : str
        GCP project hosting M-Lab public datasets.
        Defaults to ``"measurement-lab"``.
    dataset : str
        M-Lab BigQuery dataset name. Defaults to ``"ndt"``.
    default_start_date : str
        ISO date string (YYYY-MM-DD) used as the default lower bound for
        time-range queries. Defaults to ``"2020-02-18"`` — the first date
        ndt7 data was collected.
    """

    project_id: str = Field(default=_MLAB_PROJECT)
    dataset: str = Field(default="ndt7")
    default_start_date: str = Field(default="2020-02-18")

    @field_validator("default_start_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("default_start_date must be ISO format: YYYY-MM-DD")
        return v


class MLabHandler:
    """
    Handler for querying M-Lab NDT7 network measurement data from BigQuery.

    Composes ``BigQueryClient`` for all query execution against the
    ``measurement-lab.ndt.ndt7`` table. Returns results as DataFrames
    or GeoDataFrames using ``client.Geo.Latitude`` / ``client.Geo.Longitude``
    available in the ndt7 table schema.

    Parameters
    ----------
    config : MLabConfig, optional
        Dataset-level configuration. Defaults to ``MLabConfig()``.
    client_config : BigQueryClientConfig, optional
        Auth/project configuration for the underlying BigQuery client.
        Defaults to ``BigQueryClientConfig()`` which reads from global_config.

    Examples
    --------
    >>> handler = MLabHandler()
    >>> df = handler.query_ndt7("TR", start_date="2023-01-01")
    >>> df = handler.query_ndt7("KE", measurement="upload", start_date="2023-06-01")
    """

    def __init__(
        self,
        config: Optional[MLabConfig] = None,
        client_config: Optional[BigQueryClientConfig] = None,
    ) -> None:
        self.config = config or MLabConfig()
        self.bq = BigQueryClient(client_config)
        logger.info(
            "MLabHandler initialized (project=%s, dataset=%s)",
            self.config.project_id,
            self.config.dataset,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_country_code(country_code: str) -> str:
        """
        Validate and normalise an ISO 3166-1 alpha-2 country code.

        Parameters
        ----------
        country_code : str
            Two-letter country code (case-insensitive).

        Returns
        -------
        str
            Upper-cased, validated country code.

        Raises
        ------
        ValueError
            If the input is not a two-letter alphabetic string.
        """
        try:
            return pycountry.countries.lookup(country_code).alpha_2
        except LookupError as e:
            raise ValueError(f"Invalid country '{country_code}': {e}") from e

    @staticmethod
    def _validate_date(date_str: str) -> str:
        """
        Validate an ISO date string (YYYY-MM-DD).

        Parameters
        ----------
        date_str : str
            Date string to validate.

        Returns
        -------
        str
            The original string if valid.

        Raises
        ------
        ValueError
            If the string is not a valid ISO date.
        """
        try:
            datetime.fromisoformat(date_str)
        except ValueError:
            raise ValueError(f"date must be ISO format YYYY-MM-DD, got: {date_str!r}")
        return date_str

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_select_clause(measurement: MeasurementType) -> str:
        """
        Build the SELECT clause for an ndt7 query based on measurement type.

        Always includes ``id``, ``date``, all ``a.*`` scalar fields, and
        ``raw`` common fields (ServerIP, ClientIP, StartTime, EndTime).
        Measurement-specific raw fields (Download or Upload sub-records)
        are added based on the ``measurement`` argument.

        Parameters
        ----------
        measurement : {"download", "upload", "both"}
            Which measurement direction's raw fields to include.

        Returns
        -------
        str
            A formatted SQL SELECT clause (without the SELECT keyword).
        """
        fields: dict[str, str] = {
            "id": "id",
            "date": "date",
            **_NDT7_CLIENT_GEO_FIELDS,
            **_NDT7_A_FIELDS,
            **_NDT7_RAW_COMMON_FIELDS,
        }
        if measurement in ("download", "both"):
            fields.update(_NDT7_RAW_DOWNLOAD_FIELDS)
        if measurement in ("upload", "both"):
            fields.update(_NDT7_RAW_UPLOAD_FIELDS)

        return ",\n            ".join(
            f"{bq_path} AS {alias}" for bq_path, alias in fields.items()
        )

    def _build_ndt7_sql(
        self,
        country_code: str,
        start_date: str,
        end_date: Optional[str],
        measurement: MeasurementType,
    ) -> str:
        """
        Build the full SQL query for the ndt7 table.

        Parameters
        ----------
        country_code : str
            Validated ISO alpha-2 country code, filtered via ``client.Geo.CountryCode``.
        start_date : str
            ISO date lower bound for the ``date`` partition filter.
        end_date : str, optional
            ISO date upper bound. No upper bound applied if ``None``.
        measurement : {"download", "upload", "both"}
            Which measurement direction's raw fields to include.

        Returns
        -------
        str
            Complete SQL query string ready for execution.
        """
        select_clause = self._build_select_clause(measurement)

        date_filter = f"date >= '{start_date}'"
        if end_date:
            date_filter += f" AND date <= '{end_date}'"

        measurement_filter = ""
        if measurement == "download":
            measurement_filter = "AND raw.Download IS NOT NULL"
        elif measurement == "upload":
            measurement_filter = "AND raw.Upload IS NOT NULL"

        return f"""
        SELECT
            {select_clause}
        FROM {_MLAB_NDT7}
        WHERE
            client.Geo.CountryCode = '{country_code}'
            AND {date_filter}
            {measurement_filter}
        """

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def query_ndt7(
        self,
        country_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        measurement: MeasurementType = "download",
    ) -> pd.DataFrame:
        """
        Query the NDT7 table and return results as a DataFrame.

        Parameters
        ----------
        country_code : str
            ISO 3166-1 alpha-2 country code (e.g. ``"TR"``, ``"KE"``).
        start_date : str, optional
            ISO date lower bound (YYYY-MM-DD).
            Defaults to ``config.default_start_date``.
        end_date : str, optional
            ISO date upper bound (YYYY-MM-DD). No upper bound if not provided.
        measurement : {"download", "upload", "both"}
            Which measurement direction to query. Defaults to ``"download"``.

        Returns
        -------
        pd.DataFrame
            NDT7 measurements with ``id``, ``date``, derived ``a.*`` fields,
            and raw measurement fields for the selected direction.
        """
        country_code = self._validate_country_code(country_code)
        start_date = start_date or self.config.default_start_date
        if end_date:
            self._validate_date(end_date)

        logger.info(
            "Querying ndt7 %s for %s [%s → %s]",
            measurement,
            country_code,
            start_date,
            end_date or "open",
        )
        sql = self._build_ndt7_sql(country_code, start_date, end_date, measurement)
        return self.bq.query_to_dataframe(sql)

    def query_ndt7_gdf(
        self,
        country_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        measurement: MeasurementType = "download",
    ) -> gpd.GeoDataFrame:
        """
        Query the NDT7 table and return results as a GeoDataFrame.

        Convenience wrapper around ``query_ndt7`` that constructs point
        geometry from ``client.Geo.Latitude`` / ``client.Geo.Longitude``.
        Rows where lat/lon are null are dropped before geometry construction.

        Parameters
        ----------
        country_code : str
            ISO 3166-1 alpha-2 country code (e.g. ``"TR"``, ``"KE"``).
        start_date : str, optional
            ISO date lower bound (YYYY-MM-DD).
            Defaults to ``config.default_start_date``.
        end_date : str, optional
            ISO date upper bound (YYYY-MM-DD).
        measurement : {"download", "upload", "both"}
            Which measurement direction to query. Defaults to ``"download"``.

        Returns
        -------
        gpd.GeoDataFrame
            Point GeoDataFrame in EPSG:4326.
        """
        df = self.query_ndt7(country_code, start_date, end_date, measurement)
        df = df.dropna(subset=["lat", "lon"])
        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        )

    def estimate_query_cost(
        self,
        country_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        measurement: MeasurementType = "download",
    ) -> float:
        """
        Estimate the cost of a query before executing it.

        Parameters
        ----------
        country_code : str
            ISO 3166-1 alpha-2 country code.
        start_date : str, optional
            ISO date lower bound. Defaults to ``config.default_start_date``.
        end_date : str, optional
            ISO date upper bound.
        measurement : {"download", "upload", "both"}
            Which measurement direction to estimate for.

        Returns
        -------
        float
            Estimated cost in USD (on-demand pricing).

        Examples
        --------
        >>> handler.estimate_query_cost("TR", start_date="2023-01-01", end_date="2023-01-31")
        0.043
        """
        country_code = self._validate_country_code(country_code)
        start_date = start_date or self.config.default_start_date
        if end_date:
            self._validate_date(end_date)
        sql = self._build_ndt7_sql(country_code, start_date, end_date, measurement)
        cost = self.bq.get_query_cost_estimate(sql)
        logger.info("Estimated query cost: $%.4f USD", cost)
        return cost
