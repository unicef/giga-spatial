try:
    import google.auth
    from google.cloud import bigquery, bigquery_storage
    from google.oauth2 import service_account

    _HAS_BQ = True
except ImportError:
    _HAS_BQ = False

import pandas as pd
from typing import Optional
from pydantic import BaseModel, model_validator

from gigaspatial.config import config as global_config

logger = global_config.get_logger("BigQueryClient")

BIGQUERY_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# BigQuery on-demand pricing: $6.25 per TiB (us/eu regions, as of 2026)
# Note that the first 1 TiB per month is free per account.
# See: https://cloud.google.com/bigquery/pricing
_BQ_COST_PER_TIB_USD = 6.25


class BigQueryClientConfig(BaseModel):
    """
    Configuration for authenticating and initializing a BigQuery client.

    Defaults are resolved from GigaSpatial's global_config, mirroring the
    pattern used by the Google Earth Engine profiler.

    Credential resolution priority:
    1. Service account key file (``service_account_key_path``)
    2. Application Default Credentials (ADC) — for GCP-hosted environments

    Parameters
    ----------
    project : str, optional
        GCP project ID to bill queries against.
        Defaults to ``global_config.GOOGLE_CLOUD_PROJECT``.
    service_account : str, optional
        Service account email address.
        Defaults to ``global_config.GOOGLE_SERVICE_ACCOUNT``.
    service_account_key_path : str, optional
        Path to the service account JSON key file.
        Defaults to ``global_config.GOOGLE_SERVICE_ACCOUNT_KEY_PATH``.
    use_bq_storage : bool
        Enable BigQuery Storage API for faster ``query_to_dataframe`` reads.
        Defaults to ``False``.
    """

    project: Optional[str] = global_config.GOOGLE_CLOUD_PROJECT
    service_account: Optional[str] = global_config.GOOGLE_SERVICE_ACCOUNT
    service_account_key_path: Optional[str] = (
        global_config.GOOGLE_SERVICE_ACCOUNT_KEY_PATH
    )
    use_bq_storage: bool = False

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_project(self) -> "BigQueryClientConfig":
        if not self.project:
            raise ValueError(
                "`project` must be set explicitly or via "
                "GOOGLE_CLOUD_PROJECT in global_config."
            )
        return self


class BigQueryClient:
    """
    A generalizable Google BigQuery client for GigaSpatial.

    Provides generic, dataset-agnostic methods for schema inspection and
    query execution. Dataset-specific handlers (e.g. MLabHandler) should
    compose this client rather than reimplement it.

    Parameters
    ----------
    config : BigQueryClientConfig, optional
        Validated configuration object. If not provided, a default config
        is constructed from global_config.

    Examples
    --------
    >>> client = BigQueryClient()  # uses global_config defaults
    >>> client.list_datasets()
    >>> df = client.query_to_dataframe("SELECT * FROM `project.dataset.table` LIMIT 10")

    >>> custom = BigQueryClient(BigQueryClientConfig(project="my-gcp-project"))
    """

    def __init__(self, config: Optional[BigQueryClientConfig] = None) -> None:
        if not _HAS_BQ:
            raise ImportError(
                "BigQueryClient requires 'google-cloud-bigquery' and 'db-dtypes'. "
                "Install them with: pip install 'giga-spatial[bq]'"
            )
        self.config = config or BigQueryClientConfig()
        self._credentials = self._resolve_credentials()
        self.client = bigquery.Client(
            project=self.config.project,
            credentials=self._credentials,
        )
        self.storage_client = (
            bigquery_storage.BigQueryReadClient(credentials=self._credentials)
            if self.config.use_bq_storage
            else None
        )
        logger.info(
            "BigQueryClient initialized for project '%s' (service_account=%s)",
            self.config.project,
            self.config.service_account or "ADC",
        )

    def _resolve_credentials(self):
        """
        Resolve credentials using the following priority:

        1. Service account key file (``config.service_account_key_path``)
        2. Application Default Credentials (ADC)

        Returns
        -------
        google.auth.credentials.Credentials
        """
        if self.config.service_account_key_path:
            logger.debug(
                "Using service account key: %s",
                self.config.service_account_key_path,
            )
            return service_account.Credentials.from_service_account_file(
                self.config.service_account_key_path,
                scopes=BIGQUERY_SCOPES,
            )
        logger.debug("Falling back to Application Default Credentials (ADC).")
        credentials, _ = google.auth.default(scopes=BIGQUERY_SCOPES)
        return credentials

    def list_datasets(self, project_id: Optional[str] = None) -> list[str]:
        """
        List all available datasets in the configured project.

        Parameters
        ----------
        project_id : str, optional
            The BigQuery project ID. Defaults to ``config.project``.

        Returns
        -------
        list[str]
            Dataset IDs within the given project.
        """
        target_project = project_id or self.config.project
        datasets = self.client.list_datasets(target_project)
        return [ds.dataset_id for ds in datasets]

    def list_tables(self, dataset_id: str) -> list[str]:
        """
        List all tables within a dataset.

        Parameters
        ----------
        dataset_id : str
            The BigQuery dataset ID.

        Returns
        -------
        list[str]
            Table IDs within the given dataset.
        """
        tables = self.client.list_tables(dataset_id)
        return [t.table_id for t in tables]

    def get_table_schema(self, dataset_id: str, table_id: str) -> list[dict]:
        """
        Retrieve the schema for a specific table.

        Parameters
        ----------
        dataset_id : str
            The BigQuery dataset ID.
        table_id : str
            The BigQuery table ID.

        Returns
        -------
        list[dict]
            List of field descriptors with keys:
            ``name``, ``type``, ``mode``, ``description``.
        """
        table_ref = f"{dataset_id}.{table_id}"
        table = self.client.get_table(table_ref)
        return [
            {
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode,
                "description": field.description,
            }
            for field in table.schema
        ]

    def query(self, sql: str, **kwargs) -> "bigquery.table.RowIterator":
        """
        Execute a SQL query and return a raw result iterator.

        Parameters
        ----------
        sql : str
            Standard SQL query string.
        **kwargs
            Additional keyword arguments forwarded to ``bigquery.Client.query()``.

        Returns
        -------
        google.cloud.bigquery.table.RowIterator
        """
        logger.debug("Executing BigQuery query: %.200s", sql)
        return self.client.query(sql, **kwargs).result()

    def query_to_dataframe(
        self,
        sql: str,
        dtypes: Optional[dict] = None,
        max_gb_allowed: Optional[int] = 10,  # Safety rail
        is_ci: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a pandas DataFrame.

        Parameters
        ----------
        sql : str
            Standard SQL query string.
        dtypes : dict, optional
            Column dtype overrides passed to ``to_dataframe()``.
        max_gb_allowed : int, optional
            Maximum number of gigabytes to bill for the query. Defaults to 10.
        is_ci : bool, optional
            Whether the query is running in a CI environment. Defaults to False.
        **kwargs
            Additional keyword arguments forwarded to ``bigquery.Client.query()``.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the query results.
        """

        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=max_gb_allowed * (1024**3) if max_gb_allowed else None
        )

        # Merge any extra kwargs into job_config if needed
        job = self.client.query(sql, job_config=job_config, **kwargs)

        # Dataset rows can be complex; using the storage client
        return job.result().to_dataframe(
            bqstorage_client=self.storage_client,
            progress_bar_type="tqdm" if not is_ci else None,
            dtypes=dtypes,
        )

    def get_query_cost_estimate(self, sql: str) -> float:
        """
        Estimate the cost of a BigQuery query.

        Parameters
        ----------
        sql : str
            Standard SQL query string.

        Returns
        -------
        float
            Estimated cost in USD based on on-demand pricing
            (${_BQ_COST_PER_TIB_USD}/TiB). For flat-rate pricing this
            will not reflect actual cost.
        """
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = self.client.query(sql, job_config=job_config)
        return (query_job.total_bytes_processed / (1024**4)) * _BQ_COST_PER_TIB_USD
