from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import delta_sharing
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gigaspatial.config import config as global_config

LOGGER = global_config.get_logger("DeltaSharingDataStore")


class DeltaSharingConfig(BaseModel):
    """Configuration for Delta Sharing connection.

    Attributes:
        profile_file: Path to the delta-sharing profile file (JSON format).
        share_name: Name of the share to access.
        schema_name: Name of the schema within the share.
        enable_cache: Whether to cache loaded tables in memory.
    """

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    # All optional so can be composed from global_config + overrides
    profile_file: Optional[Path] = Field(
        default=None, description="Path to Delta Sharing profile configuration file"
    )
    share_name: Optional[str] = Field(
        default=None, description="Share name in Delta Sharing catalog"
    )
    schema_name: Optional[str] = Field(
        default=None, description="Schema name within the share"
    )
    enable_cache: bool = Field(
        default=True, description="Enable in-memory caching of loaded tables"
    )

    @field_validator("profile_file", mode="before")
    @classmethod
    def validate_profile_path(cls, v: Optional[Union[str, Path]]) -> Optional[Path]:
        """Ensure profile file exists and is valid if provided."""
        if v is None:
            return None
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Profile file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Profile path is not a file: {path}")
        return path


class DeltaSharingDataStore:
    """General-purpose Delta Sharing data accessor."""

    def __init__(
        self,
        *,
        # Global config values with individual override capability
        profile_file: Optional[Union[str, Path]] = global_config.API_PROFILE_FILE_PATH,
        share_name: Optional[str] = global_config.API_SHARE_NAME,
        schema_name: Optional[str] = global_config.API_SCHEMA_NAME,
        enable_cache: Optional[bool] = None,
    ):
        """
        Initialize with selective overrides of global config.

        Priority order (highest to lowest):
        1. Explicit kwargs
        2. config.* values (from env vars)
        3. Validation errors if required fields missing

        Examples:
            # Use all global config defaults
            store = DeltaSharingDataStore()

            # Use global share/profile but override schema
            store = DeltaSharingDataStore(schema_name="my-schema")

            # Fully explicit
            store = DeltaSharingDataStore(
                profile_file="/path/to/profile.json",
                share_name="gold",
                schema_name="school-master"
            )
        """
        # Validate required fields
        missing = []
        if profile_file is None:
            missing.append("profile_file")
        if share_name is None:
            missing.append("share_name")
        if schema_name is None:
            missing.append("schema_name")

        if missing:
            raise ValueError(
                f"Missing required config: {', '.join(missing)}. "
                "Set env vars (API_PROFILE_FILE_PATH, API_SHARE_NAME, API_SCHEMA_NAME) "
                "or pass them explicitly."
            )

        # Create validated config
        self.config = DeltaSharingConfig(
            profile_file=Path(profile_file),
            share_name=share_name,
            schema_name=schema_name,
            enable_cache=enable_cache if enable_cache is not None else True,
        )

        self._client: Optional[delta_sharing.SharingClient] = None
        self._cache: Dict[str, pd.DataFrame] = {}

        LOGGER.info(
            "Initialized DeltaSharingDataStore with "
            f"share={self.config.share_name}, schema={self.config.schema_name}, "
            f"profile={self.config.profile_file}"
        )

    @property
    def client(self) -> delta_sharing.SharingClient:
        """Lazy-load Delta Sharing client."""
        if self._client is None:
            self._client = delta_sharing.SharingClient(str(self.config.profile_file))
            LOGGER.debug(f"Created SharingClient from {self.config.profile_file}")
        return self._client

    def list_tables(
        self,
        schema_filter: Optional[str] = None,
        sort: bool = True,
    ) -> List[str]:
        """List all available tables in the configured schema."""
        schema = schema_filter or self.config.schema_name
        if schema is None:
            raise RuntimeError(
                "Schema name is not configured. "
                "Set 'schema_name' via env/global_config or pass it explicitly."
            )

        # 1) Try list_all_tables first
        all_tables = list(self.client.list_all_tables())
        if all_tables:
            table_names = [t.name for t in all_tables if t.schema == schema]
        else:
            # 2) Fallback: enumerate from configured share + schema
            LOGGER.debug(
                "list_all_tables() returned empty; falling back to "
                "share/schema enumeration"
            )
            # Find matching share
            shares = list(self.client.list_shares())
            try:
                share = next(s for s in shares if s.name == self.config.share_name)
            except StopIteration:
                LOGGER.warning(
                    "Configured share '%s' not found in list_shares()",
                    self.config.share_name,
                )
                return []

            # Find matching schema within that share
            schemas = list(self.client.list_schemas(share))
            try:
                schema_obj = next(s for s in schemas if s.name == schema)
            except StopIteration:
                LOGGER.warning(
                    "Configured schema '%s' not found in share '%s'",
                    schema,
                    self.config.share_name,
                )
                return []

            # List tables for that share+schema
            tables = list(self.client.list_tables(schema_obj))
            table_names = [t.name for t in tables]

        if sort:
            table_names.sort()
        return table_names

    def load_table(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Load a table from Delta Sharing with optional filtering."""
        if self.config.share_name is None or self.config.schema_name is None:
            raise RuntimeError(
                "share_name and schema_name must be configured before loading tables."
            )

        effective_cache = (
            use_cache if use_cache is not None else self.config.enable_cache
        )

        # Cache
        if effective_cache and table_name in self._cache:
            df = self._cache[table_name]
        else:
            table_url = (
                f"{self.config.profile_file}#"
                f"{self.config.share_name}."
                f"{self.config.schema_name}."
                f"{table_name}"
            )
            df = delta_sharing.load_as_pandas(table_url)
            if effective_cache:
                self._cache[table_name] = df

        if filters:
            for column, value in filters.items():
                if column not in df.columns:
                    raise ValueError(
                        f"Filter column '{column}' not found in table '{table_name}'"
                    )
                df = df[df[column] == value]

        return df

    def load_multiple_tables(
        self,
        table_names: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Load and concatenate multiple tables."""
        dfs = [self.load_table(name, filters=filters) for name in table_names]
        return pd.concat(dfs, ignore_index=True)

    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Retrieve metadata for a table."""
        df = self.load_table(table_name)
        return {
            "table_name": table_name,
            "columns": df.columns.tolist(),
            "data_types": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "num_records": len(df),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }

    def clear_cache(self, table_name: Optional[str] = None) -> None:
        """Clear cached data."""
        if table_name is None:
            self._cache.clear()
        else:
            self._cache.pop(table_name, None)

    def get_cached_tables(self) -> List[str]:
        """Get list of currently cached table names."""
        return list(self._cache.keys())

    @property
    def cache_size_mb(self) -> float:
        """Total memory used by cache in megabytes."""
        total_bytes = sum(
            df.memory_usage(deep=True).sum() for df in self._cache.values()
        )
        return total_bytes / 1024**2
