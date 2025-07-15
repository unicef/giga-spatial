from typing import List, Dict, Optional, Union, Literal

import pandas as pd
import dask.dataframe as dd

from sqlalchemy import inspect, MetaData, Table, select, create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
import warnings

from gigaspatial.config import config as global_config


class DBConnection:
    """
    A unified database connection class supporting both Trino and PostgreSQL.
    """

    DB_CONFIG = global_config.DB_CONFIG or {}

    def __init__(
        self,
        db_type: Literal["postgresql", "trino"] = DB_CONFIG.get(
            "db_type", "postgresql"
        ),
        host: Optional[str] = DB_CONFIG.get("host", None),
        port: Union[int, str] = DB_CONFIG.get("port", None),  # type: ignore
        user: Optional[str] = DB_CONFIG.get("user", None),
        password: Optional[str] = DB_CONFIG.get("password", None),
        catalog: Optional[str] = DB_CONFIG.get("catalog", None),  # For Trino
        database: Optional[str] = DB_CONFIG.get("database", None),  # For PostgreSQL
        schema: str = DB_CONFIG.get("schema", "public"),  # Default for PostgreSQL
        http_scheme: str = DB_CONFIG.get("http_scheme", "https"),  # For Trino
        sslmode: str = DB_CONFIG.get("sslmode", "require"),  # For PostgreSQL
        **kwargs,
    ):
        """
        Initialize a database connection for either Trino or PostgreSQL.

        Args:
            db_type: Either "trino" or "postgresql"
            host: Database server host
            port: Database server port
            user: Username
            password: Password
            catalog: Trino catalog name
            database: PostgreSQL database name
            schema: Default schema name
            http_scheme: For Trino ("http" or "https")
            sslmode: For PostgreSQL (e.g., "require", "verify-full")
            **kwargs: Additional connection parameters
        """
        self.db_type = db_type.lower()
        self.host = host
        self.port = str(port) if port else None
        self.user = user
        self.password = quote_plus(password) if password else None
        self.default_schema = schema

        if self.db_type == "trino":
            self.catalog = catalog
            self.http_scheme = http_scheme
            self.engine = self._create_trino_engine(**kwargs)
        elif self.db_type == "postgresql":
            self.database = database
            self.sslmode = sslmode
            self.engine = self._create_postgresql_engine(**kwargs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        self._add_event_listener()

    def _create_trino_engine(self, **kwargs) -> Engine:
        """Create a Trino SQLAlchemy engine."""
        self._connection_string = (
            f"trino://{self.user}:{self.password}@{self.host}:{self.port}/"
            f"{self.catalog}/{self.default_schema}"
        )
        return create_engine(
            self._connection_string,
            connect_args={"http_scheme": self.http_scheme},
            **kwargs,
        )

    def _create_postgresql_engine(self, **kwargs) -> Engine:
        """Create a PostgreSQL SQLAlchemy engine."""
        self._connection_string = (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/"
            f"{self.database}?sslmode={self.sslmode}"
        )
        return create_engine(self._connection_string, **kwargs)

    def _add_event_listener(self):
        """Add event listeners for schema setting."""
        if self.db_type == "trino":

            @event.listens_for(self.engine, "connect", insert=True)
            def set_current_schema(dbapi_connection, connection_record):
                cursor_obj = dbapi_connection.cursor()
                try:
                    cursor_obj.execute(f"USE {self.default_schema}")
                except Exception as e:
                    warnings.warn(f"Could not set schema to {self.default_schema}: {e}")
                finally:
                    cursor_obj.close()

    def get_connection_string(self) -> str:
        """
        Returns the connection string used to create the engine.

        Returns:
            str: The connection string.
        """
        return self._connection_string

    def get_schema_names(self) -> List[str]:
        """Get list of all schema names."""
        inspector = inspect(self.engine)
        return inspector.get_schema_names()

    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of table names in a schema."""
        schema = schema or self.default_schema
        inspector = inspect(self.engine)
        return inspector.get_table_names(schema=schema)

    def get_view_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of view names in a schema."""
        schema = schema or self.default_schema
        inspector = inspect(self.engine)
        return inspector.get_view_names(schema=schema)

    def get_column_names(
        self, table_name: str, schema: Optional[str] = None
    ) -> List[str]:
        """Get column names for a specific table."""
        if "." in table_name:
            schema, table_name = table_name.split(".")
        else:
            schema = schema or self.default_schema

        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name, schema=schema)
        return [col["name"] for col in columns]

    def get_table_info(
        self, table_name: str, schema: Optional[str] = None
    ) -> List[Dict]:
        """Get detailed column information for a table."""
        if "." in table_name:
            schema, table_name = table_name.split(".")
        else:
            schema = schema or self.default_schema

        inspector = inspect(self.engine)
        return inspector.get_columns(table_name, schema=schema)

    def get_primary_keys(
        self, table_name: str, schema: Optional[str] = None
    ) -> List[str]:
        """Get primary key columns for a table."""
        if "." in table_name:
            schema, table_name = table_name.split(".")
        else:
            schema = schema or self.default_schema

        inspector = inspect(self.engine)
        try:
            return inspector.get_pk_constraint(table_name, schema=schema)[
                "constrained_columns"
            ]
        except:
            return []  # Some databases may not support PK constraints

    def table_exists(self, table_name: str, schema: Optional[str] = None) -> bool:
        """Check if a table exists."""
        if "." in table_name:
            schema, table_name = table_name.split(".")
        else:
            schema = schema or self.default_schema

        return table_name in self.get_table_names(schema=schema)

    # PostgreSQL-specific methods
    def get_extensions(self) -> List[str]:
        """Get list of installed PostgreSQL extensions (PostgreSQL only)."""
        if self.db_type != "postgresql":
            raise NotImplementedError(
                "This method is only available for PostgreSQL connections"
            )

        with self.engine.connect() as conn:
            result = conn.execute("SELECT extname FROM pg_extension")
            return [row[0] for row in result]

    def execute_query(
        self, query: str, fetch_results: bool = True, params: Optional[Dict] = None
    ) -> Union[List[tuple], None]:
        """
        Executes a SQL query (works for both PostgreSQL and Trino).

        Args:
            query: SQL query to execute
            fetch_results: Whether to fetch results
            params: Parameters for parameterized queries

        Returns:
            Results as list of tuples or None
        """
        try:
            with self.engine.connect() as connection:
                stmt = text(query)
                result = (
                    connection.execute(stmt, params)
                    if params
                    else connection.execute(stmt)
                )

                if fetch_results and result.returns_rows:
                    return result.fetchall()
                return None
        except SQLAlchemyError as e:
            print(f"Error executing query: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Tests the database connection (works for both PostgreSQL and Trino).

        Returns:
            True if connection successful, False otherwise
        """
        test_query = (
            "SELECT 1"
            if self.db_type == "postgresql"
            else "SELECT 1 AS connection_test"
        )

        try:
            print(
                f"Attempting to connect to {self.db_type} at {self.host}:{self.port}..."
            )
            with self.engine.connect() as conn:
                conn.execute(text(test_query))
            print(f"Successfully connected to {self.db_type.upper()}.")
            return True
        except Exception as e:
            print(f"Failed to connect to {self.db_type.upper()}: {e}")
            return False

    def read_sql_to_dataframe(
        self, query: str, params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Executes query and returns results as pandas DataFrame (works for both).

        Args:
            query: SQL query to execute
            params: Parameters for parameterized queries

        Returns:
            pandas DataFrame with results
        """
        try:
            with self.engine.connect() as connection:
                return pd.read_sql_query(text(query), connection, params=params)
        except SQLAlchemyError as e:
            print(f"Error reading SQL to DataFrame: {e}")
            raise

    def read_sql_to_dask_dataframe(
        self,
        table_name: str,
        index_col: str,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Reads data to Dask DataFrame (works for both, but connection string differs).

        Args:
            table_name: Table name (schema.table or just table)
            columns: List of columns to select
            limit: Maximum rows to return
            **kwargs: Additional arguments

        Returns:
            Dask DataFrame with results
        """
        try:
            connection_string = self.get_connection_string()

            # Handle schema.table format
            if "." in table_name:
                schema, table = table_name.split(".")
            else:
                schema = self.default_schema
                table = table_name

            metadata = MetaData()
            table_obj = Table(table, metadata, schema=schema, autoload_with=self.engine)

            # Build query
            query = (
                select(*[table_obj.c[col] for col in columns])
                if columns
                else select(table_obj)
            )
            if limit:
                query = query.limit(limit)

            return dd.read_sql_query(
                sql=query, con=connection_string, index_col=index_col, **kwargs
            )
        except Exception as e:
            print(f"Error reading SQL to Dask DataFrame: {e}")
            raise ValueError(f"Failed to read SQL to Dask DataFrame: {e}") from e
